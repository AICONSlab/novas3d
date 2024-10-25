from monai.utils import (
    ensure_tuple
)
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    EnsureType,
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNETR
from monai.networks.layers import Norm
from monai.inferers import sliding_window_inference
from monai.data import ( 
    DataLoader, 
    Dataset, 
    ImageReader
)
from monai.config import (
    KeysCollection, 
    PathLike
)
import torch
import matplotlib.pyplot as plt
from os.path import exists
import os
from pathlib import Path
from re import sub
from skimage.io import imread
from typing import (
    Optional, 
    Union, 
    Sequence,  
    Dict, 
    List
)
from monai.data.utils import is_supported_format
from monai. data.image_reader import (
    _copy_compatible_dict, 
    _stack_images
)
from nibabel.nifti1 import Nifti1Image
from numpy import swapaxes, ndarray, eye, asarray, delete, empty, float16, save
from tqdm import tqdm
from monai.data import DataLoader, Dataset
from torch import no_grad
import pickle
import warnings
warnings.filterwarnings('ignore')



class TIFFReader(ImageReader):
    
    def __init__(self, npz_keys: Optional[KeysCollection] = None, channel_dim: Optional[int] = None, **kwargs):
        super().__init__()
        if npz_keys is not None:
            npz_keys = ensure_tuple(npz_keys)
        self.npz_keys = npz_keys
        self.channel_dim = channel_dim
        self.kwargs = kwargs
    
    def verify_suffix(self, filename: Union[Sequence[PathLike], PathLike]) -> bool:
            """
            Verify whether the specified file or files format is supported by Numpy reader.

            Args:
                filename: file name or a list of file names to read.
                    if a list of files, verify all the suffixes.

            Returns:
                bool: True if the file format is supported, False otherwise.
            """
            suffixes: Sequence[str] = ["tif", "tiff"]
            return is_supported_format(filename, suffixes)

    def read(self, data: Union[Sequence[PathLike], PathLike], **kwargs):
            """
            Read image data from specified file or files, it can read a list of `no-channel` data files
            and stack them together as multi-channels data in `get_data()`.
            Note that the returned object is Numpy array or list of Numpy arrays.

            Args:
                data (Union[Sequence[PathLike], PathLike]): The file name or a list of file names to read.
                **kwargs: Additional arguments for the `numpy.load` API except `allow_pickle`, which will override `self.kwargs` for existing keys.
                    More details about available arguments can be found here: https://numpy.org/doc/stable/reference/generated/numpy.load.html

            Returns:
                Union[numpy.ndarray, List[numpy.ndarray]]: The loaded image data as a Numpy array or a list of Numpy arrays.

            """
            img_: List[Nifti1Image] = []

            filenames: Sequence[PathLike] = ensure_tuple(data)
            kwargs_ = self.kwargs.copy()
            kwargs_.update(kwargs)
            for name in filenames:
                img = imread(name, **kwargs_)
                img = img.astype('float32')
                if len(img.shape)==4:
                    img = swapaxes(img,0,1)
                    img = swapaxes(img,1,3)
                img_.append(img)
            return img_ if len(img_) > 1 else img_[0]
    
    def get_data(self, img):
            """
            Extract data array and meta data from loaded image and return them.
            This function returns two objects, first is numpy array of image data, second is dict of meta data.
            It constructs `affine`, `original_affine`, and `spatial_shape` and stores them in meta dict.
            When loading a list of files, they are stacked together at a new dimension as the first dimension,
            and the meta data of the first image is used to represent the output meta data.

            Args:
                img: a Numpy array loaded from a file or a list of Numpy arrays.

            Returns:
                Tuple: A tuple containing the following two objects:
                    - img_array (numpy.ndarray): Numpy array of image data.
                    - compatible_meta (dict): Dictionary of meta data.

            """
            img_array: List[ndarray] = []
            compatible_meta: Dict = {}
            if isinstance(img, ndarray):
                img = (img,)

            for i in ensure_tuple(img):
                header = {"affine":eye(5),
                         "labels": {"0": "background",
                                    "1": "vessels",
                                    "2": "neurons",}
                         }
                if isinstance(i, ndarray):
                    # if `channel_dim` is None, can not detect the channel dim, use all the dims as spatial_shape
                    spatial_shape = asarray(i.shape)
                    if isinstance(self.channel_dim, int):
                        spatial_shape = delete(spatial_shape, self.channel_dim)
                    header["spatial_shape"] = spatial_shape
                img_array.append(i)
                header["original_channel_dim"] = self.channel_dim if isinstance(self.channel_dim, int) else "no_channel"
                _copy_compatible_dict(header, compatible_meta)

            return _stack_images(img_array, compatible_meta), compatible_meta


def get_model(parameter_file='NOVAS3D_Vessel_and_Neuron_Segmentation/parameters.pickle',
              spatial_dims=3,
              in_channels=2,
              out_channels=3,
              img_size=(128, 128, 128),
              feature_size=16,
              hidden_size=768,
              mlp_dim=3072,
              pos_embed="perceptron",
              res_block=True,
              norm_name="instance",
              gpu = False):
    """
    Get the UNETR model for prediction.

    Args:
        parameter_file (str): Path to the parameter pickle file.
        spatial_dims (int): Number of spatial dimensions.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        img_size (tuple): Size of the input image.
        feature_size (int): Size of the feature maps.
        hidden_size (int): Size of the hidden layers.
        mlp_dim (int): Dimension of the MLP layers.
        pos_embed (str): Type of positional embedding.
        res_block (bool): Whether to use residual blocks.
        norm_name (str): Name of the normalization layer.

    Returns:
        torch.nn.Module: The UNETR model for prediction.
    """

    # Load parameters from pickle file
    with open(parameter_file, 'rb') as handle:
        params = pickle.load(handle)

    # Set device for model prediction
    device = torch.device("cuda" if (torch.cuda.is_available() and gpu == True) else "cpu")

    # Create UNETR model
    model = UNETR(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        img_size=img_size,
        feature_size=feature_size,
        hidden_size=hidden_size,
        mlp_dim=mlp_dim,
        pos_embed=pos_embed,
        res_block=res_block,
        norm_name=norm_name,
        dropout_rate=params["dropout"]
    )

    # Move model to device
    model.to(device)

    if (torch.cuda.is_available() and gpu == True):
        # Use DataParallel for multi-GPU training
        model = torch.nn.DataParallel(model)
        # Load pre-trained model weights
        model.load_state_dict(torch.load(
            sub('parameters.pickle', "model.pth",parameter_file)))

    else: 
        from collections import OrderedDict
        state_dict = torch.load(sub('parameters.pickle', "model.pth",parameter_file), map_location='cpu')
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace("module.", "")] = v
        model.load_state_dict(new_state_dict)
        # Load pre-trained model weights
        #model.load_state_dict(torch.load(
        #    sub('parameters.pickle', "model.pth",parameter_file),
        #    map_location=torch.device('cpu')))
    model.eval()

    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
    return model

def predict(pred_data, 
            num_evals=20,
            num_channels_out=3, 
            model = get_model(), 
            device='cpu',
            roi_size = (128, 128, 128),
            sw_batch_size = 64):
    """
    Perform prediction using a sliding window inference approach.

    Args:
        pred_data (dict): Dictionary containing the input data for prediction.
        num_evals (int, optional): Number of evaluations to perform. Defaults to 20.
        num_channels_out (int, optional): Number of output channels. Defaults to 3.
        model (nn.Module, optional): Model to use for prediction. Defaults to get_model().
        device (str, optional): Device to use for computation. Defaults to 'cpu'.
        roi_size (tuple, optional): Size of the region of interest. Defaults to (128, 128, 128).
        sw_batch_size (int, optional): Batch size for sliding window inference. Defaults to 64.

    Returns:
        numpy.ndarray: Array containing the predicted outputs.
    """
    # Get the shape of the input image
    _shape = pred_data["image"].shape

    # Create an empty array to store the predicted outputs
    pred_array = empty((num_evals,num_channels_out,_shape[2],_shape[3],_shape[4]),dtype=float16)

    # Perform sliding window inference for the specified number of evaluations
    for j in range(num_evals):
        # define softmax for the predicted outputs
        softmax = torch.nn.Softmax(dim=1)
        # Perform sliding window inference
        pred_outputs = sliding_window_inference(
            pred_data["image"],
            roi_size, 
            sw_batch_size, 
            model,
            sw_device=device,
            device=device
        )
        
        # Apply softmax to the predicted outputs
        pred_outputs = softmax(pred_outputs)
        
        # Convert the predicted outputs to numpy array of float16 data type
        pred_outputs = float16(pred_outputs.cpu().detach().numpy())
        
        # Store the predicted outputs in the pred_array
        pred_array[j] = pred_outputs[:]
        del pred_outputs

    return pred_array


def get_pred_transforms(spacing=(1, 1, 0.375), i_min=0, i_max=1024, b_min=0.0, b_max=1.0, clip=True, channel_dim=0):
    """
    Returns a composition of transforms for preprocessing the input image for prediction.

    Args:
        spacing (tuple): Voxel spacing in the input image. Default is (1, 1, 0.375).
        i_min (int): Minimum intensity value for rescaling. Default is 0.
        imax (int): Maximum intensity value for rescaling. Default is 1024.
        b_min (float): Minimum intensity value for scaling. Default is 0.0.
        b_max (float): Maximum intensity value for scaling. Default is 1.0.
        clip (bool): Whether to clip the intensity values. Default is True.
        channel_dim (int): Dimension index for channel. Default is 0.

    Returns:
        pred_transforms (Compose): Composition of transforms for preprocessing the input image.
    """
    # Define a composition of transforms
    pred_transforms = Compose(
        [
            LoadImaged(keys=["image"], reader=TIFFReader(channel_dim=channel_dim)),  # Load the image using TIFFReader
            EnsureChannelFirstd(keys=["image"]),  # Ensure the channel dimension is first
            Spacingd(keys=["image"], pixdim=spacing, mode=("bilinear")),  # Resample the image with the given spacing
            ScaleIntensityRanged(
                keys=["image"], a_min=i_min, a_max=i_max, b_min=b_min, b_max=b_max, clip=clip
            ),  # Rescale the intensity values of the image
            EnsureTyped(keys=["image"]),  # Ensure the image is of the correct data type
        ]
    )
    return pred_transforms

def get_post_transforms():
    """
    Returns the post-processing transforms for the prediction.

    Returns:
        Compose: A composition of post-processing transforms.
    """
    # Define post-processing transforms for prediction
    post_pred = Compose(
        [
            EnsureType(), 
            AsDiscrete(argmax=True, to_onehot=3)
        ]
    )
    return post_pred

class PredictWarped:
    """
    Class for performing prediction using the registered images. Saves the predicted images.

    Args:
        data_dict (dict): A dictionary containing the data for prediction.
        config (dict): A dictionary containing the configuration parameters.
            The configuration parameters include:
                - in_dir (str): The input directory.
                - out_dir (str): The output directory.
                - num_evals (int): The number of evaluations.
                - base_file_extension (str): The base file extension.
                - pred_file_extension (str): The prediction file extension.
                - mean_file_extension (str): The mean file extension.
        parameter_file (str): The file path to the parameter file.
        spatial_dims (int): The number of spatial dimensions.
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        img_size (tuple): The size of the input image.
        feature_size (int): The size of the features.
        hidden_size (int): The size of the hidden layers.
        mlp_dim (int): The dimension of the MLP layers.
        pos_embed (bool): Whether to use positional embedding.
        res_block (bool): Whether to use residual blocks.
        norm_name (str): The name of the normalization method.
        spacing (tuple): The spacing between voxels. Reciprocal of the voxel size in um to resample to 1 um.
        i_min (float): The minimum intensity value of raw images.
        i_max (float): The maximum intensity value of raw images.
        b_min (float): The minimum background value after normalization.
        b_max (float): The maximum background value after noramlization.
        clip (bool): Whether to clip the values.
        channel_dim (int): The dimension index of the colour channel.
        gpu (bool): Whether to use the GPU.
        skip_existing (bool): Whether to skip images that have already been predicted.
    
    Methods:
        get_model(): Retrieves the model for prediction.
        get_pred_transforms(): Retrieves the prediction transforms.
        dataset_creation(): Creates the dataset for prediction.
        prediction(): Performs the prediction.

    """

    def __init__(self, data_dict, config, parameter_file, spatial_dims, in_channels, out_channels, img_size, feature_size, hidden_size, mlp_dim, pos_embed, res_block, norm_name,spacing, i_min, i_max, b_min, b_max, clip, channel_dim, gpu=False, skip_existing=False):
        self.data_dict = data_dict
        self.config = config
        self.parameter_file = parameter_file
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_size = img_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.mlp_dim = mlp_dim
        self.pos_embed = pos_embed
        self.res_block = res_block
        self.norm_name = norm_name
        self.spacing = spacing
        self.i_min = i_min
        self.i_max = i_max
        self.b_min = b_min
        self.b_max = b_max
        self.clip = clip
        self.channel_dim = channel_dim
        self.gpu = gpu
        self.skip_existing = skip_existing

    def get_model(self):
        """
        Get the model for prediction.

        Returns:
            model: The model for prediction.
        """
        # Get the model for prediction
        model = get_model(self.parameter_file, 
                          self.spatial_dims,
                          self.in_channels, 
                          self.out_channels, 
                          self.img_size, 
                          self.feature_size, 
                          self.hidden_size, 
                          self.mlp_dim, 
                          self.pos_embed, 
                          self.res_block, 
                          self.norm_name,
                          self.gpu)
        return model

    def get_pred_transforms(self):
        """
        Get the prediction transforms.

        Returns:
            pred_transforms (list): List of prediction transforms.
        """
        # Get the prediction transforms
        pred_transforms = get_pred_transforms(self.spacing, 
                                              self.i_min, 
                                              self.i_max, 
                                              self.b_min, 
                                              self.b_max, 
                                              self.clip, 
                                              self.channel_dim)
        return pred_transforms

    def dataset_creation(self):
        """
        Create the dataset for prediction.

        Returns:
            pred_ds (Dataset): The dataset for prediction.
        """
        # Create the dataset for prediction
        data_dicts = self.data_dict
        pred_transforms = get_pred_transforms(self.spacing, self.i_min, self.i_max, self.b_min, self.b_max, self.clip, self.channel_dim)
        pred_ds = Dataset(data=data_dicts, transform=pred_transforms)
        self.assertEqual(len(pred_ds), 3)
        return pred_ds

    def prediction(self):
        """
        Perform prediction using the provided model and saves the data.

        Returns:
            None
        """
        # Perform the prediction
        model = get_model(self.parameter_file, self.spatial_dims, self.in_channels, self.out_channels, self.img_size, self.feature_size, self.hidden_size, self.mlp_dim, self.pos_embed, self.res_block, self.norm_name)
        pred_transforms = get_pred_transforms(self.spacing, self.i_min, self.i_max, self.b_min, self.b_max, self.clip, self.channel_dim)
        pred_ds = Dataset(data=self.data_dict, transform=pred_transforms)
        pred_loader = DataLoader(pred_ds, batch_size=1, shuffle=False)
        config = self.config
        with no_grad():
            for i, pred_data in tqdm(enumerate(pred_loader)):
                if self.skip_existing==False:
                    new_file_name = sub(config["in_dir"], config["out_dir"], self.data_dict[i]["image"])
                    pred_array = predict(pred_data, num_evals=config["num_evals"], model=model)
                    #self.assertIsNotNone(pred_array)
                    mean = float16(pred_array.mean(axis=0))
                    save(sub(config["base_file_extension"], config['mean_file_extension'], new_file_name), mean)
                    save(sub(config["base_file_extension"], config["pred_file_extension"], new_file_name), pred_array)
                    
                elif not exists(sub(config["base_file_extension"], config["pred_file_extension"], sub(config["in_dir"], config["out_dir"], self.data_dict[i]["image"]))):
                    new_file_name = sub(config["in_dir"], config["out_dir"], self.data_dict[i]["image"])
                    pred_array = predict(pred_data, num_evals=config["num_evals"], model=model)
                    #self.assertIsNotNone(pred_array)
                    mean = float16(pred_array.mean(axis=0))
                    save(sub(config["base_file_extension"], config['mean_file_extension'], new_file_name), mean)
                    save(sub(config["base_file_extension"], config["pred_file_extension"], new_file_name), pred_array)