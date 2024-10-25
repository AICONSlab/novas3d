from numpy import uint32, int64, save, float16, zeros, load, std, copy, sum, swapaxes, load, mgrid, dstack, array, diag, copy, int16, mean, max, float32, float64, gradient, arange, meshgrid, zeros, vstack, reshape, dot, cos, deg2rad, sin, less, nanargmin, isnan, where, logical_and, sqrt, square,uint8
from numpy.random import shuffle, uniform
from skimage.io import imread, imsave
from pathlib import Path
from tqdm import tqdm
from skimage.morphology import binary_dilation, binary_closing
from scipy.ndimage import distance_transform_edt, zoom, gaussian_filter1d, map_coordinates, gaussian_filter
from scipy.signal import argrelextrema
from networkx import read_gpickle, write_gpickle, selfloop_edges, isolates
from multiprocessing.pool import ThreadPool
from scipy.stats import multivariate_normal
from skimage.restoration import richardson_lucy
from skimage.feature import peak_local_max
from novas3d.utilities import fill_holes, remove_small_comps_3d, get_mov_files, remove_terminal_edges, _rotmat, closest_node
from sknw import build_sknw
from scipy.io import savemat, loadmat
from os.path import exists
from re import sub
from scipy.signal import butter, filtfilt







def save_std_file(file,
                  in_directory='tmp',
                  out_directory='tmp',
                  in_tag='pred',
                  final_tag='std',
                  skip_existing=False):
    """
    Save the standard deviation file.

    Args:
        file (str): The path to the file containing the predicted data.
        in_directory (str): The input directory path.
        out_directory (str): The output directory path.
        in_tag (str): The initial tag for predicted data.
        final_tag (str): The final tag for standard deviation data.

    Returns:
        None
    """

    # Check if the standard deviation file already exists
    if skip_existing==False or not exists(sub(in_directory, out_directory, sub(in_tag, final_tag, file))):
        # Load the predicted data
        pred = load(file)
        # Calculate the standard deviation along the first axis
        _std = std(pred, axis=0)
        # Save the standard deviation file
        save(sub(in_directory, out_directory, sub(in_tag, final_tag, file)), _std)
        


class SaveStdFile:
    """
    Initializes an instance of the SaveStdFile class. Calculates the standard deviation for predicted data. Saves the standard deviation files.

    Args:
        files (list): List of file paths.
        in_directory (str): Input directory path.
        out_directory (str): Output directory path.
        in_tag (str): Initial tag.
        final_tag (str): Final tag.
        skip_existing (bool): Skip files where std already exists.
    
    Methods:
        std(): Save the standard deviation files.  
    """
    def __init__(self, files, in_directory, out_directory, in_tag, final_tag,skip_existing=False):
        self.files = files
        self.in_directory = in_directory
        self.out_directory = out_directory
        self.in_tag = in_tag
        self.final_tag = final_tag
        self.skip_existing = skip_existing

    def std(self):
        """
        Calculate the standard deviation for each file in the files list.

        Returns:
            None
        """
        # Iterate over the files list
        for file in tqdm(self.files):
            # Call the save_std_file function with the file path and other parameters
            save_std_file(file, self.in_directory, self.out_directory, self.in_tag, self.final_tag,self.skip_existing)



class Binarize:
    """
    Initializes the Binarize class. This class is used to binarize the files in the specified directory based on mean and standard deviation thresholds.

    Args:
        directory (str): The directory path.
        min_prob (float): The minimum probability threshold.
        max_var (float): The maximum variance threshold.
        in_directory (str): The input directory path.
        out_directory (str): The output directory path.
        mean_tag (str): The mean tag.
        std_tag (str): The standard deviation tag.
        seg_tag (str): The segmentation tag.
        skip_existing (bool): Skip existing files that have already been binarized.
    
    methods:
        binarize_files(): Binarizes the files in the directory.
        get_files(): Retrieves the list of files in the directory.
    """

    """
    Binarizes the files in the directory.

    This method performs the following steps:
    1. Retrieves the list of files.
    2. Shuffles the list of files.
    3. Iterates over each file.
    4. Checks if the segmented file already exists.
    5. If the segmented file does not exist, checks if the standard deviation file exists.
    6. If the standard deviation file exists, loads the mean and standard deviation files.
    7. Creates an empty segmentation array.
    8. Segments the image based on mean and standard deviation thresholds.
    9. Fills holes in the segmented image.
    10. Removes small connected components from the segmented image.
    11. Saves the segmented image.
    """

    def __init__(self, files, min_prob, max_var, in_directory, out_directory, mean_tag, std_tag, seg_tag, skip_existing=False):
        self.files = files
        self.min_prob = min_prob
        self.max_var = max_var
        self.in_directory = in_directory
        self.out_directory = out_directory
        self.mean_tag = mean_tag
        self.std_tag = std_tag
        self.seg_tag = seg_tag
        self.skip_existing = skip_existing

    def binarize_files(self):
        """
        Binarizes the files in the specified directory based on mean and standard deviation thresholds.

        This method iterates through the files in the directory and performs the following steps for each file:
        1. Checks if the segmented file already exists.
        2. If the segmented file does not exist, checks if the standard deviation file exists.
        3. If the standard deviation file exists, loads the mean and standard deviation arrays from the files.
        4. Creates an empty segmentation array with the same shape as the mean array.
        5. Segments the image based on mean and standard deviation thresholds.
        6. Fills holes in the segmented image.
        7. Removes small connected components from the segmented image.
        8. Saves the segmented image.

        Parameters:
        - None

        Returns:
        - None
        """
        files = self.files
        shuffle(files)

        for file in tqdm(files):
            # Check if the segmented file already exists
            seg_file = sub(self.in_directory, self.out_directory, sub(self.mean_tag, self.seg_tag, file))
            print(seg_file)
            if self.skip_existing==False or not exists(seg_file):
                # Check if the standard deviation file exists
                std_file = sub(self.mean_tag,self.std_tag,file)
                print(std_file)
                if exists(std_file):
                    print(file)
                    mean = load(file)
                    std = load(std_file)
                    seg = zeros(mean.shape[1:]).astype('int8')
                    # Segment the image based on mean and standard deviation thresholds
                    seg[(mean[1,:,:,:] > self.min_prob) * (std[1,:,:,:] < self.max_var)] = 1
                    seg[(mean[2,:,:,:] > self.min_prob) * (std[2,:,:,:] < self.max_var)] = 2
                    seg = seg.astype('int8')
                    seg = (seg==1)*1
                    # Fill holes in the segmented image
                    seg = fill_holes(seg)
                    # Remove small connected components from the segmented image
                    seg = remove_small_comps_3d(seg)
                    print(seg.shape)
                    # Save the segmented image
                    save(seg_file, seg)
    

class NeuronDistanceCalculator:
    """
    Calculates the distance between neurons based on given parameters. Saves the distance transform of the Neuron segmentation array.

    Args:
        files (list): List of file paths.
        min_prob (float): Minimum probability threshold.
        max_var (float): Maximum variance threshold.
        in_directory (str): Input directory path.
        out_directory (str): Output directory path.
        mean_tag (str): Tag for mean values.
        std_tag (str): Tag for standard deviation values.
        neuron_distance_tag (str): Tag for neuron distance values.
        seg_tag (str): Tag for segmentation values.
        seg_nrn_tag (str): Tag for segmented neuron values.
        skip_existing (bool): Whether to skip images that have already been predicted.

    Methods:
        calculate_distance(): Calculates the distance between neurons for each file in the given list.
    """
    def __init__(self, files, min_prob, max_var, in_directory, out_directory, mean_tag, std_tag, neuron_distance_tag, seg_tag, seg_nrn_tag,skip_existing=False):
        self.files = files
        self.min_prob = min_prob
        self.max_var = max_var
        self.in_directory = in_directory
        self.out_directory = out_directory
        self.mean_tag = mean_tag
        self.std_tag = std_tag
        self.neuron_distance_tag = neuron_distance_tag
        self.seg_tag = seg_tag
        self.seg_nrn_tag = seg_nrn_tag
        self.skip_existing = skip_existing

    def calculate_distance(self):
        """
        Calculate the distance between neurons in the given files.

        This method shuffles the list of files and iterates over each file.
        For each file, it checks if the corresponding output file does not exist.
        If it doesn't exist, it loads the mean and standard deviation arrays from the file.
        It then creates a segmentation array based on certain conditions.
        The segmentation array is saved as a binary mask in the output directory.
        Additionally, the distance transform of the segmentation array is calculated and saved as well.

        Args:
            None

        Returns:
            None
        """
        files = self.files
        shuffle(files)

        for file in tqdm(files):
            if not exists(sub(self.in_directory, self.out_directory, sub(self.mean_tag, self.neuron_distance_tag, file))) or self.skip_existing==False:
                if exists(sub(self.mean_tag, self.std_tag, file)):
                    mean = load(file)
                    std = load(sub(self.mean_tag, self.std_tag, file))
                    seg = zeros(mean.shape[1:])
                    seg[(mean[1, :, :, :] > self.min_prob) * (std[1, :, :, :] < self.max_var)] = 1
                    seg[(mean[2, :, :, :] > self.min_prob) * (std[2, :, :, :] < self.max_var)] = 2
                    seg = seg.astype('int8')
                    seg = (seg == 2) * 1
                    save(sub(self.in_directory, self.out_directory, sub(self.mean_tag, self.seg_nrn_tag, file)), seg)
                    save(sub(self.in_directory, self.out_directory, sub(self.mean_tag, self.neuron_distance_tag, file)),
                         distance_transform_edt(1 - seg).astype('float16'))
                    




class SegmentationMaskUnion:
    def __init__(self, files, dic, in_directory, out_directory, in_suffix, final_suffix_mat, final_suffix_npy, timepoint_suffixes, IOU_thresh, thresh_fill_holes, thresh_remove_small_comps, n_interations_binary_closing,skip_existing=False):
        self.files = files
        self.dic = dic
        self.in_directory = in_directory
        self.out_directory = out_directory
        self.in_suffix = in_suffix
        self.final_suffix_mat = final_suffix_mat
        self.final_suffix_npy = final_suffix_npy
        self.timepoint_suffixes = timepoint_suffixes
        self.IOU_thresh = IOU_thresh
        self.thresh_fill_holes = thresh_fill_holes
        self.thresh_remove_small_comps = thresh_remove_small_comps
        self.n_interations_binary_closing = n_interations_binary_closing
        self.skip_existing = skip_existing

    # Define a method called process_images
    def process_images(self):
        """
        Process a list of images by performing binary dilation, calculating intersection over union (IOU),
        performing post-processing, and saving the processed images as .mat and .npy files.

        Args:
            files (list): List of image file paths.
            dic (dict): Dictionary containing additional information.
            in_directory (str): Path to the input directory.
            out_directory (str): Path to the output directory.
            in_suffix (str): Suffix of the initial files.
            final_suffix_mat (str): Suffix of the final .mat files.
            final_suffix_npy (str): Suffix of the final .npy files.
            timepoint_suffixes (list): List of timepoint suffixes. ex: ['_0001', '_0002', '_0003']
            IOU_thresh (float): Intersection over Union (IOU) threshold.
            thresh_fill_holes (int): Threshold for filling holes in the segmented image.
            thresh_remove_small_comps (int): Threshold for removing small connected components in the segmented image.
            n_interations_binary_closing (int): Number of iterations for binary closing.
            skip_existing (bool): Skip existing files that have already been processed.

        Returns:
            None
        """
        # Iterate over each image in the list of files
        for image in tqdm(self.files):
            # Check if the processed image already exists
            if self.skip_existing==False or not exists(sub(self.in_directory, self.out_directory, sub(self.in_suffix, self.final_suffix_mat, sub('_\d{4}', '', image)))):
                # Load the image and perform binary dilation
                img = binary_dilation(load(image))
                _img = copy(img)
                # Get the list of moving files
                mov_files = get_mov_files(image, self.dic, self.in_suffix, self.timepoint_suffixes)
                # Iterate over each moving file
                for i in mov_files:
                    # Load the moving file and perform binary dilation
                    img_0001 = binary_dilation(load(i))
                    # Calculate the overlap and union of the images
                    overlap = (_img != 0) * (img_0001 != 0)  # Logical AND
                    union = (_img != 0) + (img_0001 != 0)  # Logical OR
                    # Calculate the Intersection over Union (IOU)
                    IOU = overlap.sum() / float(union.sum())
                    print(IOU)
                    # Check if the IOU is above the threshold
                    if IOU > self.IOU_thresh:
                        img += img_0001
                # Perform post-processing on the segmented image
                seg = img
                seg[seg != 0] = seg[seg != 0] / seg[seg != 0]
                seg = (seg == 1) * 1
                seg = seg.astype('int8')
                seg = fill_holes(seg, thresh=self.thresh_fill_holes)
                for i in range(self.n_interations_binary_closing):
                    seg = binary_closing(seg)
                seg = remove_small_comps_3d(seg, thresh=self.thresh_remove_small_comps)
                # Save the processed image as a .mat file
                savemat(sub(self.in_directory, self.out_directory ,sub(self.in_suffix, self.final_suffix_mat, sub('_\d{4}', '', image))),
                         {'FinalImage': fill_holes(binary_closing(seg), thresh=self.thresh_fill_holes)})
                # Save the processed image as a .npy file
                save(sub(self.in_directory, self.out_directory, sub(self.in_suffix, self.final_suffix_npy, sub('_\d{4}', '', image))),
                     fill_holes(binary_closing(seg), thresh=self.thresh_fill_holes))

class GraphGenerator:
    """
    Generates graphs based on input files and parameters. Saves the graphs as pickle files. 

    Args:
        files (list): List of input file paths.
        dic (dict): Dictionary containing paried files. Key is the reference file and value is a list of matching files.
        in_directory (str): Path to the input directory.
        out_directory (str): Path to the output directory.
        in_suffix (str): Suffix of the initial files.
        final_suffix_pickle (str): Suffix of the final pickle files.
        final_suffix_mat (str): Suffix of the final mat files.
        final_suffix_tif (str): Suffix of the final tif files.
        ref_timepoint (str): Reference timepoint.
        timepoint_suffixes (list): List of timepoint suffixes.
        IOU_thresh (float): IOU threshold.
        thresh_remove_terminal_segemnts (int): Threshold to remove terminal segments.
        skip_existing (bool): Skip existing files that have already been processed.

    Methods:
        process_files(): Process the input files and generate graphs.

    """

    def __init__(self, files, dic, in_directory, out_directory, in_suffix, final_suffix_pickle, in_suffix_mat, final_suffix_tif, ref_timepoint, timepoint_suffixes, IOU_thresh, thresh_remove_terminal_segemnts, skip_existing=False):
        self.files = files
        self.dic = dic
        self.in_directory = in_directory
        self.out_directory = out_directory
        self.in_suffix = in_suffix
        self.final_suffix_pickle = final_suffix_pickle
        self.in_suffix_mat = in_suffix_mat
        self.final_suffix_tif = final_suffix_tif
        self.ref_timepoint = ref_timepoint
        self.timepoint_suffixes = timepoint_suffixes
        self.IOU_thresh = IOU_thresh
        self.thresh_remove_terminal_segemnts = thresh_remove_terminal_segemnts
        self.skip_existing = skip_existing

    def generate_graphs(self):
        """
        Process the files in the specified directory.

        This method performs the following steps:
        1. Iterate over each file in the directory.
        2. Check if the corresponding output file does not exist.
        3. If the output file does not exist, check if the corresponding input file exists.
        4. If the input file exists, load the skeleton image from the pickle file.
        5. If the skeleton image is not empty, build a graph representation of the skeleton.
        6. Remove self-loop edges from the graph.
        7. Remove terminal edges from the graph.
        8. Save the skeleton image and the graph as pickle files.
        9. For each movement file associated with the current file:
            a. Check if the movement file exists.
            b. If the movement file exists, load the segmented image.
            c. Calculate the intersection over union (IOU) between the segmented images.
            d. If the IOU is above the threshold, save the graph as a pickle file for the movement file.

        Note: This method assumes the existence of certain variables and functions, such as `self.files`, `self.in_directory`, `self.out_directory`, `self.in_suffix`, `self.final_suffix_pickle`, `self.ref_timepoint`, `self.final_suffix_mat`, `self.thresh_remove_terminal_segemnts`, `self.final_suffix_tif`, `self.dic`, `self.timepoint_suffixes`, `self.IOU_thresh`, `loadmat`, `build_sknw`, `selfloop_edges`, `remove_terminal_edges`, `imsave`, `write_gpickle`, `get_mov_files`, `load`, `binary_dilation`, `exists`, `sub`, and `isolates`. Please make sure these variables and functions are defined before calling this method.
        """
        
        for file in tqdm(self.files):
            # Check if the output file already exists
             if self.skip_existing==False or not exists(sub(self.in_directory, self.out_directory, sub(self.in_suffix, self.final_suffix_pickle, file))):
            # Check if the corresponding input file exists
                if exists(sub(self.ref_timepoint + self.in_suffix, self.in_suffix_mat, file)):
                    # Load the skeleton image from the pickle file
                    skel_file = sub(self.ref_timepoint + self.in_suffix, self.in_suffix_mat, file) 
                    skel = loadmat(skel_file)['FilteredImage']
                    skel = skel.astype(int64)
                    # Check if the skeleton image is not empty
                    if sum(skel) != 0:
                        # Build a graph representation of the skeleton
                        graph = build_sknw(skel.astype(uint32), multi=False)
                        print(len(graph.edges))
                        # Remove self-loop edges from the graph
                        while len(list(selfloop_edges(graph))) > 0:
                            if len(list(selfloop_edges(graph))) != 0:
                                for edge in list(selfloop_edges(graph)):
                                    skel[graph[edge[0]][edge[1]]['pts'][1:-1, 0], graph[edge[0]][edge[1]]['pts'][1:-1, 1], graph[edge[0]][edge[1]]['pts'][1:-1, 2]] = 0
                                # Remove terminal edges from the graph
                                skel = remove_terminal_edges(graph, skel, self.thresh_remove_terminal_segemnts)
                                graph = build_sknw(skel.astype(uint32), multi=False)
                        skel = remove_terminal_edges(graph, skel, self.thresh_remove_terminal_segemnts)
                        graph = build_sknw(skel.astype(uint32), multi=False)
                        # Remove isolated nodes from the graph
                        graph.remove_nodes_from(list(isolates(graph)))
                        print(len(graph.edges))
                        # Save the skeleton image as a tif file
                        imsave(sub(self.in_directory, self.out_directory,sub(self.ref_timepoint + self.in_suffix, self.final_suffix_tif, file)), skel)
                        # Save the graph as a pickle file
                        write_gpickle(graph, sub(self.in_directory, self.out_directory, sub(self.in_suffix, self.final_suffix_pickle, file)))
                        # Process movement files associated with the current file
                        print(file)
                        mov_files = get_mov_files(file, self.dic, self.in_suffix, self.timepoint_suffixes)
                        print(mov_files)
                        #seg = load(sub('_\d{4}','',file))
                        seg = load(file)
                        seg = binary_closing(seg)
                        seg = fill_holes(seg)
                        seg = binary_dilation(seg)
                        seg = fill_holes(seg)
                        seg = remove_small_comps_3d(seg)
                        for _file in mov_files:
                            if exists(_file):
                                seg_0001 = load(_file)
                                seg_0001 = binary_closing(seg_0001)
                                seg_0001 = fill_holes(seg_0001)
                                seg_0001 = binary_dilation(seg_0001)
                                seg_0001 = fill_holes(seg_0001)
                                seg_0001 = remove_small_comps_3d(seg_0001)
                                # Calculate the intersection over union (IOU) between the segmented images
                                overlap = (seg != 0) * (seg_0001 != 0)  # Logical AND
                                union = (seg != 0) + (seg_0001 != 0)  # Logical OR
                                IOU = overlap.sum() / float(union.sum())
                                print(IOU)
                                # If the IOU is above the threshold, save the graph as a pickle file for the movement file
                                if IOU > self.IOU_thresh:
                                    write_gpickle(graph, sub(self.in_directory, self.out_directory, sub(self.in_suffix, self.final_suffix_pickle, _file)))
                                else:
                                    print('IOU too low')

class VesselRadiiCalc:

    '''
    Initializes an instance of the VesselRadiiCalc class. Calculates the radii of vessels based on given parameters. Saves the radii in the graph.

    Args:
        files (list): List of file paths.
        in_directory (str): Input directory path.
        img_directory (str): Image directory path.
        mean_directory (str): Mean directory path.
        std_directory (str): Standard deviation directory path.
        out_directory (str): Output directory path.
        pickle_file_suffix (str): Suffix of the pickle files.
        out_pickle_suffix (str): Suffix of the output pickle files.
        img_suffix (str): Suffix of the image files.
        seg_suffix (str): Suffix of the segmented files.
        mean_suffix (str): Suffix of the mean files.
        std_suffix (str): Suffix of the standard deviation files.
        neuron_distance_suffix (str): Suffix of the neuron distance files.
        second_channel (bool): Second channel flag. Denotes the presence of a second channel.
        neuron_channel (bool): Neuron channel flag. Denotes the presence of a neuron channel.
        psf (list): Point spread function.
        spacing (float): Spacing value for pixel size along each axis.
        vessel_segment_limit (int): Vessel segment limit. If larger than the limit, the file is skipped.
        max_pixel_value (int): Maximum pixel value.
        n_iter_deconv (int): Number of iterations for deconvolution.
        grid_size_psf_deconv (int): Grid size for deconvolution. This must be an odd number.
        sampling (float): Sampling in microns to calculate the radii.
        n_cores (int): Number of cores.
        filter_radii (bool): Filter radii with a butterworth filter.
        butter_N (int): Butterworth filter order.
        butter_Wn (float): Butterworth filter frequency.
        butter_btype (str): Butterworth filter type.
        butter_fs (float): Butterworth filter frequency.
        skip_existing (bool): Skip existing files that have already been processed.
    
        return:
            Saves the radii in the graph, along with the standard deviation of the readius estimates along the path.
            If the filter_radii flag is set to True, the radii are filtered with a butterworth filter and saved under the path_weights key in the graph. The unfiltered radii are saved under the path_weights_unfiltered key in the graph.
            If the filter_radii flag is set to False, the radii are saved as is under the path_weights key in the graph.
            In both cases, the mean radii estimate is calculated from unfilterd radii estimates and saved under the radii key in the graph.
            The standard deviation of the radii estimates is calculated and saved under the radii_std key in the graph.
            If neuron_channel is set to True, the mean distance to the closest neuron is calculated and saved under the mean_neuron_distance key in the graph.
            If neuron_channel is set to True, the standard deviation of the distance to the closest neuron is calculated and saved under the neuron_distance_std key in the graph.
            If neuron_channel is set to True, the distance to the closest neuron is saved under the neuron_distance_min key in the graph.

    '''

    def __init__(self, files, in_directory, img_directory, mean_directory, std_directory, out_directory, pickle_file_suffix, out_pickle_suffix, img_suffix, seg_suffix, mean_suffix, std_suffix, neuron_distance_suffix, second_channel, neuron_channel, psf, spacing, vessel_segment_limit, max_pixel_value, n_iter_deconv, grid_size_psf_deconv, sampling, n_cores, filter_radii = False, butter_N = None, butter_Wn = None, butter_btype = None, butter_fs = None, skip_existing=False):
        self.files = files
        self.in_directory = in_directory
        self.img_directory = img_directory
        self.mean_directory = mean_directory
        self.std_directory = std_directory
        self.out_directory = out_directory
        self.pickle_file_suffix = pickle_file_suffix
        self.out_pickle_suffix = out_pickle_suffix
        self.img_suffix = img_suffix
        self.seg_suffix = seg_suffix
        self.mean_suffix = mean_suffix
        self.std_suffix = std_suffix
        self.neuron_distance_suffix = neuron_distance_suffix
        self.second_channel = second_channel
        self.neuron_channel = neuron_channel
        self.psf = psf
        self.spacing = spacing
        self.vessel_segment_limit = vessel_segment_limit
        self.max_pixel_value = max_pixel_value
        self.n_iter_deconv = n_iter_deconv
        self.grid_size_psf_deconv = grid_size_psf_deconv
        self.sampling = sampling
        self.n_cores = n_cores
        self.filter = filter_radii
        self.butter_N = butter_N
        self.butter_Wn = butter_Wn
        self.butter_btype = butter_btype
        self.butter_fs = butter_fs
        self.skip_existing = skip_existing

    def calc_radii_vessels(self, file):

        butter_b, butter_a = butter(self.butter_N, self.butter_Wn, btype=self.butter_btype, analog=False, fs=self.butter_fs)

        if self.skip_existing==False or not exists(sub(self.in_directory, self.out_directory, sub(self.pickle_file_suffix, self.out_pickle_suffix, file))):
            graph = read_gpickle(file)
            if len(graph.edges) < self.vessel_segment_limit:
                img_file = sub(self.in_directory, self.img_directory, sub(self.pickle_file_suffix, self.img_suffix, file))
                seg_file = sub(self.pickle_file_suffix, self.seg_suffix, file)
                mean_file = sub(self.in_directory, self.mean_directory, sub(self.pickle_file_suffix, self.mean_suffix, file))
                std_file = sub(self.in_directory, self.std_directory, sub(self.pickle_file_suffix, self.std_suffix, file))
                img = imread(img_file)
                if self.second_channel:
                    img_ch2 = zoom(swapaxes(img[:, 1, :, :], 0, 2), self.spacing)
                img = zoom(swapaxes(img[:, 0, :, :], 0, 2), self.spacing)
                seg = load(seg_file)
                #mean_img = load(mean_file)
                #std_img = load(std_file)
                seg_dst = distance_transform_edt(seg)
                if self.neuron_channel:
                    nrn_dst = load(sub(self.pickle_file_suffix, self.neuron_distance_suffix, file))
                a, b, c = mgrid[-1 * self.grid_size_psf_deconv // 2:(self.grid_size_psf_deconv // 2 + 1):1,
                                -1 * self.grid_size_psf_deconv // 2:(self.grid_size_psf_deconv // 2 + 1):1,
                                -1 * self.grid_size_psf_deconv // 2:(self.grid_size_psf_deconv // 2 + 1):1]
                abc = dstack([a.flat, b.flat, c.flat])
                mu = array([0, 0, 0])
                sigma = self.psf
                covariance = diag(sigma ** 2)
                d = multivariate_normal.pdf(abc, mean=mu, cov=covariance)
                d = d.reshape((len(a), len(b), len(c)))
                deconv_img = copy(img)
                deconv_img = self.max_pixel_value * richardson_lucy(img / self.max_pixel_value, d,
                                                                    num_iter=self.n_iter_deconv)
                if self.second_channel:
                    deconv_img -= self.max_pixel_value * richardson_lucy(img_ch2 / self.max_pixel_value, d,
                                                                         num_iter=self.n_iter_deconv)
                deconv_img = float32(deconv_img)

                for i in tqdm(range(len(graph.edges))):
                    path = graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['pts']
                    _pred_radii = mean(seg_dst[path[::-1, 0], path[::-1, 1], path[::-1, 2]])
                    _pred_radii_max = max(seg_dst[path[::-1, 0], path[::-1, 1], path[::-1, 2]])
                    if _pred_radii == 0:
                        _pred_radii = 1

                    _box_fit = max([int16(_pred_radii_max) + 10, 15])
                    path_smooth = float32(copy(path))
                    for k in range(len(path[0])):
                        path_smooth[:, k] = gaussian_filter1d(float64(path[:, k]), 3, mode='nearest')

                    path_grad = gradient(path_smooth, edge_order=2)[0]
                    res_fwhm = []
                    X = arange(-1 * _box_fit, _box_fit + 1, 1)
                    Y = arange(-1 * _box_fit, _box_fit + 1, 1)
                    x, y = meshgrid(X, Y)
                    x = x.flatten()
                    y = y.flatten()
                    z = zeros(len(x))
                    xy = vstack([x, y, z])

                    def calc_fwhm_path(I):
                        point_grad = path_grad[I]
                        point = path[I]
                        if all(point_grad[0:2] == [0, 0]) and abs(point_grad[2] / point_grad[2]) == 1:
                            rotated = xy.T + point
                        else:
                            rotated = _rotmat(point_grad, xy.T) + point
                        points_img = map_coordinates(deconv_img,
                                                        rotated.T,
                                                        order=3,
                                                        mode='constant')

                        points_img = reshape(points_img, (len(X), len(Y)))
                        points_img = gaussian_filter(points_img, sigma=_pred_radii * .4)

                        _point = array(arange(0, _pred_radii + 20, self.sampling))
                        _zeros = zeros(len(_point))
                        _point = array([_point, _zeros])
                        _centre = closest_node([len(X) // 2 + 1, len(Y) // 2 + 1], peak_local_max(points_img.T))

                        _res = []

                        for deg in arange(0, 360, 10):
                            rot_point = dot(array([[cos(deg2rad(deg)), -1 * sin(deg2rad(deg))],
                                                    [sin(deg2rad(deg)), cos(deg2rad(deg))]]), _point)
                            rot_point[0] = rot_point[0] + _centre[0]
                            rot_point[1] = rot_point[1] + _centre[1]
                            points_vals = map_coordinates(points_img.T,
                                                            rot_point,
                                                            order=3,
                                                            cval=0)
                            points_vals = gaussian_filter1d(points_vals, sigma=_pred_radii * .4 / self.sampling)

                            _ = array(argrelextrema(points_vals , less, order=10))
                            #if _[0,0] == 0:
                            #    _ = array(argrelextrema(points_vals[1:]), less, order=3) + 1

                            if _.shape[1] != 0:
                                points_vals_grad = gradient(points_vals[:max((_[0, 0],int(3.5/self.sampling))) + 3])
                                _ = nanargmin(points_vals_grad)
                                _res.append(_ * self.sampling)

                            else:
                                points_vals_grad = gradient(points_vals)
                                _ = nanargmin(points_vals_grad)
                                _res.append(_ * self.sampling)

                        _res = array(_res)
                        _res = _res[~isnan(_res)]
                        _res = _res[where(_res != 0)]
                        _mean = mean(_res)
                        _std = std(_res)
                        _mask = where(logical_and(_res >= _mean - 2 * _std, _res <= _mean + 2 * _std))
                        _res = _res[_mask]
                        radii = mean(_res)
                        radii_std = std(_res)

                        return radii, radii_std

                    pool = ThreadPool(self.n_cores)
                    _vals, _vals_sigma = zip(*pool.map(calc_fwhm_path, range(len(path))))
                    if self.filter:
                        _vals_unfiltered = copy(_vals)
                        _vals = filtfilt(butter_b, butter_a, _vals, padlen=0)
                    graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['radii'] = mean(_vals)
                    graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['radii_std'] = std(_vals)
                    graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['path_weights'] = _vals
                    if self.filter:
                        graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['path_weights_unfiltered'] = _vals_unfiltered
                    graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['path_weights_uncertanty'] = _vals_sigma
                    graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['end-0z'] = path[0][0]
                    graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['end-0y'] = path[0][1]
                    graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['end-0x'] = path[0][2]
                    graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['end-1z'] = path[-1][0]
                    graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['end-1y'] = path[-1][1]
                    graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['end-1x'] = path[-1][2]
                    graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['mean_depth'] = mean(path[:, 0])
                    graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['euclidean-dst'] = sqrt(sum(square(path[-1] - path[0])))
                    if self.neuron_channel:
                        _nrn_dst_vals = nrn_dst[path[::-1, 0], path[::-1, 1], path[::-1, 2]]
                        graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['mean_neuron_distance'] = mean(_nrn_dst_vals)
                        graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['neuron_distance_std'] = std(_nrn_dst_vals)
                        graph[list(graph.edges)[i][0]][list(graph.edges)[i][1]]['neuron_distance_min'] = min(_nrn_dst_vals)
                write_gpickle(graph, sub(self.in_directory, self.out_directory, sub(self.pickle_file_suffix, self.out_pickle_suffix, file)))
        if 'pool' not in globals():
            pool=5
        del pool
    
    def process_all_files(self):
        for file in tqdm(self.files):
            print(file)
            self.calc_radii_vessels(file)

        

