from numpy import float32, expand_dims, array
from skimage.io import imread, imsave
from ants import from_numpy, registration, apply_transforms
from re import sub
from os.path import basename, dirname
from tifffile import TiffFile
from tqdm import tqdm
from os.path import exists
from novas3d.utilities import get_mov_files

def register_paired_images(fix_file, mov_files, out_dir, in_filename_extension='.tif', final_filename_extension='_warped.tif', sigma=2, flip = False):
    """
    Register paired images using ANTs registration.

    Args:
        fix_file (str): Path to the fixed image file.
        mov_files (list): List of paths to the moving image files.
        out_dir (str): Output directory to save the registered images.
        in_filename_extension (str, optional): Initial filename extension of the unregistered images. Defaults to '.tif'.
        final_filename_extension (str, optional): Final filename extension of the registered images. Defaults to '_warped.tif'.
        sigma (float, optional): Sigma value for the registration. Defaults to 2.
        flip (bool, optional): Whether to flip the moving image. Defaults to False.

    Returns:
        None
        
    """
    
    # Read the fixed image
    fix_numpy = imread(fix_file)
    fix = from_numpy(float32(fix_numpy[:,0])) # Convert images to ants 

    if flip == True:
        # Extract metadata from the fixed image
        with TiffFile(fix_file) as tif:
            tif_tags = {}
            for tag in tif.pages[0].tags.values():
                name, value = tag.name, tag.value
                tif_tags[name] = value
            image = tif.pages[0].asarray()
        #print(tif_tags)
        start_str = float([x for x in tif_tags['IJMetadata']['Info'].split('\n') if 'axis startPosition #1' in x][0].split(' ')[-1])
        end_str = float([x for x in tif_tags['IJMetadata']['Info'].split('\n') if 'axis endPosition #1' in x][0].split(' ')[-1])
        direction = end_str - start_str
        if [x for x in tif_tags['IJMetadata']['Info'].split('\n') if 'imagingParam zDriveUnitType' in x][0].split(' ')[-1] == 'Piezo':
            direction = float([x for x in tif_tags['IJMetadata']['Info'].split('\n') if 'acquisitionValue zPosition' in x][0].split(' ')[-1])
            piezo = True
        else:
            piezo = False
        #print(direction)
        
    res2 = []
    for mov_file in mov_files:
        # Read the moving image
        mov_numpy = imread(mov_file)
        
        if flip == True and piezo == False:
            with TiffFile(mov_file) as tif:
                tif_tags = {}
                for tag in tif.pages[0].tags.values():
                    name, value = tag.name, tag.value
                    tif_tags[name] = value
                image = tif.pages[0].asarray()
            start_str = float([x for x in tif_tags['IJMetadata']['Info'].split('\n') if 'axis startPosition #1' in x][0].split(' ')[-1])
            end_str = float([x for x in tif_tags['IJMetadata']['Info'].split('\n') if 'axis endPosition #1' in x][0].split(' ')[-1])
            # Flip the moving image if necessary
            if direction * (end_str - start_str) < 0:
                mov_numpy = mov_numpy[::-1]

        elif flip == True and piezo == True:
            with TiffFile(mov_file) as tif:
                tif_tags = {}
                for tag in tif.pages[0].tags.values():
                    name, value = tag.name, tag.value
                    tif_tags[name] = value
                image = tif.pages[0].asarray()
            piezo_dir = float([x for x in tif_tags['IJMetadata']['Info'].split('\n') if 'acquisitionValue zPosition' in x][0].split(' ')[-1])
            # Flip the moving image if necessary
            if direction != piezo_dir:
                mov_numpy = mov_numpy[::-1]

        mov = from_numpy(float32(mov_numpy[:,0]))

        # Register images and get displacement
        mytx = registration(fixed=fix,
                            moving=mov,
                            type_of_transform='Rigid',
                            total_sigma=sigma,
                            aff_metric='meansquares')

        # Move vascular channel
        warpedraw_1 = apply_transforms(fixed=fix,
                                       moving=from_numpy(float32(mov_numpy[:,0])),
                                       transformlist=mytx['fwdtransforms'],
                                       interpolator='linear').numpy()

        # Move neuron channel
        warpedraw_2 = apply_transforms(fixed=fix,
                                       moving=from_numpy(float32(mov_numpy[:,1])),
                                       transformlist=mytx['fwdtransforms'],
                                       interpolator='linear').numpy()
        
        # Combine moved channels into one image
        mov_numpy[:,0,:,:] = warpedraw_1[:,:,:]
        mov_numpy[:,1,:,:] = warpedraw_2[:,:,:]

        # Save warped followup image and baseline image
        imsave(out_dir + sub(in_filename_extension,final_filename_extension,basename(dirname(mov_file)) + '-' + basename(mov_file)),mov_numpy)

    # Save the fixed image
    imsave(out_dir + sub(in_filename_extension,final_filename_extension,basename(dirname(fix_file)) + '-' + basename(fix_file)),fix_numpy)

class ImageRegistration:

    """
    Register images using ANTs registration. Save the registered images to an output directory.
    
    Args:
        images (list): List of image files to register. Each timepoint should be its own file. Files should be tif stacks. Channel 1 should be the color channel. 
        out_directory (str): Output directory to save the registered images.
        in_filename_extension (str): Initial filename extension of the unregistered images.
        final_filename_extension (str): Final filename extension of the registered images.
        timepoint_suffixes (list): List of timepoint suffixes.
        sigma (float): Sigma value for the registration.
        flip (bool): Whether to flip the moving image based on metadata. (only work on images aquired with olympus systems, set to false if not sure if images were aquired on an olympus microscope)
        dic (dictionary): dictionary of the image files with a reference image as the key and a list of moving images as the value.
        skip_existing (bool): Whether to skip already registered images.
    
    Returns:
        None
    """

    def __init__(self, images, out_directory, in_filename_extension, final_filename_extension, timepoint_suffixes, sigma, flip, dic, skip_existing=False):
        self.images = images
        self.out_directory = out_directory
        self.in_filename_extension = in_filename_extension
        self.final_filename_extension = final_filename_extension
        self.timepoint_suffixes = timepoint_suffixes
        self.sigma = sigma
        self.flip = flip
        self.dic = dic
        self.skip_existing = skip_existing

    def register_images(self):
        for image in tqdm(self.images):
            fix_file = sub('_\d{4}', '', image)
            warped_file = basename(dirname(fix_file)) + '-' + basename(fix_file)
            warped_file_path = self.out_directory + sub(self.in_filename_extension, self.final_filename_extension, warped_file)
                
            if self.skip_existing==False or not exists(warped_file_path):
                mov_files = get_mov_files(image, self.dic, self.in_filename_extension, self.timepoint_suffixes)

                register_paired_images(fix_file,
                                       mov_files,
                                       self.out_directory,
                                       self.in_filename_extension,
                                       self.final_filename_extension,
                                       self.sigma,
                                       self.flip)
    