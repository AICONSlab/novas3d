from cc3d import connected_components
from numpy import unique, delete, zeros, isin, hstack, asarray, argmin, array, where
from numpy import min as npmin
from numpy import sum as npsum
from scipy.spatial.distance import cdist
from vg import normalize, perpendicular, angle, basis
from scipy.spatial.transform import Rotation
from pytransform3d.rotations import matrix_from_axis_angle
from re import sub
from numpy import unique, copy
from os.path import exists

def remove_small_comps_3d(image, thresh=500):
    """
    Remove small connected components from a 3D image.

    Parameters:
        image (ndarray): The input 3D image.
        thresh (int): The threshold value to determine the minimum size of connected components to keep.

    Returns:
        ndarray: The filtered image with small connected components removed.
    """
    # Convert the input image to a labeled image where each connected component is assigned a unique label
    img_lab, N = connected_components(image, return_N=True)

    # Count the number of pixels for each unique label
    _unique, counts = unique(img_lab, return_counts=True)

    # Keep only the labels that have pixel count greater than the threshold
    unique_keep = _unique[counts > thresh]

    # Remove the background label (label 0)
    unique_keep = delete(unique_keep, [0])

    # Create a filtered image where only the connected components with labels in unique_keep are kept
    img_filt = zeros(img_lab.shape).astype('int8')
    img_filt[isin(img_lab, unique_keep)] = 1

    return img_filt.astype('uint8')

def fill_holes(img, thresh=1000):
    """
    Fills holes in a binary image using connected component analysis.

    Args:
        img (ndarray): The input binary image.
        thresh (int): The threshold value for removing small components. Default is 1000.

    Returns:
        ndarray: The binary image with filled holes.

    """

    assert len(unique(img)==2), 'Input image must be binary'

    # Iterate over unique values in the image in reverse order
    for i in unique(img)[:]:
    #for i in unique(img)[:]:
        # Create a temporary binary image with only the current value
        _tmp = (img == i) * 1.0
        _tmp = _tmp.astype('int8')
        #print(_tmp)
        
        # Remove small components from the temporary image
        _tmp = remove_small_comps_3d(_tmp, thresh=thresh)
        
        # Update the original image with the filled holes
        img = (_tmp==i)*1 
    
    # Convert the image to int8 type and return
    res = img.astype('int8')
    return res

def _rotmat(vector, points):
    """
    Rotate the given points around the given vector.

    Args:
        vector (array-like): The vector around which the points should be rotated.
        points (array-like): The points to be rotated.

    Returns:
        array-like: The rotated points.
    """
    # Normalize the vector
    vector = normalize(vector)

    # Calculate the axis of rotation
    axis = perpendicular(basis.z, vector)

    # Calculate the angle of rotation in radians
    _angle = angle(basis.z, vector, units='rad')

    # Create the rotation matrix
    a = hstack((axis, (_angle,)))
    R = matrix_from_axis_angle(a)

    # Apply the rotation matrix to the points
    r = Rotation.from_matrix(R)
    rotmat = r.apply(points)

    return rotmat

def closest_node(node, nodes, maximal_distance=10):
    """
    Find the closest node to a given node from a list of nodes.

    Parameters:
        node (array-like): The node for which the closest node needs to be found.
        nodes (array-like): The list of nodes to search from.
        maximal_distance (float): The maximum distance allowed for a node to be considered as the closest. Default is 10.

    Returns:
        array-like: The closest node to the given node from the list of nodes.
    """
    # Convert nodes to a numpy array for efficient computation
    nodes = asarray(nodes)

    # Calculate the squared Euclidean distance between each node and the given node
    dist_2 = npsum((nodes - node)**2, axis=1)

    # Check if the minimum distance is greater than 10
    if len(dist_2) == 0:
        return node
    elif npmin(dist_2) > maximal_distance:
        # If the minimum distance is greater than 10, return the given node itself
        return node
    else:
        # If the minimum distance is less than or equal to 10, return the node with the minimum distance
        return nodes[argmin(dist_2)]

def get_mov_files(image,dic,suffix,timepoint_suffixes = ['_0001']):
    """
    Get a list of files associated with an image.

        Parameters:
            image (str): The image file name.
            dic (dict): A dictionary mapping a key to a list of file names.
            suffix (str): The file name suffix.
            timepoint_suffixes (list): A list of timepoint suffixes. Default is ['_0001'].

        Returns:
            list: A list of files associated with the image.
    """
    # Find the key in the dictionary that matches a substring of the image file name
    key = [x for x in list(dic.keys()) if x in image][0]

    # Generate a list of movie files by replacing the key with each value in the dictionary
    mov_files = [sub(key,x,image) for x in dic[key]]
    mov_files = mov_files + [image]

    # Create a copy of the original list of movie files
    mov_files_init = copy(mov_files)

    # Check if the image file name contains any of the timepoint suffixes
    if any(_suffix in image for _suffix in timepoint_suffixes):
        # If it does, find the specific timepoint suffix
        for x in timepoint_suffixes:
            if x in image:
                _suffix = x

        # Generate a new list of movie files by replacing the timepoint suffix with each value in the timepoint_suffixes list
        for timepoint_suffix in timepoint_suffixes:
            mov_files = sorted(mov_files + [sub(_suffix+suffix,timepoint_suffix + suffix,x) for x in mov_files_init])
            mov_files = sorted(mov_files + [sub(_suffix+suffix,suffix,x) for x in mov_files_init])
        mov_files = mov_files + [sub(_suffix, '',image)]
        #print(mov_files)
    else:
        # If the image file name does not contain any timepoint suffixes, generate a new list of movie files by replacing the suffix with each value in the timepoint_suffixes list
        for timepoint_suffix in timepoint_suffixes:
            mov_files = sorted(mov_files + [sub(suffix,timepoint_suffix + suffix,x) for x in mov_files_init])
            mov_files.append(sub(suffix,timepoint_suffix + suffix,image))



    # Remove the original image file name from the list of movie files
    mov_files = [x for x in mov_files if x != image]

    # Chech to see if the file exists

    mov_files = [x for x in mov_files if exists(x)]

    # Sort the list of movie files
    mov_files = sorted(mov_files)

    # Remove any duplicate movie files
    mov_files = unique(mov_files)

    # Return the final list of movie files
    return mov_files

def remove_terminal_edges(graph, skel, thresh_remove_terminal_segemnts):
    """
    Removes terminal edges from a graph based on a given threshold.

    Parameters:
        graph (networkx.Graph): The input graph.
        skel (numpy.ndarray): The skeleton array.
        thresh_remove_terminal_segemnts (float): The threshold to remove terminal segments.

    Returns:
        numpy.ndarray: The updated skeleton array after removing terminal edges.
    """
    for edge in graph.edges:
        if (graph.degree(edge[0]) == 1 and graph.degree(edge[1]) > 2) or (graph.degree(edge[1]) == 1 and graph.degree(edge[0]) > 2):
            if graph[edge[0]][edge[1]]['weight'] < thresh_remove_terminal_segemnts:
                skel[graph[edge[0]][edge[1]]['pts'][1:-1, 0], graph[edge[0]][edge[1]]['pts'][1:-1, 1], graph[edge[0]][edge[1]]['pts'][1:-1, 2]] = 0
    return skel
