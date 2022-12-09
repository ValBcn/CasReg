import numpy as np
import sys

sys.path.append("/home/valentin/Desktop/TransMorph_Transformer_for_Medical_Image_Registration/Baseline_registration_models/VoxelMorph/keras_med")
from skimage.transform import resize
from keras_med_io.utils.intensity_io import clip_upper_lower_percentile
from keras_med_io.utils.shape_io import extract_nonint_region, resample_array, extract_masked_region
from nibabel import Nifti1Image
import itertools
import scipy.signal
import numpy as np
import SimpleITK as sitk
from functools import partial

def isensee_preprocess(input_image, mask, orig_spacing, get_coords = False, ct = False, flip_label = 0, mean_patient_shape = (115, 224, 224)):
    """
    Order:
    1) Cropping to non-zero regions
    2) Resampling to the median voxel spacing of the respective dataset
    3) Normalization

    Args:
        input_image:
        mask:
        orig_spacing: list/numpy array of voxel spacings corresponding to each axis of input_image and mask (assumes they have the same spacings)
            * If it is left as None, then the images will not be resampled.
        get_coords: boolean on whether to return extraction coords or not
        mean_patient_shape: obtained from Table 1. in the nnU-Net paper
    Returns:
        preprocessed input image and mask
    """
    # converting types and axes order
    if isinstance(input_image, Nifti1Image):
        input_image = nii_to_np(input_image)
    if isinstance(mask, Nifti1Image):
        mask = nii_to_np(mask)
        
    input_image[mask == 0] = 0
    out_value = 1e-2 #-1 #input_image.max()/10.
    

    if flip_label==1:
        mask = np.flip(mask,axis=2)
        print("labels are flipped")
    # 1. Cropping
    if get_coords:
        extracted_img, extracted_mask, coords = extract_nonint_region(input_image, mask = mask, outside_value=out_value, coords = True)
    elif not get_coords:
        extracted_img, extracted_mask = extract_nonint_region(input_image, mask = mask, outside_value=out_value, coords = False)
    # 2. Resampling
    if orig_spacing is None: # renaming the variables because they don't get resampled
        resamp_img = extracted_img
        resamp_label = extracted_mask
    else:
        transposed_spacing = orig_spacing[::-1] # doing so because turning into numpy array moves the batch dimension to axis 0
        med_spacing = [np.median(transposed_spacing) for i in range(3)]
        resamp_img = resample_array(extracted_img, transposed_spacing, med_spacing, is_label = False)
        resamp_label = resample_array(extracted_mask, transposed_spacing, med_spacing, is_label = True)
    # 3. Normalization
    #norm_img = zscore_isensee(resamp_img, ct = ct, mean_patient_shape = mean_patient_shape)
    norm_img = zerone(resamp_img)
    if get_coords:
        return norm_img, resamp_label, coords,med_spacing
    elif not get_coords:
        return norm_img, resamp_label,med_spacing

def isensee_preprocess_no_label(input_image, orig_spacing, get_coords = False, ct = False, mean_patient_shape = (115, 224, 224)):

    # converting types and axes order                                                                                                                                                                       
    if isinstance(input_image, Nifti1Image):
        input_image = nii_to_np(input_image)
    # 1. Cropping                                                                                                                                                                                           
    if get_coords:
        extracted_img, coords = extract_nonint_region(input_image, mask=None, outside_value = 1e-3, coords = True)
    elif not get_coords:
        extracted_img = extract_nonint_region(input_image, mask = None, outside_value = 1e-3, coords = False)
    # 2. Resampling                                                                                                                                                                                         
    if orig_spacing is None: # renaming the variables because they don't get resampled                                                                                                                      
        resamp_img = extracted_img
    else:
        transposed_spacing = orig_spacing[::-1] # doing so because turning into numpy array moves the batch dimension to axis 0                                                                             
        med_spacing = [np.median(transposed_spacing) for i in range(3)]
        resamp_img = resample_array(extracted_img, transposed_spacing, med_spacing, is_label = False)
    # 3. Normalization                                                                                                                                                                                      
    #norm_img = zscore_isensee(resamp_img, ct = ct, mean_patient_shape = mean_patient_shape)                                                                                                                
    norm_img = zerone(resamp_img)
    if get_coords:
        return norm_img, coords,med_spacing
    elif not get_coords:
        return norm_img,med_spacing
    
def reshape_output(output,orig_spacing):
    """
    Resampling the output voxel spacing to the input voxel spacing
    """
    transposed_spacing = orig_spacing[::-1] # doing so because turning into numpy array moves the batch dimension to axis 0
    med_spacing = [np.median(transposed_spacing) for i in range(3)]
    resamp_label = resample_array(output.squeeze(), transposed_spacing, med_spacing, is_label = True)
    return resamp_label


def zscore_isensee(arr, ct, mean_patient_shape):
    """"
    Performs Z-Score normalization based on these conditions:
        CT:
            1) Clip to [0.5, 99.5] percentiles of intensity values
            2) Z-score norm on everything
        Other Modalities:
            1) Z-Score normalization individually
                * If # of voxels in crop < (mean # of voxels in orig / 4), normalization only on nonzero elements and everything else = 0
    Args:
        arr: cropped numpy array
        mean_patient_shape: list/tuple of the original input shape before cropping
    Return:
        A normalized numpy array according to the nnU-Net paper
    """
    cropped_voxels, mean_voxels = np.prod(np.array(arr.shape)), np.prod(np.array(mean_patient_shape))
    overcropped = cropped_voxels < (mean_voxels / 4)
    if ct:
        arr = clip_upper_lower_percentile(arr, percentile_lower = 0.5, percentile_upper = 99.5)
        return zscore_norm(arr)
    # Other modalities
    elif not ct:
        if overcropped:
            arr[arr != 0] = zscore_norm(arr[arr !=0]) # only zscore norm on nonzero elements
        elif not overcropped:
            arr = zscore_norm(arr)
        return arr

def zscore_norm(arr):
    """
    Mean-Var Normalization
    * mean of 0 and standard deviation of 1
    Args:
        arr: numpy array
    Returns:
        A numpy array with a mean of 0 and a standard deviation of 1
    """
    shape = arr.shape
    arr = arr.flatten()
    norm_img = (arr-np.mean(arr)) / np.std(arr)
    return norm_img.reshape(shape)

def zerone(arr):
    """
    normalize between 0 and 1
    """
    shape = arr.shape
    arr = arr.flatten()
    norm_img = (arr-arr.min())/(arr.max()-arr.min())
    return norm_img.reshape(shape)

def nii_to_np(nib_img):
    """
    Converts a 3D nifti image to a numpy array of (z, x, y) dims
    """
    np_img = np.squeeze(nib_img.get_fdata())
    return np.transpose(np_img, [-1, 0, 1])

    
def resize_data(data,new_size_x,new_size_y,new_size_z):
    initial_size_x = data.shape[0]
    initial_size_y = data.shape[1]
    initial_size_z = data.shape[2]

    delta_x = initial_size_x / float(new_size_x)
    delta_y = initial_size_y / float(new_size_y)
    delta_z = initial_size_z / float(new_size_z)

    new_data = np.zeros((new_size_x, new_size_y, new_size_z))

    for x, y, z in itertools.product(range(new_size_x),
                                     range(new_size_y),
                                     range(new_size_z)):
        new_data[x][y][z] = data[int(x * delta_x)][int(y * delta_y)][int(z * delta_z)]

    return new_data

def multi_channel_labels(label,nb_labels):
    
    multi_channel_label = np.zeros((*label.shape,nb_labels))
        
    for i in range(nb_labels):
        multi_channel_label[...,i] = np.where(label!=i,multi_channel_label[...,i],1)
    return multi_channel_label

def resize_and_uncrop(img,coords,old_shape,no_rot=False):

    size_x = coords[0][1]-coords[0][0]
    size_z = coords[1][1]-coords[1][0]
    size_y = coords[2][1]-coords[2][0]

    resized_img = resize_data(np.rot90(img,k=3,axes=(1,2)),size_x,size_y,size_z)

    print("resize image shape", resized_img.shape)
    print("sizes . %f %f %f" %(size_x,size_y, size_z))

    if no_rot == False:
        uncropped_img = np.rot90(np.zeros(old_shape),k=3,axes=(1,2))
    else:
        uncropped_img = np.zeros(old_shape)
    if coords[0][0] > 0 and coords[2][0] > 0:
        uncropped_img[coords[0][0]-1:coords[0][1]-1,coords[2][0]-1:coords[2][1]-1,coords[1][0]:coords[1][1]] = np.flip(resized_img,axis=2)
    else:
        uncropped_img[coords[0][0]:coords[0][1],coords[2][0]:coords[2][1],coords[1][0]:coords[1][1]] = np.flip(resized_img,axis=2)

    return uncropped_img


def resize(img,coords):

    size_x = coords[0][1]-coords[0][0]
    size_z = coords[1][1]-coords[1][0]
    size_y = coords[2][1]-coords[2][0]

    resized_img = resize_data(np.rot90(img,k=3,axes=(1,2)),size_x,size_y,size_z)

    return resized_img


def hist_match(source, template):
    """                                                                                                                                                                                                        Adjust the pixel values of a grayscale image such that its histogram                                                                                                                                       matches that of a target image                                                                                                                                                                              
    Arguments:                                                                                                                                                                                                 -----------                                                                                                                                                                                                     source: np.ndarray                                                                                                                                                                                             Image to transform; the histogram is computed over the flattened                                                                                                                                           array                                                                                                                                                                                           
        template: np.ndarray                                                                                                                                                                                            Template image; can have different dimensions to source                                                                                                                                         
    Returns:                                                                                                                                                                                                   -----------                                                                                                                                                                                                     matched: np.ndarray                                                                                                                                                                                            The transformed output image                                                                                                                                                                       """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and                                                                                                                                
    # counts                                                                                                                                                                                                
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to                                                                                                                                
    # get the empirical cumulative distribution functions for the source and                                                                                                                                
    # template images (maps pixel value --> quantile)                                                                                                                                                       
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image                                                                                                                                   
    # that correspond most closely to the quantiles in the source image                                                                                                                                     
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

def dice(seg1, seg2):
    dice = np.sum(seg1.flatten()[seg2.flatten()==1])*2.0 / (np.sum(seg1.flatten()) + np.sum(seg2.flatten()))
    return dice


def hd95(seg1,seg2):    
    distance_map = partial(sitk.SignedMaurerDistanceMap, squaredDistance=False, useImageSpacing=True)
    ref_contour_img = sitk.GetImageFromArray(seg1)
    test_contour_img = sitk.GetImageFromArray(seg2)
    
    gold_surface = sitk.LabelContour(ref_contour_img)
    prediction_surface = sitk.LabelContour(test_contour_img)

    ### Get distance map for contours (the distance map computes the minimum distances)
    prediction_distance_map = sitk.Abs(distance_map(prediction_surface))
    gold_distance_map = sitk.Abs(distance_map(gold_surface))

    ### Find the distances to surface points of the contour.  Calculate in both directions
    gold_to_prediction = sitk.GetArrayViewFromImage(prediction_distance_map)[sitk.GetArrayViewFromImage(gold_surface) == 1]
    prediction_to_gold = sitk.GetArrayViewFromImage(gold_distance_map)[sitk.GetArrayViewFromImage(prediction_surface) == 1]

    ### Find the 95% Distance for each direction and average
    return (np.percentile(prediction_to_gold, 95) + np.percentile(gold_to_prediction, 95)) / 2.0


def Local_CC(img1,img2,win=5,pad=2):
    hw = int((win-1)/2)
    cc = np.zeros(img1.shape)    
    for i in range(int(128/pad)):
        for j in range(int(128/pad)):
            for k in range(int(128/pad)):
                min_i = np.min((pad*i,np.abs(pad*i-hw)))
                min_j = np.min((pad*j,np.abs(pad*j-hw)))
                min_k = np.min((pad*k,np.abs(pad*k-hw)))
                max_i = np.max((pad*i,pad*i+hw))
                max_j = np.max((pad*j,pad*j+hw))
                max_k = np.max((pad*k,pad*k+hw))
                cc[pad*i:pad*(i+1),pad*j:pad*(j+1),pad*k:pad*(k+1)] = np.sum(scipy.signal.correlate(img1[min_i:max_i,min_j:max_j,min_k:max_k],img2[min_i:max_i,min_j:max_j,min_k:max_k],mode="same"))
    return cc
