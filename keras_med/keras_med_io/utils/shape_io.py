from skimage.transform import resize
import numpy as np
import SimpleITK as sitk


def reshape_img(img,newx,newy):
    new_x_size = newx #upsample
    new_y_size = newy #downsample
    new_z_size = img.GetSize()[2] #downsample
    new_size = [new_x_size, new_y_size, new_z_size]
    new_spacing = [old_sz*old_spc/new_sz  for old_sz, old_spc, new_sz in zip(img.GetSize(), img.GetSpacing(), new_size)]
    interpolator_type = sitk.sitkLinear
    new_img = sitk.Resample(img, new_size, sitk.Transform(), interpolator_type, img.GetOrigin(), new_spacing, img.GetDirection(), 0.0, img.GetPixelIDValue())
    return new_img


def reshape(orig_img, append_value=-1024, new_shape=(512, 512, 512)):
    """
    Reshapes a numpy array with the specified values where the new shape must have >= to the number of
    voxels in the original image. If # of voxels in `new_shape` is > than the # of voxels in the original image,
    then the extra will be filled in by `append_value`.
    Args:
        orig_img:
        append_value: filler value
        new_shape: must have >= to the number of voxels in the original image.
    """

    reshaped_image = np.zeros(new_shape)
    reshaped_image[...] = append_value
    x_offset = 0
    y_offset = 0  # (new_shape[1] - orig_img.shape[1]) // 2
    z_offset = 0  # (new_shape[2] - orig_img.shape[2]) // 2

    reshaped_image[x_offset:orig_img.shape[0]+x_offset, y_offset:orig_img.shape[1]+y_offset, z_offset:orig_img.shape[2]+z_offset] = orig_img

    # insert temp_img.min() as background value

    return reshaped_image

def extract_nonint_region(image, mask = None, outside_value=0, coords = False):
    """
    Resizing image around a specified region (i.e. nonzero region).
    Args:
        image:
        mask: a segmentation labeled mask that is the same shaped as 'image' (optional; default: None)
        outside_value: (optional; default: 0)
        coords: boolean on whether or not to return boundaries (bounding box coords) (optional; default: False)
    Returns:
        the resized image
        segmentation mask (when mask is not None)
        a nested list of the mins and and maxes of each axis (when coords = True)
    """
    # Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    # ===================================================================================================
    # Changes: Added the ability to return the cropping coordinates
    pos_idx = np.where(image > outside_value)
    # fetching all of the min/maxes for each axes
    pos_x, pos_y, pos_z = pos_idx[0], pos_idx[1], pos_idx[2]
    minZidx, maxZidx = int(np.min(pos_z)), int(np.max(pos_z)) + 1
    minXidx, maxXidx = int(np.min(pos_x)), int(np.max(pos_x)) + 1
    minYidx, maxYidx = int(np.min(pos_y)), int(np.max(pos_y)) + 1
    # resize images
    resizer = (slice(minXidx, maxXidx), slice(minYidx, maxYidx), slice(minZidx, maxZidx))
    coord_list = [[minXidx, maxXidx], [minYidx, maxYidx], [minZidx, maxZidx]]
    # returns cropped outputs with the bbox coordinates
    if coords:
        if mask is not None:
            return (image[resizer], mask[resizer], coord_list)
        elif mask is None:
            return (image[resizer], coord_list)
    # returns just cropped outputs
    elif not coords:
        if mask is not None:
            return (image[resizer], mask[resizer])
        elif mask is None:
            return (image[resizer])

def extract_masked_region(image, mask, masked_label=1, coords = False):
    """                                                                                                                                                                                                         Resizing image around a specified region (i.e. nonzero region)                                                                                                                                             Args:                                                                                                                                                                                                          image:                                                                                                                                                                                                     mask: a segmentation labeled mask that is the same shaped as 'image' (optional; default: None)                                                                                                             outside_value: (optional; default: 0)                                                                                                                                                                      coords: boolean on whether or not to return boundaries (bounding box coords) (optional; default: False)                                                                                                Returns:                                                                                                                                                                                                       the resized image                                                                                                                                                                                          segmentation mask (when mask is not None)                                                                                                                                                                  a nested list of the mins and and maxes of each axis (when coords = True)                                                                                                                              """
    pos_idx = np.where(mask == masked_label)
    # fetching all of the min/maxes for each axes                                                                                                                                                           
    pos_x, pos_y, pos_z = pos_idx[0], pos_idx[1], pos_idx[2]
    minZidx, maxZidx = int(np.min(pos_z)), int(np.max(pos_z)) + 1
    minXidx, maxXidx = int(np.min(pos_x)), int(np.max(pos_x)) + 1
    minYidx, maxYidx = int(np.min(pos_y)), int(np.max(pos_y)) + 1
    # resize images                                                                                                                                                                                         
    resizer = (slice(minXidx, maxXidx), slice(minYidx, maxYidx), slice(minZidx, maxZidx))
    coord_list = [[minXidx, maxXidx], [minYidx, maxYidx], [minZidx, maxZidx]]
    # returns cropped outputs with the bbox coordinates                                                                                                                                                     
    if coords:
        return (image[resizer], mask[resizer], coord_list)
    # returns just cropped outputs                                                                                                                                                                          
    elif not coords:
        return (image[resizer], mask[resizer])
        
def resample_array(src_imgs, src_spacing, target_spacing, is_label = False):
    """
    Resamples a numpy array.
    Args:
        src_imgs: numpy array
        srs_spacing: list of the voxel spacing (must be the same dimension as the input numpy array)
        target_spacing: list of the target voxel spacings (must be the same dimension as the input numpy array)
    """
    # Copyright 2018 Division of Medical Image Computing, German Cancer Research Center (DKFZ).
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    # ==============================================================================
    # CHanges: Added the ability to resample groundtruth masks (using nearest neighbor)

    # Calcuating the target shape based on the target spacing
    src_spacing = np.round(src_spacing, decimals = 3)
    target_shape = [int(src_imgs.shape[ix] * src_spacing[ix] / target_spacing[ix]) \
                    for ix in range(len(src_imgs.shape))]
    # Checking that none of the target shape is 0 or negative
    for i in range(len(target_shape)):
        try:
            assert target_shape[i] > 0
        except:
            raise AssertionError("AssertionError:", src_imgs.shape, src_spacing, target_spacing)

    img = src_imgs.astype(float)
    if is_label:
        # nearest neighbor interpolation for label
        resampled_img = resize(img, target_shape, order=0, clip=True, mode='edge').astype('float32')
    elif not is_label:
        # third-order spline interpolation for image
        resampled_img = resize(img, target_shape, order=3, clip=True, mode='edge').astype('float32')

    return resampled_img
