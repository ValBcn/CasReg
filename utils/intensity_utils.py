# coding: utf-8
# Intensity processing utilities moved from keras_med
import numpy as np

def clip_upper_lower_percentile(image, mask=None, percentile_lower=0.2, percentile_upper=99.8):
    """
    Clipping values for positive class areas.
    Args:
        image:
        mask:
        percentile_lower:
        percentile_upper:
    Return:
        Image with clipped pixel intensities
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
    # Changes: Added the ability to have the function clip at only the necessary percentiles with no mask and removed the
    # automatic generation of a mask

    # finding the percentile values
    cut_off_lower = np.percentile(image[mask != 0].ravel(), percentile_lower)
    cut_off_upper = np.percentile(image[mask != 0].ravel(), percentile_upper)
    # clipping based on percentiles
    res = np.copy(image)
    if mask is not None:
        res[(res < cut_off_lower) & (mask !=0)] = cut_off_lower
        res[(res > cut_off_upper) & (mask !=0)] = cut_off_upper
    elif mask is None:
        res[(res < cut_off_lower)] = cut_off_lower
        res[(res > cut_off_upper)] = cut_off_upper
    return res