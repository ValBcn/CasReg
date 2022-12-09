# Utils
This module contains a bunch of submodules with easy, documented, reusable, and common i/o functions. Note that these are not to fulfill the purpose of data augmentation functions (that is fulfilled by `batchgenerators.transform` and the `BaseTransformGenerator`). The functions are divided based on how they manipulate images, such as:
* `intensity_io.py`
  * contains functions that manipulate the intensity distribution of input images in some sort of way, such as `whitening` (z-score normalization), `minmax_normalize`, `clip_upper_lower_percentile`, etc.
* `shape_io.py`
  * contains functions that manipulate the shape of input images such as `reshape`, `extract_nonint_region`, and `resample_array`, etc.
* `patch_utils.py` and `patch.py`
  * Contains a bunch of patch extraction functions from [ellisdg's 3DUnetCNN repository](https://github.com/ellisdg/3DUnetCNN)
  * `patch_utils.py` is the OOP version of the functions in `patch.py`.
  * __These particular submodules need to be refactored.__
* `misc_utils.py`
  * Contains a bunch of miscellaneous functions, such as:
    * `get_list_IDs`: divides filenames into train/validation/test sets
    * `get_multi_class_labels`: one-hot encoding function for segmentation (includes the option to remove the background class)
    * `sanity_checks`: checks for NaNs, and makes sure that the labels are one-hot encoded.
    * `add_channel`: adds a gray scale channel dimension for `channels_last`
    *  `compute_pad_value`: Computes the minimum pixel intensity of the entire dataset for the pad value (if it's not 0)
    * Need to add the `KFold` function.
* `custom_augmentations.py`
  * Actually for data augmentation purposes (on-the-fly preprocessing)
  * __In the prelimary phase currently__
  * Notable utility functions:
    * __Patch extraction:__ `get_random_slice_idx`, `get_positive_idx`
