# Keras-Med-IO
Providing a fast and easy IO toolbox for on-the-fly and local preprocessing in Keras, particularly for medical image segmentation tasks.
This is currently in alpha, and I'm open to anyone who wants to contribute!

## Limitations
* `Channels_last` only
* Kind of slow? Multiprocessing doesn't really help performance.
* Single input (x,y) cases in examples and for `BaseTransformGenerator`

## Credits
* The `BaseGenerator` inspired by [Shervine's introduction to keras.utils.Sequence's](https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly).
* The data augmentation is reliant on [MIC-DKFZ's batchgenerators](https://github.com/MIC-DKFZ/batchgenerators) and their `transforms` API.
* Some of the patch extraction in `patch_utils.py` and `patch.py` are from [ellisdg's 3DUnetCNN repository](https://github.com/ellisdg/3DUnetCNN).
* Some of the I/O functions are directly from or inspired by [MedicalDetectionToolkit](https://github.com/pfjaeger/medicaldetectiontoolkit) and [Isensee's BRaTS2017 Submission](https://github.com/MIC-DKFZ/BraTS2017).
  * __Note: I still need to update my license to accommodate for their [Apache 2.0 license](https://tldrlegal.com/license/apache-license-2.0-(apache-2.0)).__

## What This Does Right
* Can achieve state-of-the-art results
* Customizable
* Positive Slice Sampling!!

## Agenda for Future Work
A lot of the suggestions is located within individual modules' `README.md`'s, but here's a list of the current top priorities:
* Revise the `n_workers` interface for `BaseTransformGenerator`
* Fix up the examples.
  * Make some easy, simple examples.
  * Provide links to my personal use cases w/Colab Notebooks.
* Tests (particularly for any I/O functions)

## Main Utilities
### Base Generators
This specific module is for low-level abstract generators for you to reuse in your in your own keras pipelines. The current base generators are: <br>
* `BaseGenerator`: Basic framework for generating thread-safe data in keras. (no preprocessing and channels_last)
  Based on https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
* `BaseTransformGenerator`: Loads data and applies data augmentation with `batchgenerators.transforms`.
  * Supports channels_last
  * Loads data with nibabel

#### Common Generator Arguments
Overall, these base generators are made with the intent to be open to a variety of potential uses (particularly targeting image segmentation i/o pipelines). Hence, I made it so that the main arguments for these generators are:
* `list_IDs`: list of all your filenames. The idea behind this is that you can just load your (image, mask) pairs for segmentation easily (assuming they have the same file name). For classification tasks, this may not be optimal, but this directory was targeting mainly segmentation users. I'll look into branching out with a separate classification base generator though.
* `data_dirs`: [x_dir, y_dir]. This was a more ambiguous approach so that you can freely play around with your directory structure.
  * Changes to look out for in the future: Dividing this into just `x_dir`, `y_dir`
  * Building a base generator for multiple inputs (iterating through `x_dir`, `y_dir`)
* `batch_size`: batch size for the network
* `n_channels`: The number of channels that your data has. This parameter exists to handle cases where the number of input channels does not match that of the output segmentation.
* `n_classes`: The number of classes. Again, this is to handle cases where the number of input channels does not match that of the output segmentation.

### Utils
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

### Inference
The goal of this module is to provide evaluation and prediction tools for medical segmentation/classification. Currently, this module is still in-the-works to make a more generalizable framework.
#### Agenda
* Patch evaluation and aggregation
* Cleaner examples
* Pipeline for easy general inference
