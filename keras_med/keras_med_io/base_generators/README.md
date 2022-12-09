# Base Generators
This specific module is for low-level abstract generators for you to reuse in your own keras pipelines. The current base generators are: <br>
* `BaseGenerator`: Basic framework for generating thread-safe data in keras. (no preprocessing and channels_last)
  Based on https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
* `BaseTransformGenerator`: Loads data and applies data augmentation with `batchgenerators.transforms`.
  * Supports channels_last
  * Loads data with nibabel

## Common Generator Arguments
Overall, these base generators are made with the intent to be open to a variety of potential uses (particularly targeting image segmentation i/o pipelines). Hence, I made it so that the main arguments for these generators are:
* `list_IDs`: list of all your filenames. The idea behind this is that you can just load your (image, mask) pairs for segmentation easily (assuming they have the same file name). For classification tasks, this may not be optimal, but this directory was targeting mainly segmentation users. I'll look into branching out with a separate classification base generator though.
* `data_dirs`: [x_dir, y_dir]. This was a more ambiguous approach so that you can freely play around with your directory structure.
  * Changes to look out for in the future: Dividing this into just `x_dir`, `y_dir`
  * Building a base generator for multiple inputs (iterating through `x_dir`, `y_dir`)
* `batch_size`: batch size for the network
* `n_channels`: The number of channels that your data has. This parameter exists to handle cases where the number of input channels does not match that of the output segmentation.
* `n_classes`: The number of classes. Again, this is to handle cases where the number of input channels does not match that of the output segmentation.
* `steps_per_epoch`: steps per epoch (# of samples per epoch = steps_per_epoch * batch size)
* `shuffle`: boolean on whether or not you want to shuffle after each epoch
