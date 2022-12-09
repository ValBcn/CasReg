import os
import numpy as np
import nibabel as nib
import pandas as pd

def get_multi_class_labels(data, n_labels, labels=None, remove_background = False):
    """
    One-hot encodes a segmentation label.
    Args:
        data: numpy array containing the label map with shape: (n_samples,..., 1).
        n_labels: number of labels
        labels: list of the integer/float values of the labels
        remove_background: option to drop the background mask (first label 0)
    Returns:
        binary numpy array of shape (n_samples,..., n_labels) or (n_samples,..., n_labels-1)
    """
    new_shape = data.shape + (n_labels,)
    y = np.zeros(new_shape, np.int8)
    for label_index in range(n_labels):
        if labels is not None:
            # with the labels specified
            y[:,:,:, label_index][data == labels[label_index]] = 1
        else:
            # automated
            y[:, :, :, label_index][data == (label_index + 1)] = 1
    if remove_background:
        without_background = n_labels - 1
        y = y[:,:,:,:without_background] # removing the background
    return y

def load_data(data_path, file_format = None):
    """
    Args:
        data_path: path to the image file
        file_format: str representing the format as shown below:
            * 'npy': data is a .npy file
            * 'nii': data is a .nii.gz or .nii file
            * Defaults to None; if it is None, it auto checks for the format
    Returns:
        A loaded numpy array (into memory) with type np.float32
    """
    assert os.path.isfile(data_path), "Please make sure that `data_path` is to a file!"
    # checking for file formats
    if file_format is None:
        if '.nii.gz' in data_path[-7:] or '.nii' in data_path[-4:]:
            file_format = 'nii'
        elif '.npy' in data_path[-4:]:
            file_format = 'npy'
    # loading the data
    if file_format == 'npy':
        return np.load(data_path).astype(np.float32)
    elif file_format == 'nii':
        return nib.load(data_path).get_fdata().astype(np.float32)
    else:
        raise Exception("Please choose a compatible file format: `npy` or `nii`.")

def get_list_IDs(data_dir, splits = [0.6, 0.2, 0.2]):
    """
    Divides filenames into train/val/test sets
    Args:
        data_dir: file path to the directory of all the files; assumes labels and training images have same names
        splits: a list with 3 elements corresponding to the decimal train/val/test splits; [train, val, test]
    Returns:
        a dictionary of file ids for each set
    """
    id_list = os.listdir(data_dir)
    total = len(id_list)
    train = round(total * splits[0])
    val_split = round(total * splits[1]) + train
    test_split = round(total * splits[2]) + val_split
    return {"train": id_list[:train], "val": id_list[train:val_split], "test": id_list[val_split:test_split]
           }

def KFold(data_dir, splits = [0.6, 0.2, 0.2], return_dict = False):
    """
    Divides list_IDs into train/val/test sets
    Args:
        data_dir: directory with files; assumes labels and training images have same names
        splits: a list with 3 elements corresponding to the decimal train/val/test splits; [train, val, test]
        return_dict: whether or not you want to return a dictionary with the filenames organized
    Returns:
        if return_dict is True:
            a dictionary of file ids for each set
        elif return_dict is False:
            tuple of lists of folds (train, validate, test)
    """
    assert np.sum(splits) == 1, "Please make sure that your splits add up to 1."
    splits = [splits[0], splits[1] + splits[0]]
    df = pd.Series(os.listdir(data_dir))
    total = len(df)
    train, validate, test = np.split(df.sample(frac=1), [int(splits[0]*len(df)), int(splits[1]*len(df))])
    assert len(train) + len(validate) + len(test) == total, "There should be no file overlap."
    if return_dict:
      fname_dict = {'train': list(train), 'val': list(validate), 'test': list(test)}
      return fname_dict
    else:
      return (list(train), list(validate), list(test))

def sanity_checks(patch_x, patch_y):
    """
    Checks for NaNs, and makes sure that the labels are one-hot encoded.
    Args:
        patch_x: a numpy array
        patch_y: a numpy array (label)
    Returns:
        True (boolean) if all the asserts run.
    """
    # sanity checks
    checks_nan_x, checks_nan_y = np.any(np.isnan(patch_x)), np.any(np.isnan(patch_y))
    assert not checks_nan_x and not checks_nan_y # NaN checks
    assert np.array_equal(np.unique(patch_y), np.array([0,1])) or np.array_equal(np.unique(patch_y), np.array([0]))
    return True

def add_channel(image):
    """
    Adds a single channel dimension to a 3D or 2D numpy array.
    Args:
        image: a numpy array without a channel dimension
    Returns:
        A single channel numpy array (channels_last)
    """
    return np.expand_dims(image.squeeze(), -1).astype(np.float32)

def compute_pad_value(input_dir, list_IDs):
    """
    Computes the minimum pixel intensity of the entire dataset for the pad value (if it's not 0)
    Args:
        input_dir: directory to input images
        list_IDs: list of filenames
    """
    print("Computing min/pad value...")
    # iterating through entire dataset
    min_list = []
    for id in list_IDs:
        x_train = load_data(os.path.join(input_dir, id))
        min_list.append(x_train.min())
    return np.asarray(min_list).min()
