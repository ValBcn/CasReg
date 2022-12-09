import numpy as np
import os
from keras_med_io.utils.misc_utils import load_data

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
