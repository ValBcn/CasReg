from builtins import range
import numpy as np

def get_random_slice_idx(arr):
    slice_dim = arr.shape[0]
    return np.random.choice(slice_dim)

def get_positive_idx(label, channels_format = "channels_last"):
    """
    Gets a random positive patch index that does not include the channels and batch_size dimensions.
    Args:
        label: one-hot encoded numpy array with the dims (n_channels, x,y,z)
    Returns:
        A numpy array representing a 3D random positive patch index
    """
    try:
        assert len(label.shape) == 4
    except AssertionError:
        # adds the channel dim if it doesn't already have it
        if channels_format == "channels_first":
            label = np.expand_dims(label, axis = 0)
        elif channels_format == "channels_last":
            label = np.expand_dims(label, axis = -1)

    # "n_dims" numpy arrays of all possible positive pixel indices for the label
    if channels_format == "channels_first":
        pos_idx_new = np.nonzero(label)[1:]
    elif channels_format == "channels_last":
        pos_idx_new = np.nonzero(label)[:-1]

    # finding random positive class index
    pos_idx = np.dstack(pos_idx_new).squeeze()
    random_coord_idx = np.random.choice(pos_idx.shape[0]) # choosing random coords out of pos_idx
    random_pos_coord = pos_idx[random_coord_idx]
    return random_pos_coord
