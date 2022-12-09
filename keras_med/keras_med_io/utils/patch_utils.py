import numpy as np

class PatchExtractor(object):
    """
    Lean patch extractor class.
    Channels_last.

    Main Methods:
        .extract_patch(): Extracting either 2D/3D patch
    """
    def __init__(self, ndim):
        self.ndim = ndim

    def extract_patch(self, data, patch_shape, patch_index):
        """
        Extracting both 2D and 3D patches depending on patch shape dimensions
        Args:
            data: a numpy array of shape (..., n_channels)
            patch_shape: a tuple representing the patch shape without the batch_size or the channels dimensions
            patch_index:
        Returns:
            a cropped version of the original image array
        """
        patch_index = np.asarray(patch_index, dtype = np.int16)
        patch_shape = np.asarray(patch_shape, dtype = np.int16)

        image_shape = data.shape[:self.ndim]
        if np.any(patch_index < 0) or np.any((patch_index + patch_shape) > image_shape):
            data, patch_index = self.fix_out_of_bound_patch_attempt(data, patch_shape, patch_index)

        if self.ndim == 2:
            return data[patch_index[0]:patch_index[0]+patch_shape[0], patch_index[1]:patch_index[1]+patch_shape[1], ...]#, ...]
        elif self.ndim == 3:
            return data[patch_index[0]:patch_index[0]+patch_shape[0], patch_index[1]:patch_index[1]+patch_shape[1],
                        patch_index[2]:patch_index[2]+patch_shape[2], ...]

    def compute_patch_indices(self, image_shape, patch_shape, overlap, start=None):
        """
        (no channel)
        Args:
            image_shape: ndarray of dimensions
            patch_shape: ndarray of patch dimensions
        Returns:
            a np array of coordinates & step
        """
        if isinstance(overlap, int):
            overlap = np.asarray([overlap] * len(image_shape))
        if start is None:
            n_patches = np.ceil(image_shape / (patch_shape - overlap))
            overflow = (patch_shape - overlap) * n_patches - image_shape + overlap
            start = -np.ceil(overflow/2)
        elif isinstance(start, int):
            start = np.asarray([start] * len(image_shape))
        stop = image_shape + start
        step = patch_shape - overlap
        return self.get_set_of_patch_indices(start, stop, step)

    def get_set_of_patch_indices(self, start, stop, step):
        """
        getting set of all possible indices with the start, stop and step.
        """
        if self.ndim == 2:
            return np.asarray(np.mgrid[start[0]:stop[0]:step[0], start[1]:stop[1]:step[1]].reshape(2, -1).T,
                             dtype=np.int)
        elif self.ndim == 3:
            return np.asarray(np.mgrid[start[0]:stop[0]:step[0], start[1]:stop[1]:step[1],
                                           start[2]:stop[2]:step[2]].reshape(3, -1).T, dtype=np.int)

    def fix_out_of_bound_patch_attempt(self, data, patch_shape, patch_index):
        """
        Pads the data and alters the corner patch index so that the patch will be correct.
        Args:
            data:
            patch_shape:
            patch_index:
        Returns:
            padded data, fixed patch index
        """
        image_shape = data.shape[:self.ndim]
        # figures out which indices need to be padded; if they're < 0
        pad_before = np.abs((patch_index < 0 ) * patch_index) # also need to check if idx-patch_shape < 0
        # checking for out of bounds if doing idx+patch shape by replacing the afflicted indices with a kinda random replacement
        pad_after = np.abs(((patch_index + patch_shape) > image_shape) * ((patch_index + patch_shape) - image_shape))
        pad_args = np.stack([pad_before, pad_after], axis=-1)
        if pad_args.shape[0] < len(data.shape):
            # adding channels dimension to padding ([0,0] so that it's ignored)
            pad_args = pad_args.tolist() + [[0, 0]] * (len(data.shape) - pad_args.shape[0])

        data = np.pad(data, pad_args, mode="edge")
        patch_index += pad_before
        return data, patch_index

    def reconstruct_from_patches(self, patches, patch_indices, data_shape, default_value=0):
        """
        [Only works for 3D patches]
        Reconstructs an array of the original shape from the lists of patches and corresponding patch indices. Overlapping
        patches are averaged.
        Args:
            patches: List of numpy array patches.
            patch_indices: List of indices that corresponds to the list of patches.
            data_shape: Shape of the array from which the patches were extracted.
            default_value: The default value of the resulting data. if the patch coverage is complete, this value will
            be overwritten.
        Returns:
            Numpy array containing the data reconstructed by the patches.
        """
        data = np.ones(data_shape) * default_value
        image_shape = data_shape[-3:]
        count = np.zeros(data_shape, dtype=np.int)
        for patch, index in zip(patches, patch_indices):
            image_patch_shape = patch.shape[-3:]
            if np.any(index < 0):
                fix_patch = np.asarray((index < 0) * np.abs(index), dtype=np.int)
                patch = patch[..., fix_patch[0]:, fix_patch[1]:, fix_patch[2]:]
                index[index < 0] = 0
            if np.any((index + image_patch_shape) >= image_shape):
                fix_patch = np.asarray(image_patch_shape - (((index + image_patch_shape) >= image_shape)
                                                            * ((index + image_patch_shape) - image_shape)), dtype=np.int)
                patch = patch[..., :fix_patch[0], :fix_patch[1], :fix_patch[2]]
            patch_index = np.zeros(data_shape, dtype=np.bool)
            patch_index[...,
                        index[0]:index[0]+patch.shape[-3],
                        index[1]:index[1]+patch.shape[-2],
                        index[2]:index[2]+patch.shape[-1]] = True
            patch_data = np.zeros(data_shape)
            patch_data[patch_index] = patch.flatten()

            new_data_index = np.logical_and(patch_index, np.logical_not(count > 0))
            data[new_data_index] = patch_data[new_data_index]

            averaged_data_index = np.logical_and(patch_index, count > 0)
            if np.any(averaged_data_index):
                data[averaged_data_index] = (data[averaged_data_index] * count[averaged_data_index] + patch_data[averaged_data_index]) / (count[averaged_data_index] + 1)
            count[patch_index] += 1
        return data

class PosRandomPatchExtractor(PatchExtractor):
    """
    Channels_last.
    Attributes:
        ndim: integer representing the number of patches\
        overlap: int representing the amount of patch overlap (default: 0)
        pos_sample_intent: boolean on if there is an intent to positively sample patches

    Main Method:
        extract_posrandom_patches: extracts a positive or random sampled (input, label) patch pair
    """
    def __init__(self, ndim, overlap = 0, pos_sample_intent = False):
        super().__init__(ndim = ndim)
        self.overlap = overlap
        if pos_sample_intent:
            if self.ndim == 2:
                self.pos_slice_dict = self.get_pos_slice_dict()

    def extract_posrandom_patches(self, image, label, patch_shape, pos_sample):
        """
        Takes both image and label and gets cropped random patches
        Args:
            image: 2D/3D single arr with no batch_size dim
            label: 2D/3D single arr with no batch_size dim
            patch_shape: 2D/3D tuple of patch shape without batch_size or n_channels
        Returns:
            a tuple of (cropped_image, cropped_label)
        """
        both = np.concatenate([image, label], axis = -1)
        image_shape = image.shape[:self.ndim]
        n_channels = image.shape[-1]
        # getting patch index
        if pos_sample:
            patch_idx = self.get_positive_idx(label)
        elif not pos_sample:
            patch_idx = self.get_random_idx(image_shape, patch_shape)
        # patch extraction
        both_crop = self.extract_patch(both, patch_shape, patch_idx)
        if self.ndim == 2:
            x, y = both_crop[:,:, :n_channels], both_crop[:, :, n_channels:]
        elif self.ndim == 3:
            x, y = both_crop[:,:, :, :n_channels], both_crop[:, :,:,  n_channels:]
        return x,y

    def get_random_idx(self, image_shape, patch_shape, start = None):
        """
        Gets a random patch index.
        Args:
            image_shape:
            patch_shape:
            start: (Optional) int representing the beginning pixel offset for patch extraction
        Returns:
            A numpy array representing a random patch index
        """
        # getting random patch index
        patch_indices = self.compute_patch_indices(image_shape, patch_shape, self.overlap, start)
        rand_idx = patch_indices[np.random.randint(0, patch_indices.shape[0]-1),:]
        return rand_idx

    def get_positive_idx(self, label, dstack = True):
        """
        Gets a random positive patch index.
        Args:
            label: one-hot encoded numpy array with the dims (x,y) or (x,y,z)
            dstack: boolean on whether or not to dstack the pos_idx for patch_idx
        Returns:
            A numpy array representing a random positive patch index
        """
        try:
            assert len(label.shape) == self.ndim + 1 # assumes that thel label has the n_channels dimensions
        except AssertionError:
            label = np.expand_dims(label, axis = -1) # adds the channel dim if it doesn't already have it
        pos_idx_dims = np.nonzero(label)[:-1] # "n_dims" numpy arrays of all possible positive pixel indices for the label
        if dstack:
            pos_idx = np.dstack(pos_idx_dims).squeeze()
            # random selection of patch
            patch_idx = pos_idx[np.random.randint(0, pos_idx.shape[0]-1), :]
            return patch_idx
        else:
            return pos_idx_dims

    def get_pos_slice_dict(self):
        """
        Returns a dictionary of all positive class slice indices for images corresponding to their ID
        """
        # pos_slice_dict = {}
        # for id in self.list_IDs:
        #     # for file_x, file_y in zip(batch_x, batch_y):
        #     file_y = nib.load(os.path.join(self.data_dirs[1] + id)).get_fdata().squeeze()
        #     pos_slice_dict[id] = self.get_positive_idx(file_y)[0]
        # return pos_slice_dict
        return NotImplementedError("Please overwrite this when making a generator with 2D positively sampled patches.")
