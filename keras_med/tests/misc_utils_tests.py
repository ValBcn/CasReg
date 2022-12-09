from keras_med_io.utils.misc_utils import load_data, get_list_IDs, add_channel, get_multi_class_labels, KFold
import unittest
import os
import nibabel as nib
import numpy as np

class Misc_Utils_Test(unittest.TestCase):
    """
    Testing all the functions in `utils.misc_utils.py`:
    * get_multi_class_labels
    * load_data
    * get_list_IDs
    * add_channel
    """
    def setUp(self):
        """
        Providing necessary paths to datasets. Please change these paths if you want to run the tests on your own machines.
        """
        # .nii.gz paths
        self.base_path = "C:\\Users\\jchen\\Desktop\\Datasets\\Task02_Heart"
        self.train_path = os.path.join(self.base_path, "imagesTr")
        self.labels_path = os.path.join(self.base_path, "labelsTr")
        # .npy paths
        self.base_npy_path = "C:\\Users\\jchen\\Desktop\\Datasets\\Heart_Isensee"
        self.train_npy_path = os.path.join(self.base_path, "imagesTr")

    def test_get_multi_class_labels(self):
        """
        Tests that the resultant arrays are one hot encoded properly
        1. 0's and 1's only
        2. Regions with non-zero values are 1's in the resultant array.
            * No point in testing this point with binary labels.
        3. Shape is correct.
        """
        sample_fname = os.path.join(self.labels_path, os.listdir(self.labels_path)[0])
        label = nib.load(sample_fname).get_fdata()
        orig_shape = label.shape
        n_labels = 2
        # counting the background class as a separate class
        one_hot = get_multi_class_labels(label, n_labels = n_labels, remove_background = False)
        unique, counts = np.unique(one_hot, return_counts = True)
        assert counts[0] > counts[1] # more zeros than ones for segmentations (for medical datasets)
        self.assertEqual(list(unique), [0,1])
        self.assertEqual(orig_shape + (2,), one_hot.shape)

    def test_get_multi_class_labels_nobackground(self):
        """
        Tests get_multi_class_labels's remove_background argument
        1. Shape check
        2. check that the removed slice was blank
        """
        sample_fname = os.path.join(self.labels_path, os.listdir(self.labels_path)[0])
        label = nib.load(sample_fname).get_fdata()
        orig_shape = label.shape
        n_labels = 2
        # Checking that the removed slice was blank (indicating a background slice)
        one_hot = get_multi_class_labels(label, n_labels = n_labels, remove_background = False)
        without_background = n_labels - 1
        blank_unique = list(np.unique(one_hot[-without_background:])) # removing the background
        self.assertEqual(blank_unique, [0])

        # counting the background class as a separate class
        one_hot = get_multi_class_labels(label, n_labels = n_labels, remove_background = True)
        unique = list(np.unique(one_hot))
        self.assertEqual(unique, [0,1])
        self.assertEqual(orig_shape + (1,), one_hot.shape)

    def test_load_data_npy(self):
        """
        Tests that load_data can actually load all its supported data formats
            * 'npy': data is a .npy file
        """
        fnames_npy = os.listdir(self.train_npy_path)
        load_npy = load_data(os.path.join(self.train_npy_path, fnames_npy[0]))
        self.assertTrue(True)

    def test_load_data_nii_gz(self):
        """
        Tests that load_data can actually load all its supported data formats
            * 'nii': data is a .nii or .nii.gz file
        """
        fnames_niigz = os.listdir(self.train_path)
        load_niigz = load_data(os.path.join(self.train_path, fnames_niigz[0]))
        self.assertTrue(True)

    def test_get_list_IDs(self):
        """
        Tests that get_list_IDs
        1. produces unique files in the folds
        2. correctly divides the folds (correct length)
        Note: We're using the path to the .npy files instead of the .nii.gz because the original directory
        had some unnecessary files that start with "._".
        """
        n_files = len(os.listdir(self.train_npy_path))
        id_dict = get_list_IDs(self.train_npy_path, splits = [0.6, 0.2, 0.2])
         # testing fold lengths
        self.assertEqual(n_files * 0.6, len(id_dict['train']))
        self.assertEqual(n_files * 0.2, len(id_dict['val']))
        self.assertEqual(n_files * 0.2, len(id_dict['test']))
        # testing that the folds have unique files
        combined = id_dict['train'] + id_dict['val'] + id_dict['test']
        self.assertEqual(n_files, len(combined))
        self.assertEqual(n_files, np.unique(combined).size)

    def test_KFold_nodict(self):
        """
        Tests that the KFold function: (return_dict = False)
        1. produces unique folds
        2. correctly divided folds (the right length wrt the provided percentages)
        3. randomly outputs folds. (Will produce different folds each time)
        """
        n_files = len(os.listdir(self.train_npy_path))
        # returns lists of files
        train, val, test = KFold(self.train_npy_path, splits = [0.6, 0.2, 0.2], return_dict = False)
        # 1. testing that the folds have unique files
        combined = train + val + test
        self.assertEqual(n_files, len(combined))
        self.assertEqual(n_files, np.unique(combined).size)
         # 2. testing fold lengths
        self.assertEqual(n_files * 0.6, len(train))
        self.assertEqual(n_files * 0.2, len(val))
        self.assertEqual(n_files * 0.2, len(test))
        # 3. testing that the folds will be different each time (for 3 iterations)
            # comparing newly generated to the original
        for i in range(3):
            train_new, val_new, test_new = KFold(self.train_npy_path, splits = [0.6, 0.2, 0.2], return_dict = False)
            self.assertTrue(train_new != train)
            self.assertTrue(val_new != val)
            self.assertTrue(test_new != test)

    def test_KFold_withdict(self):
        """
        Tests that the KFold function: (return_dict = True)
        1. produces unique folds
        2. correctly divided folds (the right length wrt the provided percentages)
        3. randomly outputs folds. (Will produce different folds each time)
        """
        n_files = len(os.listdir(self.train_npy_path))
        # returns lists of files
        id_dict = KFold(self.train_npy_path, splits = [0.6, 0.2, 0.2], return_dict = True)
        train, val, test = id_dict['train'], id_dict['val'], id_dict['test']
        # 1. testing that the folds have unique files
        combined = train + val + test
        self.assertEqual(n_files, len(combined))
        self.assertEqual(n_files, np.unique(combined).size)
         # 2. testing fold lengths
        self.assertEqual(n_files * 0.6, len(train))
        self.assertEqual(n_files * 0.2, len(val))
        self.assertEqual(n_files * 0.2, len(test))
        # 3. testing that the folds will be different each time (for 3 iterations)
            # comparing newly generated to the original
        for i in range(3):
            id_dict_new = KFold(self.train_npy_path, splits = [0.6, 0.2, 0.2], return_dict = True)
            self.assertTrue(id_dict_new['train'] != train)
            self.assertTrue(id_dict_new['val'] != val)
            self.assertTrue(id_dict_new['test'] != test)

    def test_add_channel(self):
        """
        Tests that add_channel adds a channel to the last dimension
        """
        image_shape_2D = (408, 408)
        new_shape = image_shape_2D + (1,)
        image = np.ones(image_shape_2D)
        added_channels = add_channel(image)
        self.assertEqual(added_channels.shape, new_shape)

unittest.main(argv=[''], verbosity=2, exit=False)
