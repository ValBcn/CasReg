from keras_med_io.inference.infer_utils import *
from keras_med_io.utils.shape_io import extract_nonint_region, reshape
import unittest
import os
import numpy as np

class Infer_Utils_Test(unittest.TestCase):
    """
    Testing all the functions in inference/infer_utils:
    * pad_nonint_extraction
    * undo_reshape_padding
    """
    def setUp(self):
        """
        Initializing the parameters:
            n_channels:
            shapes: tuples with channels_first shapes
            images & labels: numpy arrays
            extractors: 2D and 3D versions for all the patch extractor classes.
        """
        self.n_channels = 4
        self.image_shape_3D = (155, 240, 240, self.n_channels)
        self.label_image_3D = np.ones(self.image_shape_3D)

    def test_pad_nonint_extraction(self):
        """
        Tests that `pad_nonint_extraction`'s padding of the extracted image will produce the original shape.
        """
        # setting up a randomly padded image (pad with zeros)
        self.padded_image_3D = np.pad(self.label_image_3D, ([20, 50], [20, 50], [20,50], [0,0]), mode = 'constant')
        orig_shape = self.padded_image_3D.shape
        # extracting image
        extracted_img, coords = extract_nonint_region(self.padded_image_3D, outside_value = 0, coords = True)
        # padding the extraction
        padded_result = pad_nonint_extraction(extracted_img, orig_shape, coords)
        # checking that the original and padded array are the same
        self.assertEqual(padded_result.shape, orig_shape)
        print("padded: ", np.unique(padded_result, return_counts = True), "\norig: ", np.unique(self.padded_image_3D, \
                            return_counts = True))
        self.assertTrue(np.array_equal(self.padded_image_3D, padded_result))

    def test_undo_reshape_padding(self):
        """
        Tests that `undo_reshape_padding` will produce the original image from the padded image
        """
        # reshape(orig_img, append_value=-1024, new_shape=(512, 512, 512)
        # setting up reshaped image
        orig_shape = self.label_image_3D.shape
        new_shape = (512, 512, 512, self.n_channels)
        reshaped_img = reshape(self.label_image_3D, append_value = 0, new_shape = new_shape)
        # undo padding
        undone_img = undo_reshape_padding(reshaped_img, orig_shape)
        # checking that the original and undo-padded array are the same
        self.assertTrue(np.array_equal(undone_img, self.label_image_3D))

unittest.main(argv=[''], verbosity=2, exit=False)
