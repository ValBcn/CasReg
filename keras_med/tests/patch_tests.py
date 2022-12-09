import unittest
import numpy as np
from keras_med_io.utils.patch_utils import *

class PatchTest(unittest.TestCase):
    def setUp(self):
        """
        Initializing the parameters:
            n_channels:
            shapes: tuples with channels_first shapes
            images & labels: numpy arrays
            extractors: 2D and 3D versions for all the patch extractor classes.
        """
        self.n_channels = 4
        self.image_shape_2D = (408, 408, self.n_channels)
        self.patch_shape_2D = np.asarray((128, 128))
        self.train_image_2D = np.zeros(self.image_shape_2D)
        self.label_image_2D = np.ones(self.image_shape_2D)

        self.image_shape_3D = (155, 240, 240, self.n_channels)
        self.patch_shape_3D = np.asarray((128,128,128))
        self.train_image_3D = np.zeros(self.image_shape_3D)
        self.label_image_3D = np.ones(self.image_shape_3D)
        self.overlap = int(round(self.patch_shape_2D[0] / 2))

        self.extractor_2D_last = PatchExtractor(ndim = 2)
        self.extractor_3D_last = PatchExtractor(ndim = 3)
        self.extractor_posrandom_2D = PosRandomPatchExtractor(ndim = 2, overlap = self.overlap, pos_sample_intent = False)
        self.extractor_posrandom_3D = PosRandomPatchExtractor(ndim = 3, overlap = self.overlap, pos_sample_intent = True)

    def test_compute_patch_indices(self):
        """
        Tests the utility function to make sure it returns the correct dimensions
        """
        n_dims = 2
        patch_indices = self.extractor_2D_last.compute_patch_indices(self.image_shape_2D[:-1], self.patch_shape_2D,
                                                                     self.overlap)
        self.assertEqual(patch_indices.shape[1], n_dims)

    def test_2D_patch_extraction(self):
        """
        Tests to make sure that the outputs are the correct shapes for both channels_first and channels_last for 2D patch extraction
        """
        # testing channels_last extraction
        output_shape = tuple(list(self.patch_shape_2D) + [self.n_channels])
        patch_indices = self.extractor_2D_last.compute_patch_indices(self.image_shape_2D[:-1], self.patch_shape_2D, self.overlap)
        x = self.extractor_2D_last.extract_patch(self.train_image_2D, self.patch_shape_2D, patch_indices[0])
        self.assertEqual(x.shape, output_shape)

    def test_3D_patch_extraction(self):
        """
        Tests to make sure that the outputs are the correct shapes for both channels_first and channels_last for 3D patch extraction
        """
        # testing channels_last extraction
        output_shape = tuple(list(self.patch_shape_3D) + [self.n_channels])
        patch_indices = self.extractor_3D_last.compute_patch_indices(self.image_shape_3D[:-1], self.patch_shape_3D, self.overlap)
        x = self.extractor_3D_last.extract_patch(self.train_image_3D, self.patch_shape_3D, patch_indices[0])
        self.assertEqual(x.shape, output_shape)

    def test_posrandom_extraction_2D(self):
        """
        Testing that both pos_sample = True and pos_sample = False work for the PosRandomPatchExtractor class in 2D images
        (with the channels dim)
        """
        # testing channels_last extraction
        output_shape = tuple(list(self.patch_shape_2D) + [self.n_channels])
        x, y = self.extractor_posrandom_2D.extract_posrandom_patches(self.train_image_2D, self.label_image_2D,
                                                                     self.patch_shape_2D, pos_sample = False)
        self.assertEqual(x.shape, output_shape)

        x, y = self.extractor_posrandom_2D.extract_posrandom_patches(self.train_image_2D, self.label_image_2D,
                                                                     self.patch_shape_2D, pos_sample = True)
        self.assertEqual(x.shape, output_shape)

    def test_posrandom_extraction_3D(self):
        """
        Testing that both pos_sample = True and pos_sample = False work for the PosRandomPatchExtractor class in 3D images
        (with the channels dim)
        """
        # testing channels_last extraction
        output_shape = tuple(list(self.patch_shape_3D) + [self.n_channels])
        x, y = self.extractor_posrandom_3D.extract_posrandom_patches(self.train_image_3D, self.label_image_3D,
                                                                     self.patch_shape_3D, pos_sample = False)
        self.assertEqual(x.shape, output_shape)

        x, y = self.extractor_posrandom_3D.extract_posrandom_patches(self.train_image_3D, self.label_image_3D,
                                                                     self.patch_shape_3D, pos_sample = True)
        self.assertEqual(x.shape, output_shape)

unittest.main(argv=[''], verbosity=2, exit=False)
