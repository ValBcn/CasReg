import json
import os, sys
import nibabel as nib
import numpy as np

from keras_med_io.utils.shape_io import reshape, resample_array, extract_nonint_region
from io_utils import *
from os.path import join, isdir


class LocalPreprocessingBinarySeg(object):
    """
    Preprocessing for only binary segmentation tasks
    """
    def __init__(self, img, output_path, flip_label, seg=None):
        self.img = img
        self.seg = seg
        self.output_path = output_path
        self.flip_label = flip_label

    def gen_data(self):
        """
        Generates and saves preprocessed data
        """

        img_path = str(self.img)
        if self.seg is not None:
            label_path = str(self.seg)
            label = nib.load(label_path)
        image = nib.load(img_path)

        orig_spacing = image.header['pixdim'][1:4]
        if self.seg is not None:
            preprocessed_img, preprocessed_label, coords, spacing = isensee_preprocess(image, label, orig_spacing, get_coords = True, ct = False, flip_label=self.flip_label) #, mean_patient_shape = self.mean_patient_shape)
        else:
            preprocessed_img, coords, spacing = isensee_preprocess_no_label(image, orig_spacing, get_coords = True, ct = False) #, mean_patient_shape = self.mean_patient_shape)  

        # output dir in MSD format
        out_images_dir= join(self.output_path, 'images_preprocessed')
        out_labels_dir = join(self.output_path, 'labels_preprocessed')
        # checking to make sure that the output directories exist
        if not isdir(out_images_dir):
            os.mkdir(out_images_dir)
            print("Created directory: ", out_images_dir)
        if not isdir(out_labels_dir):
            os.mkdir(out_labels_dir)
            print("Created directory: ", out_labels_dir)

        # Saving the img and label
        img_file_name = str(self.img).split("/")[-1]
        if self.seg is not None:
            label_file_name = str(self.seg).split("/")[-1]
        
        np.save(os.path.join(out_images_dir, img_file_name), preprocessed_img)
        if self.seg is not None:
            np.save(os.path.join(out_labels_dir, label_file_name), preprocessed_label)
            return preprocessed_img, preprocessed_label, orig_spacing, coords
        else:
            return preprocessed_img, orig_spacing, coords
        
