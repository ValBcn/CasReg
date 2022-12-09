import os, sys

sys.path.append("/home/valentin/Desktop/Vox_pipeline/keras_med/")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable tensorflow warnings
import argparse
import glob
import time
import numpy as np
#import voxelmorph as vxm
from scipy.interpolate import interpn
from scipy import ndimage, misc
import nibabel as nib
import SimpleITK as sitk

from keras_med_io.inference.infer_utils import pad_nonint_extraction, undo_reshape_padding
from skimage.transform import resize

from io_img_only import LocalPreprocessingBinarySeg
from io_utils import resize_data, multi_channel_labels, nii_to_np, resize_and_uncrop, hist_match, dice

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def normalization_0_1(img):
    return (img-img.min())/(img.max()-img.min())

def z_norm(img):
    return (img-img.mean())/img.std()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="For preprocessing the dataset")
    parser.add_argument("--img_path", type = str, required = True,
                        help = "Path to the images")
    parser.add_argument("--label_path", type = str, required = True,
                        help = "Path to the labels")
    parser.add_argument("--pre_dir", type = str, required = True,
                        help = "Path to the directory where the preprocessed data are saved")
    parser.add_argument("--npz_dir", type = str, required = False,
                        help = "Path to the directory where the npz data are saved")
    parser.add_argument("--reshape", nargs="+", type = int, required = False, default = [128,128,128],
                        help = "size x of the reshaping (new shape = (x,x,x), x must be a multiple of 16.")
    parser.add_argument("--std_norm", type = float, required = False, default = .15,
                        help = "value of the std normalization")
    parser.add_argument("--auto_preprocess", type = int, required = False, default = 1,
                        help = "auto cropping / normalization")

    ########################## PREPROCESSING ############################

    args = parser.parse_args()

    t1 = time.process_time()

    # configure unet features                                                                                                                                                                           

    vol_shape = (128,128,128)
        
    # getting the files names


    for imgs,labels in zip(sorted(os.listdir(args.img_path)),sorted(os.listdir(args.label_path))):

        img_file = imgs.split("/")[-1]

        id_img = img_file.split(".")[0]

        label_file = str(labels).split("/")[-1]

        print("ID image: %s" %(str(id_img)))

        id_label = label_file.split(".")[0]
        
        # get vox spacing and origin using sitk (it's a pain in the ass with nibabel)

        img_sitk = sitk.ReadImage(os.path.join(args.img_path,imgs))
        lab_sitk = sitk.ReadImage(os.path.join(args.label_path,labels))
        spacing = img_sitk.GetSpacing()
        origin = img_sitk.GetOrigin()
        direction = img_sitk.GetDirection()

        img_nib = nib.load(os.path.join(args.img_path,imgs))
        lab_nib = nib.load(os.path.join(args.label_path,labels))
        old_shape_img = nii_to_np(img_nib).shape
        
        print("\n \nPreprocessing %s ( %s )" %(img_file, label_file))
        
        # cropping + normalizing, save the cropping and original spacing to reconstruct the image and seg in the postprocessing"
        if args.auto_preprocess == 1: 
            preprocessed = LocalPreprocessingBinarySeg(os.path.join(args.img_path,imgs), args.pre_dir, 0, os.path.join(args.label_path,labels))
            preprocessed_img, preprocessed_label, spacing, coords = preprocessed.gen_data()
        else:
            img_np = sitk.GetArrayFromImage(img_sitk)
            lab_np = sitk.GetArrayFromImage(lab_sitk)
            preprocessed_img, preprocessed_label = img_np[10:148,25:175,3:131], lab_np[20:148,25:175,3:131]
            coords = [[20,148],[25,175],[3,131]]
            # preprocessed_img = normalization_0_1(preprocessed_img)
            # preprocessed_label = normalization_0_1(preprocessed_label)

        #print("\n \nSaved preprocessed volumes and segmentation in %s" %(str(args.pre_dir)))

        # Resizing the data
    
        print("\nResizing the data to (%i,%i,%i)" %(args.reshape[0],args.reshape[1],args.reshape[2]))
        #print("%s volume" %fixed_img_file)
        resized_img = resize_data(preprocessed_img,args.reshape[0],args.reshape[1],args.reshape[2])
        #resized_img = resize(preprocessed_img, (args.reshape[0],args.reshape[1],args.reshape[2]))
        #print("%s labels" %fixed_label_file)
        print("shape resized label :", preprocessed_label.shape)
        resized_label = resize_data(preprocessed_label,args.reshape[0],args.reshape[1],args.reshape[2])
        #resized_img = resize(preprocessed_img, (args.reshape[0],args.reshape[1],args.reshape[2]))

        # std normalization

        print("Old std = %f" %(resized_img.std()))
        #resized_img = resized_img/resized_img.std() * args.std_norm

        # Converting to npz
    
        print("\n \nConverting to .npz format")
        vol ,seg = resized_img, resized_label
        np.savez(str(os.path.join(args.npz_dir, str(id_img) + ".npz")), vol=vol, seg=seg, coords = coords, spacing = spacing, origin = origin, direction = direction, old_shape = old_shape_img, vol_original = img_nib, seg_original = lab_nib)
    
        print("Min/Max : %f/%f" %(vol.min(),vol.max()))

        
