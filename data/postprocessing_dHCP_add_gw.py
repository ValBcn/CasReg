import os, sys

sys.path.append("/data/CasReg/keras_med/keras_med_io/")
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

from skimage.transform import resize


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
    parser.add_argument("--lab_path", type = str, required = True,
                        help = "Path to the images")
    parser.add_argument("--npz_dir", type = str, required = False,
                        help = "Path to the directory where the npz data are saved")



    ########################## PREPROCESSING ############################

    args = parser.parse_args()

    t1 = time.process_time()

    # configure unet features
    info = np.load("/data/datasets/dHCP/longitudinal_info.npy")
                                                                                                                                                                         

    vol_shape = (192,256,256)
        
    # getting the files names


    for imgs,labs in zip(sorted(os.listdir(args.img_path)),sorted(os.listdir(args.lab_path))):
     
        sub = imgs.split("_")[0]
        sess = imgs.split("_")[1].split(".")[0].split("-")[1]
        
        for imgs2,labs2 in zip(sorted(os.listdir(args.img_path)),sorted(os.listdir(args.lab_path))):
            
            sub2 = imgs2.split("_")[0]
            sess2 = imgs2.split("_")[1].split(".")[0].split("-")[1]
            
            
            if sub == sub2:
                if sess != sess2:
                    
                    # get vox spacing and origin using sitk (it's a pain in the ass with nibabel)
                    
                    for i in range(info.shape[0]):
                        if sub == info[i,0] and sess == info[i,1]:
                            gw1 = float(info[i,2])/7
                            sex = info[i,3]
                        if sub2 == info[i,0] and sess2 == info[i,1]:
                            gw2 = float(info[i,2])/7
                            sex = info[i,3]
            
                    img_sitk = sitk.ReadImage(os.path.join(args.img_path,imgs))
                    lab_sitk = sitk.ReadImage(os.path.join(args.lab_path,labs))
                    img_np = sitk.GetArrayFromImage(img_sitk)[8:200,10:266,10:266]
                    lab_np = sitk.GetArrayFromImage(lab_sitk)[8:200,10:266,10:266]
                    img_np = (img_np-img_np.min())/(img_np.max()-img_np.min())
                    
                    img_sitk2 = sitk.ReadImage(os.path.join(args.img_path,imgs2))
                    lab_sitk2 = sitk.ReadImage(os.path.join(args.lab_path,labs2))
                    img_np2 = sitk.GetArrayFromImage(img_sitk2)[8:200,10:266,10:266]
                    lab_np2 = sitk.GetArrayFromImage(lab_sitk2)[8:200,10:266,10:266]
                    img_np2 = (img_np2-img_np2.min())/(img_np2.max()-img_np2.min())
                    
                    if gw1 < gw2:
                        np.savez(str(os.path.join(args.npz_dir, str(sub) + ".npz")), vol1=img_np, vol2=img_np2, lab1 = lab_np, lab2 = lab_np2, gw1 = gw1, gw2 = gw2, sex = sex)
                        print("gw1 : %.2f, gw2 : %.2f" %(gw1,gw2))
                    if gw1 > gw2:
                        np.savez(str(os.path.join(args.npz_dir, str(sub) + ".npz")), vol1=img_np2, vol2=img_np, lab1 = lab_np2, lab2 = lab_np, gw1 = gw2, gw2 = gw1, sex = sex)
                        print("gw1 : %.2f, gw2 : %.2f" %(gw2,gw1))

    
        
