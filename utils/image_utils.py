"""
Image processing and transformation utilities.
"""

import re
import numpy as np
from scipy.interpolate import interpn


def process_label():
    """Process labeling information for FreeSurfer."""
    seg_table = [0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26,
                 28, 30, 31, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62,
                 63, 72, 77, 80, 85, 251, 252, 253, 254, 255]

    file1 = open('label_info.txt', 'r')
    Lines = file1.readlines()
    dict = {}
    seg_i = 0
    seg_look_up = []
    for seg_label in seg_table:
        for line in Lines:
            line = re.sub(' +', ' ', line).split(' ')
            try:
                int(line[0])
            except:
                continue
            if int(line[0]) == seg_label:
                seg_look_up.append([seg_i, int(line[0]), line[1]])
                dict[seg_i] = line[1]
        seg_i += 1
    return dict


def create_gaussian_flow(img, seg, nb_gauss):
    """
    Create Gaussian flow field for data augmentation.

    Args:
        img: Input image
        seg: Input segmentation
        nb_gauss: Number of Gaussian components

    Returns:
        Warped image, warped segmentation, and flow field
    """
    img_size = img.shape
    sigma = []
    centroid_list = []
    amplitudes = []

    for i in range(nb_gauss):
        centroid_list.append((int(img_size[0]*np.random.rand()),
                              int(img_size[1]*np.random.rand()),
                              int(img_size[2]*np.random.rand())))
        amplitudes.append(20*(np.random.rand(3)-0.5))
        sigma.append(1000*(np.random.rand()))

    x, y, z = np.meshgrid(np.arange(img_size[1]),
                          np.arange(img_size[0]),
                          np.arange(img_size[2]))

    u,v,w = np.zeros(img_size),np.zeros(img_size),np.zeros(img_size)
    for i in range(nb_gauss):
        xnode = centroid_list[i][0]
        ynode = centroid_list[i][1]
        znode = centroid_list[i][2]
        dist = np.sqrt((x-xnode)**2+(y-ynode)**2+(z-znode)**2)
        u += amplitudes[i][0]*np.exp(-(dist)**2/sigma[i])
        v += amplitudes[i][1]*np.exp(-(dist)**2/sigma[i])
        w += amplitudes[i][2]*np.exp(-(dist)**2/sigma[i])

    flow=np.stack((u,v,w))
    flow=np.transpose(flow,(1,2,3,0))

    xx = np.arange(img_size[1])
    yy = np.arange(img_size[0])
    zz = np.arange(img_size[2])
    grid = np.rollaxis(np.array(np.meshgrid(xx, yy, zz)), 0, 4)

    sample = np.zeros(flow.shape)
    sample[:,:,:,0] = flow[:,:,:,1] + grid[:,:,:,0]
    sample[:,:,:,1] = flow[:,:,:,0] + grid[:,:,:,1]
    sample[:,:,:,2] = flow[:,:,:,2] + grid[:,:,:,2]

    sample = np.stack((sample[:, :, :, 1], sample[:, :, :, 0], sample[:, :, :, 2]), 3)

    # warping the labels
    warped = interpn((yy, xx, zz), img, sample, method='linear', bounds_error=False, fill_value=img.min())
    warped_seg = interpn((yy, xx, zz), seg, sample, method='nearest', bounds_error=False, fill_value=seg.min())
    return warped, warped_seg, flow