import os, glob
import torch, sys
from torch.utils.data import Dataset
from .data_utils import pkload
import matplotlib.pyplot as plt

import numpy as np

def volgen(vol_names, transforms, batch_size=2, get_file_name=False):
    # convert glob path to filenames
    if isinstance(vol_names, str):
        if os.path.isdir(vol_names):
            vol_names = os.path.join(vol_names, '*')
        vol_names = glob.glob(vol_names)

    indices = np.random.randint(len(vol_names), size=batch_size)
    while indices[0] == indices[1]:
        indices = np.random.randint(len(vol_names), size=batch_size)

    vol1, seg1 = np.load(vol_names[indices[0]])["vol"], np.load(vol_names[indices[0]])["seg"]
    vol2, seg2 = np.load(vol_names[indices[1]])["vol"], np.load(vol_names[indices[1]])["seg"]

    vol1, seg1, vol2, seg2 = vol1[None,None, ...], seg1[None,None, ...], vol2[None,None, ...], seg2[None,None, ...]

    vol1,seg1 = transforms([vol1, seg1])
    vol2, seg2 = transforms([vol2, seg2])

    vol1 = np.ascontiguousarray(vol1)
    seg1 = np.ascontiguousarray(seg1)
    vol2 = np.ascontiguousarray(vol2)
    seg2 = np.ascontiguousarray(seg2)

    if get_file_name == False:
        yield torch.tensor(vol1,requires_grad=True).cuda(), torch.tensor(vol2,requires_grad=True).cuda(), torch.tensor(seg1,requires_grad=True).cuda(), torch.tensor(seg2,requires_grad=True).cuda(),
    if get_file_name == True:
        yield torch.tensor(vol1,requires_grad=True).cuda(), torch.tensor(vol2,requires_grad=True).cuda(), torch.tensor(seg1,requires_grad=True).cuda(), torch.tensor(seg2,requires_grad=True).cuda(), str(vol_names[indices[0]]).split("/")[-1], str(vol_names[indices[1]]).split("/")[-1]


def volgen_one(vol_names, transforms, batch_size=2, get_file_name=False):
    # convert glob path to filenames
    if isinstance(vol_names, str):
        if os.path.isdir(vol_names):
            vol_names = os.path.join(vol_names, '*')
        vol_names = glob.glob(vol_names)

    indices = np.random.randint(len(vol_names), size=batch_size)
    while indices[0] == indices[1]:
        indices = np.random.randint(len(vol_names), size=batch_size)

    vol1, seg1 = np.load(vol_names[indices[0]])["vol"], np.load(vol_names[indices[0]])["seg"]

    vol1, seg1= vol1[None,None, ...], seg1[None,None, ...]

    vol1,seg1 = transforms([vol1, seg1])

    vol1 = np.ascontiguousarray(vol1)
    seg1 = np.ascontiguousarray(seg1)

    if get_file_name == False:
        yield torch.tensor(vol1,requires_grad=True).cuda(), torch.tensor(seg1,requires_grad=True).cuda()
    if get_file_name == True:
        yield torch.tensor(vol1,requires_grad=True).cuda(), torch.tensor(seg1,requires_grad=True).cuda(), str(vol_names[indices[0]]).split("/")[-1]



def npzload(fname):
    with open(fname, 'rb') as f:
        return np.load(f)["vol"], np.load(f)["seg"]

class volgen_batch(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms
        
    def __getitem__(self, index):
        path = self.paths[index]
        vol, seg = npzload(path)

        vol, seg = vol[None, ...], seg[None, ...]
        
        vol, seg = self.transforms([vol, seg])
       
        vol = np.ascontiguousarray(vol)
        seg = np.ascontiguousarray(seg)
    
        return vol, seg
    
    def __len__(self):
        return len(self.paths)

class JHUBrainDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, y = pkload(path)
        #print(x.shape)
        #print(x.shape)
        #print(np.unique(y))
        # print(x.shape, y.shape)#(240, 240, 155) (240, 240, 155)
        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]
        # print(x.shape, y.shape)#(1, 240, 240, 155) (1, 240, 240, 155)
        x,y = self.transforms([x, y])
        #y = self.one_hot(y, 2)
        #print(y.shape)
        #sys.exit(0)
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        #plt.figure()
        #plt.subplot(1, 2, 1)
        #plt.imshow(x[0, :, :, 8], cmap='gray')
        #plt.subplot(1, 2, 2)
        #plt.imshow(y[0, :, :, 8], cmap='gray')
        #plt.show()
        #sys.exit(0)
        #y = np.squeeze(y, axis=0)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.paths)


class JHUBrainInferDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, y, x_seg, y_seg = pkload(path)
        #print(x.shape)
        #print(x.shape)
        #print(np.unique(y))
        # print(x.shape, y.shape)#(240, 240, 155) (240, 240, 155)
        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]
        # print(x.shape, y.shape)#(1, 240, 240, 155) (1, 240, 240, 155)
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        #y = self.one_hot(y, 2)
        #print(y.shape)
        #sys.exit(0)
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        #plt.figure()
        #plt.subplot(1, 2, 1)
        #plt.imshow(x[0, :, :, 8], cmap='gray')
        #plt.subplot(1, 2, 2)
        #plt.imshow(y[0, :, :, 8], cmap='gray')
        #plt.show()
        #sys.exit(0)
        #y = np.squeeze(y, axis=0)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)
