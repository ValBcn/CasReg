import os, glob
import torch, sys
from torch.utils.data import Dataset
from .data_utils import pkload
import matplotlib.pyplot as plt
import random
from scipy.ndimage import gaussian_filter

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
    
    seg1[seg1==4] = 0
    seg1[seg1==5] = 4
    seg1[seg1==6] = 5
    seg1[seg1==7] = 6
    seg1[seg1==9] = 6
    seg1[seg1==8] = 7
    
    seg2[seg2==4] = 0
    seg2[seg2==5] = 4
    seg2[seg2==6] = 5
    seg2[seg2==7] = 6
    seg2[seg2==9] = 6
    seg2[seg2==8] = 7
    
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


def volgen_dhcp(vol_names, batch_size=2, get_file_name=False):
    # convert glob path to filenames
    if isinstance(vol_names, str):
        if os.path.isdir(vol_names):
            vol_names = os.path.join(vol_names, '*')
        vol_names = glob.glob(vol_names)

    indices = np.random.randint(len(vol_names), size=1)
    
    print("processing", vol_names[indices[0]].split('/')[-1])

    vol1 = np.load(vol_names[indices[0]])["vol1"]
    vol2 = np.load(vol_names[indices[0]])["vol2"]

    vol1, vol2 = vol1[None,None, ...], vol2[None,None, ...]
    
    vol1 = np.ascontiguousarray(vol1)
    vol2 = np.ascontiguousarray(vol2)

    yield torch.tensor(vol1,requires_grad=True, dtype=torch.float32).cuda(), torch.tensor(vol2,requires_grad=True, dtype=torch.float32).cuda()


def volgen_dhcp_hybrid(vol_names, batch_size=2, get_file_name=False):
    # OPTIMIZATION B: Cache NPZ data in memory to eliminate disk I/O bottleneck
    if not hasattr(volgen_dhcp_hybrid, '_cached_data'):
        print("OPTIMIZATION: Loading all NPZ files into memory for volgen_dhcp_hybrid...")

        # convert glob path to filenames
        if isinstance(vol_names, str):
            if os.path.isdir(vol_names):
                vol_names = os.path.join(vol_names, '*')
            vol_names = glob.glob(vol_names)

        # Load all NPZ files into memory
        volgen_dhcp_hybrid._cached_data = {}
        volgen_dhcp_hybrid._file_list = []

        for i, file_path in enumerate(vol_names):
            if i % 10 == 0:
                print(f"Loading file {i+1}/{len(vol_names)}: {os.path.basename(file_path)}")

            try:
                npz_data = np.load(file_path)
                filename = os.path.basename(file_path)

                # OPTIMIZATION C: Pre-process segmentations (using lab2 for this function)
                seg2 = npz_data["lab2"].copy()
                seg2[seg2 == 0] = 1
                seg2[seg2 == 4] = 1
                seg2[seg2!=1] = 1e-5
                seg2 = gaussian_filter(seg2, sigma=.5)
                seg2 = (seg2-seg2.min())/(seg2.max()-seg2.min())

                # Store preprocessed data
                volgen_dhcp_hybrid._cached_data[file_path] = {
                    'vol1': npz_data["vol1"],
                    'vol2': npz_data["vol2"],
                    'seg2_processed': seg2[:-1,:-1,:-1],
                    'gw1': npz_data.get("gw1", 0.0),
                    'gw2': npz_data.get("gw2", 0.0),
                    'filename': filename
                }

                volgen_dhcp_hybrid._file_list.append(file_path)
                npz_data.close()

            except Exception as e:
                print(f"WARNING: Failed to load {file_path}: {e}")
                continue

        print(f"OPTIMIZATION: Loaded {len(volgen_dhcp_hybrid._file_list)} files into memory cache")

    # INFINITE GENERATOR: Keep yielding random samples forever
    while True:
        # Fast random sampling from cached data
        indices = np.random.randint(len(volgen_dhcp_hybrid._file_list), size=1)
        selected_file = volgen_dhcp_hybrid._file_list[indices[0]]

        # Get pre-loaded and pre-processed data
        cached_data = volgen_dhcp_hybrid._cached_data[selected_file]

        print("processing %s", cached_data['filename'])

        # Extract pre-processed data (normal order for this function)
        vol1 = cached_data["vol1"]
        vol2 = cached_data["vol2"]
        seg2_processed = cached_data["seg2_processed"]
        gw1 = cached_data["gw1"]
        gw2 = cached_data["gw2"]
        filename = cached_data["filename"]

        # Add batch and channel dimensions
        vol1 = vol1[None,None, ...]
        vol2 = vol2[None,None, ...]

        vol1 = np.ascontiguousarray(vol1)
        vol2 = np.ascontiguousarray(vol2)

        # Create tensors efficiently
        vol1_tensor = torch.from_numpy(vol1).float().cuda().requires_grad_(True)
        vol2_tensor = torch.from_numpy(vol2).float().cuda().requires_grad_(True)

        yield vol1_tensor, vol2_tensor, seg2_processed, gw1, gw2, filename


def volgen_dhcp_hybrid_inv(vol_names, batch_size=2, get_file_name=False):
    # OPTIMIZATION B: Cache NPZ data in memory to eliminate disk I/O bottleneck
    if not hasattr(volgen_dhcp_hybrid_inv, '_cached_data'):
        print("OPTIMIZATION: Loading all NPZ files into memory (one-time setup)...")

        # convert glob path to filenames
        if isinstance(vol_names, str):
            if os.path.isdir(vol_names):
                vol_names = os.path.join(vol_names, '*')
            vol_names = glob.glob(vol_names)

        # Load all NPZ files into memory
        volgen_dhcp_hybrid_inv._cached_data = {}
        volgen_dhcp_hybrid_inv._file_list = []

        for i, file_path in enumerate(vol_names):
            if i % 10 == 0:  # Progress indicator
                print(f"Loading file {i+1}/{len(vol_names)}: {os.path.basename(file_path)}")

            try:
                npz_data = np.load(file_path)
                filename = os.path.basename(file_path)

                # OPTIMIZATION C: Pre-process segmentations (expensive operations done once)
                seg2 = npz_data["lab1"].copy()
                seg2[seg2 == 0] = 1
                seg2[seg2 == 4] = 1
                seg2[seg2!=1] = 1e-5
                seg2 = gaussian_filter(seg2, sigma=.5)
                seg2 = (seg2-seg2.min())/(seg2.max()-seg2.min())

                # Store preprocessed data
                volgen_dhcp_hybrid_inv._cached_data[file_path] = {
                    'vol1': npz_data["vol1"],  # moving
                    'vol2': npz_data["vol2"],  # fixed
                    'seg2_processed': seg2[:-1,:-1,:-1],  # Pre-processed segmentation
                    'gw1': npz_data.get("gw1", 0.0),
                    'gw2': npz_data.get("gw2", 0.0),
                    'filename': filename
                }

                volgen_dhcp_hybrid_inv._file_list.append(file_path)
                npz_data.close()  # Free memory

            except Exception as e:
                print(f"WARNING: Failed to load {file_path}: {e}")
                continue

        print(f"OPTIMIZATION: Loaded {len(volgen_dhcp_hybrid_inv._file_list)} files into memory cache")

    # INFINITE GENERATOR: Keep yielding random samples forever
    while True:
        # Fast random sampling from cached data
        indices = np.random.randint(len(volgen_dhcp_hybrid_inv._file_list), size=1)
        selected_file = volgen_dhcp_hybrid_inv._file_list[indices[0]]

        # Get pre-loaded and pre-processed data (no disk I/O or processing!)
        cached_data = volgen_dhcp_hybrid_inv._cached_data[selected_file]

        print("processing %s", cached_data['filename'])

        # Extract pre-processed data (inverted as per original function name)
        vol1 = cached_data["vol2"]  # fixed -> moving (inverted)
        vol2 = cached_data["vol1"]  # moving -> fixed (inverted)
        seg2_processed = cached_data["seg2_processed"]
        gw1 = cached_data["gw1"]
        gw2 = cached_data["gw2"]
        filename = cached_data["filename"]

        # Add batch and channel dimensions efficiently
        vol1 = vol1[None,None, ...]
        vol2 = vol2[None,None, ...]

        # Convert to contiguous arrays (faster GPU transfer)
        vol1 = np.ascontiguousarray(vol1)
        vol2 = np.ascontiguousarray(vol2)

        # Create tensors efficiently
        vol1_tensor = torch.from_numpy(vol1).float().cuda().requires_grad_(True)
        vol2_tensor = torch.from_numpy(vol2).float().cuda().requires_grad_(True)

        yield vol1_tensor, vol2_tensor, seg2_processed, gw1, gw2, filename
        
        
def volgen_dhcp_hybrid_inv_old(vol_names, batch_size=2, get_file_name=False):
    # convert glob path to filenames
    if isinstance(vol_names, str):
        if os.path.isdir(vol_names):
            vol_names = os.path.join(vol_names, '*')
        vol_names = glob.glob(vol_names)

    indices = np.random.randint(len(vol_names), size=1)

    npz_data = np.load(vol_names[indices[0]])
    vol1 = npz_data["vol2"]
    vol2 = npz_data["vol1"]
    seg2 = npz_data["lab1"]

    # Get gestational ages if available
    gw1 = npz_data.get("gw1", 0.0)
    gw2 = npz_data.get("gw2", 0.0)
    filename = vol_names[indices[0]].split('/')[-1]
    
    seg2[seg2 == 0] = 1
    seg2[seg2 == 4] = 1
    seg2[seg2!=1] = 1e-5

    seg2 = gaussian_filter(seg2,sigma=.5)
    seg2 = (seg2-seg2.min())/(seg2.max()-seg2.min())

    vol1, vol2 = vol1[None,None, ...], vol2[None,None, ...]
    
    vol1 = np.ascontiguousarray(vol1)
    vol2 = np.ascontiguousarray(vol2)

    yield torch.tensor(vol1,requires_grad=True, dtype=torch.float32).cuda(), torch.tensor(vol2,requires_grad=True, dtype=torch.float32).cuda(), seg2[:-1,:-1,:-1], gw1, gw2, filename


def volgen_mas_dhcp(vol_names1, vol_names2, batch_size=2, get_file_name=False):
    # convert glob path to filenames
    
    vol_names1 = os.path.join(vol_names1, '*')
    vol_names1 = glob.glob(vol_names1)
    vol_names2 = os.path.join(vol_names2, '*')
    vol_names2 = glob.glob(vol_names2)

    indices1 = np.random.randint(len(vol_names1))
    indices2 = np.random.randint(len(vol_names2))
    
    gw_ghol = np.load(vol_names1[indices1])["gw"]
    gw_dhcp = np.load(vol_names2[indices2])["gw"]
    
    while np.abs(gw_dhcp-gw_ghol) > 3:
        indices1 = np.random.randint(len(vol_names1))
        indices2 = np.random.randint(len(vol_names2))
        
        gw_ghol = np.load(vol_names1[indices1])["gw"]
        gw_dhcp = np.load(vol_names2[indices2])["gw"]
        
    print("Gestational weeks %.2f - %.2f" %(gw_dhcp,gw_ghol))

    vol1, seg1 = np.load(vol_names1[indices1])["vol"], np.load(vol_names1[indices1])["seg"]
    vol2 = np.load(vol_names2[indices2])["vol"]

    print(vol_names1[indices1].split("/")[-1])
    print(vol_names2[indices2].split("/")[-1])

    vol1, seg1, vol2 = vol1[None,None, ...], seg1[None,None, ...], vol2[None,None, ...]


    vol1 = np.ascontiguousarray(vol1)
    seg1 = np.ascontiguousarray(seg1)
    vol2 = np.ascontiguousarray(vol2)

    yield torch.tensor(vol1,requires_grad=True, dtype=torch.float32).cuda(), torch.tensor(seg1,requires_grad=True, dtype=torch.float32).cuda(), torch.tensor(vol2,requires_grad=True, dtype=torch.float32).cuda()



def volgen_acdc_3D(vol_names, transforms, batch_size=1, get_file_name=False):
    
    # convert glob path to filenames
   if isinstance(vol_names, str):
       if os.path.isdir(vol_names):
           vol_names = os.path.join(vol_names, '*')
       vol_names = glob.glob(vol_names)

   patient_ids = sorted(set([os.path.basename(name).split('_')[0] for name in vol_names]))
   while len(patient_ids) < batch_size:
       patient_ids *= 2
   patient_id = random.choice(patient_ids)
   print("patient id : ", patient_id)
   filenames = [name for name in vol_names if os.path.basename(name).startswith(patient_id)]
   assert len(filenames) > 1, f"patient {patient_id} has less than 2 frames"
   while len(filenames) < 2:
       filenames *= 2
   frame_indices = np.random.randint(len(filenames), size=2)
   while frame_indices[0] == frame_indices[1]:
       frame_indices = np.random.randint(len(filenames), size=2)
   print("frames :", frame_indices)

   vols = []
   segs = []
   for i, idx in enumerate(frame_indices):
       filename = filenames[idx]
       vol = np.load(filename)["vol"]
       seg = np.load(filename)["vol"]
       vol, seg = vol[None,None, ...], seg[None,None, ...]
       vol,seg = transforms([vol, seg])
       vols.append(np.ascontiguousarray(vol))
       segs.append(np.ascontiguousarray(seg))
       
   yield torch.tensor(vols[0], requires_grad=True).cuda(), torch.tensor(vols[1], requires_grad=True).cuda()

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
        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]
        x,y = self.transforms([x, y])
        #y = self.one_hot(y, 2)
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
        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        #y = self.one_hot(y, 2)
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
