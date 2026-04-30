import glob
import argparse
import sys, os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import utils.utils as utils
from utils.spatial import register_model
from data.data_utils import cpu
from utils.metrics import dice_multiclass
from utils.io_utils import hd95
from models import losses
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from natsort import natsorted
from models.models import CasReg, CasReg_nhe
from scipy import ndimage
import SimpleITK as sitk
from collections import Counter
import time
from utils.io_utils import hd95, resize_data
from torch import optim
import torch.nn as nn
from scipy.ndimage import gaussian_filter, median_filter, gaussian_filter1d
from skimage import measure
import matplotlib.colors as colors

# import traitsui
from mayavi import mlab
from mayavi.modules.surface import Surface
import trimesh
from utils.io_utils import resize_data
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.colors import ListedColormap

import seaborn as sb
import pandas as pd


from utils.io_utils import resize_data, multi_channel_labels, nii_to_np, resize_and_uncrop, hist_match, dice




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="For training ViT-VNet for registration of 3D fetal brain MRI")
    
    parser.add_argument("--npz_files", type = str, required = True,
                        help = "Path to the base directory where the preprocessed/reshaped npz files for validation are")
    parser.add_argument("--in_weights", type = str, required = False,
                        help = " path to trained weights for biomechanical reg")
    parser.add_argument("--in_weights2", type = str, required = False,
                        help = " path to trained weights for registration to template")
    parser.add_argument("--save_plot_dir", type = str, required = False)
    parser.add_argument("--template", type = str, required = True)
    parser.add_argument("--template_atlas", type = str, required = True)

    

    args = parser.parse_args()

    imgsize = (192,256,256)
            
    # Biomechanical model
    mode = torch.load(args.in_weights)["mode"]
    model = CasReg_nhe(imgsize, batch_size=2, coef_ratio = 1, n_casc=5, mode = "cascade", layer_mode = "contracted")
    best_model = torch.load(args.in_weights)['state_dict']
    model.load_state_dict(best_model)
    model.cuda()
    
    # grad model
    mode = torch.load(args.in_weights2)["mode"]
    model2 = CasReg_nhe(imgsize, batch_size=2, coef_ratio = 1, n_casc=5, mode = "cascade", layer_mode = "contracted")
    best_model2 = torch.load(args.in_weights2)['state_dict']
    model2.load_state_dict(best_model2)
    model2.cuda()

    # Registration tools
    reg_model = register_model((192, 256, 256), 'nearest')
    reg_model_bilin = register_model((192, 256, 256), 'bilinear')
    reg_model.cuda()
    reg_model_bilin.cuda()

    # Common template
    template = np.load(args.template)["vol2"]
    template_seg = np.load(args.template)["lab2"]
    template_st = np.load(args.template_atlas)["struct"]

    template = template[None,None,...]
    template = np.ascontiguousarray(template)
    template = torch.tensor(template,requires_grad=True, dtype=torch.float32).cuda()
    
    Exx_list, Eyy_list, Ezz_list, detJ_list, ADG_list, mu_list, kappa_list, warped_list, nhe1_list, nhe2_list, Tr_list, phi_xy_list, phi_xz_list, phi_yz_list = [], [], [], [], [], [], [], [], [], [], [], [], [], []
    Exy_list = []
    Exz_list = []
    Eyz_list = []
    flows_list = []
    
    structures_list = ['Hippocampus L',
                 'Hippocampus R',
                 'Amygdala L',
                 'Amygdala R',
                 'Ant. temp. lobe, Med. L GM',
                 'Ant. temp. lobe, Med. R GM',
                 'Ant. temp. lobe, Lat. L GM',
                 'Ant. temp. lobe, Lat. R GM',
                 'Gyri parahip. et amb. Ant. L GM',
                 'Gyri parahip. et amb. Ant. R GM',
                 'Sup. temp. gyrus, mid. L GM',
                 'Sup. temp. gyrus, mid. R GM',
                 'Med. & Inf. temp. gyri Ant. L GM',
                 'Med. & Inf. temp. gyri Ant. R GM',
                 'Lat. occipitotemp. gyrus, gyrus fus. Ant. L GM',
                 'Lat. occipitotemp. gyrus, gyrus fus. Ant. R GM',
                 'Cerebellum L',
                 'Cerebellum R',
                 'Brainstem',
                 'Insula R GM',
                 'Insula L GM',
                 'Occip. lobe R GM',
                 'Occip. lobe L GM',
                 'Gyri parahip. et amb. Post. R GM',
                 'Gyri parahip. et amb. Post. L GM',
                 'Lat. occipitotemp. & fusi. gyri Post. R GM',
                 'Lat. occipitotemp. & fusi. gyri Post. L GM',
                 'Med. & Inf. temp. gyri Post. R GM',
                 'Med. & Inf. temp. gyri Post. L GM',
                 'Sup. temp. gyrus, Post. R GM',
                 'Sup. temp. gyrus, Post. L GM',
                 'Cing. gyrus, Ant. R GM',
                 'Cing. gyrus, Ant. L GM',
                 'Cing. gyrus, Post. R GM',
                 'Cing. gyrus, Post. L GM',
                 'Frontal lobe R GM',
                 'Frontal lobe L GM',
                 'Parietal lobe R GM',
                 'Parietal lobe L GM',
                 'Caudate nucleus R',
                 'Caudate nucleus L',
                 'Thalamus R, high intensity in T2',
                 'Thalamus L, high intensity in T2',
                 'Subthalamic nucleus R',
                 'Subthalamic nucleus L ',
                 'Lentiform Nucleus R',
                 'Lentiform Nucleus L',
                 'Corpus Callosum',
                 'Lat. Ventricle L',
                 'Lat. Ventricle R',
                 'Ant. temp. lobe, Med. L WM',
                 'Ant. temp. lobe, Med. R WM',
                 'Ant. temp. lobe, Lat. L WM',
                 'Ant. temp. lobe, Lat. R WM',
                 'Gyri parahip. et amb. Ant. L WM',
                 'Gyri parahip. et amb. Ant. R WM',
                 'Sup. temp. gyrus, middle L WM',
                 'Sup. temp. gyrus, middle R WM',
                 'Med. & Inf. temp. gyri Ant. L WM',
                 'Med. & Inf. temp. gyri Ant. R WM',
                 'Lat. occipitotemp. & fusi. gyri Ant. L WM',
                 'Lat. occipitotemp. & fusi. gyri Ant. R WM',
                 'Insula R WM',
                 'Insula L WM',
                 'Occip. lobe R WM',
                 'Occip. lobe L WM',
                 'Gyri parahip. et amb. Post. R WM',
                 'Gyri parahip. et amb. Post. L WM',
                 'Lat. occipitotemp. & fusi. gyri Post. R WM',
                 'Lat. occipitotemp. & fusi. gyri Post. L WM',
                 'Med. & Inf. temp. gyri Post. R WM',
                 'Med. & Inf. temp. gyri Post. L WM',
                 'Sup. temp. gyrus, Post. R WM',
                 'Sup. temp. gyrus, Post. L WM',
                 'Cing. gyrus, Ant. R WM',
                 'Cing. gyrus, Ant. L WM',
                 'Cing. gyrus, Post. R WM',
                 'Cing. gyrus, Post. L WM',
                 'Frontal lobe R WM',
                 'Frontal lobe L WM',
                 'Parietal lobe R WM',
                 'Parietal lobe L WM',
                 'CSF',
                 'Extra-cranial background',
                 'Intra-cranial background',
                 'Thalamus R, low intensity in T2',
                 'Thalamus L, low intensity in T2']
    
    patients = []
    vol_cgm_ratio = []
    vol_wm_ratio = []
    vol_th_ratio = []
    percJ_list = []
    dice_list = []
    hd95_list = []
    ncc_list = []
    
    
    for npz_file in os.listdir(args.npz_files):
        if npz_file != "sub-CC00385XX15.npz":
            
            data = np.load(os.path.join(args.npz_files, npz_file))
            
            if data["gw2"] - data["gw1"] > 4 and data["gw2"] - data["gw1"] < 8:
                
                if npz_file[:11] not in patients:
                    patients.append(npz_file[:11])
    
                    moving = data["vol2"]
                    moving_seg = data["lab2"]
                    
                    fixed = data["vol1"]
                    fixed_seg = data["lab1"]
                    fixed_seg4 = data["lab1"]
                    
                    vol_cgm_ratio.append(len(moving_seg[moving_seg==2])/len(fixed_seg[fixed_seg==2]))
                    vol_wm_ratio.append(len(moving_seg[moving_seg==3])/len(fixed_seg[fixed_seg==3]))
                    vol_th_ratio.append(len(moving_seg[moving_seg==6])/len(fixed_seg[fixed_seg==6]))
                    
                    moving, moving_seg = moving[None,None,...], moving_seg[None,None,...]
                    moving, moving_seg = np.ascontiguousarray(moving), np.ascontiguousarray(moving_seg)
                    moving = torch.tensor(moving,requires_grad=True, dtype=torch.float32).cuda()
                    moving_seg = torch.tensor(moving_seg,requires_grad=True, dtype=torch.float32).cuda()
                    
                    
                    fixed = fixed[None,None,...]
                    fixed = np.ascontiguousarray(fixed)
                    fixed = torch.tensor(fixed,requires_grad=True, dtype=torch.float32).cuda()
                    
                    fixed_seg[fixed_seg == 0] = 1
                    fixed_seg[fixed_seg == 4] = 1
                    fixed_seg[fixed_seg!=1] = 1e-5
                
                    fixed_seg = gaussian_filter(fixed_seg,sigma=.5)
                    fixed_seg = (fixed_seg-fixed_seg.min())/(fixed_seg.max()-fixed_seg.min())
                    fixed_seg = torch.tensor(fixed_seg)
                    
        
                    loss_nhe = losses.local_grad_neo_hookean_dHCP(shape=(192,256,256), return_maps=False, window=7, use_entropy=True, weight_entropy = 0, normalize=True)
                    
        
                    # predicting biomechanical parameters
                    
                    with torch.no_grad():
                    
                        x_in = torch.cat((moving,fixed), dim=1)
                        
                        output = model(x_in)                    
                        warped, flow = output[0], output[1]
                        nhe, ratio, F, J = loss_nhe.forward(flow,output[-1], fixed_seg[...,:-1,:-1,:-1])
                        
                        ncc_full = losses.NCC_full(win=5).forward(fixed,warped)
                        
                        # fixed_cpu = cpu(fixed)
                        # moving_cpu = cpu(moving)
                        # warped_cpu = cpu(warped)
                        
                        # plt.figure()
                        # plt.subplot(1,3,1)
                        # plt.imshow(moving_cpu[moving_cpu.shape[0]//2,...],cmap="gray")
                        # plt.subplot(1,3,2)
                        # plt.imshow(fixed_cpu[fixed_cpu.shape[0]//2,...],cmap="gray")
                        # plt.subplot(1,3,3)
                        # plt.imshow(warped_cpu[warped_cpu.shape[0]//2,...],cmap="gray")
                        # plt.show()
                        
                        # Registering everything to common templace
                        
                        x_in2 = torch.cat((fixed,template), dim=1)
                        output2 = model2(x_in2)                    
                        warped2, flow2 = output2[0], output2[1]
                        
                                    
                    def register_data(data,flow):
                        warped_data = reg_model_bilin([data, flow])
                        return warped_data
                    
                    def register_dataN(data,flow):
                        warped_data = reg_model([data, flow])
                        return warped_data
                    
                    def perc(data):
                        data[data<np.percentile(data,0)] = np.percentile(data,0)
                        data[data>np.percentile(data,100)] = np.percentile(data,100)
                        return data
                    
                    
                    warped_seg = ndimage.median_filter(cpu(register_dataN(moving_seg, flow)),size=5)

                    dice_list.append(dice_multiclass(warped_seg, fixed_seg4, len(np.unique(cpu(moving_seg)))))
                    hd95_list.append(hd95(warped_seg.astype(np.int32), fixed_seg4.astype(np.int32)))
                    
                    mu = np.zeros((192,256,256))
                    kappa = np.zeros((192,256,256))
                    
                    mu[:-1,:-1,:-1] = cpu(ratio[1])
                    kappa[:-1,:-1,:-1] = cpu(ratio[0])
                    
                    fixed_seg4[fixed_seg4==1] = 0
                    fixed_seg4[fixed_seg4==4] = 0
                    fixed_seg4[fixed_seg4==5] = 0
                    fixed_seg4[fixed_seg4==7] = 0
                    fixed_seg4[fixed_seg4!=0] = 1
                    # mu[fixed_seg4==0] = 0.1
                    # kappa[fixed_seg4==0] = 0
                    
                    # mu = median_filter(mu,size=3)
                    # kappa = median_filter(kappa,size=3)

                    
                    mu = torch.tensor(mu)[None,None,...]
                    kappa = torch.tensor(kappa)[None,None,...]
                    
                    mu = mu.to(torch.float32).cuda()
                    kappa = kappa.to(torch.float32).cuda()
    
                    warped_list.append(cpu(warped2))
                    Exx_list.append(cpu(register_data(F[0],flow2)))
                    Eyy_list.append(cpu(register_data(F[1],flow2)))
                    Ezz_list.append(cpu(register_data(F[2],flow2)))
                    ADG_list.append(cpu(register_data(F[3],flow2)))
                    Exy_list.append(cpu(register_data(F[4],flow2)))
                    Exz_list.append(cpu(register_data(F[5],flow2)))
                    Eyz_list.append(cpu(register_data(F[6],flow2)))
                    percJ_list.append(F[7])
                    detJ_list.append(cpu(register_data(J,flow2)))
                    mu_list.append(cpu(register_data(mu,flow2)))
                    kappa_list.append(cpu(register_data(kappa,flow2)))
                    nhe1_list.append(cpu(register_data(ratio[3],flow2)))
                    nhe2_list.append(cpu(register_data(ratio[4],flow2)))
                    Tr_list.append(cpu(register_data(ratio[5],flow2)))
                    phi_xy_list.append(perc(cpu(register_data(ratio[6],flow2))))
                    phi_xz_list.append(perc(cpu(register_data(ratio[7],flow2))))
                    phi_yz_list.append(perc(cpu(register_data(ratio[8],flow2))))
                    ncc_list.append(cpu(register_data(ncc_full,flow2)))
                    
                    flow_cp = torch.clone(flow)

                    for i in range(3):
                        flow_cp[:,i,...] -= flow_cp[:,i,...].mean()
                        flow_cp[:,i,...] = register_dataN(flow_cp[:,i:i+1,...],flow2)
                    flows_list.append(cpu(flow_cp))
                
    np.save(os.path.join(args.save_plot_dir, "mu.png"), np.asarray(mu_list))
    np.save(os.path.join(args.save_plot_dir, "kappa.png"), np.asarray(kappa_list))
                
                
    ######################## 3D PLOTS
    
    fixed_seg2 = np.load(args.template)["lab2"]
    # fixed_seg2[fixed_seg2==1] = 0
    # fixed_seg2[fixed_seg2==4] = 0
    # fixed_seg2[fixed_seg2==5] = 0
    # fixed_seg2[fixed_seg2==6] = 0
    # fixed_seg2[fixed_seg2==8] = 0
    # fixed_seg2[fixed_seg2!=0] = 1
    
    fixed_seg2[fixed_seg2==1] = 0
    # fixed_seg2[fixed_seg2==2] = 0
    fixed_seg2[fixed_seg2==4] = 0
    fixed_seg2[fixed_seg2==5] = 0
    fixed_seg2[fixed_seg2==7] = 0
    fixed_seg2[fixed_seg2!=0] = 1
                
    def make_3d_plot(data, seg, savename, crop, color=False, view = "axial", vmin=0, vmax=1, colormap="jet"):
        
        array_data = np.copy(data)
        seg2 = np.copy(seg)
        
        if view == "axial":
            seg2[..., :crop] = 0
        if view == "sagital":
            seg2[:crop, ...] = 0
            seg2 = np.rot90(seg2, k=2, axes=(1,2))
            array_data = np.rot90(array_data, k=2, axes=(1,2))
        if view == "coronal":
            seg2[:, -crop:, :] = 0
            seg2 = np.rot90(seg2, k=2, axes=(1,2))
            array_data = np.rot90(array_data, k=2, axes=(1,2))
            
        # Generate mesh from binary mask    
        vertices, faces, _, _ = measure.marching_cubes(seg2[:-1,:-1,:-1], allow_degenerate=False, method="lorensen")
        vertex_values = np.array([array_data[int(v[0]), int(v[1]), int(v[2])] for v in vertices])
        
        # Smooth mesh
        if color == True:
            mesh = trimesh.Trimesh(vertices,faces)
            mesh = trimesh.smoothing.filter_laplacian(mesh,lamb=0.1,iterations=30)
            vertices = mesh.vertices
            faces = mesh.faces
        
        mlab.figure(size=(1024,1024))
        # Create mesh
        mesh = mlab.triangular_mesh([vert[0] for vert in vertices],
                                  [vert[1] for vert in vertices],
                                  [vert[2] for vert in vertices],
                                  faces,
                                  scalars=vertex_values, colormap=colormap, vmin=vmin, vmax=vmax)
        
        mesh.actor.property.representation = "surface"
        
        if color == True:
            mesh.actor.property.interpolation = "phong"
        else:
            mesh.actor.property.interpolation = "phong"
            
        # mesh.actor.property.metallic = .2
        
        mesh.scene.light_manager.light_mode = "vtk"
     
        # Display mesh with color coding
        f = mlab.gcf()  
        if view == "axial":              
            mlab.view(azimuth=0, elevation=180, distance=500)
        if view == "sagital":              
            mlab.view(azimuth=0, elevation=-90, distance=500)
        if view == "coronal":              
            mlab.view(azimuth=270, elevation=90, distance=500)
        f.scene.background = (1, 1, 1)
        f.scene.foreground = (0, 0, 0)
        if color == True:
            mlab.colorbar(orientation="vertical")

        mlab.savefig(savename)
        mlab.close()

    def make_subplots(data, seg, vmin, vmax, cmap, savedir, savename):
        
        axial = [25, 90, 110, 120, 140, 170]
        sagittal = [0, 40, 60, 70, 85, 98]
        coronal = [10, 35, 75, 90, 110, 160]
    
        # plt.style.use("dark_background")
        plt.figure(figsize=(16,7))
        for i in range(6):
            plt.subplot(3,6,i+1)
            if i == 0:
                make_3d_plot(data,fixed_seg2, os.path.join(savedir,"%i_ax.png" %i), axial[i], color=True, view="axial", vmin=vmin, vmax=vmax, colormap=cmap)
                img = plt.imread(savedir+"%i_ax.png" %i)
            else:
                make_3d_plot(data,fixed_seg2, os.path.join(savedir,"%i.png" %i), axial[i], color=False, view="axial", vmin=vmin, vmax=vmax, colormap=cmap)
                img = plt.imread(savedir+"%i.png" %i)
                img = img/img.max()
                # img = img**1.2
            img = img[:,:-100]
            plt.imshow(img)
            plt.axis("off")
    
            
        for i in range(6):
            plt.subplot(3,6,i+7)
            if i == 0:
                make_3d_plot(data,fixed_seg2, os.path.join(savedir,"%i_sg.png" %i), sagittal[i], color=True, view="sagital", vmin=vmin, vmax=vmax, colormap=cmap)
                img = plt.imread(savedir+"%i_sg.png" %i)
            else:
                make_3d_plot(data,fixed_seg2, os.path.join(savedir,"%i.png" %i), sagittal[i], color=False, view="sagital", vmin=vmin, vmax=vmax, colormap=cmap)
                img = plt.imread(savedir+"%i.png" %i)
                img = img/img.max()
            img = img[:-100,:-50]
            plt.imshow(img)
            plt.axis("off")
            
        for i in range(6):
            plt.subplot(3,6,i+13)
            if i == 0:
                make_3d_plot(data,fixed_seg2, os.path.join(savedir,"%i_cr.png" %i), coronal[i], color=True, view="coronal", vmin=vmin, vmax=vmax, colormap=cmap)
                img = plt.imread(savedir+"%i_cr.png" %i)
            else:
                make_3d_plot(data,fixed_seg2, os.path.join(savedir,"%i.png" %i), coronal[i], color=False, view="coronal", vmin=vmin, vmax=vmax, colormap=cmap)
                img = plt.imread(savedir+"%i.png" %i)
                img = img/img.max()
            
            img = img[:-100,:-100]
            plt.imshow(img)
            plt.axis("off")
        
        plt.tight_layout()
        # plt.subplots_adjust(bottom=-0.01, top=1.01)
        plt.savefig(os.path.join(savedir,savename),dpi=300)
        plt.close()

    def make_subplots_flow(img, flow, mask, vmin, vmax, cmap, savename):
        
        axial = [70, 110, 130, 140, 155, 165]
        sagittal = [45, 55, 63, 66, 75, 98]
        coronal = [40, 70, 90, 110, 140, 150]

        # flow[2,...] = -flow[2,...]
        # flow[1,...] = flow[1,...]-10
        flow = flow.transpose(0,3,1,2)
        s = 4
        plt.style.use("dark_background")
        plt.figure(figsize=(16,7))
        xx, yy, zz = np.meshgrid(np.linspace(0,192,192), np.linspace(0,256,256), np.linspace(0,256,256))
        grid = np.stack((xx,yy,zz), axis=0) 
        
        # img = np.rot90(img, axes=(0,2),k=3)
        # img = np.flip(img,axis=2)
        # img = np.ma.masked_where(mask==0,img)
        img = img.transpose(2,0,1)
        mask = mask.transpose(2,0,1)
        
        def msk(data,mask):
            return np.ma.masked_where(mask==0,data)
        
        width = 3e-3
        pivot = "tail"
        hw = 4
        win=7
        scale = 2e2
        
        jet = plt.cm.jet(np.linspace(0,1,512))
        jet[:1,:] = [0,0,0,1]
        jet = ListedColormap(jet)
        
        for i in range(6):
            plt.subplot(3,6,i+1)
            
            plt.imshow(np.rot90(img[axial[i],...]), cmap="gray", vmin=0.2, vmax=0.65)
            
            Y = grid[2,axial[i],::s,::s]
            Y = Y.max()-Y
            X = grid[0,axial[i],::s,::s]
            V = flow[2,axial[i],::s,::s]
            U = flow[0,axial[i],::s,::s]
            
            for x in range(V.shape[0]):
                for y in range(U.shape[1]):
                    V[x,y] = V[x,y] - V[max(0,x-win):min(V.shape[0],x+win),max(0,y-win):min(V.shape[1],y+win)].mean()
                    U[x,y] = U[x,y] - U[max(0,x-win):min(U.shape[0],x+win),max(0,y-win):min(U.shape[1],y+win)].mean()
            
            V[mask[axial[i],::s,::s]==0] = 1e-5*V[mask[axial[i],::s,::s]==0]
            U[mask[axial[i],::s,::s]==0] = 1e-5*U[mask[axial[i],::s,::s]==0]
            modulus = 1e-5+0.8**2*(U**2+V**2)**0.5
            modulus = (modulus-modulus.min())/(modulus.max()-modulus.min())

            norm = plt.Normalize()

            C = norm(modulus.flatten())
            
            plt.quiver(X,Y,U,V,C, scale=0.7*scale, cmap=jet, width=width, pivot=pivot, headwidth=hw)
            if i==5:
                plt.colorbar(shrink=0.7)
            
            plt.axis("off")
            
            
        for i in range(6):
            plt.subplot(3,6,i+7)
            
            plt.imshow(img[...,-coronal[i]], cmap="gray", vmin=0.2, vmax=0.65)
            
            X = grid[0,::s,::s,-coronal[i]]
            Y = grid[1,::s,::s,-coronal[i]]
            U = flow[0,::s,::s,-coronal[i]]
            V = flow[1,::s,::s,-coronal[i]]
            
            for x in range(V.shape[0]):
                for y in range(U.shape[1]):
                    V[x,y] = V[x,y] - V[max(0,x-win):min(V.shape[0],x+win),max(0,y-win):min(V.shape[1],y+win)].mean()
                    U[x,y] = U[x,y] - U[max(0,x-win):min(U.shape[0],x+win),max(0,y-win):min(U.shape[1],y+win)].mean()
            
            V[mask[::s,::s,-coronal[i]]==0] = 1e-5*V[mask[::s,::s,-coronal[i]]==0]
            U[mask[::s,::s,-coronal[i]]==0] = 1e-5*U[mask[::s,::s,-coronal[i]]==0]
            modulus = 1e-5+0.8**2*(U**2+V**2)**0.5    
            modulus = (modulus-modulus.min())/(modulus.max()-modulus.min())

            C = norm(modulus.flatten())
            
            plt.quiver(X,Y,U,V,C, scale=0.5*scale, cmap=jet, width=width, pivot=pivot, headwidth=hw)
            if i==5:
                plt.colorbar(shrink=0.7)
            # plt.gca().set_aspect("equal")
            plt.axis("off")

            
        for i in range(6):
            plt.subplot(3,6,i+13)
            plt.imshow(img[:,sagittal[i],:], cmap="gray", vmin=0.2, vmax=0.65)
            
            X = grid[2,::s,sagittal[i],::s]
            Y = grid[1,::s,sagittal[i],::s]
            U = flow[2,::s,sagittal[i],::s]
            V = flow[1,::s,sagittal[i],::s]
            
            for x in range(V.shape[0]):
                for y in range(U.shape[1]):
                    V[x,y] = V[x,y] - V[max(0,x-win):min(V.shape[0],x+win),max(0,y-win):min(V.shape[1],y+win)].mean()
                    U[x,y] = U[x,y] - U[max(0,x-win):min(U.shape[0],x+win),max(0,y-win):min(U.shape[1],y+win)].mean()
            
            V[mask[::s,sagittal[i],::s]==0] = 1e-5*V[mask[::s,sagittal[i],::s]==0]
            U[mask[::s,sagittal[i],::s]==0] = 1e-5*U[mask[::s,sagittal[i],::s]==0]
            modulus = 1e-5+0.8**2*(U**2+V**2)**0.5   
            modulus = (modulus-modulus.min())/(modulus.max()-modulus.min())
            

            C = norm(modulus.flatten())
            
            plt.quiver(X,Y,U,V,C, scale=scale, cmap=jet, width=width, pivot=pivot, headwidth=hw)
            if i==5:
                plt.colorbar(shrink=0.7)
            plt.axis("off")

        plt.tight_layout()
        # plt.subplots_adjust(bottom=-0.01, top=1.01)
        plt.savefig(os.path.join(savename),dpi=300)
        plt.close()
        

    def make_subplots_flow2(img, flow, mask, vmin, vmax, cmap, savename):
        
        axial =  [140]
        sagittal = [85]
        coronal = [150]

        # flow[2,...] = -flow[2,...]
        # flow[1,...] = flow[1,...]-10
        flow = flow.transpose(0,3,1,2)
        s = 3
        plt.style.use("dark_background")
        plt.figure(figsize=(16,7))
        xx, yy, zz = np.meshgrid(np.linspace(0,192,192), np.linspace(0,256,256), np.linspace(0,256,256))
        grid = np.stack((xx,yy,zz), axis=0) 
        
        # img = np.rot90(img, axes=(0,2),k=3)
        # img = np.flip(img,axis=2)
        # img = np.ma.masked_where(mask==0,img)
        img = img.transpose(2,0,1)
        mask = mask.transpose(2,0,1)
        
        def msk(data,mask):
            return np.ma.masked_where(mask==0,data)
        
        width = 1.5e-3
        pivot = "tail"
        hw = 4
        hl = 3
        
        jet = plt.cm.jet(np.linspace(0,1,512))
        jet[:1,:] = [0,0,0,1]
        jet = ListedColormap(jet)
        win=12
        
        for i in range(1):
            plt.subplot(1,3,i+1)
            
            plt.imshow(np.rot90(img[axial[i],...]), cmap="gray")
            
            Y = grid[2,axial[i],::s,::s]
            Y = Y.max()-Y
            X = grid[0,axial[i],::s,::s]
            V = flow[2,axial[i],::s,::s]
            U = flow[0,axial[i],::s,::s]
            
            for x in range(V.shape[0]):
                for y in range(U.shape[1]):
                    V[x,y] = V[x,y] - V[max(0,x-win):min(V.shape[0],x+win),max(0,y-win):min(V.shape[1],y+win)].mean()
                    U[x,y] = U[x,y] - U[max(0,x-win):min(U.shape[0],x+win),max(0,y-win):min(U.shape[1],y+win)].mean()
                    
            
            V[mask[axial[i],::s,::s]==0] = 1e-5*V[mask[axial[i],::s,::s]==0]
            U[mask[axial[i],::s,::s]==0] = 1e-5*U[mask[axial[i],::s,::s]==0]
            modulus = 1e-5+(U**2+V**2)**0.5

            norm = plt.Normalize()

            C = norm(modulus.flatten())
            
            plt.quiver(X,Y,U,V,C, scale=scale, cmap=jet, width=width, pivot=pivot, headwidth=hw, headlength=hl)
            
            plt.axis("off")
            
            
        for i in range(1):
            plt.subplot(1,3,i+2)
            
            plt.imshow(img[...,-coronal[i]], cmap="gray")
            
            # print("shape grid :", grid.shape)
            # print("shape img :", img.shape)
            # print("shape flow :", flow.shape)
            
            X = grid[0,::s,::s,-coronal[i]]
            Y = grid[1,::s,::s,-coronal[i]]
            U = flow[0,::s,::s,-coronal[i]]
            V = flow[1,::s,::s,-coronal[i]]
            
            for x in range(V.shape[0]):
                for y in range(U.shape[1]):
                    V[x,y] = V[x,y] - V[max(0,x-win):min(V.shape[0],x+win),max(0,y-win):min(V.shape[1],y+win)].mean()
                    U[x,y] = U[x,y] - U[max(0,x-win):min(U.shape[0],x+win),max(0,y-win):min(U.shape[1],y+win)].mean()
            
            V[mask[::s,::s,-coronal[i]]==0] = 1e-5*V[mask[::s,::s,-coronal[i]]==0]
            U[mask[::s,::s,-coronal[i]]==0] = 1e-5*U[mask[::s,::s,-coronal[i]]==0]
            modulus = 1e-5+(U**2+V**2)**0.5      

            C = norm(modulus.flatten())
            
            plt.quiver(X,Y,U,V,C, scale=2e2, cmap=jet, width=width, pivot=pivot, headwidth=hw, headlength=hl)
            # plt.gca().set_aspect("equal")
            plt.axis("off")

            
        for i in range(1):
            plt.subplot(1,3,i+3)
            plt.imshow(img[:,sagittal[i],:], cmap="gray")
            
            X = grid[2,::s,sagittal[i],::s]
            Y = grid[1,::s,sagittal[i],::s]
            U = flow[2,::s,sagittal[i],::s]
            V = flow[1,::s,sagittal[i],::s]
            
            for x in range(V.shape[0]):
                for y in range(U.shape[1]):
                    V[x,y] = V[x,y] - V[max(0,x-win):min(V.shape[0],x+win),max(0,y-win):min(V.shape[1],y+win)].mean()
                    U[x,y] = U[x,y] - U[max(0,x-win):min(U.shape[0],x+win),max(0,y-win):min(U.shape[1],y+win)].mean()
            
            V[mask[::s,sagittal[i],::s]==0] = 1e-5*V[mask[::s,sagittal[i],::s]==0]
            U[mask[::s,sagittal[i],::s]==0] = 1e-5*U[mask[::s,sagittal[i],::s]==0]
            modulus = 1e-5+(U**2+V**2)**0.5       
            

            C = norm(modulus.flatten())
            
            plt.quiver(X,Y,U,V,C, scale=2e2, cmap=jet, width=width, pivot=pivot, headwidth=hw, headlength=hl)
            plt.axis("off")

        plt.tight_layout()
        # plt.subplots_adjust(bottom=-0.01, top=1.01)
        plt.savefig(os.path.join(savename),dpi=500)
        plt.close()

        
    def norm_0_1(array):
        return (array-array.min())/(array.max()-array.min())
        
    def makeplots_anim(data, seg, savename, crop, color=False, view = "axial", vmin=0, vmax=1, colormap="jet"):
        
        array_data = np.copy(data)
        seg2 = np.copy(seg)
        
        if view == "axial":
            seg2[..., :crop] = 0
        if view == "sagital":
            seg2[:crop, ...] = 0
            seg2 = np.rot90(seg2, k=2, axes=(1,2))
            array_data = np.rot90(array_data, k=2, axes=(1,2))
        if view == "coronal":
            seg2[:, -crop:, :] = 0
            seg2 = np.rot90(seg2, k=2, axes=(1,2))
            array_data = np.rot90(array_data, k=2, axes=(1,2))
            
        # Generate mesh from binary mask    
        vertices, faces, _, _ = measure.marching_cubes(seg2[:-1,:-1,:-1], allow_degenerate=False, method="lorensen")
        vertex_values = np.array([array_data[int(v[0]), int(v[1]), int(v[2])] for v in vertices])
        
        # Smooth mesh
        if color == True:
            mesh = trimesh.Trimesh(vertices,faces)
            mesh = trimesh.smoothing.filter_laplacian(mesh,lamb=0.1,iterations=30)
            vertices = mesh.vertices
            faces = mesh.faces
        
        mlab.figure(size=(1024,1024))
        # Create mesh
        mesh = mlab.triangular_mesh([vert[0] for vert in vertices],
                                  [vert[1] for vert in vertices],
                                  [vert[2] for vert in vertices],
                                  faces,
                                  scalars=vertex_values, colormap=colormap, vmin=vmin, vmax=vmax)
        
        mesh.actor.property.representation = "surface"
        
        if color == True:
            mesh.actor.property.interpolation = "phong"
        else:
            mesh.actor.property.interpolation = "phong"
            
        # mesh.actor.property.metallic = .2
        
        mesh.scene.light_manager.light_mode = "vtk"
     
        # Display mesh with color coding
        f = mlab.gcf()  
        if view == "axial":              
            mlab.view(azimuth=0, elevation=180, distance=500)
        if view == "sagital":              
            mlab.view(azimuth=0, elevation=-90, distance=500)
        if view == "coronal":              
            mlab.view(azimuth=270, elevation=90, distance=500)
        f.scene.background = (1, 1, 1)
        f.scene.foreground = (0, 0, 0)
        if color == True:
            mlab.colorbar(orientation="vertical")

        mlab.savefig(savename)
        mlab.close()        

    warped = np.mean(np.asarray(warped_list),axis=0)
    flow = np.mean(np.asarray(flows_list),axis=0)
    Exx = np.mean(np.asarray(Exx_list),axis=0)
    Eyy = np.mean(np.asarray(Eyy_list),axis=0)
    Ezz = np.mean(np.asarray(Ezz_list),axis=0)
    Exy = np.mean(np.asarray(Exy_list),axis=0)
    Exz = np.mean(np.asarray(Exz_list),axis=0)
    Eyz = np.mean(np.asarray(Eyz_list),axis=0)
    ADG = np.mean(np.asarray(ADG_list),axis=0)
    
    detJ = np.mean(np.asarray(detJ_list),axis=0)
    mu = norm_0_1(np.mean(np.asarray(mu_list),axis=0))
    kappa = norm_0_1(np.mean(np.asarray(kappa_list),axis=0))
    ncc_tot = np.mean(np.asarray(ncc_list),axis=0)
    
    ratio = kappa/(mu+1e-3)
    
    nhe1 = np.mean(np.asarray(nhe1_list),axis=0)
    nhe2 = np.mean(np.asarray(nhe2_list),axis=0)
    Tr = np.mean(np.asarray(Tr_list),axis=0)
    phi_xy = np.mean(np.asarray(phi_xy_list),axis=0)
    phi_xz = np.mean(np.asarray(phi_xz_list),axis=0)
    phi_yz = np.mean(np.asarray(phi_yz_list),axis=0)
    
    np.save("flow.npz", flow)
    
    # ratio = kappa/mu
    
    # fs = np.load(args.template)["lab2"]
    
    # make_subplots_flow(warped, flow, fs, vmin=-5, vmax=5, cmap="jet", savename="flow_baby.png")
    # make_subplots_flow2(warped, flow, fixed_seg2, vmin=-5, vmax=5, cmap="jet", savename="flow_babyg.png")
    def make(data, seg, n, label, vmin,vmax):
        counts, bins = np.histogram(data[seg==n].flatten(), bins=1000, density=True, range=(vmin,vmax))
        bin_centers = (bins[:-1]+bins[1:])/2
        bin_width = bins[1]-bins[0]
        sigma = 0.1*np.std(data[seg==n].flatten()) / bin_width
        smoothed = gaussian_filter1d(counts, sigma=sigma)
        plt.plot(bin_centers, smoothed, label=label, linewidth=2)
    
    def hist_tissues(data,seg,namedata,vmin,vmax):
        
        make(data, seg, 2, "CGM", vmin,vmax)
        make(data, seg, 3, "WM", vmin,vmax)
        make(data, seg, 6, "DGM", vmin,vmax)
        
        plt.xlim(vmin,vmax)
        plt.xlabel(namedata, fontsize=15)
        plt.ylabel("Density", fontsize=15)
        plt.tick_params(labelsize=10)
        plt.grid()
        plt.legend()
        
    def hist_structures(data,structure,structure_list, namedata,savename, cmap):
        rows=[]
        for i, structure_name in enumerate(structure_list, start=1):
            if structure_name not in ['CSF','Extra-cranial background','Intra-cranial background', 'Lat. Ventricle R', 'Lat. Ventricle L']:
                st_data = data[structure==i]
                # pc05 = np.percentile(st_data,5)
                # pc95 = np.percentile(st_data,95)
                # st_data = st_data[(st_data>=pc05) & (st_data<=pc95)]
                
                rows.append({"Structure" : structure_name, "Values" : st_data})
        df = pd.DataFrame(rows)
        df = df.explode("Values")
        df["Values"] = df["Values"].astype(float)
        
        median_values = df.groupby("Structure")["Values"].median().sort_values(ascending=False)
        df = df.set_index("Structure").loc[median_values.index].reset_index()
        
        palette = sb.color_palette(cmap, len(median_values))
        
        plt.figure(figsize=(30,10))
        sb.boxplot(x="Structure", y="Values", data=df, palette=palette, showfliers = False)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(fontsize=20)
        plt.ylabel(namedata, fontsize=20)
        plt.xlabel("Structure", fontsize=20)
        plt.grid()
        plt.subplots_adjust(bottom=0.25)
        plt.tight_layout()
        plt.savefig(savename, dpi=300)
        plt.show()
        plt.close()
        

    fixed_seg3 = np.load(args.template)["lab2"]
    
    #
    
    print("MU CGM : %.3f +- %.3f" %(mu[fixed_seg3 == 2].mean(),mu[fixed_seg3 == 2].std()))
    print("MU WM : %.3f +- %.3f" %(mu[fixed_seg3 == 3].mean(),mu[fixed_seg3 == 3].std()))
    print("MU THL : %.3f +- %.3f" %(mu[fixed_seg3 == 6].mean(),mu[fixed_seg3 == 6].std()))
    print("KAPPA CGM : %.3f +- %.3f" %(kappa[fixed_seg3 == 2].mean(),kappa[fixed_seg3 == 2].std()))
    print("KAPPA WM : %.3f +- %.3f" %(kappa[fixed_seg3 == 3].mean(),kappa[fixed_seg3 == 3].std()))
    print("KAPPA THL : %.3f +- %.3f" %(kappa[fixed_seg3 == 6].mean(),kappa[fixed_seg3 == 6].std()))
    
    print("DET CGM : %.3f +- %.3f" %(detJ[fixed_seg3 == 2].mean(),detJ[fixed_seg3 == 2].std()))
    print("DET WM : %.3f +- %.3f" %(detJ[fixed_seg3 == 3].mean(),detJ[fixed_seg3 == 3].std()))
    print("DET THL : %.3f +- %.3f" %(detJ[fixed_seg3 == 6].mean(),detJ[fixed_seg3 == 6].std()))
    
    print("DET2 CGM : %.3f +- %.3f" %(np.asarray(vol_cgm_ratio).mean(),np.asarray(vol_cgm_ratio).std()))
    print("DET2 WM : %.3f +- %.3f" %(np.asarray(vol_wm_ratio).mean(),np.asarray(vol_wm_ratio).std()))
    print("DET2 THL : %.3f +- %.3f" %(np.asarray(vol_th_ratio).mean(),np.asarray(vol_th_ratio).std()))
    
    print("percentage neg Jac : %.5f +- %.5f" %(np.asarray(percJ_list).mean(),np.asarray(percJ_list).std()))
    print("DICE: %.3f +- %.3f" %(np.asarray(dice_list).mean(),np.asarray(dice_list).std()))
    print("HD95: %.3f +- %.3f" %(np.asarray(hd95_list).mean(),np.asarray(hd95_list).std()))

    
    # hist_structures(data=detJ, structure=template_st,structure_list=structures_list, namedata=r"$det(F)$",savename="hist__structures_detF.png", cmap = "rainbow_r")
    # hist_structures(data=Exx, structure=template_st,structure_list=structures_list, namedata=r"$E_{yy}$",savename="hist__structures_Eyy.png", cmap = "jet_r")
    # hist_structures(data=Eyy, structure=template_st,structure_list=structures_list, namedata=r"$E_{xx}$",savename="hist__structures_Exx.png", cmap = "jet_r")
    # hist_structures(data=Ezz, structure=template_st,structure_list=structures_list, namedata=r"$E_{zz}$",savename="hist__structures_Ezz.png", cmap = "jet_r")
    # hist_structures(data=phi_xy, structure=template_st,structure_list=structures_list, namedata=r"$\phi_{xy}$",savename="hist__structures_phi_xy.png", cmap = "bwr_r")
    # hist_structures(data=phi_xz, structure=template_st,structure_list=structures_list, namedata=r"$\phi_{yz}$",savename="hist__structures_phi_yz.png", cmap = "bwr_r")
    # hist_structures(data=phi_yz, structure=template_st,structure_list=structures_list, namedata=r"$\phi_{xz}$",savename="hist__structures_phi_xz.png", cmap = "bwr_r")
    # hist_structures(data=ADG, structure=template_st,structure_list=structures_list, namedata=r"$ADG$",savename="hist__structures_ADG.png", cmap = "jet_r")
    # hist_structures(data=Tr, structure=template_st,structure_list=structures_list, namedata=r"$Tr(F^T F)-3$",savename="hist__structures_Tr.png", cmap = "rainbow_r")
    # hist_structures(data=ADG, structure=template_st,structure_list=structures_list, namedata=r"$Tr(F^T F) \cdot J^{-2/3}$",savename="hist__structures_nhe1.png", cmap = "rainbow_r")
    # hist_structures(data=nhe1, structure=template_st,structure_list=structures_list, namedata=r"$(J-1)^2$",savename="hist__structures_nhe2.png", cmap = "rainbow_r")
    
    plt.rcParams['font.family'] = 'serif'
    
    plt.figure(figsize=(15,15))
    plt.subplot(3,3,1)
    hist_tissues(data=detJ, seg=fixed_seg3, namedata="J", vmin=0.5, vmax=2.5)
    plt.subplot(3,3,2)
    hist_tissues(data=ncc_tot, seg=fixed_seg3, namedata="NCC", vmin=0.5, vmax=1)
    plt.subplot(3,3,3)
    hist_tissues(data=kappa, seg=fixed_seg3, namedata=r"$\kappa$", vmin=0, vmax=1)
    plt.subplot(3,3,4)
    hist_tissues(data=Eyy, seg=fixed_seg3, namedata=r"$E_{xx}$", vmin=-0.2, vmax=0.8)
    plt.subplot(3,3,5)
    hist_tissues(data=Exx, seg=fixed_seg3, namedata=r"$E_{yy}$", vmin=-0.2, vmax=0.8)
    plt.subplot(3,3,6)
    hist_tissues(data=Ezz, seg=fixed_seg3, namedata=r"$E_{zz}$", vmin=-0.2, vmax=0.8)
    plt.subplot(3,3,7)
    hist_tissues(data=phi_xy, seg=fixed_seg3, namedata=r"$\phi_{xy}(rad)$", vmin=-0.4, vmax=0.4)
    plt.subplot(3,3,8)
    hist_tissues(data=phi_xz, seg=fixed_seg3, namedata=r"$\phi_{yz}(rad)$", vmin=-0.4, vmax=0.4)
    plt.subplot(3,3,9)
    hist_tissues(data=phi_yz, seg=fixed_seg3, namedata=r"$\phi_{xz}(rad)$", vmin=-0.4, vmax=0.4)

    
    # plt.tight_layout()
    plt.savefig(os.path.join(args.save_plot_dir, "hist.png"),dpi=300)
    plt.show()
    
    
    # for i in range(40,160):
    #     makeplots_anim(detJ, fixed_seg2, "anim/detJ/%03d.png" %i, i, color=True, view = "axial", vmin=0.5, vmax=2.75, colormap="rainbow")
    
    # for i in range(40,160):
    #     makeplots_anim(kappa, fixed_seg2, "anim/kappa/%03d.png" %i, "anim/%03d.png" %(int(500-i)), i, color=True, view = "axial", vmin=0, vmax=1, colormap="rainbow")
    
    # for i in range(40,160):
    #     makeplots_anim(Exx, fixed_seg2, "anim/Exx/%03d.png" %i, i, color=True, view = "axial", vmin=-0.1, vmax=1.2, colormap="jet")
        
    # for i in range(40,160):
    #     makeplots_anim(Eyy, fixed_seg2, "anim/Eyy/%03d.png" %i, "anim/%03d.png" %(int(500-i)), i, color=True, view = "axial", vmin=-0.2, vmax=1, colormap="jet")
        
    # for i in range(40,160):
    #     makeplots_anim(Ezz, fixed_seg2, "anim/Ezz/%03d.png" %i, "anim/%03d.png" %(int(500-i)), i, color=True, view = "axial", vmin=-0.2, vmax=1.2, colormap="jet")
        
    # for i in range(40,160):
    #     makeplots_anim(ADG, fixed_seg2, "anim/ADG/%03d.png" %i, "anim/%03d.png" %(int(500-i)), i, color=True, view = "axial", vmin=0.05, vmax=0.35, colormap="jet")
        
    # for i in range(40,160):
    #     makeplots_anim(Tr, fixed_seg2, "anim/Tr/%03d.png" %i, "anim/%03d.png" %(int(500-i)), i, color=True, view = "axial", vmin=0, vmax=4, colormap="rainbow")
        
    # for i in range(40,160):
    #     makeplots_anim(phi_xy, fixed_seg2, "anim/phi_xy/%03d.png" %i, "anim/%03d.png" %(int(500-i)), i, color=True, view = "axial", vmin=-0.4, vmax=0.4, colormap="bwr")
        
    # for i in range(40,160):
    #     makeplots_anim(phi_xy, fixed_seg2, "anim/phi_xz/%03d.png" %i, i, color=True, view = "axial", vmin=-0.4, vmax=0.4, colormap="bwr")
        
    # for i in range(40,160):
    #     makeplots_anim(phi_yz, fixed_seg2, "anim/phi_yz/%03d.png" %i, i, color=True, view = "axial", vmin=-0.4, vmax=0.5, colormap="bwr")
    
    
    # make_subplots(data=ratio, seg=fixed_seg2, vmin=0, vmax=2, cmap="bwr", savedir=args.save_plot_dir, savename="ratio.png")
    make_subplots(data=mu, seg=fixed_seg2, vmin=0, vmax=1, cmap="rainbow", savedir=args.save_plot_dir, savename="mu.png")
    make_subplots(data=kappa, seg=fixed_seg2, vmin=0, vmax=1, cmap="rainbow", savedir=args.save_plot_dir, savename="kappa.png")
    

    
    make_subplots(data=Exx, seg=fixed_seg2, vmin=-0.2, vmax=0.8, cmap="jet", savedir=args.save_plot_dir, savename="Eyy.png")
    make_subplots(data=Eyy, seg=fixed_seg2, vmin=-0.1, vmax=0.6, cmap="jet", savedir=args.save_plot_dir, savename="Exx.png")
    make_subplots(data=Ezz, seg=fixed_seg2, vmin=-0.2, vmax=0.8, cmap="jet", savedir=args.save_plot_dir, savename="Ezz.png")
    # make_subplots(data=Exy, seg=fixed_seg2, vmin=-0.4, vmax=0.4, cmap="jet", savedir=args.save_plot_dir, savename="Exy.png")
    # make_subplots(data=Exz, seg=fixed_seg2, vmin=-0.4, vmax=0.5, cmap="jet", savedir=args.save_plot_dir, savename="Eyz.png")
    # make_subplots(data=Eyz, seg=fixed_seg2, vmin=-0.4, vmax=0.4, cmap="jet", savedir=args.save_plot_dir, savename="Exz.png")
    make_subplots(data=ADG, seg=fixed_seg2, vmin=0., vmax=0.3, cmap="jet", savedir=args.save_plot_dir, savename="ADG.png")
    
    make_subplots(data=detJ, seg=fixed_seg2, vmin=0.8, vmax=2.2, cmap="rainbow", savedir=args.save_plot_dir, savename="detJ.png")
    # make_subplots(data=nhe1, seg=fixed_seg2, vmin=0., vmax=1.5, cmap="rainbow", savedir=args.save_plot_dir, savename="nhe1.png")
    # make_subplots(data=nhe2, seg=fixed_seg2, vmin=0., vmax=3.5, cmap="rainbow", savedir=args.save_plot_dir, savename="nhe2.png")
    # make_subplots(data=Tr, seg=fixed_seg2, vmin=0, vmax=4, cmap="rainbow", savedir=args.save_plot_dir, savename="Tr.png")
    make_subplots(data=phi_xy, seg=fixed_seg2, vmin=-.3, vmax=.3, cmap="bwr", savedir=args.save_plot_dir, savename="phi_xy.png")
    make_subplots(data=phi_xz, seg=fixed_seg2, vmin=-.3, vmax=.3, cmap="bwr", savedir=args.save_plot_dir, savename="phi_yz.png")
    make_subplots(data=phi_yz, seg=fixed_seg2, vmin=-.3, vmax=.3, cmap="bwr", savedir=args.save_plot_dir, savename="phi_xz.png")
    make_subplots(data=warped, seg=fixed_seg2, vmin=0.1, vmax=0.5, cmap="gray", savedir=args.save_plot_dir, savename="template.png")
    # make_subplots(data=ratio, seg=fixed_seg2, vmin=0.5, vmax=1.3, cmap="coolwarm", savedir=args.save_plot_dir, savename="ratio.png")
    
    
    
    # mlab.show()

    # mlab.savefig("3d_plots/detJ/det_J.png")
    # mlab.close()
    
    
    """
    warped_seg = reg_model([moving_seg, flow])
    
    warped_seg_cpu = cpu(warped_seg)
    fixed_seg_cpu = fixed_seg
    moving_seg_cpu = ndimage.median_filter(cpu(moving_seg), size=5)
    
    
    warped_cpu = cpu(warped)
    fixed_cpu = cpu(fixed)
    moving_cpu = cpu(moving)
    
    # x = np.arange(100)
    # plt.figure()
    # for i in range(N):
    #     plt.plot(x, ncc_list[i,:], label="ratio = %i" %i)
    #     plt.legend()
    # plt.show()
    
    
    ### Plots
    
    plt.subplot(2,3,1)
    plt.imshow(moving_cpu[...,128], cmap="gray")
    plt.subplot(2,3,2)
    plt.imshow(fixed_cpu[...,128], cmap="gray")
    plt.subplot(2,3,3)
    plt.imshow(warped_cpu[...,128], cmap="gray")
    plt.subplot(2,3,4)
    plt.imshow(moving_seg_cpu[...,128])
    plt.subplot(2,3,5)
    plt.imshow(fixed_seg_cpu[...,128])
    plt.subplot(2,3,6)
    plt.imshow(warped_seg_cpu[...,128])
    plt.show()
    
    
  
    ### grid
    plt.style.use("dark_background")
    s=1
    
    grid_img = utils.mk_grid_img(grid_step=1, grid_sz=moving.shape[-3:])
    warped_grid = cpu(utils.mk_grid(moving.shape[-3:], -flow)[0])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    
    ax.plot_wireframe(warped_grid[0,::s,::s,128], warped_grid[1,::s,::s,128],warped_grid[2,::s,::s,128], linewidth=.3, rstride=2, cstride=2)
    ax.set_axis_off()
    plt.show()
    
    ### deformation field
    
    grid_img = utils.mk_grid_img(grid_step=8, grid_sz=moving.shape[-3:])
    warped_grid = cpu(utils.mk_grid(moving.shape[-3:], flow)[0])
    
    norm_flow = utils.norm_flow(cpu(flow)).transpose(1,2,3,0)
    norm_flow = norm_flow**2
    
    s=1
    
    colors = np.stack([norm_flow[::s,::s,128,0].flatten(),
                       norm_flow[::s,::s,128,1].flatten(),
                       norm_flow[::s,::s,128,2].flatten()], axis=-1)
    
    print(colors)
    print(colors.shape)
    print(warped_grid[0,::s,::s,128].shape)
    
    
    
    plt.quiver(warped_grid[0,::s,::s,128], warped_grid[1,::s,::s,128],
               color=colors, pivot="mid", scale=1000., width=0.01)
    plt.show()
    
    """
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
    