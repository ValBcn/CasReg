import glob
import argparse
import os, losses, utils
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from natsort import natsorted
from models import VxmDense_2_cascade
from scipy import ndimage
import SimpleITK as sitk
from collections import Counter
import time
import losses
from io_utils import hd95
from torch import optim
import torch.nn as nn
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').cuda()
from io_utils import resize_data, multi_channel_labels, nii_to_np, resize_and_uncrop, hist_match, dice


def Most_Common(lst):
    data = Counter(lst)
    # print(data.most_common(1))
    return data.most_common(1)[0][0]

def weighted_fusion(sim_metric, labels, nb_labels):
    weights = sim_metric
    weighted_labels = np.zeros(nb_labels)
    labels = labels.astype(int)
    for i in range(len(weights)):
        weighted_labels[labels[i]] += weights[i]**1.
    return weighted_labels.argmax()

def mk_grid_img(line_thickness=1, grid_sz=128):
    grid_img = np.zeros((grid_sz,grid_sz,grid_sz))
    grid_step = int(grid_sz/(8*args.grid_factor))
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="For training ViT-VNet for registration of 3D fetal brain MRI")
    
    parser.add_argument("--npz_files_val", type = str, required = True,
                        help = "Path to the base directory where the preprocessed/reshaped npz files for validation are")
    parser.add_argument("--fixed", type = str, required = True )
    parser.add_argument("--nb_labels", type = int, required = False, default = 8,
                        help = "Nb of labels")
    parser.add_argument("--output_dir", type = str, required = False,
                        help = "Path to the directory where the results are saved")
    parser.add_argument("--lr", type = float, required = False, default = 1e-3,
                        help = "The learning rate")
    parser.add_argument("--in_weights", type = str, required = False,
                        help = " path to trained weights")
    parser.add_argument("--supervised", type=int, required = False, default = 1,
                        help="Supervised training or not (supervised = 1, unsupervised=0).")
    parser.add_argument("--grid_factor", type=int, required = False, default = 4,
                        help="Scaling factor for the grid representation")
    parser.add_argument("--grid", type=int, required = False, default = 0,
                        help="whether or not to produce a deformation grid (1/0)")
    parser.add_argument("--dice_file", type=str, required=True,
                        help="path + name of the file where the dice results are saved")
#    parser.add_argument("--mode", type=str, required = False, default = "parallel",
#                        help="Mode of the cascade training, parallel or cascade")
    parser.add_argument("--cascades", type=int, required = False, default = 0,
                        help="Cascading networks or not (1/0)")
    parser.add_argument("--nb_channel", type=int, required = False, default = 1,
                        help="number of channel(s)")
    parser.add_argument("--smoothing", type=int, required = False, default = 0,
                        help="smooting the segmentation with a median filter")
    parser.add_argument("--rep", type=int, required = False, default = 1,
                        help="number of registration per image")
    parser.add_argument("--flip", type=int, required = False, default = 0,
                        help="number of registration per image")
    parser.add_argument("--flip_axis", type=int, required = False, default = 0,
                        help="number of registration per image")
    



    args = parser.parse_args()

    #test_dir = 'D:/DATA/JHUBrain/Test/'
    #model_idx = -1
    #weights = [1, 0.02]
    #model_folder = 'TransMorph_mse_{}_diffusion_{}/'.format(weights[0], weights[1])
    #model_dir = 'experiments/' + model_folder
    #dict = utils.process_label()
    #if os.path.exists('experiments/'+model_folder[:-1]+'.csv'):
        #os.remove('experiments/'+model_folder[:-1]+'.csv')
    #csv_writter(model_folder[:-1], 'experiments/' + model_folder[:-1])
    #line = ''
    #for i in range(46):
    #    line = line + ',' + dict[i]
    #csv_writter(line, 'experiments/' + model_folder[:-1])

    imgsize = (128,128,128)
    
    if args.supervised == 1:
        mode = torch.load(args.in_weights)["mode"]
        layer_sizes = torch.load(args.in_weights)["layer sizes"]
        nb_cascades = len(layer_sizes)
        model = VxmDense_2_cascade_sup(imgsize, n_casc = nb_cascades, mode = mode, layer_sizes = layer_sizes)
    else:
        if args.cascades == 0:
            model = VxmDense_2(imgsize)
        if args.cascades == 1:
            mode = torch.load(args.in_weights)["mode"]
            layer_sizes = torch.load(args.in_weights)["layer sizes"]
            nb_cascades = len(layer_sizes)
            model = VxmDense_2_cascade(imgsize, batch_size=1, n_casc = nb_cascades, mode = mode, layer_sizes = layer_sizes)

    best_model = torch.load(args.in_weights)['state_dict'] #model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
        
    #print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()
    reg_model = utils.register_model((128, 128, 128), 'nearest')
    reg_model_bilin = utils.register_model((128, 128, 128), 'bilinear')
    reg_model.cuda()
    reg_model_bilin.cuda()
    test_composed = transforms.Compose([trans.Identity(),trans.NumpyType((np.float32, np.float32))])
    #test_set = datasets.JHUBrainInferDataset(glob.glob(test_dir + '*.pkl'), transforms=test_composed)
    #test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    eval_dsc_def = utils.AverageMeter()
    eval_dsc_raw = utils.AverageMeter()
    eval_det = utils.AverageMeter()

    max_test = 20
    nb_test = args.nb_channel
    Dice_before_list = []
    Dice_after_list = []
    warped_list = []
    ncc_list = []
    sim_list = []
    warped_list_smooth = []
    
    with torch.no_grad():
        stdy_idx = 0
    
    t1 = time.time()
    
    best_model = torch.load(args.in_weights)['state_dict']
    model.load_state_dict(best_model)
   
    for files in sorted(os.listdir(args.npz_files_val)):
        
        print("nb_test", nb_test)
        model.eval()
        
        vol1, seg1 = np.load(os.path.join(args.npz_files_val, files))["vol"], np.load(os.path.join(args.npz_files_val, files))["seg"]
        vol2, seg2 = np.load(args.fixed)["vol"], np.load(args.fixed)["seg"]

        
        coords = np.load(args.fixed)["coords"]
        old_shape = np.load(args.fixed)["old_shape"]
        spacing = tuple(np.load(args.fixed)["spacing"])
        origin = tuple(np.load(args.fixed)["origin"])
        direction = tuple(np.load(args.fixed)["direction"])
        
        vol1, seg1, vol2, seg2 = vol1[None,None, ...], seg1[None,None, ...], vol2[None,None, ...], seg2[None,None, ...]
    
        vol1,seg1 = test_composed([vol1, seg1])
        vol2, seg2 = test_composed([vol2, seg2])
    
        vol1 = np.ascontiguousarray(vol1)
        seg1 = np.ascontiguousarray(seg1)
        vol2 = np.ascontiguousarray(vol2)
        seg2 = np.ascontiguousarray(seg2)
    
        moving, moving_seg = torch.tensor(vol1,requires_grad=True).cuda(), torch.tensor(seg1,requires_grad=True).cuda()
        fixed, fixed_seg = torch.tensor(vol2,requires_grad=True).cuda(), torch.tensor(seg2,requires_grad=True).cuda()
        
        model.train()
        x_in = torch.cat((moving,fixed), dim=1)
        x_seg_in = torch.cat((moving_seg,fixed_seg), dim = 1)
        if args.supervised == 0:
            output = model(x_in)
        if args.supervised == 1:
            output = model([x_in,x_seg_in])
        loss_function = losses.NCC().forward        
        ncc = loss_function(output[0], fixed).item()
        
        print("NCC = ", ncc)
        
        ncc_list.append(ncc)
        
    del model
    
    if args.supervised == 1:
        model = VxmDense_2_sup(imgsize)
    else:
        if args.cascades == 0:
            model = VxmDense_2(imgsize)
        if args.cascades == 1:
            mode = torch.load(args.in_weights)["mode"]
            layer_sizes = torch.load(args.in_weights)["layer sizes"]
            nb_cascades = len(layer_sizes)
            model = VxmDense_2_cascade(imgsize, batch_size=1, n_casc = nb_cascades, mode = mode, layer_sizes = layer_sizes)

    best_model = torch.load(args.in_weights)['state_dict'] 
    model.load_state_dict(best_model)
    model.cuda()
    reg_model = utils.register_model((128, 128, 128), 'nearest')
    reg_model_bilin = utils.register_model((128, 128, 128), 'bilinear')
    reg_model.cuda()
    reg_model_bilin.cuda()
    test_composed = transforms.Compose([trans.Identity(),trans.NumpyType((np.float32, np.float32))])
    eval_dsc_def = utils.AverageMeter()
    eval_dsc_raw = utils.AverageMeter()
    eval_det = utils.AverageMeter()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0, amsgrad=True)
    
    ncc_list = np.asarray(ncc_list)

    print(ncc_list)
    
    selected = ncc_list.argsort()[1:nb_test+1]
    
    print(selected)
    
    for i in selected:
        files = sorted(os.listdir(args.npz_files_val))[i]
        
        model.eval()
        
        vol1, seg1 = np.load(os.path.join(args.npz_files_val, files))["vol"], np.load(os.path.join(args.npz_files_val, files))["seg"]
        vol2, seg2 = np.load(args.fixed)["vol"], np.load(args.fixed)["seg"]
        
        seg1[seg1>7] = 0
        seg2[seg2>7] = 0
        
        for i in range(0):
            model.train()
            x_in = torch.cat((moving,fixed), dim=1)
            x_seg_in = torch.cat((moving_seg,fixed_seg), dim = 1)
            if args.supervised == 0:
                output = model(x_in)
            if args.supervised == 1:
                output = model([x_in,x_seg_in])
            loss_function = losses.NCC().forward(output[0], fixed)
            loss_function.backward()
            optimizer.step()
            optimizer.zero_grad()
            print("NCC = ", losses.NCC().forward(output[0], fixed).item())
        
        
        # vol2 = vol1.std() * vol2 / vol2.std()
        
        print("std fixed :", vol2.std())
        print("std moving :", vol1.std())
    
        vol1, seg1, vol2, seg2 = vol1[None,None, ...], seg1[None,None, ...], vol2[None,None, ...], seg2[None,None, ...]
    
        vol1,seg1 = test_composed([vol1, seg1])
        vol2, seg2 = test_composed([vol2, seg2])
    
        vol1 = np.ascontiguousarray(vol1)
        seg1 = np.ascontiguousarray(seg1)
        vol2 = np.ascontiguousarray(vol2)
        seg2 = np.ascontiguousarray(seg2)
    
        moving, moving_seg = torch.tensor(vol1,requires_grad=True).cuda(), torch.tensor(seg1,requires_grad=True).cuda()
        fixed, fixed_seg = torch.tensor(vol2,requires_grad=True).cuda(), torch.tensor(seg2,requires_grad=True).cuda()
        
        for i in range(0):
            model.train()
            x_in = torch.cat((moving,fixed), dim=1)
            x_seg_in = torch.cat((moving_seg,fixed_seg), dim = 1)
            if args.supervised == 0:
                output = model(x_in)
            if args.supervised == 1:
                output = model([x_in,x_seg_in])
            loss_function = losses.NCC().forward(output[0], fixed)
            loss_function.backward()
            optimizer.step()
            optimizer.zero_grad()
            print("NCC = ", losses.NCC().forward(output[0], fixed).item())
        
        ncc_moving = losses.NCC().forward(fixed, moving) #losses.NCC().forward(fixed, moving)
                    
        moving2 = torch.clone(moving)
        fixed2 = torch.clone(fixed)
        
        #data_generator = datasets.volgen(args.npz_files_val, transforms = test_composed, get_file_name=True)
        #moving, fixed, moving_seg, fixed_seg, moving_file, fixed_file = next(data_generator)
        
        # moving[moving_seg == 0] = 0
        # fixed[fixed_seg == 0] = 0

        print("Registering %s and %s" %(files ,args.fixed))


        x_in = torch.cat((moving,fixed),dim=1)
        warped, flow, flow_list, _ = model(x_in)


        flow_cpu = flow.detach().cpu().numpy()
        
        
        ncc_warped = losses.NCC().forward(fixed, warped)
        
        warped_seg = reg_model([moving_seg.cuda().float(), flow.cuda()])
        warped = reg_model_bilin([moving2.cuda().float(), flow.cuda()])
           
        fixed_seg_cpu = fixed_seg.detach().cpu().numpy()
        
        print("uniques fixed ,", np.unique(fixed_seg_cpu))
        
        moving_seg_cpu = moving_seg.detach().cpu().numpy()
        
        if args.smoothing == 1:
            warped_seg_cpu = ndimage.median_filter(warped_seg.detach().cpu().numpy(),size=3)
        if args.smoothing == 0:
            warped_seg_cpu = warped_seg.detach().cpu() 
        print("uniques moving ,", np.unique(moving_seg_cpu))
        print("uniques warped ,", np.unique(warped_seg_cpu))

        sim_list.append(np.squeeze(losses.NCC_full().forward(fixed, warped).detach().cpu().numpy()))
        # sim_list.append(np.squeeze(1-nn.MSELoss(reduction="none").forward(fixed, warped).detach().cpu().numpy()))
        warped_list.append(warped_seg_cpu)
        
        fixed_seg_multi = utils.multi_channel_labels(fixed_seg_cpu,8)
        moving_seg_multi = utils.multi_channel_labels(moving_seg_cpu,8)
        warped_seg_multi = utils.multi_channel_labels(warped_seg_cpu,8)

        dice_after_list = []
        dice_before_list = []
        for i in range(args.nb_labels):
            dice_before = utils.dice(moving_seg_multi[...,i], fixed_seg_multi[...,i])
            dice_after = utils.dice(fixed_seg_multi[...,i], warped_seg_multi[...,i])
            dice_after_list.append(dice_after)
            dice_before_list.append(dice_before)
            print("\n dice before : %.4f" %dice_before)
            print("\n dice after : %.4f" %dice_after)

        Dice_after_list.append(np.asarray(dice_after_list))
        Dice_before_list.append(np.asarray(dice_before_list))
        dsc_trans = np.mean(np.asarray(dice_after_list))

        print("Dice list shape :", np.asarray(Dice_after_list).shape)
        print("Dice list shape :", np.asarray(Dice_before_list).shape)
        tar = fixed.detach().cpu().numpy()[0, 0, :, :, :]
        jac_det = utils.jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, :, :, :, :])
        eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), moving.size(0))
        print("det =", 100*len(np.unique(jac_det[jac_det<=0]))/128**3)
        np.save("jac_det.npy", jac_det)
        np.save("flow.npy", flow_cpu)
        np.save("warped.npy", np.squeeze(warped.detach().cpu().numpy()))
        np.save("moving.npy", np.squeeze(moving.detach().cpu().numpy()))
        stdy_idx += 1
        
        file_cc_dice = open("cc_dice.txt","a")
        file_cc_dice.write("%f %f %f\n" %(ncc_moving.item(),ncc_warped.item(), np.mean(np.asarray(dice_after_list)[1:])))
        file_cc_dice.close()

                
    all_warped = np.zeros((128,128,128,len(warped_list)))
    # warped_final = np.zeros((128,128,128))
    warped_final2 = np.zeros((128,128,128))
    all_ncc = np.zeros((128,128,128,len(warped_list)))
    sim_array = np.asarray(sim_list)
    for i in range(len(warped_list)):
        all_warped[...,i] = warped_list[i]
               
    for i in range(128):
        for j in range(128):
            for k in range(128):
                # print(all_warped[i,j,k,:])
                warped_final2[i,j,k] = Most_Common(all_warped[i,j,k,:])
                # print(all_ncc[i,j,k,:])
                # warped_final2[i,j,k] = weighted_fusion(sim_array[:,i,j,k], all_warped[i,j,k,:], 8)
                # print(warped_final[i,j,k])
                
        
    
    print("uniques")
    print(np.unique(warped_final2))
    
    # reshaping
    
    for i in range(len(flow_list)):
        flow_list[i] = flow_list[i].detach().cpu().numpy()
    np.save("flow_list.npy", flow_list)
    
    warped_final2 = resize_and_uncrop(np.squeeze(warped_final2), coords , old_shape)
    warped = resize_and_uncrop(np.squeeze(warped.detach().cpu()), coords, old_shape)
    fixed_seg_cpu = resize_and_uncrop(np.squeeze(fixed_seg_cpu), coords , old_shape)
        
    fixed_seg_multi = utils.multi_channel_labels(fixed_seg_cpu,8)
    # warped_final_multi = utils.multi_channel_labels(warped_final,8)
    warped_final_multi2 = utils.multi_channel_labels(warped_final2,8)
    
    t2 = time.time()
    
    for i in range(args.nb_labels):
        # dice_after2 = utils.dice(fixed_seg_multi[...,i], warped_final_multi[...,i])
        dice_after3 = utils.dice(fixed_seg_multi[...,i], warped_final_multi2[...,i])
        # hd95_after = hd95(np.uint32(np.squeeze(fixed_seg_multi[...,i])), np.uint32(np.squeeze(warped_final_multi[...,i])))
        # print("dice: %.4f" %dice_after2)
        print("dice: %.4f" %dice_after3)
        # print("hd95 : %.4f" %hd95_after)
    
    # print("mv")
    # for i in range(args.nb_labels):
    #     dice_after2 = utils.dice(fixed_seg_multi[...,i], warped_final_multi[...,i])
    #     print(dice_after2)
        
    print("ncc")
    for i in range(args.nb_labels):
        dice_after3 = utils.dice(fixed_seg_multi[...,i], warped_final_multi2[...,i])
        print(dice_after3)
        # print("hd95 : %.4f" %hd95_after)
        
    print(t2-t1)
    print("(time)")
    warped = sitk.GetImageFromArray(warped)
    warped_seg_smoothed2 = sitk.GetImageFromArray(warped_final2)
    
    warped.SetSpacing(np.array(spacing, dtype="float").tolist())
    warped.SetOrigin(origin)
    warped.SetDirection(direction)
    
    warped_seg_smoothed2.SetSpacing(np.array(spacing, dtype="float").tolist())
    warped_seg_smoothed2.SetOrigin(origin)
    warped_seg_smoothed2.SetDirection(direction)
    
    fixed = sitk.GetImageFromArray(np.squeeze(fixed.detach().cpu()))
    fixed_seg = sitk.GetImageFromArray(np.squeeze(fixed_seg.detach().cpu()))
    
    ID = (args.fixed).split("/")[-1].split(".")[0]

    sitk.WriteImage(warped, os.path.join(args.output_dir, str(ID) + "-warped" + ".nii"))
    sitk.WriteImage(warped_seg_smoothed2, os.path.join(args.output_dir, str(ID) + "-labels" + ".nii"))
    # sitk.WriteImage(fixed, os.path.join(args.output_dir, "fixed_" + str(nb_test) + "_dice_" + str(dsc_trans) + ".nii"))
    # sitk.WriteImage(fixed_seg, os.path.join(args.output_dir, "fixed_seg_" + str(nb_test) + "_dice_" + str(dsc_trans) + ".nii"))

