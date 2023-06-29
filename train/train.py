import glob
import sys
import numpy as np
import torch
import nibabel as nib
from torchvision import transforms
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os, glob
import argparse


torch.cuda.empty_cache()
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

sys.path.append("..")

from models import losses
from utils.utils import *
import utils.utils as utils
from data import datasets, trans
from models.models import CasReg
from data import datasets, trans


        
if __name__ == "__main__":

    """                                                                                                                                                                                                           
    GPU configuration                                                                                                                                                                                             
                                                                                                                                                                                                                  
    GPU_iden = 0                                                                                                                                                                                                  
    GPU_num = torch.cuda.device_count()                                                                                                                                                                           
    print('Number of GPU: ' + str(GPU_num))                                                                                                                                                                       
    for GPU_idx in range(GPU_num):                                                                                                                                                                                
        GPU_name = torch.cuda.get_device_name(GPU_idx)                                                                                                                                                            
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)                                                                                                                                                      
    torch.cuda.set_device(GPU_iden)                                                                                                                                                                               
    GPU_avai = torch.cuda.is_available()                                                                                                                                                                          
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))                                                                                                                                             
    print('If the GPU is available? ' + str(GPU_avai))
    GPU_avai = torch.cuda.is_available()                                                                                                                                                                          
    print('If the GPU is available? ' + str(GPU_avai))                                                                                                                                                            
    Cuda_version = torch.version.cuda                                                                                                                                                                             
    print('version ? ' + str(Cuda_version))                                                                                                                                                                       
    """

    parser = argparse.ArgumentParser(description="For training ViT-VNet for registration of 3D fetal brain MRI")
    
    parser.add_argument("--in_weights", type = str, required = False,
                        help = " path to pretrained weights")
    parser.add_argument("--save_dir", type = str, required = False,
                        help = "Path to the directory where the weights are saved")
    
    parser.add_argument("--npz_train", type = str, required = True,
                        help = "Path to the base directory where the preprocessed/reshaped npz files are")
    parser.add_argument("--npz_val", type = str, required = True,
                        help = "Path to the base directory where the preprocessed/reshaped npz files for validation are")
    
    parser.add_argument("--nb_labels", type = int, required = False, default = 8,
                        help = "Nb of labels")
    parser.add_argument('--img_size', nargs="+", type=int, required = False, default = (128,128,128),
                        help="Sizes of the layers in case of parallel or cascading training")
    
    parser.add_argument("--epochs", type = int, required = False, default = 500,
                        help = "Number of epochs")
    parser.add_argument("--lr", type = float, required = False, default = 1e-3,
                        help = "The learning rate")
    parser.add_argument("--lambda_", type = float, required = False, default = 0.01,
                        help = " weight of the gradient normalization of the registration field")
    parser.add_argument("--dice_weight", type = float, required = False, default = 0.01,
                        help = " weight of the dice loss (if supervised)")
    
    parser.add_argument("--supervised", type=int, required = False, default = 1,
                        help="Supervised training or not (supervised = 1, unsupervised=0).")
    parser.add_argument("--nb_cascades", type=int, required = False, default = 5,
                        help="number of cascaded network")
    parser.add_argument("--mode", type=str, required = False, default = "parallel",
                        help="Mode of the cascade training, parallel or cascade")
    parser.add_argument('--contracted', type=int, required = False, default = 1,
                        help="Contracted architecture or not (1/0)")
    parser.add_argument('--regularization', type=str, required = True, default = "grad",
                        help="Which regularization of the deformation, can be grad or NHE (neo-hookean energy strain")
    
    args = parser.parse_args()

    
    lr = args.lr
    epoch_start = 0
    max_epoch = args.epochs
    img_size = args.img_size
    
    file_suffix = make_suffix(args.supervised, args.contracted, args.nb_cascades, args.mode, args.lambda_)
    
    if not os.path.exists(args.save_dir):
        if not os.path.exists((str(args.save_dir)+ file_suffix)):
            os.makedirs(str(args.save_dir)+ file_suffix)
        save_dir = os.path.join("weights/"+ file_suffix + "/")
        
    if not os.path.exists('logs/' + args.save_dir):
        os.makedirs('logs/' + args.save_dir)
    sys.stdout = Logger('logs/' + args.save_dir)
    cont_training = False

    '''
    Initialize model
    '''    
    model = CasReg(img_size, n_casc=args.nb_cascades, mode = args.mode, layer_mode = args.mode)

    '''
    Initialize spatial transformation function
    '''
    
    reg_model = register_model(img_size, 'nearest')
    reg_model.cuda()
    reg_model_bilin = register_model(img_size, 'bilinear')
    reg_model_bilin.cuda()

    
    '''
    Initialize training
    '''
    
    train_composed = transforms.Compose([trans.Identity(),trans.NumpyType((np.float32, np.float32))])
    val_composed = transforms.Compose([trans.Seg_norm(), trans.NumpyType((np.float32, np.int16))])

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)
    best_dsc = 0

    epoch=0
    lr_steps=10
    
    while epoch < max_epoch:
        epoch += 1
        
        lr = lr_decay(lr,epoch,max_epoch)
        
        # Loss                                                                                                                                                                                                    
        if args.supervised == 0:
            if args.regularization == "NHE":
                if args.with_maps == False:
                    loss_list = [losses.NCC().forward, losses.local_grad_neo_hookean(mu=0.11,lambda_=1.1).forward]
                else:
                    loss_list = [losses.NCC().forward, losses.local_grad_neo_hookean_with_maps().forward]
            elif args.regularization == "grad":
                loss_list = [losses.NCC().forward, losses.Grad3d(penalty='l2').forward]
            weights = [1.,args.lambda_]
        if args.supervised == 1:
            loss_list = [losses.NCC().forward, losses.Grad3d(penalty='l2').forward, losses.Dice_Weighted(np.ones(len(weights_dice))).loss]
            weights = [1.,0.1,1.]
        
        print('Training Starts')
        '''                                                                                                                                                                                                       
        Training                                                                                                                                                                                                  
        '''
        
        loss_all = AverageMeter()
        steps = 0
        steps2 = 0
        max_steps_train = 100
        max_steps_val = 20

        while steps <max_steps_train:
            steps += 1
            model.train()
            
            data_generator = datasets.volgen(args.npz_train, transforms = train_composed)
            fixed, moving, fixed_seg, moving_seg = next(data_generator)
            
            x_in = torch.cat((moving,fixed), dim=1)
            x_seg_in = torch.cat((moving_seg,fixed_seg), dim=1)

            if args.supervised == 0:
                output = model(x_in)
            if args.supervised == 1:
                output = model([x_in,x_seg_in])
            
            flow = output[1]
            loss = 0
            loss_vals = []
            
            for n, loss_function in enumerate(loss_list):
                if n == 0:
                    curr_loss = loss_function(output[n], fixed) * weights[n]
                if n == 1:
                    curr_loss = loss_function(output[n], moving) * weights[n]
                if n == 2:
                    curr_loss = loss_function(output[n], fixed_seg) * weights[n]
                loss_vals.append(curr_loss)

                loss += curr_loss

            loss_np = loss.cpu().detach().numpy()

            loss_all.update(loss.item(), fixed.numel())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if args.supervised == 1:
                print('\n \nIter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}, Dice: {:.6f}, Learning Rate: {:.6f}'.format(steps, max_steps_train, loss.item(), loss_vals[0].item()/weights[0], loss_vals[1].item(), loss_vals[2].item(),  lr))
            if args.supervised == 0:
                print('\n \nIter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}, Learning Rate: {:.6f}'.format(steps, max_steps_train, loss.item(), loss_vals[0].item()/weights[0], loss_vals[1].item(),  lr))
                
        writer.add_scalar("loss/train", loss, epoch)
       
        '''
        Validation
        '''

        ncc = utils.AverageMeter()
        Dice_before_list = []
        Dice_after_list = []
        
        with torch.no_grad():
            while steps2 < max_steps_val:
                steps2 += 1
                
                data_generator_val = datasets.volgen(args.npz_val, transforms = train_composed)
                model.eval()
                
                fixed, moving, fixed_seg, moving_seg = next(data_generator_val)
                
                x_in = torch.cat((moving, fixed), dim=1)
                x_seg_in = torch.cat((moving_seg, fixed_seg), dim=1)
                
                if args.supervised == 0:
                    output = model(x_in)
                if args.supervised == 1:
                    output = model([x_in,x_seg_in])
                    
                ncc = -loss_function[0](output[0],fixed)

                warped_seg = reg_model([moving_seg.cuda().float(), output[1].cuda()])
                np.savez("segs_vxm.npz", fixed_seg=fixed_seg.cpu().numpy(), moving_seg=moving_seg.cpu().numpy(), warped_seg=warped_seg.cpu().numpy())
                dsc = utils.dice_val(warped_seg.long(), fixed_seg.long(), 46)
                
                ncc.update(dsc.item(), moving.size(0))
                
                # write a function for the dice score
                warped_seg_multi = utils.multi_channel_labels(np.squeeze(warped_seg.detach().cpu().numpy()),args.nb_labels)
                moving_seg_multi = utils.multi_channel_labels(np.squeeze(moving_seg.detach().cpu().numpy()),args.nb_labels)
                fixed_seg_multi = utils.multi_channel_labels(np.squeeze(fixed_seg.detach().cpu().numpy()),args.nb_labels)

                dice_before_list = []
                dice_after_list = []
                for i in range(args.nb_labels):
                    dice_before = utils.dice2(fixed_seg[...,i],moving_seg[...,i], args.nb_labels)
                    dice_after = utils.dice2(warped_seg[...,i],fixed_seg[...,i], args.nb_labels)
                    dice_after_list.append(dice_after)
                    dice_before_list.append(dice_before)
                print("Dice before : ", dice_before_list)
                print("Dice after : ", dice_after_list)
                Dice_after_list.append(dice_after_list)
                Dice_before_list.append(dice_before_list)
        
        for i in range(args.nb_labels):
            dice_before = np.mean(np.asarray(Dice_before_list)[:,i])
            std_before = np.std(np.asarray(Dice_before_list)[:,i])/np.sqrt(max_steps_val)
            dice_after = np.mean(np.asarray(Dice_after_list)[:,i])
            std_after = np.std(np.asarray(Dice_after_list)[:,i])/np.sqrt(max_steps_val)
            print('\nDICE before %i = %f +- %f' %(i+1, dice_before, std_before))
            print('DICE after %i = %f +- %f' %(i+1, dice_after, std_after))

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_dsc': best_dsc,
            'optimizer': optimizer.state_dict(),
            'number cascades': args.nb_cascades,
            'supornot': args.supervised,
            'cascade': args.cascades,
            'mode': args.mode
        }, save_dir=save_dir, filename='dsc{:.3f}.pth.tar'.format(ncc.avg))
        writer.add_scalar('DSC/validate', ncc.avg, epoch)




