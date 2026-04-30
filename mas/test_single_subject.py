#!/usr/bin/env python3
"""
Test single subject to identify matplotlib and other errors
"""

import argparse
import os
import sys
import glob
from pathlib import Path

# Add parent directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# Core imports
import numpy as np
import torch
from natsort import natsorted
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Project imports
from models import losses
from models.models import CasReg_nhe
from utils.spatial import register_model
from utils.biomech_utils import (
    load_brain_structures, make_3d_plot, make_subplots, make_subplots_flow,
    hist_tissues, hist_structures, ensure_figure_dir, save_results_to_file
)
from utils.metrics import dice_multiclass
from utils.io_utils import hd95
from data.data_utils import cpu

def test_single_subject():
    """Test processing of a single subject to identify all errors"""

    # Configuration
    imgsize = (192, 256, 256)

    # Create figure directories
    figure_dir = os.path.abspath("./test_figures")
    os.makedirs(figure_dir, exist_ok=True)
    print(f"Test figures will be saved to: {figure_dir}")

    # Load brain structures list
    structures_file = os.path.join(os.path.dirname(__file__), 'brain_structures.txt')
    structures_list = load_brain_structures(structures_file)
    print(f"Loaded {len(structures_list)} brain structures")

    # Load models
    print("Loading biomechanical model...")
    model1 = CasReg_nhe(imgsize, batch_size=2, coef_ratio=1, n_casc=5,
                       mode="cascade", layer_mode="contracted")
    best_model1 = torch.load("../../registration_experiments/dHCP/forward/hybrid/CasReg2/results/weights/ncc0.947.pth.tar")['state_dict']
    model1.load_state_dict(best_model1)
    model1.cuda()

    print("Loading template registration model...")
    model2 = CasReg_nhe(imgsize, batch_size=2, coef_ratio=1, n_casc=5,
                       mode="cascade", layer_mode="contracted")
    best_model2 = torch.load("/eos/jeodpp/data/projects/DIGLIFE/comteva/registration_experiments/dHCP/forward/nhe/CasReg2/results/weights/ncc0.924.pth.tar")['state_dict']
    model2.load_state_dict(best_model2)
    model2.cuda()

    # Setup registration tools
    reg_model = register_model(imgsize, 'nearest')
    reg_model_bilin = register_model(imgsize, 'bilinear')
    reg_model.cuda()
    reg_model_bilin.cuda()

    # Load template data
    template_data = np.load("../../data/dHCP/npz/sub-CC00385XX15.npz")
    template_atlas = np.load("../../data/dhcp-volumetric-atlas-groupwise/npz/dhcp_42.npz")

    template = template_data["vol2"][None, None, ...]
    template_seg = template_data["lab2"]
    template_st = template_atlas["struct"]

    template = np.ascontiguousarray(template)
    template = torch.tensor(template, requires_grad=True, dtype=torch.float32).cuda()

    # Setup loss function
    loss_nhe = losses.local_grad_neo_hookean_dHCP(
        shape=imgsize, return_maps=False, window=7,
        use_entropy=True, weight_entropy=0, normalize=True
    )

    # Process a single test subject
    npz_file = "sub-CC00135AN12.npz"
    npz_dir = "../../data/dHCP/npz/"

    print(f"Processing test subject: {npz_file}")
    try:
        # Load subject data
        data = np.load(os.path.join(npz_dir, npz_file))

        moving = data["vol1"][None, None, ...]
        moving_seg = data["lab1"][None, None, ...]
        fixed_seg_original = data["lab2"].copy()
        fixed_seg = data["lab2"].copy()

        # Process fixed_seg for loss computation
        fixed_seg[fixed_seg == 4] = 1
        fixed_seg[fixed_seg != 1] = 1e-5

        from scipy.ndimage import gaussian_filter
        fixed_seg = gaussian_filter(fixed_seg, sigma=0.5)
        fixed_seg = (fixed_seg - fixed_seg.min()) / (fixed_seg.max() - fixed_seg.min())
        fixed_seg = torch.tensor(fixed_seg)

        moving = torch.tensor(moving, requires_grad=True, dtype=torch.float32).cuda()
        moving_seg = torch.tensor(moving_seg, requires_grad=True, dtype=torch.float32).cuda()

        # Registration 1: Biomechanical
        print("  Running biomechanical registration...")
        with torch.no_grad():
            x_in = torch.cat((moving, template), dim=1)
            output = model1(x_in)
            warped, flow = output[0], output[1]

            # Compute biomechanical properties
            nhe, ratio, F, J = loss_nhe.forward(flow, output[-1], fixed_seg[...,:-1,:-1,:-1])

        # Registration 2: Template registration
        print("  Running template registration...")
        with torch.no_grad():
            x_in2 = torch.cat((moving, template), dim=1)
            output2 = model2(x_in2)
            warped2_temp, flow2 = output2[0], output2[1]

        # Warp images and compute metrics
        print("  Computing metrics...")
        warped = reg_model_bilin([moving, flow])
        warped2 = reg_model_bilin([moving, flow2])

        warped_seg = ndimage.median_filter(cpu(reg_model([moving_seg, flow])), size=5)

        # Calculate metrics
        dice_score = dice_multiclass(warped_seg, fixed_seg_original, len(np.unique(cpu(moving_seg))))
        hd95_score = hd95(warped_seg.astype(np.int32), fixed_seg_original.astype(np.int32))

        print(f"  Dice: {np.mean(dice_score) if isinstance(dice_score, (list, np.ndarray)) else dice_score:.3f}, HD95: {hd95_score:.3f}")

        # Extract biomechanical parameters
        print("  Extracting biomechanical parameters...")
        mu = np.zeros((192, 256, 256))
        kappa = np.zeros((192, 256, 256))
        mu[:-1, :-1, :-1] = cpu(ratio[1])
        kappa[:-1, :-1, :-1] = cpu(ratio[0])

        # Create visualizations - this is where matplotlib errors likely occur
        print("  Creating visualizations...")
        subject_id = npz_file.replace('.npz', '')
        subj_fig_dir = os.path.join(figure_dir, subject_id)
        os.makedirs(subj_fig_dir, exist_ok=True)

        # Test make_subplots - this likely has the matplotlib error
        print("    Testing make_subplots...")
        try:
            make_subplots(cpu(warped[0,0]), fixed_seg_original, 0, 1, 'gray',
                         subj_fig_dir, f'{subject_id}_registered.png')
            print("    ✓ make_subplots successful")
        except Exception as e:
            print(f"    ✗ make_subplots failed: {e}")
            import traceback
            traceback.print_exc()

        # Test make_subplots_flow
        print("    Testing make_subplots_flow...")
        try:
            make_subplots_flow(cpu(warped[0,0]), cpu(flow[0]), fixed_seg_original, -5, 5, 'jet',
                              os.path.join(subj_fig_dir, f'{subject_id}_flow.png'))
            print("    ✓ make_subplots_flow successful")
        except Exception as e:
            print(f"    ✗ make_subplots_flow failed: {e}")
            import traceback
            traceback.print_exc()

        print(f"✓ Subject {subject_id} processed successfully!")

    except Exception as e:
        print(f"✗ Error processing {npz_file}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_subject()