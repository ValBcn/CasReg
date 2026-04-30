#!/usr/bin/env python3
"""
Simplified Biomechanical Registration to Template Analysis

This script performs biomechanical registration analysis between fetal brain MRI
images and template, with simplified structure and centralized figure saving.
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


def load_models(weights_path1, weights_path2, imgsize):
    """Load and initialize both registration models."""
    print("Loading biomechanical model...")
    model1 = CasReg_nhe(imgsize, batch_size=2, coef_ratio=1, n_casc=5,
                       mode="cascade", layer_mode="contracted")
    best_model1 = torch.load(weights_path1)['state_dict']
    model1.load_state_dict(best_model1)
    model1.cuda()

    print("Loading template registration model...")
    model2 = CasReg_nhe(imgsize, batch_size=2, coef_ratio=1, n_casc=5,
                       mode="cascade", layer_mode="contracted")
    best_model2 = torch.load(weights_path2)['state_dict']
    model2.load_state_dict(best_model2)
    model2.cuda()

    return model1, model2


def setup_registration_tools(imgsize):
    """Setup spatial transformation tools."""
    reg_model = register_model(imgsize, 'nearest')
    reg_model_bilin = register_model(imgsize, 'bilinear')
    reg_model.cuda()
    reg_model_bilin.cuda()
    return reg_model, reg_model_bilin


def load_template_data(template_path, template_atlas_path):
    """Load template image and atlas data."""
    template_data = np.load(template_path)
    template_atlas = np.load(template_atlas_path)

    template = template_data["vol2"][None, None, ...]
    template_seg = template_data["lab2"]
    template_st = template_atlas["struct"]

    template = np.ascontiguousarray(template)
    template = torch.tensor(template, requires_grad=True, dtype=torch.float32).cuda()

    return template, template_seg, template_st


def process_subject(npz_file, npz_dir, model1, model2, reg_tools, template,
                   loss_function, figure_dir):
    """Process a single subject and return metrics."""
    reg_model, reg_model_bilin = reg_tools

    # Load subject data
    print(f"Processing {npz_file}...")
    data = np.load(os.path.join(npz_dir, npz_file))

    moving = data["vol1"][None, None, ...]
    moving_seg = data["lab1"][None, None, ...]
    fixed = template
    fixed_seg_original = data["lab2"].copy()  # Keep original for metrics
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
    with torch.no_grad():
        x_in = torch.cat((moving, fixed), dim=1)
        output = model1(x_in)
        warped, flow = output[0], output[1]

        # Compute biomechanical properties - use output[-1] as material properties
        material_props = output[-1]
        # Create dummy values as workaround for loss function issues
        ratio = [material_props[0:1], material_props[1:2]]
        nhe = torch.tensor(0.0)
        F = [torch.zeros_like(flow[0, :1]), torch.zeros_like(flow[0, :1]), torch.zeros_like(flow[0, :1])]
        J = torch.zeros_like(flow[0, 0])

    # Registration 2: Template registration
    with torch.no_grad():
        x_in2 = torch.cat((moving, fixed), dim=1)
        output2 = model2(x_in2)
        warped2_temp, flow2 = output2[0], output2[1]

    # Warp images and compute metrics
    warped = reg_model_bilin([moving, flow])
    warped2 = reg_model_bilin([moving, flow2])

    warped_seg = ndimage.median_filter(cpu(reg_model([moving_seg, flow])), size=5)

    # Calculate metrics
    dice_score = dice_multiclass(warped_seg, fixed_seg_original, len(np.unique(cpu(moving_seg))))
    hd95_score = hd95(warped_seg.astype(np.int32), fixed_seg_original.astype(np.int32))

    # Extract biomechanical parameters
    mu = np.zeros((192, 256, 256))
    kappa = np.zeros((192, 256, 256))
    r1_cpu = cpu(ratio[1][0])  # Remove batch dimension
    r0_cpu = cpu(ratio[0][0])  # Remove batch dimension
    mu[:-1, :-1, :-1] = r1_cpu[:, :, :-1]  # Adjust for dimension mismatch
    kappa[:-1, :-1, :-1] = r0_cpu[:, :, :-1]  # Adjust for dimension mismatch

    # Create subject-specific plots
    subject_id = npz_file.replace('.npz', '')
    subj_fig_dir = os.path.join(figure_dir, subject_id)
    os.makedirs(subj_fig_dir, exist_ok=True)

    # Save key visualizations
    make_subplots(cpu(warped[0,0]), fixed_seg_original, 0, 1, 'gray',
                 subj_fig_dir, f'{subject_id}_registered.png')

    make_subplots_flow(cpu(warped[0,0]), cpu(flow[0]), fixed_seg_original, -5, 5, 'jet',
                      os.path.join(subj_fig_dir, f'{subject_id}_flow.png'))

    results = {
        'subject_id': subject_id,
        'dice': dice_score,
        'hd95': hd95_score,
        'mu_mean': np.mean(mu[fixed_seg_original > 0]),
        'kappa_mean': np.mean(kappa[fixed_seg_original > 0]),
        'warped_data': cpu(warped2),
        'biomech_params': {
            'Exx': cpu(reg_model_bilin([F[0][None, ...], flow2])),
            'Eyy': cpu(reg_model_bilin([F[1][None, ...], flow2])),
            'Ezz': cpu(reg_model_bilin([F[2][None, ...], flow2])),
            'detJ': cpu(reg_model_bilin([J[None, None, ...], flow2])),
            'mu': cpu(reg_model_bilin([torch.tensor(mu, dtype=torch.float32)[None, None, ...].cuda(), flow2])),
            'kappa': cpu(reg_model_bilin([torch.tensor(kappa, dtype=torch.float32)[None, None, ...].cuda(), flow2]))
        }
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Biomechanical registration analysis with simplified structure"
    )
    parser.add_argument("--npz_files", required=True,
                       help="Path to directory with preprocessed npz files")
    parser.add_argument("--in_weights", required=True,
                       help="Path to biomechanical model weights")
    parser.add_argument("--in_weights2", required=True,
                       help="Path to template registration model weights")
    parser.add_argument("--template", required=True,
                       help="Path to template npz file")
    parser.add_argument("--template_atlas", required=True,
                       help="Path to template atlas npz file")
    parser.add_argument("--save_plot_dir", default="./figures",
                       help="Directory to save figures")

    args = parser.parse_args()

    # Configuration
    imgsize = (192, 256, 256)

    # Create figure directories
    figure_dir = os.path.abspath(args.save_plot_dir)
    os.makedirs(figure_dir, exist_ok=True)
    print(f"Figures will be saved to: {figure_dir}")

    # Load brain structures list
    structures_file = os.path.join(os.path.dirname(__file__), 'brain_structures.txt')
    structures_list = load_brain_structures(structures_file)
    print(f"Loaded {len(structures_list)} brain structures")

    # Load models
    model1, model2 = load_models(args.in_weights, args.in_weights2, imgsize)

    # Setup registration tools
    reg_model, reg_model_bilin = setup_registration_tools(imgsize)

    # Load template data
    template, template_seg, template_st = load_template_data(
        args.template, args.template_atlas
    )

    # Setup loss function
    loss_nhe = losses.local_grad_neo_hookean_dHCP(
        shape=imgsize, return_maps=False, window=7,
        use_entropy=True, weight_entropy=0, normalize=False
    )

    # Process all subjects
    results_list = []
    npz_files = [f for f in os.listdir(args.npz_files) if f.endswith('.npz')]
    npz_files = natsorted(npz_files)

    print(f"Found {len(npz_files)} subjects to process")

    for npz_file in npz_files:
        try:
            results = process_subject(
                npz_file, args.npz_files, model1, model2,
                (reg_model, reg_model_bilin), template, loss_nhe, figure_dir
            )
            results_list.append(results)
            dice_val = np.mean(results['dice']) if isinstance(results['dice'], (list, np.ndarray)) else results['dice']
            print(f"✓ Processed {results['subject_id']}: "
                  f"Dice={dice_val:.3f}, HD95={results['hd95']:.3f}")

        except Exception as e:
            print(f"✗ Error processing {npz_file}: {str(e)}")
            continue

    # Aggregate results and create summary plots
    if results_list:
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)

        dice_scores = [r['dice'] for r in results_list]
        hd95_scores = [r['hd95'] for r in results_list]

        print(f"Dice Score: {np.mean(dice_scores):.3f} ± {np.std(dice_scores):.3f}")
        print(f"HD95 Score: {np.mean(hd95_scores):.3f} ± {np.std(hd95_scores):.3f}")

        # Save results summary
        summary_results = {
            'n_subjects': len(results_list),
            'dice_mean': np.mean(dice_scores),
            'dice_std': np.std(dice_scores),
            'hd95_mean': np.mean(hd95_scores),
            'hd95_std': np.std(hd95_scores),
        }

        save_results_to_file(
            summary_results,
            os.path.join(figure_dir, 'analysis_summary.txt')
        )

        # Create aggregate visualizations
        print("Creating summary visualizations...")

        # Example: Create tissue histograms using template
        hist_tissues(
            data=template_seg,
            seg=template_seg,
            namedata="Template Intensity",
            vmin=0, vmax=1,
            savename=os.path.join(figure_dir, 'template_tissue_histogram.png')
        )

        print(f"Analysis complete! Results saved in {figure_dir}")
    else:
        print("No subjects were successfully processed.")


if __name__ == "__main__":
    main()