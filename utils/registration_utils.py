"""
Unified Registration Utilities
Created by refactoring common functions from mas/ directory
Contains all shared functionality for registration workflows
"""

import os
import glob
import time
import numpy as np
import torch
from collections import Counter
from scipy import ndimage
from scipy.ndimage import gaussian_filter, median_filter
from skimage import measure
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors
from torch import optim
import torch.nn as nn

try:
    from mayavi import mlab
    from mayavi.modules.surface import Surface
    MAYAVI_AVAILABLE = True
except ImportError:
    MAYAVI_AVAILABLE = False

import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import utils.utils as utils
from models import losses
from models.models import CasReg, CasReg_nhe


# ==========================================
# DATA LOADING AND PREPROCESSING
# ==========================================

def load_npz_data(npz_path, data_keys=None, tensor_convert=True, gpu=True):
    """
    Load NPZ file data with proper tensor conversion and GPU transfer

    Args:
        npz_path: Path to NPZ file
        data_keys: List of keys to load (default: ['vol', 'seg', 'vol1', 'vol2', 'lab1', 'lab2'])
        tensor_convert: Convert to torch tensors
        gpu: Move to GPU if tensor_convert=True

    Returns:
        Dictionary of loaded data
    """
    if data_keys is None:
        data_keys = ['vol', 'seg', 'vol1', 'vol2', 'lab1', 'lab2']

    data = {}
    npz_data = np.load(npz_path)

    for key in data_keys:
        if key in npz_data:
            data[key] = npz_data[key]

            if tensor_convert:
                # Add batch and channel dimensions if needed
                if len(data[key].shape) == 3:
                    data[key] = data[key][None, None, ...]
                elif len(data[key].shape) == 4:
                    data[key] = data[key][None, ...]

                # Ensure memory contiguity
                data[key] = np.ascontiguousarray(data[key])

                # Convert to tensor
                data[key] = torch.tensor(data[key], requires_grad=True, dtype=torch.float32)

                if gpu and torch.cuda.is_available():
                    data[key] = data[key].cuda()

    # Also load metadata if available
    for meta_key in ['coords', 'spacing', 'origin', 'direction', 'old_shape']:
        if meta_key in npz_data:
            data[meta_key] = npz_data[meta_key]

    return data


def prepare_data_for_registration(moving_data, fixed_data, channels=['vol'], seg_channels=['seg']):
    """
    Prepare data tensors for registration

    Args:
        moving_data: Dict containing moving image data
        fixed_data: Dict containing fixed image data
        channels: List of image channels to concatenate
        seg_channels: List of segmentation channels

    Returns:
        x_in, x_seg_in, individual tensors
    """
    # Prepare image input
    moving_imgs = [moving_data[ch] for ch in channels if ch in moving_data]
    fixed_imgs = [fixed_data[ch] for ch in channels if ch in fixed_data]

    if moving_imgs and fixed_imgs:
        moving_concat = torch.cat(moving_imgs, dim=1) if len(moving_imgs) > 1 else moving_imgs[0]
        fixed_concat = torch.cat(fixed_imgs, dim=1) if len(fixed_imgs) > 1 else fixed_imgs[0]
        x_in = torch.cat((moving_concat, fixed_concat), dim=1)
    else:
        x_in = None

    # Prepare segmentation input if available
    x_seg_in = None
    if seg_channels and all(ch in moving_data and ch in fixed_data for ch in seg_channels):
        moving_segs = [moving_data[ch] for ch in seg_channels]
        fixed_segs = [fixed_data[ch] for ch in seg_channels]

        moving_seg_concat = torch.cat(moving_segs, dim=1) if len(moving_segs) > 1 else moving_segs[0]
        fixed_seg_concat = torch.cat(fixed_segs, dim=1) if len(fixed_segs) > 1 else fixed_segs[0]
        x_seg_in = torch.cat((moving_seg_concat, fixed_seg_concat), dim=1)

    return x_in, x_seg_in, moving_concat, fixed_concat


# ==========================================
# MODEL MANAGEMENT
# ==========================================

def initialize_model(model_type, imgsize, weights_path=None, **kwargs):
    """
    Initialize registration model based on type and configuration

    Args:
        model_type: 'casreg' or 'casreg_nhe'
        imgsize: Tuple of image dimensions
        weights_path: Path to pretrained weights
        **kwargs: Additional model parameters

    Returns:
        Initialized model
    """
    if weights_path and os.path.exists(weights_path):
        checkpoint = torch.load(weights_path)
        mode = checkpoint.get("mode", "cascade")
        layer_sizes = checkpoint.get("layer sizes", None)
        n_casc = len(layer_sizes) if layer_sizes else kwargs.get("n_casc", 2)
    else:
        mode = kwargs.get("mode", "cascade")
        n_casc = kwargs.get("n_casc", 2)

    if model_type.lower() == 'casreg':
        model = CasReg(
            imgsize,
            batch_size=kwargs.get("batch_size", 1),
            n_casc=n_casc,
            mode=mode,
            layer_mode=kwargs.get("layer_mode", "cascade"),
            return_list=kwargs.get("return_list", False)
        )
    elif model_type.lower() == 'casreg_nhe':
        model = CasReg_nhe(
            imgsize,
            batch_size=kwargs.get("batch_size", 2),
            coef_ratio=kwargs.get("coef_ratio", 1),
            n_casc=n_casc,
            mode=mode,
            layer_mode=kwargs.get("layer_mode", "contracted")
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if weights_path and os.path.exists(weights_path):
        checkpoint = torch.load(weights_path)
        model.load_state_dict(checkpoint['state_dict'])

    if torch.cuda.is_available():
        model.cuda()

    return model


def setup_registration_models(imgsize, interpolation_modes=['nearest', 'bilinear']):
    """
    Setup spatial transformer models for applying deformations

    Args:
        imgsize: Tuple of image dimensions
        interpolation_modes: List of interpolation modes to setup

    Returns:
        Dictionary of registration models
    """
    reg_models = {}
    for mode in interpolation_modes:
        reg_models[mode] = utils.register_model(imgsize, mode)
        if torch.cuda.is_available():
            reg_models[mode].cuda()

    return reg_models


# ==========================================
# FUSION ALGORITHMS
# ==========================================

def most_common(lst):
    """
    Find most common element in a list (majority voting)

    Args:
        lst: List of values

    Returns:
        Most common value
    """
    data = Counter(lst)
    return data.most_common(1)[0][0]


def weighted_fusion(sim_metric, labels, nb_labels):
    """
    Weighted fusion based on similarity metrics

    Args:
        sim_metric: Array of similarity values for each label
        labels: Array of label values
        nb_labels: Total number of labels

    Returns:
        Fused label based on weighted voting
    """
    weighted_labels = np.zeros(nb_labels)
    labels = labels.astype(int)

    for i in range(len(sim_metric)):
        if labels[i] < nb_labels:  # Safety check
            weighted_labels[labels[i]] += sim_metric[i]

    return weighted_labels.argmax()


def multi_atlas_fusion(warped_segs, similarity_scores, fusion_method='majority'):
    """
    Fuse multiple atlas segmentations

    Args:
        warped_segs: List of warped segmentation arrays
        similarity_scores: List of similarity scores for each atlas
        fusion_method: 'majority', 'weighted', or 'best'

    Returns:
        Fused segmentation array
    """
    if not warped_segs:
        return None

    # Stack all segmentations
    all_segs = np.stack(warped_segs, axis=-1)
    result_shape = all_segs.shape[:-1]
    fused_seg = np.zeros(result_shape)

    if fusion_method == 'majority':
        # Majority voting
        for idx in np.ndindex(result_shape):
            fused_seg[idx] = most_common(all_segs[idx])

    elif fusion_method == 'weighted':
        # Weighted fusion
        sim_array = np.array(similarity_scores)
        nb_labels = int(np.max([np.max(seg) for seg in warped_segs])) + 1

        for idx in np.ndindex(result_shape):
            fused_seg[idx] = weighted_fusion(
                sim_array[:, idx] if len(sim_array.shape) > 1 else sim_array,
                all_segs[idx],
                nb_labels
            )

    elif fusion_method == 'best':
        # Use best atlas only
        best_idx = np.argmax(similarity_scores)
        fused_seg = warped_segs[best_idx]

    return fused_seg


# ==========================================
# METRICS AND EVALUATION
# ==========================================

def calculate_metrics(warped, fixed, warped_seg=None, fixed_seg=None, flow=None, nb_labels=8):
    """
    Calculate comprehensive registration metrics

    Args:
        warped: Warped moving image
        fixed: Fixed image
        warped_seg: Warped segmentation (optional)
        fixed_seg: Fixed segmentation (optional)
        flow: Deformation field (optional)
        nb_labels: Number of segmentation labels

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Image similarity metrics
    metrics['ncc'] = losses.NCC().forward(warped, fixed).item()

    if warped_seg is not None and fixed_seg is not None:
        # Convert to CPU numpy if needed
        if isinstance(warped_seg, torch.Tensor):
            warped_seg = utils.cpu(warped_seg)
        if isinstance(fixed_seg, torch.Tensor):
            fixed_seg = utils.cpu(fixed_seg)

        # DICE scores per label
        dice_scores = utils.dice(warped_seg, fixed_seg, nb_labels)
        metrics['dice_per_label'] = dice_scores
        metrics['mean_dice'] = np.mean(dice_scores[1:])  # Exclude background

    # Jacobian determinant (deformation quality)
    if flow is not None:
        flow_cpu = utils.cpu(flow)
        jac_det = utils.jacobian_determinant_vxm(flow_cpu[0, :, :, :, :])
        metrics['folding_ratio'] = np.sum(jac_det <= 0) / np.prod(jac_det.shape)
        metrics['jac_det_std'] = np.std(jac_det)
        metrics['jac_det_mean'] = np.mean(jac_det)

    return metrics


# ==========================================
# VISUALIZATION
# ==========================================

def create_deformation_grid(imgsize, grid_step=8, line_thickness=1):
    """
    Create a grid image for deformation visualization

    Args:
        imgsize: Tuple of image dimensions
        grid_step: Spacing between grid lines
        line_thickness: Thickness of grid lines

    Returns:
        Grid tensor
    """
    grid_img = np.zeros(imgsize)

    for i in range(0, grid_img.shape[0], grid_step):
        grid_img[i:i+line_thickness, :, :] = 1
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j:j+line_thickness, :] = 1
    for k in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, k:k+line_thickness] = 1

    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img)

    if torch.cuda.is_available():
        grid_img = grid_img.cuda()

    return grid_img


def plot_registration_results(moving, fixed, warped, moving_seg=None, fixed_seg=None, warped_seg=None,
                            slice_idx=None, save_path=None):
    """
    Create comprehensive registration result plots

    Args:
        moving, fixed, warped: Image arrays
        moving_seg, fixed_seg, warped_seg: Segmentation arrays (optional)
        slice_idx: Slice index for visualization (default: middle slice)
        save_path: Path to save the plot
    """
    # Convert tensors to numpy if needed
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return utils.cpu(x)
        return x

    moving = to_numpy(moving)
    fixed = to_numpy(fixed)
    warped = to_numpy(warped)

    if slice_idx is None:
        slice_idx = moving.shape[-1] // 2

    # Determine subplot layout
    n_rows = 2 if any(x is not None for x in [moving_seg, fixed_seg, warped_seg]) else 1
    n_cols = 3

    plt.figure(figsize=(12, 8))

    # Image plots
    plt.subplot(n_rows, n_cols, 1)
    plt.imshow(moving[..., slice_idx], cmap="gray")
    plt.title("Moving")
    plt.axis("off")

    plt.subplot(n_rows, n_cols, 2)
    plt.imshow(fixed[..., slice_idx], cmap="gray")
    plt.title("Fixed")
    plt.axis("off")

    plt.subplot(n_rows, n_cols, 3)
    plt.imshow(warped[..., slice_idx], cmap="gray")
    plt.title("Warped")
    plt.axis("off")

    # Segmentation plots if available
    if n_rows == 2:
        if moving_seg is not None:
            moving_seg = to_numpy(moving_seg)
            plt.subplot(n_rows, n_cols, 4)
            plt.imshow(moving_seg[..., slice_idx])
            plt.title("Moving Seg")
            plt.axis("off")

        if fixed_seg is not None:
            fixed_seg = to_numpy(fixed_seg)
            plt.subplot(n_rows, n_cols, 5)
            plt.imshow(fixed_seg[..., slice_idx])
            plt.title("Fixed Seg")
            plt.axis("off")

        if warped_seg is not None:
            warped_seg = to_numpy(warped_seg)
            plt.subplot(n_rows, n_cols, 6)
            plt.imshow(warped_seg[..., slice_idx])
            plt.title("Warped Seg")
            plt.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_deformation_grid_3d(flow, imgsize, subsample=2):
    """
    3D visualization of deformation grid

    Args:
        flow: Deformation field
        imgsize: Image dimensions
        subsample: Subsampling factor for visualization
    """
    plt.style.use("dark_background")

    grid_img = create_deformation_grid(imgsize, grid_step=8)
    warped_grid = utils.cpu(utils.mk_grid(imgsize, flow)[0])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    s = subsample
    slice_idx = imgsize[2] // 2

    ax.plot_wireframe(
        warped_grid[0, ::s, ::s, slice_idx],
        warped_grid[1, ::s, ::s, slice_idx],
        warped_grid[2, ::s, ::s, slice_idx],
        linewidth=0.5,
        rstride=2,
        cstride=2
    )
    ax.set_axis_off()
    plt.show()


# ==========================================
# POSTPROCESSING
# ==========================================

def smooth_segmentation(seg_array, method='median', kernel_size=3):
    """
    Apply smoothing to segmentation

    Args:
        seg_array: Segmentation array
        method: 'median', 'gaussian'
        kernel_size: Kernel size for smoothing

    Returns:
        Smoothed segmentation
    """
    if method == 'median':
        return ndimage.median_filter(seg_array, size=kernel_size)
    elif method == 'gaussian':
        return gaussian_filter(seg_array.astype(float), sigma=kernel_size/3)
    else:
        return seg_array


def save_results_nifti(data_dict, output_dir, subject_id, spacing=None, origin=None, direction=None):
    """
    Save results as NIFTI files

    Args:
        data_dict: Dictionary with 'image', 'segmentation', etc.
        output_dir: Output directory
        subject_id: Subject identifier
        spacing, origin, direction: Image metadata
    """
    os.makedirs(output_dir, exist_ok=True)

    for data_type, data in data_dict.items():
        if data is None:
            continue

        # Convert to numpy if needed
        if isinstance(data, torch.Tensor):
            data = utils.cpu(data)

        # Remove extra dimensions
        data = np.squeeze(data)

        # Create SITK image
        sitk_img = sitk.GetImageFromArray(data)

        # Set metadata if available
        if spacing is not None:
            sitk_img.SetSpacing(np.array(spacing, dtype="float").tolist())
        if origin is not None:
            sitk_img.SetOrigin(origin)
        if direction is not None:
            sitk_img.SetDirection(direction)

        # Save
        filename = f"{subject_id}_{data_type}.nii.gz"
        sitk.WriteImage(sitk_img, os.path.join(output_dir, filename))


# ==========================================
# WORKFLOW HELPERS
# ==========================================

class RegistrationWorkflow:
    """
    Base class for registration workflows
    """
    def __init__(self, config):
        self.config = config
        self.model = None
        self.reg_models = None

    def setup(self):
        """Setup models and configurations"""
        self.model = initialize_model(
            self.config['model_type'],
            self.config['imgsize'],
            self.config.get('weights_path'),
            **self.config.get('model_params', {})
        )

        self.reg_models = setup_registration_models(
            self.config['imgsize'],
            self.config.get('interpolation_modes', ['nearest', 'bilinear'])
        )

    def run_registration(self, moving_data, fixed_data):
        """Run registration - to be implemented by subclasses"""
        raise NotImplementedError

    def evaluate_results(self, results):
        """Evaluate registration results"""
        return calculate_metrics(
            results['warped'],
            results['fixed'],
            results.get('warped_seg'),
            results.get('fixed_seg'),
            results.get('flow'),
            self.config.get('nb_labels', 8)
        )


def setup_optimizer(model, lr=3e-4, optimizer_type='adamw'):
    """
    Setup optimizer for fine-tuning

    Args:
        model: PyTorch model
        lr: Learning rate
        optimizer_type: 'adamw', 'adam', 'sgd'

    Returns:
        Optimizer instance
    """
    if optimizer_type.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)
    elif optimizer_type.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")