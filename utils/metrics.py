"""
Metrics and evaluation functions for medical image analysis.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.spatial.distance import directed_hausdorff


def dice_val(y_pred, y_true, num_clus):
    """
    Calculate Dice coefficient for multi-class segmentation.

    Args:
        y_pred: Predicted segmentation
        y_true: Ground truth segmentation
        num_clus: Number of classes

    Returns:
        Average Dice coefficient
    """
    y_pred = nn.functional.one_hot(y_pred, num_classes=num_clus)
    y_pred = torch.squeeze(y_pred, 1)
    y_pred = y_pred.contiguous()
    y_true = nn.functional.one_hot(y_true, num_classes=num_clus)
    y_true = torch.squeeze(y_true, 1)
    y_true = y_true.contiguous()
    intersection = y_pred * y_true
    intersection = intersection.sum(dim=[2, 3, 4])
    union = y_pred.sum(dim=[2, 3, 4]) + y_true.sum(dim=[2, 3, 4])
    dsc = (2.*intersection) / (union + 1e-5)
    return torch.mean(torch.mean(dsc, dim=1))


def dice_val_substruct(y_pred, y_true, std_idx):
    """
    Calculate Dice coefficient for each structure separately.

    Args:
        y_pred: Predicted segmentation
        y_true: Ground truth segmentation
        std_idx: Standard index

    Returns:
        Dice scores for each structure as string
    """
    with torch.no_grad():
        y_pred = nn.functional.one_hot(y_pred, num_classes=46)
        y_pred = torch.squeeze(y_pred, 1)
        y_pred = y_pred.permute(0, 4, 1, 2, 3).contiguous()
        y_true = nn.functional.one_hot(y_true, num_classes=46)
        y_true = torch.squeeze(y_true, 1)
        y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()

    line = 'p_{}'.format(std_idx)
    for i in range(46):
        pred_clus = y_pred[0, i, ...]
        true_clus = y_true[0, i, ...]
        intersection = pred_clus * true_clus
        intersection = intersection.sum()
        union = pred_clus.sum() + true_clus.sum()
        dsc = (2.*intersection) / (union + 1e-5)
        line = line+','+str(dsc)
    return line


def dice(y_pred, y_true):
    """
    Calculate Dice coefficient for binary segmentation.

    Args:
        y_pred: Predicted segmentation
        y_true: Ground truth segmentation

    Returns:
        Dice coefficient
    """
    intersection = y_pred * y_true
    intersection = np.sum(intersection)
    union = np.sum(y_pred) + np.sum(y_true)
    dsc = (2.*intersection) / (union + 1e-5)
    return dsc


def hd95(seg1, seg2, voxel_spacing=None):
    """
    Calculate 95th percentile Hausdorff distance.

    Args:
        seg1: First segmentation
        seg2: Second segmentation
        voxel_spacing: Voxel spacing information

    Returns:
        95th percentile Hausdorff distance
    """
    if voxel_spacing is None:
        voxel_spacing = [1.0, 1.0, 1.0]

    # Get surface points using distance transform
    dt1 = distance_transform_edt(~seg1, sampling=voxel_spacing)
    dt2 = distance_transform_edt(~seg2, sampling=voxel_spacing)

    # Find surfaces
    surface1 = np.logical_and(seg1, dt1 <= np.sqrt(3))
    surface2 = np.logical_and(seg2, dt2 <= np.sqrt(3))

    # Get coordinates
    coords1 = np.array(np.nonzero(surface1)).T * voxel_spacing
    coords2 = np.array(np.nonzero(surface2)).T * voxel_spacing

    if len(coords1) == 0 or len(coords2) == 0:
        return float('inf')

    # Calculate Hausdorff distance
    hd1 = directed_hausdorff(coords1, coords2)[0]
    hd2 = directed_hausdorff(coords2, coords1)[0]

    # Return 95th percentile
    distances = np.concatenate([
        np.min(np.linalg.norm(coords1[:, None] - coords2[None, :], axis=2), axis=1),
        np.min(np.linalg.norm(coords2[:, None] - coords1[None, :], axis=2), axis=1)
    ])

    return np.percentile(distances, 95)


def uceloss(errors, uncert, n_bins=15, outlier=0.0, range=None):
    """
    Uncertainty Calibration Error loss function.

    Args:
        errors: Error values
        uncert: Uncertainty values
        n_bins: Number of bins for calibration
        outlier: Outlier threshold
        range: Range for binning

    Returns:
        UCE loss and bin statistics
    """
    device = errors.device
    if range == None:
        bin_boundaries = torch.linspace(uncert.min().item(), uncert.max().item(), n_bins + 1, device=device)
    else:
        bin_boundaries = torch.linspace(range[0], range[1], n_bins + 1, device=device)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    errors_in_bin_list = []
    avg_uncert_in_bin_list = []
    prop_in_bin_list = []

    uce = torch.zeros(1, device=device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |uncertainty - error| in each bin
        in_bin = uncert.gt(bin_lower.item()) * uncert.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()  # |Bm| / n
        prop_in_bin_list.append(prop_in_bin)
        if prop_in_bin.item() > outlier:
            errors_in_bin = errors[in_bin].float().mean()  # err()
            avg_uncert_in_bin = uncert[in_bin].mean()  # uncert()
            uce += torch.abs(avg_uncert_in_bin - errors_in_bin) * prop_in_bin

            errors_in_bin_list.append(errors_in_bin)
            avg_uncert_in_bin_list.append(avg_uncert_in_bin)

    err_in_bin = torch.tensor(errors_in_bin_list, device=device)
    avg_uncert_in_bin = torch.tensor(avg_uncert_in_bin_list, device=device)
    prop_in_bin = torch.tensor(prop_in_bin_list, device=device)

    return uce, err_in_bin, avg_uncert_in_bin, prop_in_bin


def multi_channel_labels(label, nb_labels):
    """
    Convert single-channel labels to multi-channel format.

    Args:
        label: Input label array
        nb_labels: Number of label classes

    Returns:
        Multi-channel label array
    """
    multi_channel_label = np.zeros((*label.shape, nb_labels))

    for i in range(nb_labels):
        multi_channel_label[..., i] = np.where(label != i, multi_channel_label[..., i], 1)
    return multi_channel_label


def dice_multiclass(seg1, seg2, nlabels):
    """
    Calculate Dice coefficient for multi-class segmentation.

    Args:
        seg1: First segmentation
        seg2: Second segmentation
        nlabels: Number of labels

    Returns:
        List of Dice coefficients for each class
    """
    seg1 = multi_channel_labels(seg1, nlabels)
    seg2 = multi_channel_labels(seg2, nlabels)

    dices = []

    for i in range(nlabels):
        dice_val = np.sum(seg1[..., i].flatten()[seg2[..., i].flatten() == 1]) * 2.0 / (
                np.sum(seg1[..., i].flatten()) + np.sum(seg2[..., i].flatten()))
        dices.append(dice_val)
    return dices


def smooth_seg(binary_img, sigma=1.5, thresh=0.4):
    """
    Smooth binary segmentation using Gaussian filter.

    Args:
        binary_img: Binary segmentation image
        sigma: Gaussian filter sigma
        thresh: Threshold for binarization

    Returns:
        Smoothed binary segmentation
    """
    binary_img = gaussian_filter(binary_img.astype(np.float32()), sigma=sigma)
    binary_img = binary_img > thresh
    return binary_img