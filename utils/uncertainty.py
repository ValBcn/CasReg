"""
Uncertainty estimation utilities for deep learning models.
"""

import torch
import torch.nn as nn


def get_mc_preds(net, inputs, mc_iter: int = 25):
    """
    Convenience fn. for MC integration for uncertainty estimation.

    Args:
        net: DIP model (can be standard, MFVI or MCDropout)
        inputs: Input to net
        mc_iter: Number of MC samples

    Returns:
        Lists of image and flow predictions
    """
    img_list = []
    flow_list = []
    with torch.no_grad():
        for _ in range(mc_iter):
            img, flow = net(inputs)
            img_list.append(img)
            flow_list.append(flow)
    return img_list, flow_list


def calc_uncert(tar, img_list):
    """
    Calculate uncertainty from Monte Carlo predictions.

    Args:
        tar: Target tensor
        img_list: List of prediction tensors

    Returns:
        Uncertainty tensor
    """
    sqr_diffs = []
    for i in range(len(img_list)):
        sqr_diff = (img_list[i] - tar)**2
        sqr_diffs.append(sqr_diff)
    uncert = torch.mean(torch.cat(sqr_diffs, dim=0)[:], dim=0, keepdim=True)
    return uncert


def calc_error(tar, img_list):
    """
    Calculate error from Monte Carlo predictions.

    Args:
        tar: Target tensor
        img_list: List of prediction tensors

    Returns:
        Error tensor
    """
    sqr_diffs = []
    for i in range(len(img_list)):
        sqr_diff = (img_list[i] - tar)**2
        sqr_diffs.append(sqr_diff)
    uncert = torch.mean(torch.cat(sqr_diffs, dim=0)[:], dim=0, keepdim=True)
    return uncert


def get_mc_preds_w_errors(net, inputs, target, mc_iter: int = 25):
    """
    Convenience fn. for MC integration for uncertainty estimation with errors.

    Args:
        net: DIP model (can be standard, MFVI or MCDropout)
        inputs: Input to net
        target: Target tensor
        mc_iter: Number of MC samples

    Returns:
        Lists of predictions and error values
    """
    img_list = []
    flow_list = []
    MSE = nn.MSELoss()
    err = []
    with torch.no_grad():
        for _ in range(mc_iter):
            img, flow = net(inputs)
            img_list.append(img)
            flow_list.append(flow)
            err.append(MSE(img, target).item())
    return img_list, flow_list, err


def get_diff_mc_preds(net, inputs, mc_iter: int = 25):
    """
    Convenience fn. for MC integration for uncertainty estimation with displacement.

    Args:
        net: DIP model (can be standard, MFVI or MCDropout)
        inputs: Input to net
        mc_iter: Number of MC samples

    Returns:
        Lists of image, flow, and displacement predictions
    """
    img_list = []
    flow_list = []
    disp_list = []
    with torch.no_grad():
        for _ in range(mc_iter):
            img, _, flow, disp = net(inputs)
            img_list.append(img)
            flow_list.append(flow)
            disp_list.append(disp)
    return img_list, flow_list, disp_list


def uncert_regression_gal(img_list, reduction='mean'):
    """
    Calculate aleatoric and epistemic uncertainty using Gal's method.

    Args:
        img_list: List of prediction tensors
        reduction: Reduction method ('mean', 'sum', or None)

    Returns:
        Aleatoric, epistemic, and total uncertainty
    """
    img_list = torch.cat(img_list, dim=0)
    mean = img_list[:,:-1].mean(dim=0, keepdim=True)
    ale = img_list[:,-1:].mean(dim=0, keepdim=True)
    epi = torch.var(img_list[:,:-1], dim=0, keepdim=True)
    epi = epi.mean(dim=1, keepdim=True)
    uncert = ale + epi
    if reduction == 'mean':
        return ale.mean().item(), epi.mean().item(), uncert.mean().item()
    elif reduction == 'sum':
        return ale.sum().item(), epi.sum().item(), uncert.sum().item()
    else:
        return ale.detach(), epi.detach(), uncert.detach()