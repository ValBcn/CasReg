"""
Visualization utilities for medical images and registration.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt


def comput_fig(img):
    """
    Create figure from image tensor for visualization.

    Args:
        img: Input image tensor

    Returns:
        Matplotlib figure
    """
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12, 12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    """
    Create a grid image for deformation visualization.

    Args:
        grid_step: Step size for grid lines
        line_thickness: Thickness of grid lines
        grid_sz: Size of the grid

    Returns:
        Grid image tensor
    """
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img


def mk_grid_img_2d(grid_step, line_thickness=1, grid_sz=(128, 128, 128)):
    """
    Create a 2D grid image for deformation visualization.

    Args:
        grid_step: Step size for grid lines
        line_thickness: Thickness of grid lines
        grid_sz: Size of the grid

    Returns:
        Grid image tensor
    """
    grid_img = np.zeros((grid_sz[0],grid_sz[1],grid_sz[2]))
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for k in range(0, grid_img.shape[0], grid_step):
        grid_img[k+line_thickness-1,...] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img