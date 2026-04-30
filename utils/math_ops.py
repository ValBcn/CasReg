"""
Mathematical operations and utility functions.
"""

import numpy as np
import torch
import pystrum.pynd.ndutils as nd


def jacobian_determinant_vxm(disp):
    """
    Jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.

    Args:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims

    Returns:
        Jacobian determinant (scalar)
    """

    # check inputs
    disp = disp.transpose(1, 2, 3, 0)
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D flow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2
        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]


def MSE_torch(x, y):
    """
    Calculate Mean Squared Error using PyTorch.

    Args:
        x: First tensor
        y: Second tensor

    Returns:
        MSE value
    """
    return torch.mean((x - y) ** 2)


def mk_grid(shape, flow):
    """
    Create a warped grid from flow field.

    Args:
        shape: Grid shape
        flow: Flow field

    Returns:
        Warped grid
    """
    vectors = [torch.arange(0, s) for s in shape]
    grids = torch.meshgrid(vectors)
    grid = torch.stack(grids).cuda()

    warped_grid = grid + flow
    return warped_grid