import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import math
import torch.nn as nn
import pystrum.pynd.ndutils as nd

def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size,
                                                                  window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def _ssim_3D(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv3d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv3d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


class SSIM3D(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window_3D(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return 1-_ssim_3D(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def ssim3D(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim_3D(img1, img2, window, window_size, channel, size_average)


class Grad(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        super(Grad, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred, y_true):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

class Grad3d(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        super(Grad3d, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred, y_true):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

class Grad3DiTV(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self):
        super(Grad3DiTV, self).__init__()
        a = 1

    def forward(self, y_pred, y_true):
        dy = torch.abs(y_pred[:, :, 1:, 1:, 1:] - y_pred[:, :, :-1, 1:, 1:])
        dx = torch.abs(y_pred[:, :, 1:, 1:, 1:] - y_pred[:, :, 1:, :-1, 1:])
        dz = torch.abs(y_pred[:, :, 1:, 1:, 1:] - y_pred[:, :, 1:, 1:, :-1])
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz
        d = torch.mean(torch.sqrt(dx+dy+dz+1e-6))
        grad = d / 3.0
        return grad

class DisplacementRegularizer(torch.nn.Module):
    def __init__(self, energy_type):
        super().__init__()
        self.energy_type = energy_type

    def gradient_dx(self, fv): return (fv[:, 2:, 1:-1, 1:-1] - fv[:, :-2, 1:-1, 1:-1]) / 2

    def gradient_dy(self, fv): return (fv[:, 1:-1, 2:, 1:-1] - fv[:, 1:-1, :-2, 1:-1]) / 2

    def gradient_dz(self, fv): return (fv[:, 1:-1, 1:-1, 2:] - fv[:, 1:-1, 1:-1, :-2]) / 2

    def gradient_txyz(self, Txyz, fn):
        return torch.stack([fn(Txyz[:,i,...]) for i in [0, 1, 2]], dim=1)

    def compute_gradient_norm(self, displacement, flag_l1=False):
        dTdx = self.gradient_txyz(displacement, self.gradient_dx)
        dTdy = self.gradient_txyz(displacement, self.gradient_dy)
        dTdz = self.gradient_txyz(displacement, self.gradient_dz)
        if flag_l1:
            norms = torch.abs(dTdx) + torch.abs(dTdy) + torch.abs(dTdz)
        else:
            norms = dTdx**2 + dTdy**2 + dTdz**2
        return torch.mean(norms)/3.0

    def compute_bending_energy(self, displacement):
        dTdx = self.gradient_txyz(displacement, self.gradient_dx)
        dTdy = self.gradient_txyz(displacement, self.gradient_dy)
        dTdz = self.gradient_txyz(displacement, self.gradient_dz)
        dTdxx = self.gradient_txyz(dTdx, self.gradient_dx)
        dTdyy = self.gradient_txyz(dTdy, self.gradient_dy)
        dTdzz = self.gradient_txyz(dTdz, self.gradient_dz)
        dTdxy = self.gradient_txyz(dTdx, self.gradient_dy)
        dTdyz = self.gradient_txyz(dTdy, self.gradient_dz)
        dTdxz = self.gradient_txyz(dTdx, self.gradient_dz)
        return torch.mean(dTdxx**2 + dTdyy**2 + dTdzz**2 + 2*dTdxy**2 + 2*dTdxz**2 + 2*dTdyz**2)

    def forward(self, disp, _):
        if self.energy_type == 'bending':
            energy = self.compute_bending_energy(disp)
        elif self.energy_type == 'gradient-l2':
            energy = self.compute_gradient_norm(disp)
        elif self.energy_type == 'gradient-l1':
            energy = self.compute_gradient_norm(disp, flag_l1=True)
        else:
            raise Exception('Not recognised local regulariser!')
        return energy

class NCC(torch.nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        super(NCC, self).__init__()
        self.win = win

    def forward(self, y_pred, y_true):

        I = y_true
        J = y_pred

        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0]/2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1,1)
            padding = (pad_no, pad_no)
        else:
            stride = (1,1,1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)
    
class NCC_full(torch.nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        super(NCC_full, self).__init__()
        self.win = win

    def forward(self, y_pred, y_true):

        I = y_true
        J = y_pred

        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [7] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0]/2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1,1)
            padding = (pad_no, pad_no)
        else:
            stride = (1,1,1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return cc

class MIND_loss(torch.nn.Module):
    """
        Local (over window) normalized cross correlation loss.
        """

    def __init__(self, win=None):
        super(MIND_loss, self).__init__()
        self.win = win

    def pdist_squared(self, x):
        xx = (x ** 2).sum(dim=1).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
        dist[dist != dist] = 0
        dist = torch.clamp(dist, 0.0, np.inf)
        return dist

    def MINDSSC(self, img, radius=2, dilation=2):
        # see http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for details on the MIND-SSC descriptor

        # kernel size
        kernel_size = radius * 2 + 1

        # define start and end locations for self-similarity pattern
        six_neighbourhood = torch.Tensor([[0, 1, 1],
                                          [1, 1, 0],
                                          [1, 0, 1],
                                          [1, 1, 2],
                                          [2, 1, 1],
                                          [1, 2, 1]]).long()

        # squared distances
        dist = self.pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)

        # define comparison mask
        x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
        mask = ((x > y).view(-1) & (dist == 2).view(-1))

        # build kernel
        idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :]
        idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :]
        mshift1 = torch.zeros(12, 1, 3, 3, 3).cuda()
        mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:, 0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
        mshift2 = torch.zeros(12, 1, 3, 3, 3).cuda()
        mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:, 0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
        rpad1 = nn.ReplicationPad3d(dilation)
        rpad2 = nn.ReplicationPad3d(radius)

        # compute patch-ssd
        ssd = F.avg_pool3d(rpad2(
            (F.conv3d(rpad1(img), mshift1, dilation=dilation) - F.conv3d(rpad1(img), mshift2, dilation=dilation)) ** 2),
                           kernel_size, stride=1)

        # MIND equation
        mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
        mind_var = torch.mean(mind, 1, keepdim=True)
        mind_var = torch.clamp(mind_var, (mind_var.mean() * 0.001).item(), (mind_var.mean() * 1000).item())
        mind /= mind_var
        mind = torch.exp(-mind)

        # permute to have same ordering as C++ code
        mind = mind[:, torch.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]

        return mind

    def forward(self, y_pred, y_true):
        return torch.mean((self.MINDSSC(y_pred) - self.MINDSSC(y_true)) ** 2)

class MutualInformation(torch.nn.Module):
    """
    Mutual Information
    """

    def __init__(self, sigma_ratio=1, minval=0., maxval=1., num_bin=32):
        super(MutualInformation, self).__init__()

        """Create bin centers"""
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = Variable(torch.linspace(minval, maxval, num_bin), requires_grad=False).cuda()
        num_bins = len(bin_centers)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
        print(sigma)

        self.preterm = 1 / (2 * sigma ** 2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers

    def mi(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, 0., self.max_clip)
        y_true = torch.clamp(y_true, 0, self.max_clip)

        y_true = y_true.view(y_true.shape[0], -1)
        y_true = torch.unsqueeze(y_true, 2)
        y_pred = y_pred.view(y_pred.shape[0], -1)
        y_pred = torch.unsqueeze(y_pred, 2)

        nb_voxels = y_pred.shape[1]  # total num of voxels

        """Reshape bin centers"""
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o).cuda()

        """compute image terms by approx. Gaussian dist."""
        I_a = torch.exp(- self.preterm * torch.square(y_true - vbc))
        I_a = I_a / torch.sum(I_a, dim=-1, keepdim=True)

        I_b = torch.exp(- self.preterm * torch.square(y_pred - vbc))
        I_b = I_b / torch.sum(I_b, dim=-1, keepdim=True)

        # compute probabilities
        pab = torch.bmm(I_a.permute(0, 2, 1), I_b)
        pab = pab / nb_voxels
        pa = torch.mean(I_a, dim=1, keepdim=True)
        pb = torch.mean(I_b, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)
        return mi.mean()  # average across batch

    def forward(self, y_true, y_pred):
        return -self.mi(y_true, y_pred)

class localMutualInformation(torch.nn.Module):
    """
    Local Mutual Information for non-overlapping patches
    """

    def __init__(self, sigma_ratio=1, minval=0., maxval=1., num_bin=32, patch_size=5):
        super(localMutualInformation, self).__init__()

        """Create bin centers"""
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = Variable(torch.linspace(minval, maxval, num_bin), requires_grad=False).cuda()
        num_bins = len(bin_centers)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio

        self.preterm = 1 / (2 * sigma ** 2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers
        self.patch_size = patch_size

    def local_mi(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, 0., self.max_clip)
        y_true = torch.clamp(y_true, 0, self.max_clip)

        """Reshape bin centers"""
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o).cuda()

        """Making image paddings"""
        if len(list(y_pred.size())[2:]) == 3:
            ndim = 3
            x, y, z = list(y_pred.size())[2:]
            # compute padding sizes
            x_r = -x % self.patch_size
            y_r = -y % self.patch_size
            z_r = -z % self.patch_size
            padding = (z_r // 2, z_r - z_r // 2, y_r // 2, y_r - y_r // 2, x_r // 2, x_r - x_r // 2, 0, 0, 0, 0)
        elif len(list(y_pred.size())[2:]) == 2:
            ndim = 2
            x, y = list(y_pred.size())[2:]
            # compute padding sizes
            x_r = -x % self.patch_size
            y_r = -y % self.patch_size
            padding = (y_r // 2, y_r - y_r // 2, x_r // 2, x_r - x_r // 2, 0, 0, 0, 0)
        else:
            raise Exception('Supports 2D and 3D but not {}'.format(list(y_pred.size())))
        y_true = F.pad(y_true, padding, "constant", 0)
        y_pred = F.pad(y_pred, padding, "constant", 0)

        """Reshaping images into non-overlapping patches"""
        if ndim == 3:
            y_true_patch = torch.reshape(y_true, (y_true.shape[0], y_true.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size,
                                                  (z + z_r) // self.patch_size, self.patch_size))
            y_true_patch = y_true_patch.permute(0, 1, 2, 4, 6, 3, 5, 7)
            y_true_patch = torch.reshape(y_true_patch, (-1, self.patch_size ** 3, 1))

            y_pred_patch = torch.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size,
                                                  (z + z_r) // self.patch_size, self.patch_size))
            y_pred_patch = y_pred_patch.permute(0, 1, 2, 4, 6, 3, 5, 7)
            y_pred_patch = torch.reshape(y_pred_patch, (-1, self.patch_size ** 3, 1))
        else:
            y_true_patch = torch.reshape(y_true, (y_true.shape[0], y_true.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size))
            y_true_patch = y_true_patch.permute(0, 1, 2, 4, 3, 5)
            y_true_patch = torch.reshape(y_true_patch, (-1, self.patch_size ** 2, 1))

            y_pred_patch = torch.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size))
            y_pred_patch = y_pred_patch.permute(0, 1, 2, 4, 3, 5)
            y_pred_patch = torch.reshape(y_pred_patch, (-1, self.patch_size ** 2, 1))

        """Compute MI"""
        I_a_patch = torch.exp(- self.preterm * torch.square(y_true_patch - vbc))
        I_a_patch = I_a_patch / torch.sum(I_a_patch, dim=-1, keepdim=True)

        I_b_patch = torch.exp(- self.preterm * torch.square(y_pred_patch - vbc))
        I_b_patch = I_b_patch / torch.sum(I_b_patch, dim=-1, keepdim=True)

        pab = torch.bmm(I_a_patch.permute(0, 2, 1), I_b_patch)
        pab = pab / self.patch_size ** ndim
        pa = torch.mean(I_a_patch, dim=1, keepdim=True)
        pb = torch.mean(I_b_patch, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)

        return mi.mean()

    def forward(self, y_true, y_pred):
        return -self.local_mi(y_true, y_pred)

class Dice_Weighted:
    """                                                                                                                                                                                                    
    N-Dice for segmention, with evolving weights
    """
    def __init__(self, weights_dc):
        self.weights_dc = weights_dc
    def loss(self, y_true, y_pred):
        DICE = 0

        for i in range(self.weights_dc.shape[0]):
            y_true_ = torch.zeros(y_true.shape).cuda()
            y_true_[y_true==i] = y_true[y_true==i]+1
            y_true_ = y_true_/y_true_.max()
            y_pred_ = torch.zeros(y_pred.shape).cuda()
            y_pred_[y_pred==i] = y_pred[y_pred==i]+1
            y_pred_ = y_pred_/y_pred_.max()
            top = 0
            bottom = 0
            top = 2 * torch.sum(y_true_[0,0,...]*y_pred_[0,0,...])
            bottom = torch.sum(y_true_[0,0,...]+y_pred_[0,0,...])
            DICE += self.weights_dc[i] * torch.div(top,bottom)

        return -DICE/(len(self.weights_dc)-1)


class Dice:
    """                                                                                                                                                                                                    
    N-Dice for segmention, with evolving weights
    """
    def __init__(self,weights_dc):
        self.weights_dc = weights_dc
    def loss(self, y_pred, y_true, num_clus=8):
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

from torch import Tensor


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())
    
    



class GDL(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False, square_volumes=False):
        """
        square_volumes will square the weight term. The paper recommends square_volumes=True; I don't (just an intuition)
        """
        super(GDL, self).__init__()

        self.square_volumes = square_volumes
        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        shp_y = y.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if len(shp_x) != len(shp_y):
            y = y.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(x.shape, y.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = y
        else:
            gt = y.long()
            y_onehot = torch.zeros(shp_x)
            if x.device.type == "cuda":
                y_onehot = y_onehot.cuda(x.device.index)
            y_onehot.scatter_(1, gt, 1)

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        if not self.do_bg:
            x = x[:, 1:]
            y_onehot = y_onehot[:, 1:]

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y_onehot, axes, loss_mask, self.square)

        # GDL weight computation, we use 1/V
        volumes = sum_tensor(y_onehot, axes) + 1e-6 # add some eps to prevent div by zero

        if self.square_volumes:
            volumes = volumes ** 2

        # apply weights
        tp = tp / volumes
        fp = fp / volumes
        fn = fn / volumes

        # sum over classes
        if self.batch_dice:
            axis = 0
        else:
            axis = 1

        tp = tp.sum(axis, keepdim=False)
        fp = fp.sum(axis, keepdim=False)
        fn = fn.sum(axis, keepdim=False)

        # compute dice
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)

        dc = dc.mean()

        return -dc


def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x, device=net_output.device)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn


class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc


class MCCLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_mcc=False, do_bg=True, smooth=0.0):
        """
        based on matthews correlation coefficient
        https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
        Does not work. Really unstable. F this.
        """
        super(MCCLoss, self).__init__()

        self.smooth = smooth
        self.do_bg = do_bg
        self.batch_mcc = batch_mcc
        self.apply_nonlin = apply_nonlin

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        voxels = np.prod(shp_x[2:])

        if self.batch_mcc:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, tn = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)
        tp /= voxels
        fp /= voxels
        fn /= voxels
        tn /= voxels

        nominator = tp * tn - fp * fn + self.smooth
        denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5 + self.smooth

        mcc = nominator / denominator

        if not self.do_bg:
            if self.batch_mcc:
                mcc = mcc[1:]
            else:
                mcc = mcc[:, 1:]
        mcc = mcc.mean()

        return -mcc


class SoftDiceLossSquared(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        """
        squares the terms in the denominator as proposed by Milletari et al.
        """
        super(SoftDiceLossSquared, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        shp_y = y.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(x.shape, y.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                y = y.long()
                y_onehot = torch.zeros(shp_x)
                if x.device.type == "cuda":
                    y_onehot = y_onehot.cuda(x.device.index)
                y_onehot.scatter_(1, y, 1).float()

        intersect = x * y_onehot
        # values in the denominator get smoothed
        denominator = x ** 2 + y_onehot ** 2

        # aggregation was previously done in get_tp_fp_fn, but needs to be done here now (needs to be done after
        # squaring)
        intersect = sum_tensor(intersect, axes, False) + self.smooth
        denominator = sum_tensor(denominator, axes, False) + self.smooth

        dc = 2 * intersect / denominator

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc


softmax_helper = lambda x: F.softmax(x, 1)

class DC_and_CE_loss(nn.Module):
    def __init__(self, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1,
                 log_dice=False, ignore_label=None):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()

        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate
        self.ce = RobustCrossEntropyLoss()

        self.ignore_label = ignore_label

        if not square_dice:
            self.dc = SoftDiceLoss(apply_nonlin=softmax_helper)
        else:
            self.dc = SoftDiceLossSquared(apply_nonlin=softmax_helper)

    def forward(self, net_output, target):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        dc_loss = self.dc(net_output, target, loss_mask=mask) if self.weight_dice != 0 else 0
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)

        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0
        if self.ignore_label is not None:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()

        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result


class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, aggregate="sum"):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!
        THIS LOSS IS INTENDED TO BE USED FOR BRATS REGIONS ONLY
        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()

        self.aggregate = aggregate
        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output, target):
        ce_loss = self.ce(net_output, target)
        dc_loss = self.dc(net_output, target)

        if self.aggregate == "sum":
            result = ce_loss + dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)

        return result

class MSE_full(torch.nn.Module):
    """
    Local (over window) MSE loss.
    """

    def __init__(self, win=None):
        super(MSE_full, self).__init__()
        self.win = win

    def forward(self, y_pred, y_true):

        I = y_true
        J = y_pred

        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0]/2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1,1)
            padding = (pad_no, pad_no)
        else:
            stride = (1,1,1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I * I
        J2 = J * J

        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        
        mse = 1-torch.abs(I2_sum - J2_sum)/ win_size
        
        return torch.mean(mse)
    
def jacobian_determinant_vxm(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    disp = np.squeeze(disp).transpose(1, 2, 3, 0)
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    
    J = np.gradient(disp+grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = (dx[..., 0]+1) * ((dy[..., 1]+1) * (dz[..., 2]+1) - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * (dz[..., 2]+1) - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - (dy[..., 1]+1) * dz[..., 0])
        
        Jdet = Jdet0 - Jdet1 + Jdet2
        
        # compute trace
        Tr1 = dx[..., 0]**2 + dx[..., 1]**2 + dx[..., 2]**2
        Tr2 = dy[..., 0]**2 + dy[..., 1]**2 + dy[..., 2]**2
        Tr3 = dz[..., 0]**2 + dz[..., 1]**2 + dz[..., 2]**2
        
        Tr = Tr1 + Tr2 + Tr3

        return Jdet

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]

class local_grad_neo_hookean(torch.nn.Module):
    def __init__(self, ratio=5, return_maps=False, window = 3, mode = "exp", facx = None):
        super(local_grad_neo_hookean, self).__init__()
        self.ratio = ratio
        self.lambda_ = lambda_
        self.return_maps = return_maps
        self.window = window
        self.mode = mode
        self.facx = facx
        
    def forward(self, y_pred, y_true):
        
        # Define the window size for the convolution
        win = [self.window] * 3
        
        pos_idx = np.where(y_true.detach().cpu().numpy() > 0)
        y_pred2 = torch.zeros((1,3,np.max(pos_idx)-np.min(pos_idx),128,128))
        y_pred2 = y_pred[:,:,np.min(pos_idx):np.max(pos_idx),...]
        
        if self.facx is not None:
            y_pred2[:,0,...] *= self.facx
        # Compute the padding size, stride and output shape
        pad_no = math.floor(win[0]/2)
        stride = (1,1,1)
        padding = (pad_no, pad_no, pad_no)

        # compute the gradient along x,y,z
        dy = torch.abs(y_pred[:, :, 1:, :-1, :-1] - y_pred[:, :, :-1, :-1, :-1]) 
        dx = torch.abs(y_pred[:, :, :-1, 1:, :-1] - y_pred[:, :, :-1, :-1, :-1]) 
        dz = torch.abs(y_pred[:, :, :-1, :-1, 1:] - y_pred[:, :, :-1, :-1, :-1])
        
        # dy = torch.abs(y_pred[:, :, 2:, :-2, :-2] + 2*y_pred[:, :, 1:-1, :-1, :-1] - y_pred[:, :, :-2, :-2, :-2]) 
        # dx = torch.abs(y_pred[:, :, :-2, 2:, :-2] + 2*y_pred[:, :, :-2, 1:-1, :-2] - y_pred[:, :, :-2, :-2, :-2])
        # dz = torch.abs(y_pred[:, :, :-2, :-2, 2:] + 2*y_pred[:, :, :-2, :-2, 1:-1] - y_pred[:, :, :-2, :-2, :-2])
        
        # Create a filter that will be used to convolve the deformation field
        sum_filt = torch.ones([1,1, *win]).to("cuda")
        sum_filt = sum_filt/(self.window**3) # divide by window^3 to get the average over the window
                
        # convolve the gradients to get the partial derivatives of the deformation fields
        dxdx = torch.nn.functional.conv3d(dx[:,1:2,...], sum_filt, stride=stride, padding=padding) # dx/dX
        dxdy = torch.nn.functional.conv3d(dy[:,1:2,...], sum_filt, stride=stride, padding=padding) # dx/dY
        dxdz = torch.nn.functional.conv3d(dz[:,1:2,...], sum_filt, stride=stride, padding=padding) # dx/dZ
        dydx = torch.nn.functional.conv3d(dx[:,0:1,...], sum_filt, stride=stride, padding=padding) # dy/dX
        dydy = torch.nn.functional.conv3d(dy[:,0:1,...], sum_filt, stride=stride, padding=padding) # dy/dY
        dydz = torch.nn.functional.conv3d(dz[:,0:1,...], sum_filt, stride=stride, padding=padding) # dy/dZ
        dzdx = torch.nn.functional.conv3d(dx[:,2:3,...], sum_filt, stride=stride, padding=padding) # dz/dX
        dzdy = torch.nn.functional.conv3d(dy[:,2:3,...], sum_filt, stride=stride, padding=padding) # dz/dY
        dzdz = torch.nn.functional.conv3d(dz[:,2:3,...], sum_filt, stride=stride, padding=padding) # dz/dZ
        
        # compute the determinant of the deformation gradient tensor
        det_1 = (dxdx+1) * ((dydy+1) * (dzdz+1) - dydz * dzdy)
        det_2 = -dxdy * (dydx * (dzdz+1) - dydz * dzdx)
        det_3 = dxdz * (dydx * dzdy - (dydy+1) * dzdx)
        J = det_1 + det_2 + det_3
        
        # compute the trace of the right Cauchy-Green tensor
        tr_1 = (dxdx+1)**2 + dxdy**2 + dxdz**2
        tr_2 = dydx**2 + (dydy+1)**2 + dydz**2
        tr_3 = dzdx**2 + dzdy**2 +(dzdz+1)**2
        Tr = tr_1 + tr_2 + tr_3
        
        # adding a term based on gradient to avoid divergence
        grad = dx**2 + dy**2 + dz **2

        # compute the determinant of the transformation's Jacobian to check if the transformation is invertible
        det_J = J.flatten()
        neg = len(det_J[det_J<0])
        print("percent neg jac :", 100*neg/len(det_J))
        
        
        det2 = jacobian_determinant_vxm(y_pred[0,...].detach().cpu().numpy())
        det2 = det2.flatten()
        # print("percent neg jac 2:", 100*len(det2[det2<0])/len(det2))
        
        
        if self.mode == "exp":
            stretch = Tr*torch.exp(-J+1)-3
            
        if self.mode == "original":
            J[J<=0]=1e-6
            stretch = Tr*J**(-2./3.)-3
            # stretch -= stretch.min()
            
        vol_change = (J-1)**2 #torch.exp(-J+1)+J-2
        
        # compute the Neo-Hookean energy
        mu = 1
        lambda_ = mu * self.ratio
        mu = mu / (mu+lambda_)
        lambda_ = lambda_ / (mu+lambda_)
        
        U = (mu/2) * stretch + (lambda_/2) * vol_change
        
        # save the first and second term of the energy, which represent the amount of deformation and the volume change, respectively
        np.save("strech_c.npy", stretch.detach().cpu().numpy())
        np.save("volumechange_c.npy", vol_change.detach().cpu().numpy())
        
        
        # return the loss and the first and second term of the energy, if return_maps==True
        if self.return_maps == True:
            return torch.sum(U), stretch.detach().cpu().numpy(), vol_change.detach().cpu().numpy()
        else:
            return torch.mean(U)
        
class local_grad_neo_hookean2D(torch.nn.Module):
    def __init__(self, mu=0.2, lambda_=1., return_maps=False, window = 3, mode = "exp"):
        super(local_grad_neo_hookean2D, self).__init__()
        self.mu = mu
        self.lambda_ = lambda_
        self.return_maps = return_maps
        self.window = window
        self.mode = mode
        
    def forward(self, y_pred, y_true):
        
        # Define the window size for the convolution
        win = [self.window] * 2
        
        pos_idx = np.where(y_true.detach().cpu().numpy() > 0)
        
        # Compute the padding size, stride and output shape
        pad_no = math.floor(win[0]/2)
        stride = (1,1)
        padding = (pad_no, pad_no)

        # compute the gradient along x,y,z
        dy = torch.abs(y_pred[:, :, 1:, :-1] - y_pred[:, :, :-1, :-1]) 
        dx = torch.abs(y_pred[:, :, :-1, 1:] - y_pred[:, :, :-1, :-1])
        
  
        # Create a filter that will be used to convolve the deformation field
        sum_filt = torch.ones([1,1, *win]).to("cuda")
        sum_filt = sum_filt/(self.window**2) # divide by window^3 to get the average over the window
                
        # convolve the gradients to get the partial derivatives of the deformation fields
        dxdx = torch.nn.functional.conv2d(dx[:,1:2,...], sum_filt, stride=stride, padding=padding) # dx/dX
        dxdy = torch.nn.functional.conv2d(dy[:,1:2,...], sum_filt, stride=stride, padding=padding) # dx/dY
        dydx = torch.nn.functional.conv2d(dx[:,0:1,...], sum_filt, stride=stride, padding=padding) # dy/dX
        dydy = torch.nn.functional.conv2d(dy[:,0:1,...], sum_filt, stride=stride, padding=padding) # dy/dY
        
        # compute the determinant of the deformation gradient tensor
        det_1 = (dxdx+1) * (dydy+1)
        det_2 = -dxdy * dydx 
        J = det_1 + det_2 
        
        # compute the trace of the right Cauchy-Green tensor
        tr_1 = (dxdx+1)**2 + dxdy**2
        tr_2 = dydx**2 + (dydy+1)**2
        Tr = tr_1 + tr_2
        

        # compute the determinant of the transformation's Jacobian to check if the transformation is invertible
        # det_J, Tr = jacobian_determinant_vxm(y_pred.detach().cpu().numpy())
        det_J = J.flatten()
        neg = len(det_J[det_J<0])
        print("percent neg jac :", 100*neg/len(det_J))
        
        
        # convolve det_F and TrC to get their average value over window for each voxel        
        # Ic =  torch.nn.functional.conv3d(Ic, sum_filt, stride=stride, padding=padding)
        # J =  torch.nn.functional.conv3d(J, sum_filt, stride=stride, padding=padding)
        # vol_change =  torch.nn.functional.conv3d(vol_change, sum_filt, stride=stride, padding=padding)
        
        if self.mode == "exp":
            stretch = (self.mu/2)*(Tr*torch.exp(-J+1)-3)
            
        if self.mode == "original":
            J[J<=0]=1e-6
            stretch = (self.mu/2)*(Tr*J**(-2./3.)-3) 
            stretch -= stretch.min()
            
        vol_change = (self.lambda_/2)*(J-1)**2 #torch.exp(-J+1)+J-2
        # vol_change -= vol_change.min()
        
        # compute the Neo-Hookean energy
        U = (self.mu/2) * stretch + (self.lambda_/2) * vol_change
        
        # save the first and second term of the energy, which represent the amount of deformation and the volume change, respectively
        np.save("strech_c.npy", stretch.detach().cpu().numpy())
        np.save("volumechange_c.npy", vol_change.detach().cpu().numpy())
        
        
        # return the loss and the first and second term of the energy, if return_maps==True
        if self.return_maps == True:
            return torch.sum(U), stretch.detach().cpu().numpy(), vol_change.detach().cpu().numpy()
        else:
            return torch.mean(U)
        
class local_grad_neo_hookean_with_maps(torch.nn.Module):
    def __init__(self, mu=0.2, lambda_=1., return_maps=False, window = 3):
        super(local_grad_neo_hookean_with_maps, self).__init__()
        self.mu = mu
        self.lambda_ = lambda_
        self.return_maps = return_maps
        self.window = window
        
    def forward(self, y_pred, mu, lambda_):
        
        # Define the window size for the convolution
        win = [self.window] * 3
        
        # Compute the padding size, stride and output shape
        pad_no = math.floor(win[0]/2)
        stride = (1,1,1)
        padding = (pad_no, pad_no, pad_no)

        # compute the gradient along x,y,z
        dy = torch.abs(y_pred[:, :, 1:, :-1, :-1] - y_pred[:, :, :-1, :-1, :-1]) 
        dx = torch.abs(y_pred[:, :, :-1, 1:, :-1] - y_pred[:, :, :-1, :-1, :-1]) 
        dz = torch.abs(y_pred[:, :, :-1, :-1, 1:] - y_pred[:, :, :-1, :-1, :-1]) 
        
        # Create a filter that will be used to convolve the deformation field
        sum_filt = torch.ones([1,1, *win]).to("cuda")
        sum_filt = sum_filt/(self.window**3) # divide by window^3 to get the average over the window
                
        # convolve the gradients to get the partial derivatives of the deformation fields
        dxdx = torch.nn.functional.conv3d(dx[:,1:2,...], sum_filt, stride=stride, padding=padding) # dx/dX
        dxdy = torch.nn.functional.conv3d(dy[:,1:2,...], sum_filt, stride=stride, padding=padding) # dx/dY
        dxdz = torch.nn.functional.conv3d(dz[:,1:2,...], sum_filt, stride=stride, padding=padding) # dx/dZ
        dydx = torch.nn.functional.conv3d(dx[:,0:1,...], sum_filt, stride=stride, padding=padding) # dy/dX
        dydy = torch.nn.functional.conv3d(dy[:,0:1,...], sum_filt, stride=stride, padding=padding) # dy/dY
        dydz = torch.nn.functional.conv3d(dz[:,0:1,...], sum_filt, stride=stride, padding=padding) # dy/dZ
        dzdx = torch.nn.functional.conv3d(dx[:,2:3,...], sum_filt, stride=stride, padding=padding) # dz/dX
        dzdy = torch.nn.functional.conv3d(dy[:,2:3,...], sum_filt, stride=stride, padding=padding) # dz/dY
        dzdz = torch.nn.functional.conv3d(dz[:,2:3,...], sum_filt, stride=stride, padding=padding) # dz/dZ
        
        # compute the determinant of the deformation gradient tensor
        det_1 = (dxdx+1) * ((dydy+1) * (dzdz+1) - dydz * dzdy)
        det_2 = -dxdy * (dydx * (dzdz+1) - dydz * dzdx)
        det_3 = dxdz * (dydx * dzdy - (dydy+1) * dzdx)
        J = det_1 + det_2 + det_3
        
        # compute the trace of the right Cauchy-Green tensor
        tr_1 = (dxdx+1)**2 + dxdy**2 + dxdz**2
        tr_2 = dydx**2 + (dydy+1)**2 + dydz**2
        tr_3 = dzdx**2 + dzdy**2 +(dzdz+1)**2
        Tr = tr_1 + tr_2 + tr_3
        
        # adding a term based on gradient to avoid divergence
        grad = dx**2 + dy**2 + dz **2

        # compute the determinant of the transformation's Jacobian to check if the transformation is invertible
        # det_J, Tr = jacobian_determinant_vxm(y_pred.detach().cpu().numpy())
        det_J = J.flatten()
        neg = len(det_J[det_J<0])
        print("percent neg jac :", 100*neg/len(det_J))
        
        # disp = np.zeros((3,128,128,128))
        
        det2 = jacobian_determinant_vxm(y_pred[0,...].detach().cpu().numpy())
        det2 = det2.flatten()
        print("percent neg jac 2:", 100*len(det2[det2<0])/len(det2))
        
        # convolve det_F and TrC to get their average value over window for each voxel        
        # Ic =  torch.nn.functional.conv3d(Ic, sum_filt, stride=stride, padding=padding)
        # J =  torch.nn.functional.conv3d(J, sum_filt, stride=stride, padding=padding)
        # vol_change =  torch.nn.functional.conv3d(vol_change, sum_filt, stride=stride, padding=padding)
        
        J[J<=0]=1e-6
        
        stretch = (Tr*torch.exp(-J+1)-3)
        vol_change = (J-1)**2 #torch.exp(-J+1)+J-2
        
        mu = 0.167*mu/mu.mean()
        lambda_ = 0.833*lambda_/lambda_.mean()
        
        
        U = (mu/2) * stretch + (lambda_/2) * vol_change
        
        # save the first and second term of the energy, which represent the amount of deformation and the volume change, respectively
        np.save("strech_c.npy", (Tr-3).detach().cpu().numpy())
        np.save("volumechange_c.npy", vol_change.detach().cpu().numpy())
        
        # return the loss and the first and second term of the energy, if return_maps==True
        if self.return_maps == True:
            return torch.mean(U), stretch.detach().cpu().numpy(), vol_change.detach().cpu().numpy()
        else:
            return torch.mean(U)
        
class jacobian_map(torch.nn.Module):
    def __init__(self, window = 3):
        super(jacobian_map, self).__init__()
        self.window = window
    def forward(self, y_pred):
        
        # Define the window size for the convolution
        win = [self.window] * 3
        pad_no = math.floor(win[0]/2)
        stride = (1,1,1)
        padding = (pad_no, pad_no, pad_no)
        
        # Create a filter that will be used to convolve the deformation field
        sum_filt = torch.ones([1,1, *win]).to("cuda")
        sum_filt = sum_filt/(self.window**3) # divide by window^3 to get the average over the window
        
        dy = torch.abs(y_pred[:, :, 1:, :-1, :-1] - y_pred[:, :, :-1, :-1, :-1]) 
        dx = torch.abs(y_pred[:, :, :-1, 1:, :-1] - y_pred[:, :, :-1, :-1, :-1]) 
        dz = torch.abs(y_pred[:, :, :-1, :-1, 1:] - y_pred[:, :, :-1, :-1, :-1])
        
        # convolve the gradients to get the partial derivatives of the deformation fields
        dxdx = torch.nn.functional.conv3d(dx[:,1:2,...], sum_filt, stride=stride, padding=padding) # dx/dX
        dxdy = torch.nn.functional.conv3d(dy[:,1:2,...], sum_filt, stride=stride, padding=padding) # dx/dY
        dxdz = torch.nn.functional.conv3d(dz[:,1:2,...], sum_filt, stride=stride, padding=padding) # dx/dZ
        dydx = torch.nn.functional.conv3d(dx[:,0:1,...], sum_filt, stride=stride, padding=padding) # dy/dX
        dydy = torch.nn.functional.conv3d(dy[:,0:1,...], sum_filt, stride=stride, padding=padding) # dy/dY
        dydz = torch.nn.functional.conv3d(dz[:,0:1,...], sum_filt, stride=stride, padding=padding) # dy/dZ
        dzdx = torch.nn.functional.conv3d(dx[:,2:3,...], sum_filt, stride=stride, padding=padding) # dz/dX
        dzdy = torch.nn.functional.conv3d(dy[:,2:3,...], sum_filt, stride=stride, padding=padding) # dz/dY
        dzdz = torch.nn.functional.conv3d(dz[:,2:3,...], sum_filt, stride=stride, padding=padding) # dz/dZ


        
        # compute the determinant of the deformation gradient tensor
        det_1 = (dxdx+1) * ((dydy+1) * (dzdz+1) - dydz * dzdy)
        det_2 = -dxdy * (dydx * (dzdz+1) - dydz * dzdx)
        det_3 = dxdz * (dydx * dzdy - (dydy+1) * dzdx)
        J = det_1 + det_2 + det_3

        return np.squeeze(J.detach().cpu().numpy())