"""
Training utilities and helper functions.
"""

import sys
import os
import glob
import numpy as np
import torch
from natsort import natsorted


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    """
    Adjust learning rate using polynomial decay.

    Args:
        optimizer: PyTorch optimizer
        epoch: Current epoch
        MAX_EPOCHES: Maximum number of epochs
        INIT_LR: Initial learning rate
        power: Power for polynomial decay
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)


def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    """
    Save model checkpoint and manage checkpoint files.

    Args:
        state: State dictionary to save
        save_dir: Directory to save checkpoints
        filename: Checkpoint filename
        max_model_num: Maximum number of checkpoints to keep
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))


class Logger(object):
    """Logger that writes to both terminal and file."""

    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def make_suffix(supervised, contracted, nb_cascades, mode, lambda_):
    """
    Create filename suffix from training parameters.

    Args:
        supervised: Supervision flag
        contracted: Architecture flag
        nb_cascades: Number of cascades
        mode: Training mode
        lambda_: Lambda parameter

    Returns:
        Formatted suffix string
    """
    if supervised == 0:
        supornot = "unsup"
    if supervised == 1:
        supornot = "sup"

    if contracted == 0:
        archit = "original"
    if contracted == 1:
        archit = "contracted"

    file_suffix = str(supornot)+ "_" +  str(archit) + "_" + str(mode) + "_nb_casc_" + str(nb_cascades) + "_lambda_" + str(lambda_)
    return file_suffix


def lr_decay(lr, epoch, max_epoch):
    """
    Calculate learning rate decay.

    Args:
        lr: Initial learning rate
        epoch: Current epoch
        max_epoch: Maximum epochs

    Returns:
        Decayed learning rate
    """
    return lr*np.exp(-3.*float(epoch)/float(max_epoch))


def boolean_string(s):
    """
    Parse boolean string.

    Args:
        s: String to parse

    Returns:
        Boolean value

    Raises:
        ValueError: If string is not valid boolean
    """
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'