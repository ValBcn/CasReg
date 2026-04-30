import random
import pickle
import numpy as np
import torch

M = 2 ** 32 - 1


def init_fn(worker):
    seed = torch.LongTensor(1).random_().item()
    seed = (seed + worker) % M
    np.random.seed(seed)
    random.seed(seed)


def add_mask(x, mask, dim=1):
    mask = mask.unsqueeze(dim)
    shape = list(x.shape);
    shape[dim] += 21
    new_x = x.new(*shape).zero_()
    new_x = new_x.scatter_(dim, mask, 1.0)
    s = [slice(None)] * len(shape)
    s[dim] = slice(21, None)
    new_x[s] = x
    return new_x


def sample(x, size):
    # https://gist.github.com/yoavram/4134617
    i = random.sample(range(x.shape[0]), size)
    return torch.tensor(x[i], dtype=torch.int16)
    # x = np.random.permutation(x)
    # return torch.tensor(x[:size])


def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


_shape = (240, 240, 155)


def get_all_coords(stride):
    return torch.tensor(
        np.stack([v.reshape(-1) for v in
                  np.meshgrid(
                      *[stride // 2 + np.arange(0, s, stride) for s in _shape],
                      indexing='ij')],
                 -1), dtype=torch.int16)


_zero = torch.tensor([0])


def gen_feats():
    x, y, z = 240, 240, 155
    feats = np.stack(
        np.meshgrid(
            np.arange(x), np.arange(y), np.arange(z),
            indexing='ij'), -1).astype('float32')
    shape = np.array([x, y, z])
    feats -= shape / 2.0
    feats /= shape

    return feats


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


def write2csv(line, name):
    """
    Write a line to CSV file.

    Args:
        line: Line to write
        name: Filename (without extension)
    """
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')


def cpu(tensor):
    """
    Convert tensor to CPU numpy array.

    Args:
        tensor: Input tensor

    Returns:
        CPU numpy array
    """
    return np.squeeze(tensor.detach().cpu().numpy())


def norm_flow(flow):
    """
    Normalize flow field to [0, 1] range.

    Args:
        flow: Input flow field

    Returns:
        Normalized flow field
    """
    norm_flow = np.zeros(flow.shape)
    for i in range(flow.shape[0]):
        norm_flow[i,...] = (flow[i,...] - flow[i,...].min()) / (flow[i,...].max() - flow[i,...].min())
    return norm_flow