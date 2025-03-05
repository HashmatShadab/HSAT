import torch.nn as nn
import torch
import torch.nn.functional as F
import csv
import torchvision.transforms as transforms
import os
import torchvision

def plot_grid(w):
    import matplotlib.pyplot as plt
    grid_img = torchvision.utils.make_grid(w)
    plt.imshow(grid_img.permute(1,2,0).cpu())
    plt.show()

def patchify(imgs, patch_size):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    p = patch_size
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
    return x

def unpatchify(x, patch_size):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    p = patch_size
    h = w = int(x.shape[1] ** .5)
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    return imgs


def mask_batch(images, mask_ratio, patch_size):
    patches = patchify(images, patch_size=patch_size)
    num_masked = int(mask_ratio * patches.shape[1])
    masked_indices = torch.rand(patches.shape[0], patches.shape[1]).topk(k=num_masked, dim=-1).indices
    masked_bool_mask = torch.zeros((patches.shape[0], patches.shape[1])).scatter_(-1, masked_indices, 1).bool()
    mask_patches = torch.zeros_like(patches)
    patches = torch.where(masked_bool_mask[..., None], mask_patches, patches)
    unpatches = unpatchify(patches, patch_size=patch_size)

    return unpatches

def get_mask_batch(images, mask_ratio, patch_size):
    mask = torch.ones_like(images)
    patches = patchify(mask, patch_size=patch_size)
    num_masked = int(mask_ratio * patches.shape[1])
    masked_indices = torch.rand(patches.shape[0], patches.shape[1]).topk(k=num_masked, dim=-1).indices
    masked_bool_mask = torch.zeros((patches.shape[0], patches.shape[1])).scatter_(-1, masked_indices, 1).bool()
    mask_patches = torch.zeros_like(patches)
    patches = torch.where(masked_bool_mask[..., None], mask_patches, patches)
    unpatches = unpatchify(patches, patch_size=patch_size)

    return unpatches


if __name__ == "__main__":
    images = torch.randn(4, 3, 224, 224)
    images = mask_batch_check(images, mask_ratio=0.50, patch_size=16)
    a=2