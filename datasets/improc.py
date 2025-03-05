"""Image processing functions designed to work with OpenSRH datasets.

Copyright (c) 2022 University of Michigan. All rights reserved.
Licensed under the MIT License. See LICENSE for license information.
"""

from typing import Optional, List, Tuple, Dict
from functools import partial

import random
import tifffile
import numpy as np

import torch
from torch.nn import ModuleList
from torchvision.transforms import (
    Normalize, RandomApply, Compose, RandomHorizontalFlip, RandomVerticalFlip,
    Resize, RandAugment, RandomErasing, RandomAutocontrast, Grayscale,
    RandomSolarize, ColorJitter, RandomAdjustSharpness, GaussianBlur,
    RandomAffine, RandomResizedCrop)

from torchvision import transforms
import logging

log = logging.getLogger(__name__)


class GetThirdChannel(torch.nn.Module):
    """Computes the third channel of SRH image

    Compute the third channel of SRH images by subtracting CH3 and CH2. The
    channel difference is added to the subtracted_base.

    """

    def __init__(self, subtracted_base: float = 5000 / 65536.0):
        super().__init__()
        self.subtracted_base = subtracted_base

    def __call__(self, two_channel_image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            two_channel_image: a 2 channel np array in the shape H * W * 2
            subtracted_base: an integer to be added to (CH3 - CH2)

        Returns:
            A 3 channel np array in the shape H * W * 3
        """
        ch2 = two_channel_image[0, :, :]
        ch3 = two_channel_image[1, :, :]
        ch1 = ch3 - ch2 + self.subtracted_base

        return torch.stack((ch1, ch2, ch3), dim=0)

    def __repr__(self):
        return self.__class__.__name__ + f"(subtracted_base={self.subtracted_base})"


class MinMaxChop(torch.nn.Module):
    """Clamps the images to float (0,1) range."""

    def __init__(self, min_val: float = 0.0, max_val: float = 1.0):
        super().__init__()
        self.min_ = min_val
        self.max_ = max_val

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return image.clamp(self.min_, self.max_)

    def __repr__(self):
        return self.__class__.__name__ + f"(min_val={self.min_}, max_val={self.max_})"


class GaussianNoise(torch.nn.Module):
    """Adds guassian noise to images."""

    def __init__(self, min_var: float = 0.01, max_var: float = 0.1):
        super().__init__()
        self.min_var = min_var
        self.max_var = max_var

    def __call__(self, tensor):
        var = random.uniform(self.min_var, self.max_var)
        noisy = tensor + torch.randn(tensor.size()) * var
        noisy = torch.clamp(noisy, min=0., max=1.)
        return noisy

    def __repr__(self):
        return self.__class__.__name__ + f"(min_var={self.min_var}, max_var={self.max_var})"


def process_read_im(imp: str) -> torch.Tensor:
    """Read in two channel image

    Args:
        imp: a string that is the path to the tiff image

    Returns:
        A 2 channel torch Tensor in the shape 2 * H * W
    """
    # reference: https://github.com/pytorch/vision/blob/49468279d9070a5631b6e0198ee562c00ecedb10/torchvision/transforms/functional.py#L133

    return torch.from_numpy(tifffile.imread(imp).astype(
        np.float32)).contiguous()


def get_srh_base_aug() -> List:
    """Base processing augmentations for all SRH images"""
    u16_min = (0, 0)
    u16_max = (65536, 65536)  # 2^16
    return [Normalize(u16_min, u16_max), GetThirdChannel(), MinMaxChop()]


def get_srh_vit_base_aug() -> List:
    """Base processing augmentations for all SRH images, with resize to 224"""
    u16_min = (0, 0)
    u16_max = (65536, 65536)  # 2^16
    return [
        Normalize(u16_min, u16_max),
        GetThirdChannel(),
        MinMaxChop(),
        Resize((224, 224))
    ]



def get_dynacl_aug_v1(strength):
    u16_min = (0, 0)
    u16_max = (65536, 65536)  # 2^16

    # Base augmentations
    strong_aug = [Normalize(u16_min, u16_max), GetThirdChannel(), MinMaxChop()]

    # Append transformations to the list
    strong_aug.append(transforms.RandomResizedCrop((300,300) , scale=(1.0 - 0.9 * strength, 1.0), interpolation=3))
    strong_aug.append(transforms.RandomHorizontalFlip())
    strong_aug.append(transforms.RandomApply([transforms.ColorJitter(0.4 * strength, 0.4 * strength, 0.4 * strength, 0.1 * strength)], p=0.8 * strength))
    strong_aug.append(transforms.RandomGrayscale(p=0.2 * strength))

    log.info(f"Fixed probability augmentations:")
    log.info(f"RandomHorizontalFlip with probability {0.5}")

    log.info(f"Variable probability augmentations and Varying augmentation strength:")
    log.info(f"ColorJitter with brightness: {0.4 * strength}, contrast: {0.4 * strength}, saturation: {0.4 * strength}, hue: {0.1 * strength},"
        f" RandomApply with probability {0.8 * strength}")
    log.info(f"RandomGrayscale with probability {0.2 * strength}")
    log.info(f"RandomResizedCrop with size: 300, 300, scale: {1.0 - 0.9 * strength}, 1.0, interpolation: 3")


    return strong_aug

def get_ft_aug_v1():
    u16_min = (0, 0)
    u16_max = (65536, 65536)  # 2^16

    # Base augmentations
    strong_aug = [Normalize(u16_min, u16_max), GetThirdChannel(), MinMaxChop()]

    # Append transformations to the list
    strong_aug.append(transforms.Resize((300,300)))
    strong_aug.append(transforms.RandomHorizontalFlip())



    return strong_aug




def get_strong_aug(augs, rand_prob) -> List:
    """Strong augmentations for OpenSRH training"""
    rand_apply = lambda which, **kwargs: RandomApply(
        ModuleList([which(**kwargs)]), p=rand_prob)

    callable_dict = {
        "resize": Resize,
        "random_horiz_flip": partial(RandomHorizontalFlip, p=rand_prob),
        "random_vert_flip": partial(RandomVerticalFlip, p=rand_prob),
        "gaussian_noise": partial(rand_apply, which=GaussianNoise),
        "color_jitter": partial(rand_apply, which=ColorJitter),
        "random_autocontrast": partial(RandomAutocontrast, p=rand_prob),
        "random_solarize": partial(RandomSolarize, p=rand_prob),
        "random_sharpness": partial(RandomAdjustSharpness, p=rand_prob),
        "drop_color": partial(rand_apply, which=Grayscale),
        "gaussian_blur": partial(rand_apply, GaussianBlur),
        "random_erasing": partial(RandomErasing, p=rand_prob),
        "random_affine": partial(rand_apply, RandomAffine),
        "random_resized_crop": partial(rand_apply, RandomResizedCrop)
    }

    return [callable_dict[a["which"]](**a["params"]) for a in augs]


def get_srh_aug_list(augs, rand_prob=0.5, dyanamic_aug=False, strength=1.0) -> List:
    """Combine base and strong augmentations for OpenSRH training"""
    if dyanamic_aug:
        return get_dynamic_augs(rand_prob, strength)
    else:
        return get_srh_base_aug() + get_strong_aug(augs, rand_prob)


# def get_dynamic_augs(augs, rand_prob, strength) -> List:
#     """Strong augmentations for OpenSRH training"""
#
#     u16_min = (0, 0)
#     u16_max = (65536, 65536)  # 2^16
#
#     base_aug = [Normalize(u16_min, u16_max), GetThirdChannel(), MinMaxChop()]
#
#     random_horiz_flip = RandomHorizontalFlip(p=rand_prob)
#     random_vert_flip = RandomVerticalFlip(p=rand_prob)
#     gaussian_noise = transforms.RandomApply([GaussianNoise(min_var=0.01 * strength, max_var=0.1 * strength)], p=rand_prob * strength)
#     color_jitter = transforms.RandomApply([transforms.ColorJitter(0.4 * strength, 0.4 * strength, 0.4 * strength, 0.1 * strength)],
#                                           p=rand_prob * strength)
#     autocontrast = RandomAutocontrast(p=rand_prob * strength)
#     solarize = RandomSolarize(threshold=0.2, p=rand_prob * strength)
#     sharpness = RandomAdjustSharpness(sharpness_factor=2, p=rand_prob * strength)
#     gaussian_blur = transforms.RandomApply([GaussianBlur(kernel_size=(5, 5), sigma=(1.0 * strength, 1.0 * strength))],
#                                            p=rand_prob * strength)
#     random_affine = transforms.RandomApply(
#         [RandomAffine(degrees=(-10.0 * strength, 10.0 * strength), translate=(0.1 * strength, 0.3 * strength))], p=rand_prob * strength)
#     random_resized_crop = transforms.RandomApply([RandomResizedCrop(size=(300, 300), scale=(0.08, 1.0), ratio=(0.75, 1.333))],
#                                                  p=rand_prob * strength)
#     random_erasing = RandomErasing(p=rand_prob * strength, scale=(0.02 * strength, 0.33 * strength), ratio=(0.3 * strength, 3.3 * strength),
#                                    value=0, inplace=False)
#
#     strong_aug = [random_horiz_flip, random_vert_flip, gaussian_noise, color_jitter, autocontrast, solarize, sharpness, gaussian_blur,
#                   random_affine, random_resized_crop, random_erasing]
#
#     return base_aug + strong_aug


def get_dynamic_augs(rand_prob, strength):
    """Strong augmentations for OpenSRH training"""
    # Initialize list of transformations

    u16_min = (0, 0)
    u16_max = (65536, 65536)  # 2^16

    # Base augmentations
    strong_aug = [Normalize(u16_min, u16_max), GetThirdChannel(), MinMaxChop()]

    # Append transformations to the list
    strong_aug.append(transforms.RandomHorizontalFlip(p=rand_prob))
    strong_aug.append(transforms.RandomVerticalFlip(p=rand_prob))

    strong_aug.append(transforms.RandomApply([GaussianNoise(min_var=0.01 * strength, max_var=0.1 * strength)], p=rand_prob))
    strong_aug.append(transforms.RandomApply([transforms.ColorJitter(0.4 * strength, 0.4 * strength, 0.4 * strength, 0.1 * strength)]
                                             , p=rand_prob * strength))
    strong_aug.append(transforms.RandomAutocontrast(p=rand_prob))
    strong_aug.append(transforms.RandomSolarize(threshold=0.2 + 0.8 * (1 - strength), p=rand_prob * strength))
    strong_aug.append(transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=rand_prob))
    strong_aug.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(1.0 * strength, 1.0 * strength))],
                               p=rand_prob * strength))
    strong_aug.append(transforms.RandomErasing(p=rand_prob * strength, scale=(0.02 * strength, 0.33 * strength),
                                               ratio=(0.3, 3.3), value=0, inplace=False))
    strong_aug.append(transforms.RandomApply(
        [transforms.RandomAffine(degrees=(-10.0 * strength, 10.0 * strength),
                                 translate=(0.1 * strength, 0.3 * strength))], p=rand_prob * strength))
    strong_aug.append(
        transforms.RandomApply([transforms.RandomResizedCrop(size=(300, 300), scale=(1 - strength * 0.92, 1.0), ratio=(0.75, 1.333))],
                               p=rand_prob))

    log.info(f"Fixed probability augmentations:")
    log.info(f"RandomHorizontalFlip with probability {rand_prob}")
    log.info(f"RandomVerticalFlip with probability {rand_prob}")
    log.info(f"GaussianNoise with min_var: {0.01 * strength}, max_var: {0.1 * strength}, RandomApply with probability {rand_prob}")
    log.info(f"RandomAutocontrast with probability {rand_prob}")
    log.info("Fixed probability augmentations and Varying augmentation strength:")
    log.info(f"RandomAdjustSharpness with sharpness_factor: {2.0 * strength}, with probability {rand_prob}")
    log.info(
        f"RandomResizedCrop with size: 300, 300, scale: {1 - strength * 0.92}, {1.0}, ratio: 0.75, 1.333, RandomApply with probability {rand_prob}")

    log.info(f"Variable probability augmentations and Varying augmentation strength:")
    log.info(
        f"ColorJitter with brightness: {0.4 * strength}, contrast: {0.4 * strength}, saturation: {0.4 * strength}, hue: {0.1 * strength},"
        f" RandomApply with probability {rand_prob * strength}")
    log.info(f"Solarize threshold: {0.2 + 0.8 * (1 - strength)}, with probability {rand_prob * strength}")
    log.info(f"Gaussian blur sigma: {1.0 * strength}, {1.0 * strength}, RandomApply with probability {rand_prob * strength}")
    log.info(
        f"RandomAffine with probability {rand_prob * strength}, Affine degrees: {-10.0 * strength}, {10.0 * strength}, Affine translate: {0.1 * strength}, {0.3 * strength}")
    log.info(
        f"RandomErasing with probability {rand_prob * strength}, Erasing scale: {0.02 * strength}, {0.33 * strength}, Erasing ratio: {0.3 * strength}, {3.3 * strength}")

    return strong_aug
