"""
Image Transform Utilities
=========================

Shared preprocessing used during both
training and inference.

Author:
    Yogeshwaran
"""

from typing import Callable

import torch


class ToTensor:
    """
    Convert NumPy arrays to PyTorch tensors.
    """

    def __call__(self, image, mask):

        image = torch.from_numpy(image).float().unsqueeze(0)
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        return {
            "image": image,
            "mask": mask,
        }


class Normalize:
    """
    Normalize image using mean and std.

    By default this is disabled because
    the challenge dataset already contains
    normalized floating-point images.
    """

    def __init__(self, mean: float, std: float):

        self.mean = mean
        self.std = std

    def __call__(self, image, mask):

        image = (image - self.mean) / self.std

        return {
            "image": image,
            "mask": mask,
        }


class Compose:
    """
    Compose multiple transforms.
    """

    def __init__(self, transforms: list[Callable]):

        self.transforms = transforms

    def __call__(self, image, mask):

        sample = {
            "image": image,
            "mask": mask,
        }

        for transform in self.transforms:

            sample = transform(
                sample["image"],
                sample["mask"],
            )

        return sample