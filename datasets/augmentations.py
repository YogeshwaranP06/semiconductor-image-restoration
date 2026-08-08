"""
Paired augmentations for LR-HR restoration.
"""

import random
import numpy as np

from datasets.transforms import Compose, ToTensor


class PairedAugmentation:

    def __call__(self, image, mask):

        # Horizontal Flip
        if random.random() < 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        # Vertical Flip
        if random.random() < 0.5:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()

        # Random Rotation
        k = random.randint(0, 3)

        image = np.rot90(image, k).copy()
        mask = np.rot90(mask, k).copy()

        return Compose([
            ToTensor(),
        ])(image, mask)


def get_train_transforms():

    return PairedAugmentation()


def get_validation_transforms():

    return Compose([
        ToTensor(),
    ])