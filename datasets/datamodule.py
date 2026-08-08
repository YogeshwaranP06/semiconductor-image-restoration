"""
DataLoader Utilities
====================

Creates train/validation datasets and DataLoaders.

Author:
    Yogeshwaran
"""

from torch.utils.data import DataLoader, random_split

from datasets.dataset import SemiconductorDataset
from datasets.augmentations import (
    get_train_transforms,
    get_validation_transforms,
)


def create_dataloaders(
    gt_dir: str,
    noisy_dir: str,
    batch_size: int = 8,
    val_split: float = 0.1,
    num_workers: int = 0,
    seed: int = 42,
    debug: bool = False,
    debug_train_samples: int = 32,
    debug_val_samples: int = 8,
):
    """
    Create train and validation DataLoaders.
    """

    # Full dataset without augmentation
    full_dataset = SemiconductorDataset(
        gt_dir=gt_dir,
        noisy_dir=noisy_dir,
    )

    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    generator = __import__("torch").Generator().manual_seed(seed)

    train_indices, val_indices = random_split(
        range(total_size),
        [train_size, val_size],
        generator=generator,
    )

    train_dataset = SemiconductorDataset(
        gt_dir=gt_dir,
        noisy_dir=noisy_dir,
        transform=get_train_transforms(),
    )

    val_dataset = SemiconductorDataset(
        gt_dir=gt_dir,
        noisy_dir=noisy_dir,
        transform=get_validation_transforms(),
    )

    train_dataset = __import__("torch").utils.data.Subset(
        train_dataset,
        train_indices.indices,
    )

    val_dataset = __import__("torch").utils.data.Subset(
        val_dataset,
        val_indices.indices,
    )
    if debug:
     train_dataset = __import__("torch").utils.data.Subset(
        train_dataset,
        range(min(debug_train_samples, len(train_dataset))),
    )

    val_dataset = __import__("torch").utils.data.Subset(
        val_dataset,
        range(min(debug_val_samples, len(val_dataset))),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader