"""
DataLoader Utilities
====================

Creates train/validation datasets and DataLoaders.

Project:
    Semiconductor Image Restoration

Author:
    Yogeshwaran
"""

from pathlib import Path

import torch
from torch.utils.data import (
    DataLoader,
    Subset,
    random_split,
)

from datasets.dataset import SemiconductorDataset
from datasets.augmentations import (
    get_train_transforms,
    get_validation_transforms,
)


def create_dataloaders(
    gt_dir: str | Path,
    noisy_dir: str | Path,
    batch_size: int = 8,
    val_split: float = 0.1,
    num_workers: int = 0,
    seed: int = 42,
    debug: bool = False,
    debug_train_samples: int = 32,
    debug_val_samples: int = 8,
):
    """
    Create training and validation DataLoaders.

    Parameters
    ----------
    gt_dir : str | Path
        Directory containing ground-truth images.

    noisy_dir : str | Path
        Directory containing noisy low-resolution images.

    batch_size : int
        Number of samples per batch.

    val_split : float
        Fraction of the dataset used for validation.

    num_workers : int
        Number of DataLoader worker processes.

    seed : int
        Random seed used for reproducible splitting.

    debug : bool
        If True, limit the dataset to a small number of
        training and validation samples.

    debug_train_samples : int
        Number of training samples used in debug mode.

    debug_val_samples : int
        Number of validation samples used in debug mode.

    Returns
    -------
    tuple[DataLoader, DataLoader]
        Training and validation DataLoaders.
    """

    # --------------------------------------------------
    # Base dataset
    # --------------------------------------------------

    full_dataset = SemiconductorDataset(
        gt_dir=gt_dir,
        noisy_dir=noisy_dir,
    )

    total_size = len(full_dataset)

    if total_size == 0:
        raise ValueError(
            "Dataset is empty. "
            "Please check the GT and noisy image directories."
        )

    # --------------------------------------------------
    # Train / validation split
    # --------------------------------------------------

    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    if val_size == 0:
        raise ValueError(
            "Validation split produced zero validation samples."
        )

    generator = torch.Generator().manual_seed(seed)

    train_indices, val_indices = random_split(
        range(total_size),
        [train_size, val_size],
        generator=generator,
    )

    # --------------------------------------------------
    # Training dataset
    # --------------------------------------------------

    train_dataset = SemiconductorDataset(
        gt_dir=gt_dir,
        noisy_dir=noisy_dir,
        transform=get_train_transforms(),
    )

    train_dataset = Subset(
        train_dataset,
        train_indices.indices,
    )

    # --------------------------------------------------
    # Validation dataset
    # --------------------------------------------------

    val_dataset = SemiconductorDataset(
        gt_dir=gt_dir,
        noisy_dir=noisy_dir,
        transform=get_validation_transforms(),
    )

    val_dataset = Subset(
        val_dataset,
        val_indices.indices,
    )

    # --------------------------------------------------
    # Debug mode
    # --------------------------------------------------

    if debug:

        train_dataset = Subset(
            train_dataset,
            range(
                min(
                    debug_train_samples,
                    len(train_dataset),
                )
            ),
        )

        val_dataset = Subset(
            val_dataset,
            range(
                min(
                    debug_val_samples,
                    len(val_dataset),
                )
            ),
        )

    # --------------------------------------------------
    # DataLoader settings
    # --------------------------------------------------

    # Pin memory only provides a benefit when using CUDA.
    pin_memory = torch.cuda.is_available()

    # --------------------------------------------------
    # Training DataLoader
    # --------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # --------------------------------------------------
    # Validation DataLoader
    # --------------------------------------------------

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader