"""
Semiconductor Dataset Loader
============================

Loads paired Ground Truth (GT) and Noisy Low Resolution (NoisyLR)
images stored as NumPy (.npy) arrays.

Project:
    Semiconductor Image Restoration

Author:
    Yogeshwaran

License:
    MIT
"""

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset


class SemiconductorDataset(Dataset):
    """
    PyTorch Dataset for Semiconductor Image Restoration.

    Each sample contains:
        - Noisy Low Resolution image
        - Ground Truth High Resolution image
        - Filename

    Example:
        sample = dataset[0]

        sample["noisy"]
        sample["gt"]
        sample["filename"]
    """

    def __init__(
        self,
        gt_dir: str | Path,
        noisy_dir: str | Path,
        transform: Callable | None = None,
    ) -> None:

        self.gt_dir = Path(gt_dir)
        self.noisy_dir = Path(noisy_dir)

        self.transform = transform

        # --------------------------------------------------
        # Check folders
        # --------------------------------------------------

        if not self.gt_dir.exists():
            raise FileNotFoundError(
                f"Ground Truth folder not found:\n{self.gt_dir}"
            )

        if not self.noisy_dir.exists():
            raise FileNotFoundError(
                f"Noisy folder not found:\n{self.noisy_dir}"
            )

        # --------------------------------------------------
        # Collect files
        # --------------------------------------------------

        self.gt_files = sorted(self.gt_dir.glob("*.npy"))
        self.noisy_files = sorted(self.noisy_dir.glob("*.npy"))

        if len(self.gt_files) == 0:
            raise RuntimeError(
                f"No GT images found inside:\n{self.gt_dir}"
            )

        if len(self.noisy_files) == 0:
            raise RuntimeError(
                f"No Noisy images found inside:\n{self.noisy_dir}"
            )

        if len(self.gt_files) != len(self.noisy_files):
            raise RuntimeError(
                f"""
Dataset mismatch

GT Images     : {len(self.gt_files)}
Noisy Images  : {len(self.noisy_files)}
"""
            )

        # --------------------------------------------------
        # Verify filenames
        # --------------------------------------------------

        for gt, noisy in zip(self.gt_files, self.noisy_files):

            if gt.stem != noisy.stem:

                raise RuntimeError(
                    f"""
Filename mismatch detected

GT    : {gt.name}

Noisy : {noisy.name}
"""
                )

    def __len__(self) -> int:
        """Return total number of image pairs."""
        return len(self.gt_files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        """
        Load one training sample.

        Returns
        -------
        dict
            {
                "noisy": Tensor,
                "gt": Tensor,
                "filename": str
            }
        """

        gt = np.load(self.gt_files[index]).astype(np.float32)
        noisy = np.load(self.noisy_files[index]).astype(np.float32)

        if self.transform is not None:

            transformed = self.transform(
                image=noisy,
                mask=gt,
            )

            noisy = transformed["image"]
            gt = transformed["mask"]

        else:

            noisy = torch.from_numpy(noisy).unsqueeze(0)
            gt = torch.from_numpy(gt).unsqueeze(0)

        return {

            "noisy": noisy,

            "gt": gt,

            "filename": self.gt_files[index].stem,

        }

    def __repr__(self) -> str:
        """Pretty representation of the dataset."""

        return (
            f"{self.__class__.__name__}(\n"
            f"  Samples : {len(self)}\n"
            f"  GT Dir  : {self.gt_dir}\n"
            f"  Noisy   : {self.noisy_dir}\n"
            f")"
        )