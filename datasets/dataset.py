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

        if not self.gt_dir.exists():
            raise FileNotFoundError(f"Ground Truth folder not found:\n{self.gt_dir}")

        if not self.noisy_dir.exists():
            raise FileNotFoundError(f"Noisy folder not found:\n{self.noisy_dir}")

        gt_files = sorted(self.gt_dir.glob("*.npy"))
        noisy_files = sorted(self.noisy_dir.glob("*.npy"))

        if len(gt_files) == 0:
            raise RuntimeError(f"No GT images found inside:\n{self.gt_dir}")

        if len(noisy_files) == 0:
            raise RuntimeError(f"No Noisy images found inside:\n{self.noisy_dir}")

        # Keep only paired files using filename stems.
        gt_map = {f.stem: f for f in gt_files}
        noisy_map = {f.stem: f for f in noisy_files}

        paired_names = sorted(gt_map.keys() & noisy_map.keys())

        if len(paired_names) == 0:
            raise RuntimeError(
                f"No paired files found between:\nGT: {self.gt_dir}\nNoisy: {self.noisy_dir}"
            )

        self.gt_files = [gt_map[name] for name in paired_names]
        self.noisy_files = [noisy_map[name] for name in paired_names]

        self.unpaired_gt = sorted(gt_map.keys() - noisy_map.keys())
        self.unpaired_noisy = sorted(noisy_map.keys() - gt_map.keys())

    def __len__(self) -> int:
        return len(self.gt_files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        gt = np.load(self.gt_files[index]).astype(np.float32)
        noisy = np.load(self.noisy_files[index]).astype(np.float32)

        if self.transform is not None:
            transformed = self.transform(image=noisy, mask=gt)
            noisy = transformed["image"]
            gt = transformed["mask"]
        else:
            noisy = torch.from_numpy(noisy).unsqueeze(0)
            gt = torch.from_numpy(gt).unsqueeze(0)

        if not isinstance(noisy, torch.Tensor):
            noisy = torch.from_numpy(noisy).unsqueeze(0)
        if not isinstance(gt, torch.Tensor):
            gt = torch.from_numpy(gt).unsqueeze(0)

        return {
            "noisy": noisy,
            "gt": gt,
            "filename": self.gt_files[index].stem,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  Samples : {len(self)}\n"
            f"  GT Dir  : {self.gt_dir}\n"
            f"  Noisy   : {self.noisy_dir}\n"
            f")"
        )