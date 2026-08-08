import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dataset import SemiconductorDataset
from datasets.augmentations import get_train_transforms

dataset = SemiconductorDataset(
    gt_dir="data/raw/train/GT",
    noisy_dir="data/raw/train/NoisyLR",
    transform=get_train_transforms(),
)

sample = dataset[0]

print("Noisy :", sample["noisy"].shape)
print("GT    :", sample["gt"].shape)
print("File  :", sample["filename"])