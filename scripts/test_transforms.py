import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dataset import SemiconductorDataset
from datasets.transforms import Compose, ToTensor

transform = Compose([
    ToTensor(),
])

dataset = SemiconductorDataset(
    gt_dir="data/raw/train/GT",
    noisy_dir="data/raw/train/NoisyLR",
    transform=transform,
)

sample = dataset[0]

print("Noisy:", sample["noisy"].shape, sample["noisy"].dtype)
print("GT   :", sample["gt"].shape, sample["gt"].dtype)