import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dataset import SemiconductorDataset

dataset = SemiconductorDataset(
    gt_dir="data/raw/train/GT",
    noisy_dir="data/raw/train/NoisyLR",
)

print(dataset)

sample = dataset[0]

print("\nSample Information")
print("-" * 40)

for key, value in sample.items():

    if hasattr(value, "shape"):
        print(f"{key:10} : {value.shape}")

    else:
        print(f"{key:10} : {value}")