"""
=========================================================
Dataset Analyzer
=========================================================

Project:
    Semiconductor Image Restoration

Purpose:
    - Verify dataset integrity
    - Compute dataset statistics
    - Check GT <-> Noisy pairs
    - Visualize sample images
    - Generate dataset report

Author:
    Yogeshwaran

=========================================================
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# Dataset Paths
# ==========================================================

TRAIN_GT = Path("data/raw/train/GT")
TRAIN_NOISY = Path("data/raw/train/NoisyLR")
TEST_NOISY = Path("data/raw/Test_NoisyLR/NoisyLR")

REPORT_PATH = Path("docs/dataset_report.md")


# ==========================================================
# Utility Functions
# ==========================================================

def get_npy_files(folder: Path):
    """Return sorted list of .npy files."""
    return sorted(folder.glob("*.npy"))


def folder_statistics(folder: Path):
    """
    Compute statistics over an entire folder.
    """

    files = get_npy_files(folder)

    if len(files) == 0:
        return None

    means = []
    stds = []

    global_min = float("inf")
    global_max = float("-inf")

    shapes = set()
    dtypes = set()

    for file in files:

        image = np.load(file)

        shapes.add(image.shape)
        dtypes.add(str(image.dtype))

        means.append(image.mean())
        stds.append(image.std())

        global_min = min(global_min, float(image.min()))
        global_max = max(global_max, float(image.max()))

    return {
        "count": len(files),
        "shape": shapes,
        "dtype": dtypes,
        "mean": np.mean(means),
        "std": np.mean(stds),
        "min": global_min,
        "max": global_max,
    }


def compare_pairs(gt_folder: Path, noisy_folder: Path):
    """
    Check whether GT and Noisy files match.
    """

    gt_files = {f.name for f in get_npy_files(gt_folder)}
    noisy_files = {f.name for f in get_npy_files(noisy_folder)}

    missing_gt = noisy_files - gt_files
    missing_noisy = gt_files - noisy_files

    return missing_gt, missing_noisy


def visualize_sample(index=0):
    """
    Display one GT/Noisy pair.
    """

    gt_files = get_npy_files(TRAIN_GT)
    noisy_files = get_npy_files(TRAIN_NOISY)

    gt = np.load(gt_files[index])
    noisy = np.load(noisy_files[index])

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(gt, cmap="gray")
    ax[0].set_title("Ground Truth")
    ax[0].axis("off")

    ax[1].imshow(noisy, cmap="gray")
    ax[1].set_title("Noisy LR")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()


def save_report(train_gt_stats, train_noisy_stats, test_stats,
                missing_gt, missing_noisy):

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:

        f.write("# Dataset Report\n\n")

        f.write("## Ground Truth\n")
        f.write(f"- Images: {train_gt_stats['count']}\n")
        f.write(f"- Shape: {train_gt_stats['shape']}\n")
        f.write(f"- Dtype: {train_gt_stats['dtype']}\n")
        f.write(f"- Min: {train_gt_stats['min']:.6f}\n")
        f.write(f"- Max: {train_gt_stats['max']:.6f}\n")
        f.write(f"- Mean: {train_gt_stats['mean']:.6f}\n")
        f.write(f"- Std: {train_gt_stats['std']:.6f}\n\n")

        f.write("## Training Noisy\n")
        f.write(f"- Images: {train_noisy_stats['count']}\n")
        f.write(f"- Shape: {train_noisy_stats['shape']}\n")
        f.write(f"- Dtype: {train_noisy_stats['dtype']}\n")
        f.write(f"- Min: {train_noisy_stats['min']:.6f}\n")
        f.write(f"- Max: {train_noisy_stats['max']:.6f}\n")
        f.write(f"- Mean: {train_noisy_stats['mean']:.6f}\n")
        f.write(f"- Std: {train_noisy_stats['std']:.6f}\n\n")

        f.write("## Test Noisy\n")
        f.write(f"- Images: {test_stats['count']}\n")
        f.write(f"- Shape: {test_stats['shape']}\n")
        f.write(f"- Dtype: {test_stats['dtype']}\n")
        f.write(f"- Min: {test_stats['min']:.6f}\n")
        f.write(f"- Max: {test_stats['max']:.6f}\n")
        f.write(f"- Mean: {test_stats['mean']:.6f}\n")
        f.write(f"- Std: {test_stats['std']:.6f}\n\n")

        f.write("## Pair Verification\n")

        if len(missing_gt) == 0 and len(missing_noisy) == 0:
            f.write("✅ All GT and Noisy pairs match.\n")
        else:
            f.write(f"Missing GT: {len(missing_gt)}\n")
            f.write(f"Missing Noisy: {len(missing_noisy)}\n")


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("SEMICONDUCTOR IMAGE RESTORATION DATASET ANALYZER")
    print("=" * 70)

    gt_stats = folder_statistics(TRAIN_GT)
    noisy_stats = folder_statistics(TRAIN_NOISY)
    test_stats = folder_statistics(TEST_NOISY)

    print("\nGround Truth")
    print(gt_stats)

    print("\nTraining Noisy")
    print(noisy_stats)

    print("\nTesting Noisy")
    print(test_stats)

    missing_gt, missing_noisy = compare_pairs(TRAIN_GT, TRAIN_NOISY)

    print("\nPair Verification")
    print("-----------------------------")

    if len(missing_gt) == 0 and len(missing_noisy) == 0:
        print("All GT and Noisy pairs match.")
    else:
        print("Missing GT :", len(missing_gt))
        print("Missing Noisy :", len(missing_noisy))

    visualize_sample(index=0)

    save_report(
        gt_stats,
        noisy_stats,
        test_stats,
        missing_gt,
        missing_noisy
    )

    print("\nDataset report saved to docs/dataset_report.md")