from pathlib import Path
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys

# ============================================================
# PROJECT
# ============================================================

PROJECT = Path(__file__).resolve().parent

CHECKPOINT = (
    PROJECT
    / "demo_checkpoint"
    / "exp2_epoch50_best_model.pth"
)

CONFIG = PROJECT / "configs" / "default.yaml"

NOISY_DIR = (
    PROJECT
    / "data"
    / "raw"
    / "train"
    / "NoisyLR"
)

GT_DIR = (
    PROJECT
    / "data"
    / "raw"
    / "train"
    / "GT"
)

OUTPUT_DIR = PROJECT / "demo_output" / "train_comparisons"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# MODEL IMPORT
# ============================================================

sys.path.insert(0, str(PROJECT))

from models.factory import create_model


# ============================================================
# SETTINGS
# ============================================================

SAMPLES = [
    "000000",
    "000025",
    "000050",
    "000100",
    "000200",
]


# ============================================================
# LOAD CONFIG
# ============================================================

with open(CONFIG, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("TRAIN IMAGE RESTORATION VISUALIZATION")
print("=" * 70)

print("Device :", device)

if device.type == "cuda":
    print("GPU    :", torch.cuda.get_device_name(0))


# ============================================================
# LOAD MODEL
# ============================================================

model = create_model(config).to(device)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Checkpoint epoch :", checkpoint["epoch"])
print("Checkpoint loss  :", checkpoint["loss"])

print(
    "Parameters       :",
    sum(p.numel() for p in model.parameters())
)


# ============================================================
# PROCESS SAMPLES
# ============================================================

for sample_id in SAMPLES:

    noisy_path = NOISY_DIR / f"{sample_id}.npy"
    gt_path = GT_DIR / f"{sample_id}.npy"

    if not noisy_path.exists():
        print(f"SKIP {sample_id}: NoisyLR not found")
        continue

    if not gt_path.exists():
        print(f"SKIP {sample_id}: GT not found")
        continue

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    noisy = np.load(noisy_path).astype(np.float32)
    gt = np.load(gt_path).astype(np.float32)

    # --------------------------------------------------------
    # Model inference
    # --------------------------------------------------------

    x = (
        torch.from_numpy(noisy)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        prediction = model(x)

    prediction = (
        prediction
        .squeeze()
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    # --------------------------------------------------------
    # Bicubic baseline
    # --------------------------------------------------------

    bicubic_tensor = F.interpolate(
        torch.from_numpy(noisy)
        .unsqueeze(0)
        .unsqueeze(0),
        size=(256, 256),
        mode="bicubic",
        align_corners=False
    )

    bicubic = (
        bicubic_tensor
        .squeeze()
        .numpy()
        .astype(np.float32)
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    assert noisy.shape == (128, 128)
    assert gt.shape == (256, 256)
    assert prediction.shape == (256, 256)

    assert np.isfinite(prediction).all()

    # --------------------------------------------------------
    # Common display range
    # --------------------------------------------------------

    all_values = np.concatenate([
        gt.ravel(),
        bicubic.ravel(),
        prediction.ravel()
    ])

    vmin, vmax = np.percentile(
        all_values,
        [1, 99]
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4.5)
    )

    axes[0].imshow(
        noisy,
        cmap="gray"
    )
    axes[0].set_title(
        f"Degraded Input\n{sample_id} — 128×128"
    )
    axes[0].axis("off")

    axes[1].imshow(
        bicubic,
        cmap="gray",
        vmin=vmin,
        vmax=vmax
    )
    axes[1].set_title(
        "Bicubic\n256×256"
    )
    axes[1].axis("off")

    axes[2].imshow(
        prediction,
        cmap="gray",
        vmin=vmin,
        vmax=vmax
    )
    axes[2].set_title(
        "SRDnCNN Output\n256×256"
    )
    axes[2].axis("off")

    axes[3].imshow(
        gt,
        cmap="gray",
        vmin=vmin,
        vmax=vmax
    )
    axes[3].set_title(
        "Ground Truth\n256×256"
    )
    axes[3].axis("off")

    fig.suptitle(
        f"Semiconductor Image Restoration — Sample {sample_id}",
        fontsize=16
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"{sample_id}_train_comparison.png"
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.show()

    plt.close(fig)

    print()
    print(f"Sample {sample_id}")
    print("  Input      :", noisy.shape)
    print("  Prediction :", prediction.shape)
    print("  GT         :", gt.shape)
    print("  Saved      :", output_path)


print()
print("=" * 70)
print("TRAIN VISUALIZATION COMPLETE")
print("=" * 70)
print("Output directory:")
print(OUTPUT_DIR)