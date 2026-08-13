import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from models.factory import create_model


# ============================================================
# DEFAULT SUBMISSION ARTIFACTS
# ============================================================

DEFAULT_CONFIG = (
    PROJECT
    / "configs"
    / "default.yaml"
)

DEFAULT_CHECKPOINT = (
    PROJECT
    / "demo_checkpoint"
    / "exp2_epoch50_best_model.pth"
)


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Semiconductor Image Restoration inference "
        "using the Exp2 SRDnCNN model."
    )
)

parser.add_argument(
    "--input_dir",
    required=True,
    help="Directory containing degraded .npy images."
)

parser.add_argument(
    "--output_dir",
    required=True,
    help="Directory where restored .npy images will be saved."
)

parser.add_argument(
    "--config",
    default=str(DEFAULT_CONFIG),
    help="Path to model configuration YAML."
)

parser.add_argument(
    "--checkpoint",
    default=str(DEFAULT_CHECKPOINT),
    help="Path to trained model checkpoint."
)

args = parser.parse_args()


# ============================================================
# PATHS
# ============================================================

input_dir = Path(args.input_dir)
output_dir = Path(args.output_dir)
config_path = Path(args.config)
checkpoint_path = Path(args.checkpoint)


print("=" * 70)
print("SEMICONDUCTOR IMAGE RESTORATION — STANDALONE INFERENCE")
print("=" * 70)

print("Input directory :", input_dir)
print("Output directory:", output_dir)
print("Config          :", config_path)
print("Checkpoint      :", checkpoint_path)


# ============================================================
# VALIDATE PATHS
# ============================================================

if not input_dir.exists():
    raise FileNotFoundError(
        f"Input directory does not exist:\n{input_dir}"
    )

if not input_dir.is_dir():
    raise NotADirectoryError(
        f"Input path is not a directory:\n{input_dir}"
    )

if not config_path.exists():
    raise FileNotFoundError(
        f"Configuration file does not exist:\n{config_path}"
    )

if not checkpoint_path.exists():
    raise FileNotFoundError(
        f"Checkpoint does not exist:\n{checkpoint_path}"
    )

output_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FIND INPUT IMAGES
# ============================================================

input_files = sorted(
    input_dir.glob("*.npy")
)

if len(input_files) == 0:
    raise RuntimeError(
        f"No .npy files found in:\n{input_dir}"
    )

print()
print("Input images    :", len(input_files))


# ============================================================
# LOAD CONFIG
# ============================================================

with open(
    config_path,
    "r",
    encoding="utf-8"
) as f:
    config = yaml.safe_load(f)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device          :", device)

if device.type == "cuda":
    print(
        "GPU             :",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# CREATE MODEL
# ============================================================

model = create_model(config).to(device)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False
)

if "model_state_dict" not in checkpoint:
    raise RuntimeError(
        "Checkpoint does not contain model_state_dict."
    )

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print()
print("Checkpoint epoch:", checkpoint.get("epoch"))
print("Checkpoint loss :", checkpoint.get("loss"))
print(
    "Model parameters:",
    sum(p.numel() for p in model.parameters())
)


# ============================================================
# INFERENCE
# ============================================================

print()
print("-" * 70)
print("PROCESSING")
print("-" * 70)

processed = 0

with torch.no_grad():

    for input_file in input_files:

        image_id = input_file.stem

        arr = np.load(
            input_file
        ).astype(np.float32)

        if arr.shape != (128, 128):
            raise RuntimeError(
                f"{input_file.name}: "
                f"expected (128,128), got {arr.shape}"
            )

        x = (
            torch.from_numpy(arr)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

        prediction = model(x)

        output = (
            prediction
            .squeeze()
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        if output.shape != (256, 256):
            raise RuntimeError(
                f"{input_file.name}: "
                f"expected (256,256), got {output.shape}"
            )

        if not np.isfinite(output).all():
            raise RuntimeError(
                f"{input_file.name}: "
                "output contains NaN or Inf."
            )

        output_file = (
            output_dir
            / f"{image_id}.npy"
        )

        np.save(
            output_file,
            output
        )

        processed += 1

        if (
            processed <= 5
            or processed % 50 == 0
            or processed == len(input_files)
        ):
            print(
                f"Processed {processed}/{len(input_files)} "
                f"| {input_file.name}"
            )


# ============================================================
# FINAL VERIFICATION
# ============================================================

output_files = sorted(
    output_dir.glob("*.npy")
)

missing = sorted(
    {
        f.stem for f in input_files
    }
    -
    {
        f.stem for f in output_files
    }
)

extra = sorted(
    {
        f.stem for f in output_files
    }
    -
    {
        f.stem for f in input_files
    }
)

bad_shapes = 0
bad_finite = 0

for output_file in output_files:

    arr = np.load(
        output_file
    )

    if arr.shape != (256, 256):
        bad_shapes += 1

    if not np.isfinite(arr).all():
        bad_finite += 1


# ============================================================
# RESULT
# ============================================================

print()
print("=" * 70)
print("INFERENCE COMPLETE")
print("=" * 70)

print("Input files     :", len(input_files))
print("Output files    :", len(output_files))
print("Missing outputs :", len(missing))
print("Extra outputs   :", len(extra))
print("Bad shapes      :", bad_shapes)
print("NaN/Inf files   :", bad_finite)

if missing:
    print("Missing IDs     :", missing[:10])

if extra:
    print("Extra IDs       :", extra[:10])

if (
    len(output_files) != len(input_files)
    or missing
    or extra
    or bad_shapes
    or bad_finite
):
    raise RuntimeError(
        "FINAL INFERENCE VERIFICATION FAILED."
    )

print()
print("128×128 -> 256×256 : PASSED")
print("All outputs valid  : PASSED")
print("Standalone inference: PASSED")
print("=" * 70)
