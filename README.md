AI-Based Restoration of Degraded Images for Semiconductor Inspection

> **Semiconductor India Hackathon 2026 — KLA Problem Statement**

An AI-based image restoration system for recovering high-resolution semiconductor inspection images from degraded low-resolution inputs affected by noise and spatial-resolution loss.

The project uses an **SRDnCNN** architecture and evaluates restoration using **PSNR, SSIM, and LPIPS**, with Bicubic comparison, runtime benchmarking, visual inspection, failure analysis, and reproducibility verification. The final selected model is **SRDnCNN + L1Loss**, restoring 128 × 128 NoisyLR images to 256 × 256 outputs.

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Problem Statement](#2-problem-statement)
- [3. Objective](#3-objective)
- [4. Dataset](#4-dataset)
- [5. Degradation Setting](#5-degradation-setting)
- [6. Proposed Pipeline](#6-proposed-pipeline)
- [7. Model Architecture](#7-model-architecture)
- [8. Experimental Design](#8-experimental-design)
- [9. Experiment 1 — SRDnCNN Baseline](#9-experiment-1--srdncnn-baseline)
- [10. Experiment 2 — L1Loss Ablation](#10-experiment-2--l1loss-ablation)
- [11. Experiment 3 — Synthetic Degradation Training](#11-experiment-3--synthetic-degradation-training)
- [12. Final Selected Model](#12-final-selected-model)
- [13. Evaluation Metrics](#13-evaluation-metrics)
- [14. Bicubic Baseline](#14-bicubic-baseline)
- [15. Failure Analysis](#15-failure-analysis)
- [16. Final Phase 9 Pipeline](#16-final-phase-9-pipeline)
- [17. Runtime Benchmark](#17-runtime-benchmark)
- [18. Final Test Inference](#18-final-test-inference)
- [19. Final Prediction Verification](#19-final-prediction-verification)
- [20. Visual Inspection](#20-visual-inspection)
- [21. Reproducibility Audit](#21-reproducibility-audit)
- [22. Repository Structure](#22-repository-structure)
- [23. Installation](#23-installation)
- [24. Running Inference](#24-running-inference)
- [25. Training](#25-training)
- [26. Checkpoint](#26-checkpoint)
- [27. Research Artifacts](#27-research-artifacts)
- [28. Limitations](#28-limitations)
- [29. Future Work](#29-future-work)
- [30. Final Project Status](#30-final-project-status)

## 1. Project Overview

Semiconductor inspection systems rely on high-quality microscopic images to identify and analyze fine structures. Inspection images can be degraded by speckle noise, Gaussian noise, downsampling, and reduced spatial resolution.

The restoration mapping is:

    128 × 128 degraded image
              │
              ▼
          SRDnCNN model
              │
              ▼
    256 × 256 restored image

## 2. Problem Statement

The KLA semiconductor image restoration problem provides degraded semiconductor inspection images and corresponding high-resolution training targets. The model must learn:

    Degraded NoisyLR image
              ↓
       Image restoration
              ↓
    High-resolution output

## 3. Objective

1. Remove degradation from semiconductor inspection images.
2. Recover the original spatial resolution.
3. Preserve fine structural details.
4. Improve reconstruction quality over Bicubic interpolation.
5. Produce valid 256 × 256 restored images.
6. Maintain stable inference performance.
7. Provide a reproducible inference pipeline.
8. Analyze failure cases and document limitations.

## 4. Dataset

The project uses paired degraded and ground-truth semiconductor inspection images.

| Dataset | Number of images |
|---|---:|
| Paired samples | 3,200 |
| Training split | 2,880 |
| Validation split | 320 |
| Test NoisyLR | 400 |

| Data | Shape | Dtype |
|---|---|---|
| Ground Truth | 256 × 256 | float32 |
| Training NoisyLR | 128 × 128 | float32 |
| Test NoisyLR | 128 × 128 | float32 |

Ground Truth statistics:

    Images : 3200
    Shape  : 256 × 256
    Dtype  : float32
    Min    : 0.000000
    Max    : 1.000000
    Mean   : 0.433528
    Std    : 0.187629

Training NoisyLR statistics:

    Images : 3200
    Shape  : 128 × 128
    Dtype  : float32
    Min    : -0.278563
    Max    : 2.158005
    Mean   : 0.433536
    Std    : 0.205795

Test NoisyLR statistics:

    Images : 400
    Shape  : 128 × 128
    Dtype  : float32
    Min    : -0.224881
    Max    : 2.158016
    Mean   : 0.442742
    Std    : 0.220269

All GT and NoisyLR pairs were verified to match. The dataset is excluded from Git. See [docs/dataset_report.md](docs/dataset_report.md).

## 5. Degradation Setting

The setting includes speckle noise, Gaussian noise, downsampling, reduced spatial resolution, and combinations of these degradations. Components may occur in different orders, and NoisyLR values can be outside the conventional [0, 1] range.

## 6. Proposed Pipeline

    Degraded NoisyLR (128 × 128)
                  ↓
           SRDnCNN restoration
                  ↓
    Restored output (256 × 256)
                  ↓
    PSNR / SSIM / LPIPS / Bicubic / visual analysis

## 7. Model Architecture

| Parameter | Value |
|---|---:|
| Model | SRDnCNN |
| Input channels | 1 |
| Output channels | 1 |
| Number of features | 64 |
| Residual blocks | 8 |
| Upscale factor | 2 |
| Number of parameters | 739,777 |

Implementation files:

    models/srdncnn.py
    models/common/
    models/factory.py

## 8. Experimental Design

    Experiment 1: SRDnCNN baseline
              ↓
    Experiment 2: L1Loss ablation and frozen final configuration
              ↓
    Experiment 3: Synthetic degradation training
              ↓
    Phase 8: Failure analysis
              ↓
    Phase 9: Runtime, inference, verification, and artifact audit

## 9. Experiment 1 — SRDnCNN Baseline

Experiment 1 established the initial SRDnCNN restoration baseline.

Report:

see  [the full report](results/experiment1/Experiment_1_SRDnCNN_Baseline.docx).

## 10. Experiment 2 — L1Loss Ablation

Experiment 2 evaluated SRDnCNN using L1Loss and became the frozen final configuration.

| Parameter | Value |
|---|---|
| Model | SRDnCNN |
| Input | 1 × 128 × 128 NoisyLR |
| Output | 1 × 256 × 256 restored image |
| Loss | L1Loss |
| Epochs | 50 |
| Batch size | 8 |
| Learning rate | 0.0001 |
| Optimizer | Adam |
| Scheduler | CosineAnnealingLR |
| Seed | 42 |
| Training device | NVIDIA Tesla T4 |

    Paired training samples : 3200
    Training split          : 2880
    Validation split        : 320
    Test NoisyLR samples    : 400

    Best epoch              : 50
    Best validation L1 loss : 0.0311003166

| Metric | Result |
|---|---:|
| Mean Bicubic PSNR | 22.7294 dB |
| Mean SRDnCNN PSNR | 28.0301 dB |
| Mean PSNR gain | +5.3007 dB |
| Median PSNR gain | +4.1676 dB |
| Mean SSIM | 0.9440 |
| Mean SSIM gain | +0.05885 |
| Validation improved over Bicubic | 320 / 320 (100%) |
| Validation worse than Bicubic | 0 / 320 (0%) |

| PSNR gain over Bicubic | Samples | Percentage |
|---|---:|---:|
| ≥ 6 dB | 113 | 35.31% |
| 3–6 dB | 106 | 33.12% |
| 1–3 dB | 90 | 28.12% |
| 0–1 dB | 11 | 3.44% |
| < 0 dB | 0 | 0.00% |

See [results/experiment2/README.md](results/experiment2/README.md) and [the full report](results/experiment2/Experiment_2_L1Loss_Ablation_Report.docx).

## 11. Experiment 3 — Synthetic Degradation Training

Experiment 3 investigated training with synthetic degradation to study whether controlled degradation generation could improve restoration behavior.

Report:

see   [the full report](results/experiment3/EXP3_Synthetic_Degradation_Training_Report.docx).

## 12. Final Selected Model

    SRDnCNN + L1Loss

    Epochs        : 50
    Batch size    : 8
    Learning rate : 0.0001
    Optimizer     : Adam
    Scheduler     : CosineAnnealingLR
    Seed          : 42
    Training GPU  : NVIDIA Tesla T4

## 13. Evaluation Metrics

- **PSNR** — pixel-level reconstruction quality; higher is better.
- **SSIM** — structural similarity; higher is better.
- **LPIPS** — learned perceptual similarity; lower is better.

## 14. Bicubic Baseline

Bicubic upsampling is the conventional reference baseline. Comparisons include Bicubic PSNR, SRDnCNN PSNR, PSNR gain, SSIM, per-image improvement, and improvement distribution. The final validation comparison improved over Bicubic for all 320 validation samples.

## 15. Failure Analysis

Phase 8 evaluated all 320 validation samples using per-image PSNR/SSIM, Bicubic comparison, degradation-severity analysis, PSNR-gain correlations, visual comparison panels, objective failure classification, and final visual evaluation.

The main limitation identified was fine and high-frequency texture reconstruction under challenging degradation.

Report:

    results/phase8_failure_analysis/Phase_8_Failure_Analysis_Semiconductor_Image_Restoration.docx

## 16. Final Phase 9 Pipeline

    Phase 9.2  Runtime benchmark
    Phase 9.3  Final inference script and 400-image test inference
    Phase 9.4  Prediction verification
    Phase 9.5  Final visual inspection
    Phase 9.6  Artifact and reproducibility audit

## 17. Runtime Benchmark

The first recorded runtime benchmark was performed on an NVIDIA Tesla T4 with CUDA.

    Model-only:
    Images       : 320
    Batch size   : 1
    Warm-up runs : 20
    Mean latency : 4.861 ms
    Median       : 4.726 ms
    Throughput   : 205.700 images/s

    End-to-end:
    Mean latency : 6.342 ms
    Median       : 6.337 ms
    Throughput   : 157.673 images/s

## 18. Final Test Inference

    Device          : NVIDIA Tesla T4
    Checkpoint      : Epoch 50
    Checkpoint loss : 0.031100316578522323
    Test images     : 400
    Output files    : 400
    Mean inference  : 7.924 ms
    Median inference: 7.149 ms
    Total inference : 3.170 s
    Throughput      : 126.192 images/s

## 19. Final Prediction Verification

    Test inputs      : 400
    Prediction files : 400
    Missing IDs      : 0
    Extra IDs        : 0
    Bad shapes       : 0
    Bad dtypes       : 0
    NaN/Inf files    : 0

    First prediction : 000000.npy
    Last prediction  : 000399.npy
    Global minimum   : -0.22802016139030457
    Global maximum   : 1.265357255935669
    Mean of means    : 0.4422313290741295

    PHASE 9.4 — FINAL PREDICTION VERIFICATION PASSED

## 20. Visual Inspection

Representative samples were 000000, 000025, 000050, 000075, 000100, 000150, 000200, 000250, 000300, 000325, 000350, and 000399.

The inspection generated 12 individual comparison images and one contact sheet:

    results/experiment2/final_visual_inspection/
    └── phase9_5_final_test_contact_sheet.png

## 21. Reproducibility Audit

    OK : train.py
    OK : SRDnCNN model and factory
    OK : Exp2 configuration and checkpoint
    OK : training history and log
    OK : LPIPS summary and per-image records
    OK : runtime summary and per-image records
    OK : final inference script
    OK : final predictions and timing
    OK : visual inspection

    Backup files             : 821
    Backup final predictions : 400
    Backup visual artifacts  : 13

    PHASE 9.6 — FINAL ARTIFACT AUDIT: PASSED
    All event-critical artifacts are present.
    The computational experiment is frozen.

## 22. Repository Structure

    semiconductor-image-restoration/
    ├── .github/
    ├── .gitattributes
    ├── .gitignore
    ├── checkpoint/
    │   └── exp2_epoch50_best_model.pth
    ├── configs/
    │   └── default.yaml
    ├── datasets/
    │   ├── __init__.py
    │   ├── augmentations.py
    │   ├── datamodule.py
    │   ├── dataset.py
    │   └── transforms.py
    ├── docs/
    │   └── dataset_report.md
    ├── losses/
    │   ├── __init__.py
    │   ├── factory.py
    │   └── losses.py
    ├── models/
    │   ├── factory.py
    │   ├── srdncnn.py
    │   └── common/
    │       ├── blocks.py
    │       └── layers.py
    ├── scripts/
    │   ├── analyze_dataset.py
    │   ├── test_augmentations.py
    │   ├── test_checkpoint.py
    │   ├── test_config.py
    │   ├── test_dataloader.py
    │   ├── test_dataset.py
    │   ├── test_logger.py
    │   ├── test_loss.py
    │   ├── test_model.py
    │   ├── test_optimizer.py
    │   ├── test_scheduler.py
    │   ├── test_seed.py
    │   └── test_transforms.py
    ├── training/
    │   ├── checkpoint.py
    │   ├── engine.py
    │   ├── optimizer.py
    │   ├── scheduler.py
    │   └── trainer.py
    ├── utils/
    │   ├── config.py
    │   ├── logger.py
    │   └── seed.py
    ├── train.py
    ├── inference.py
    ├── demo.py
    ├── view_train_outputs.py
    ├── requirements.txt
    ├── results/
    │   ├── comparisons/
    │   ├── experiment1/
    │   ├── experiment2/
    │   ├── experiment3/
    │   ├── phase8_failure_analysis/
    │   └── results/
    └── README.md

The original dataset, generated prediction arrays, temporary outputs, and large experiment archives are not part of the normal source tree.

## 23. Installation

Dependencies:

    torch
    torchvision
    numpy
    PyYAML
    tqdm
    matplotlib

Install:

    py -m pip install -r requirements.txt

## 24. Running Inference

The evaluator-facing entry point is [inference.py](inference.py). Defaults:

    Configuration: configs/default.yaml
    Checkpoint  : checkpoint/exp2_epoch50_best_model.pth

Run:

    py inference.py --input_dir "PATH_TO_INPUT_DIRECTORY" --output_dir "PATH_TO_OUTPUT_DIRECTORY"

Example:

    py inference.py --input_dir "E:\semiconductor-image-restoration\data\raw\Test_NoisyLR\NoisyLR" --output_dir "E:\semiconductor-image-restoration\submission_output"

To override the checkpoint:

    py inference.py --input_dir "path\to\NoisyLR" --output_dir "path\to\output" --checkpoint "path\to\exp2_epoch50_best_model.pth"

The script validates inputs, loads the checkpoint, writes one .npy prediction per input, and verifies count, shape, dtype, and numerical validity.

Expected shape:

    Input  : 128 × 128
    Output : 256 × 256

## 25. Training

The training entry point is [train.py](train.py):

    py train.py --config configs/default.yaml

The frozen experiment used 50 epochs, batch size 8, Adam, learning rate 0.0001, CosineAnnealingLR, and seed 42.

## 26. Checkpoint

    checkpoint/exp2_epoch50_best_model.pth

    Epoch            : 50
    Validation loss  : 0.031100316578522323
    Model parameters : 739,777

Large binary artifacts should generally be stored with Git LFS or an external experiment archive rather than ordinary Git blobs.

## 27. Research Artifacts

- [docs/dataset_report.md](docs/dataset_report.md) — dataset statistics and pair verification.
- [results/experiment2/README.md](results/experiment2/README.md) — frozen Experiment 2 configuration and validation results.
- [results/experiment1/Experiment_1_SRDnCNN_Baseline.docx](results/experiment1/Experiment_1_SRDnCNN_Baseline.docx) — baseline report.
- [results/experiment2/Experiment_2_L1Loss_Ablation_Report.docx](results/experiment2/Experiment_2_L1Loss_Ablation_Report.docx) — L1Loss ablation report.
- [results/experiment3/EXP3_Synthetic_Degradation_Training_Report.docx](results/experiment3/EXP3_Synthetic_Degradation_Training_Report.docx) — synthetic degradation report.
- [results/phase8_failure_analysis/Phase_8_Failure_Analysis_Semiconductor_Image_Restoration.docx](results/phase8_failure_analysis/Phase_8_Failure_Analysis_Semiconductor_Image_Restoration.docx) — failure-analysis report.
- [results/results/FINAL_EXPERIMENTAL_RECORD.md](results/results/FINAL_EXPERIMENTAL_RECORD.md) — final experimental record.
- [results/results/research_decision.md](results/results/research_decision.md) — research decision record.

The complete local Experiment 2 archive is EXP2_RESULTS_FINAL.zip.

## 28. Limitations

- Fine and high-frequency textures remain difficult under severe degradation.
- The test set has no public ground truth for direct objective evaluation.
- Results depend on the degradation distribution represented by the training data.
- A single-channel model may not generalize to color or multi-modal imagery without architectural changes.
- Large prediction arrays and experiment archives are intentionally kept outside the normal source tree.

## 29. Future Work

1. Train with a broader range of real degradation processes.
2. Test perceptual, frequency-domain, and hybrid reconstruction losses.
3. Add multi-scale or attention-based restoration blocks.
4. Evaluate on additional semiconductor inspection datasets.
5. Export the model to an optimized deployment format.
6. Add automated production quality-control checks.
7. Expand evaluation with expert semiconductor-inspection review.

## 30. Final Project Status

    Final model          : SRDnCNN + L1Loss
    Input resolution     : 128 × 128
    Output resolution    : 256 × 256
    Test images          : 400
    Predictions verified : 400 / 400
    Missing outputs      : 0
    Bad shapes           : 0
    NaN/Inf outputs      : 0
    Inference pipeline   : PASSED
    Artifact audit       : PASSED

The project is ready for repository review and evaluator-facing inference.
