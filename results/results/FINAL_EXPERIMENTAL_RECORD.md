FINAL EXPERIMENTAL RECORD

AI-Based Restoration of Degraded Images for Semiconductor Inspection

Primary model: SRDnCNNValidation set: 320 imagesRandom seed: 42Experiments: Exp1, Exp2, Exp3Failure analysis: Phase 8

1. Project Objective

The project investigates AI-based restoration of degraded semiconductor inspection images. The task is to restore a high-resolution ground-truth image from a degraded low-resolution input containing documented degradation mechanisms:

Speckle noise

Gaussian noise

Downsampling

Evaluation metrics:

PSNR

SSIM

MAE

LPIPS

Higher PSNR/SSIM are better. Lower MAE/LPIPS are better.

2. Dataset

Component

Count

Resolution

Training GT

3200

256 × 256

Training NoisyLR

3200

128 × 128

Test NoisyLR

400

128 × 128

Images are stored as float32.

The fixed validation split is:

Split

Samples

Training

2880

Validation

320

The validation split was verified to contain 320 unique filenames.

3. Common Experimental Protocol

Validation samples : 320
Validation split   : 10%
Random seed        : 42
Input              : 128 × 128
Target             : 256 × 256

Exp1 and Exp2 use the same validation images for paired comparison.

4. Experiment 1 — SRDnCNN Baseline

4.1 Purpose

Exp1 establishes the baseline restoration performance of the SRDnCNN architecture.

4.2 Aggregate Results

Metric

Exp1

PSNR

28.0357 dB

SSIM

0.749338

LPIPS

0.324483

MAE

Not recovered in frozen record

4.3 Representative Cases

Case

Index

PSNR (dB)

SSIM

Best PSNR

219

44.0977

0.970090

Worst PSNR

124

11.1695

0.275663

Best SSIM

107

34.1106

0.983356

Worst SSIM

272

20.8467

0.197963

4.4 Observed Failure Characteristics

The frozen Exp1 analysis identifies:

Over-smoothing

Incorrect texture reconstruction

Loss of fine structural information

Substantial texture/intensity mismatch in difficult cases

These are documented observations, not proven causal explanations.

5. Experiment 2 — L1Loss Ablation

5.1 Purpose

Exp2 investigates the effect of the L1Loss training objective under the controlled experimental setup.

5.2 Aggregate Results

Metric

Exp1

Exp2

PSNR

28.0357 dB

28.0315 dB

SSIM

0.749338

0.749582

MAE

Not recovered

0.031087

LPIPS

0.324483

0.319640

The aggregate PSNR values are nearly identical. Exp2 has slightly higher SSIM and lower LPIPS.

6. Exp1 vs Exp2 Paired Analysis

Exp1 and Exp2 were evaluated on the same 320 validation images.

6.1 Per-Metric Win Counts

Metric

Exp1 wins

Exp2 wins

PSNR

167

153

SSIM

123

197

LPIPS

30

290

Interpretation:

PSNR is very close, with a small Exp1 advantage in per-image wins.

Exp2 wins substantially more images on SSIM.

Exp2 wins overwhelmingly more images on LPIPS.

These are descriptive results from the tested validation set. No statistical significance testing was performed.

6.2 Largest Per-Image Differences

PSNR

Case

Index

Exp1

Exp2

Largest Exp1 advantage

38

28.269480

28.107425

Largest Exp2 advantage

114

42.066628

42.202264

SSIM

Case

Index

Exp1

Exp2

Largest Exp1 advantage

37

0.575235

0.566451

Largest Exp2 advantage

287

0.741836

0.747059

LPIPS

Case

Index

Exp1

Exp2

Largest Exp2 advantage

67

0.336790

0.315396

Largest Exp1 advantage

272

0.958732

0.970777

7. Important Metric Disagreement — Index 272

Index 272 is an important failure-analysis case.

It is the documented worst-SSIM case for Exp1:

PSNR : 20.8467 dB
SSIM : 0.197963

The paired LPIPS comparison also records an Exp1 advantage over Exp2 for this image.

This demonstrates that one image can behave differently depending on the evaluation metric. Failure analysis should therefore not rely on a single metric.

8. Experiment 3 — Synthetic Degradation Training

8.1 Purpose

Exp3 investigates training with synthetic degradation based on the documented degradation mechanisms:

Speckle

Gaussian noise

Downsampling

The tested training configuration used a 50/50 mixture of original and synthetically degraded training data.

8.2 Best Checkpoint

Best epoch : 45
Best Val L1: 0.031645628483965994

8.3 Original Validation Results

Metric

Exp2

Exp3

PSNR

28.0315 dB

27.899917 dB

SSIM

0.749582

0.739837

MAE

0.031087

0.031635

LPIPS

0.319640

0.338273

Differences relative to Exp2:

PSNR : -0.131583 dB
SSIM : -0.009745
MAE  : +0.000548
LPIPS: +0.018633

Therefore, Exp3 did not improve the original validation distribution under the tested configuration.

This is a conclusion about the tested 50/50 configuration, not a general rejection of synthetic degradation.

9. Exp3 Test Inference

The frozen Exp3 test evaluation processed 400 test images:

Device          : CUDA
GPU             : Tesla T4
Test images     : 400
Checkpoint      : Epoch 45
Input images    : 400
Output images   : 400
Output size     : 256 × 256
Output dtype    : float32

The test inference completed successfully.

10. Exp3 LPIPS Evaluation

Exp3

Mean   : 0.338273
Median : 0.297124
Std    : 0.221602
Min    : 0.044400
Max    : 1.111020

Bicubic

Mean   : 0.435090
Median : 0.412678
Std    : 0.157307
Min    : 0.103440
Max    : 0.920526

LPIPS improvement : +0.096817
LPIPS wins        : 223 / 320

Exp3 therefore achieves lower mean LPIPS than Bicubic on the evaluated validation set.

11. Exp3 vs Bicubic

Metric

Exp3

Bicubic

PSNR

27.899917 dB

22.729405 dB

SSIM

0.739837

0.541810

MAE

0.031635

0.057429

Improvement relative to Bicubic:

PSNR improvement : +5.170512 dB
SSIM improvement : +0.198027
MAE reduction    : +0.025794

This confirms substantial improvement over Bicubic on the evaluated validation set.

12. OOD Robustness Dataset

An optional OOD dataset was generated from 100 source GT images.

Conditions:

moderate

strong

alternate_order

mixed

Mechanisms:

Speckle

Gaussian

Downsampling

The OOD definition represents variation in severity and/or ordering of documented mechanisms, not a new degradation mechanism.

Source GT images : 100
Conditions       : 4
Generated inputs : 400
Seed             : 42

Each condition contains 100 generated inputs.

The manifest contains:

source_gt
condition
severity
degradation_order
seed
input_file
gt_file
shape
outside_0_1_percent

The generated inputs were verified as:

Shape    : 128 × 128
Dtype    : float32
NaN/Inf  : None

OOD dataset generation and verification passed.

13. OOD Evaluation Status

The OOD dataset was successfully generated, but the original frozen Exp2/Exp3 checkpoints were not recovered from the Kaggle runtime.

Therefore a complete quantitative OOD comparison of the frozen experiments has not been established.

OOD dataset generation      : COMPLETED
OOD quantitative evaluation : NOT ESTABLISHED

The OOD dataset remains a future evaluation resource.

14. Phase 8 — Failure Analysis

14.1 Failure Taxonomy

F1 — Over-smoothing

Observed in difficult Exp1 cases, particularly the worst-SSIM example.

F2 — Texture reconstruction mismatch

Difficult cases can exhibit substantial differences in texture and intensity patterns.

F3 — Fine-structure loss

Loss of fine structural information is identified among difficult cases.

F4 — Metric disagreement

Different metrics can rank the same image differently. Index 272 is a documented example.

F5 — Synthetic-training trade-off

The tested 50/50 synthetic-degradation configuration did not improve the original validation distribution.

15. Phase 8 Visual Re-analysis Status

Exact visual reproduction of the frozen Exp1/Exp2/Exp3 outputs was investigated.

The original Kaggle runtime artifacts were not available in the current recovery environment.

A local checkpoint was found:

checkpoints/best_model.pth

Inspection showed:

Epoch : 1
Loss  : 0.18943360447883606

This does not match the frozen Exp3 checkpoint:

Epoch      : 45
Best Val L1: 0.031645628483965994

Therefore the local checkpoint was not substituted for the frozen experiments.

Current status:

Frozen aggregate results           : AVAILABLE
Exp1 representative failure cases : AVAILABLE
Exp1 vs Exp2 paired evidence       : AVAILABLE
Exact frozen visual outputs        : NOT RECOVERED
Exact Exp1/Exp2 checkpoints        : NOT RECOVERED
Exact Exp3 checkpoint              : NOT RECOVERED

No reconstructed model is presented as an original frozen experiment.

16. Scientific Interpretation

16.1 SRDnCNN effectiveness

The restoration model substantially outperforms Bicubic interpolation on the evaluated validation data.

For Exp3:

PSNR : +5.170512 dB
SSIM : +0.198027
MAE  : -0.025794

relative to Bicubic.

16.2 Exp1 vs Exp2

Exp1 and Exp2 have nearly identical aggregate PSNR.

Exp2 has:

Slightly higher SSIM

Lower LPIPS

More per-image SSIM wins

Much more per-image LPIPS wins

Based on the frozen original-validation evidence, Exp2 is preferred over Exp1.

16.3 Exp3

Exp3 successfully implemented synthetic-degradation training.

However, under the tested 50/50 configuration, it did not improve the original validation distribution.

This result should not be generalized beyond the tested configuration.

17. Limitations

Statistical significance testing was not performed.

Repeated-seed evaluation was not performed.

Exp1 frozen MAE was not recovered.

Exp3 per-image PSNR/SSIM/MAE arrays were lost after the Kaggle runtime reset.

Exact original visual outputs/checkpoints were not recovered.

Complete 320-image individual failure ranking for Exp3 is unavailable.

Quantitative OOD robustness for the frozen models was not established.

Exp3 conclusions apply specifically to the tested synthetic-degradation configuration and original validation distribution.

Exact frozen visual comparisons cannot be claimed without the original outputs/checkpoints.

18. Artifact Preservation Lessons

The Kaggle workflow demonstrated that artifacts stored only under /kaggle/working may become unavailable after a runtime/session reset.

Future experiments must preserve artifacts immediately.

For every future experiment, save:

Experiment configuration
Model checkpoint
Training log
Validation metrics
Per-image metrics
LPIPS results
Representative outputs
Evaluation manifest
Summary JSON
Summary Markdown

Recommended storage separation:

GitHub

Source code

Configuration

Small CSV/JSON/Markdown records

Documentation

Persistent Drive/storage

Model checkpoints

Large arrays

Generated image outputs

Large experiment artifacts

19. Final Experimental Status

Component

Status

Dataset analysis

COMPLETED

Dataset loader

COMPLETED

Baseline Exp1

COMPLETED

Exp2 loss ablation

COMPLETED

Exp1 vs Exp2 paired analysis

COMPLETED

Exp3 synthetic degradation

COMPLETED

Exp3 test inference

COMPLETED

Exp3 LPIPS evaluation

COMPLETED

OOD dataset generation

COMPLETED

OOD quantitative evaluation

NOT ESTABLISHED

Phase 8 evidence-based failure analysis

COMPLETED

Exact frozen visual re-analysis

NOT RECOVERED

20. Final Conclusions

The project established a working SRDnCNN-based restoration pipeline for degraded semiconductor inspection images.

The experiments demonstrate that:

Learned restoration substantially outperforms Bicubic interpolation.

Exp1 and Exp2 have nearly identical PSNR performance.

Exp2 provides stronger SSIM and LPIPS behavior on the frozen validation set.

The tested Exp3 50/50 synthetic-degradation configuration did not improve the original validation distribution.

Difficult restoration cases involve over-smoothing, texture mismatch and fine-structure loss.

Restoration quality cannot be fully characterized by a single metric.

OOD robustness remains insufficiently established.

Exact frozen visual outputs/checkpoints were not recovered and were not replaced with newly trained models.

21. Recommended Next Research Question

The current experiments motivate:

How can restoration robustness be improved while preserving performance on the original degradation distribution?

Any future experiment should evaluate both:

Original validation distribution

Controlled robustness/OOD distribution

It must also preserve all checkpoints, per-image metrics, outputs and manifests from the beginning.

END OF FINAL EXPERIMENTAL RECORD