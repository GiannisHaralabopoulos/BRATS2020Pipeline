# 🧠 BraTS2020 Unified Segmentation Pipeline

A single-file PyTorch pipeline that unifies training, evaluation, and cross-validation for brain tumour segmentation on the [BraTS 2020](https://www.med.upenn.edu/cbica/brats2020/data.html) dataset. Six model architectures — from a lightweight 2-D U-Net to a diffusion-embedded 3-D network — are available behind one consistent CLI.

---

## Table of Contents

- [Overview](#overview)
- [Source Repositories](#source-repositories)
- [Segmentation Targets](#segmentation-targets)
- [Requirements](#requirements)
- [Dataset Setup](#dataset-setup)
- [Quick Start](#quick-start)
- [Models](#models)
- [Training](#training)
- [Evaluation](#evaluation)
- [Cross-Validation](#cross-validation)
- [Early Stopping](#early-stopping)
- [Output Structure](#output-structure)
- [Configuration Reference](#configuration-reference)

---

## Overview

This pipeline consolidates five independent research repositories into a single, consistent codebase. It provides:

- **6 model architectures** selectable with one flag
- **Unified data loading** — NIfTI volumes, z-score normalisation, foreground-biased patch cropping, and on-the-fly augmentation
- **Combined Dice + BCE loss** (MSE noise-prediction loss for Diff-UNet)
- **Mixed-precision training** (AMP) with gradient clipping
- **Early stopping** — configurable patience, streak survives checkpoint resume
- **10-fold cross-validation** with a clean 10 % held-out test set
- **TensorBoard logging** and per-run `config.json` for full reproducibility
- **CSV evaluation reports** — per-patient Dice and HD95 for each tumour region

---

## Source Repositories

| Repository | Architecture | Framework |
|---|---|---|
| [renugadevi26/HVU_Code](https://github.com/renugadevi26/HVU_Code) | Hybrid ViT-UNet | Keras / TF → **ported to PyTorch** |
| [mahdizynali/BraTS2020-Tensorflow-Brain-Tumor-Segmentation](https://github.com/mahdizynali/BraTS2020-Tensorflow-Brain-Tumor-Segmentation) | Attention U-Net | TF → **ported to PyTorch** |
| [lescientifik/open_brats2020](https://github.com/lescientifik/open_brats2020) | EquiUnet (Top-10 BraTS 2020) | PyTorch |
| [ge-xing/Diff-UNet](https://github.com/ge-xing/Diff-UNet) | Diffusion-embedded U-Net | PyTorch |
| [mtancak/PyTorch-UNet-Brain-Cancer-Segmentation](https://github.com/mtancak/PyTorch-UNet-Brain-Cancer-Segmentation) | Vanilla 3-D U-Net | PyTorch |

---

## Segmentation Targets

BraTS uses a three-region hierarchical labelling convention:

| Region | Label | Description |
|---|---|---|
| **WT** — Whole Tumour | 1 + 2 + 4 | Full tumour extent |
| **TC** — Tumour Core | 1 + 4 | Core without oedema |
| **ET** — Enhancing Tumour | 4 | Actively enhancing tissue |

The pipeline produces **three binary output maps** (one per region) and reports **Dice** and **HD95** for each independently.

---

## Requirements

```bash
pip install torch torchvision nibabel scipy tensorboard
```

| Package | Purpose |
|---|---|
| `torch` | Model training and inference |
| `nibabel` | Load `.nii` / `.nii.gz` MRI volumes |
| `scipy` | HD95 computation (`directed_hausdorff`) |
| `tensorboard` | Live training metrics dashboard |

Python ≥ 3.9 and PyTorch ≥ 2.0 are recommended. CUDA is auto-detected; CPU fallback is supported.

---

## Dataset Setup

Download the BraTS 2020 training data from [Kaggle](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation) and unpack it. The pipeline expects the standard BraTS directory layout:

```
BraTS2020/
├── BraTS20_Training_001/
│   ├── BraTS20_Training_001_flair.nii.gz
│   ├── BraTS20_Training_001_t1.nii.gz
│   ├── BraTS20_Training_001_t1ce.nii.gz
│   ├── BraTS20_Training_001_t2.nii.gz
│   └── BraTS20_Training_001_seg.nii.gz
├── BraTS20_Training_002/
│   └── ...
```

Any subfolder containing a `*_flair.nii*` file is automatically detected. Nested subdirectories (e.g. a `Training/` wrapper) are handled transparently.

---

## Quick Start

```bash
# List all available models
python brats_pipeline.py --list_models

# Train the lightweight 2-D U-Net (great for quick experiments)
python brats_pipeline.py --model unet2d --data_dir ./BraTS2020 \
    --batch_size 16 --epochs 50

# Train the top-10 BraTS 2020 EquiUnet
python brats_pipeline.py --model equiunet --data_dir ./BraTS2020 --epochs 200

# Evaluate a saved checkpoint
python brats_pipeline.py --mode eval --model equiunet \
    --data_dir ./BraTS2020 --checkpoint runs/equiunet_20250101_120000/best.pth

# Run 10-fold cross-validation
python brats_pipeline.py --cv --model unet3d --data_dir ./BraTS2020
```

---

## Models

| Flag | Architecture | Dims | Approx. Params | Notes |
|---|---|---|---|---|
| `unet2d` | Classic 2-D U-Net | 2-D slices | ~1 M | Fastest; good for prototyping |
| `unet3d` | Vanilla 3-D U-Net | 3-D patches | ~19 M | Solid baseline |
| `attention_unet` | Attention U-Net | 3-D patches | ~22 M | Soft attention gates at each skip |
| `equiunet` | EquiUnet | 3-D patches | ~18 M | Top-10 BraTS 2020; residual-equivariant blocks |
| `diff_unet` | Diff-UNet | 3-D patches | ~25 M | DDPM training; DDIM inference (10 steps) |
| `hvu` | Hybrid ViT-UNet | 3-D patches | ~28 M | CNN encoder + Transformer global context branch |

### UNet2D — slice mode

`unet2d` decomposes each 3-D volume into axial slices before training. ~90 % of fully-background slices are skipped during dataset construction to keep class balance. At validation and test time, slice-level Dice scores are aggregated back to patient level before reporting — so metrics are comparable with 3-D models.

### Diff-UNet — diffusion mode

During **training** the model receives the image concatenated with a noised segmentation map and learns to predict the added noise (standard DDPM objective). During **inference** it starts from Gaussian noise and iteratively denoises via DDIM in 10 steps — no segmentation ground truth is required.

---

## Training

```bash
python brats_pipeline.py \
    --model equiunet \
    --data_dir ./BraTS2020 \
    --epochs 200 \
    --batch_size 2 \
    --lr 1e-4 \
    --scheduler cosine \
    --amp \
    --save_dir ./runs
```

Key training features:

- **Foreground-biased cropping** — patches are centred on labelled voxels with high probability, ensuring the tumour is always in view
- **On-the-fly augmentation** — random flips (all three axes), per-modality intensity shift/scale
- **Mixed-precision (AMP)** — enabled by default; disable with `--no_augment` / remove `--amp`
- **Gradient clipping** — global norm clipped to 1.0
- **Resume from checkpoint** — pass `--checkpoint path/to/last.pth` to continue a run; the early-stopping streak is preserved

### Learning-rate schedulers

| `--scheduler` | Behaviour |
|---|---|
| `cosine` (default) | Cosine annealing over the full epoch budget |
| `plateau` | Reduce LR by 0.5× after 10 epochs without val-loss improvement |
| `none` | Constant learning rate |

---

## Evaluation

```bash
python brats_pipeline.py \
    --mode eval \
    --model equiunet \
    --data_dir ./BraTS2020 \
    --checkpoint runs/equiunet_20250101_120000/best.pth
```

Evaluation loads the checkpoint, runs inference on the validation split, and writes `eval_results.csv` to `--save_dir`. Each row is one patient; a final `MEAN` row summarises across the cohort.

**Reported metrics per region (WT / TC / ET):**

- **Dice** — volumetric overlap coefficient
- **HD95** — 95th-percentile Hausdorff distance in voxel units (requires `scipy`)

---

## Cross-Validation

```bash
# 10-fold CV with 10% held-out test set (defaults)
python brats_pipeline.py --cv --model unet3d --data_dir ./BraTS2020

# Custom configuration
python brats_pipeline.py --cv --n_folds 5 --test_ratio 0.15 \
    --model unet2d --data_dir ./BraTS2020 --epochs 100
```

### How it works

```
All patients (N)
│
├── 10% ──► Held-out test set  (locked away, never used during CV)
│
└── 90% ──► k-fold split
             ├── Fold 1: train on folds 2–10, validate on fold 1
             ├── Fold 2: train on folds 1, 3–10, validate on fold 2
             ├── ...
             └── Fold k: train on folds 1–(k-1), validate on fold k
                 │
                 └── After each fold: evaluate best.pth on held-out test set
```

Each fold trains a **completely fresh model** with early stopping active. After all folds complete, every `best.pth` is evaluated on the held-out test patients and results are aggregated.

### Output files

| File | Contents |
|---|---|
| `cv_split.json` | Full patient-to-fold assignment (reproducible) |
| `cv_fold_summary.csv` | Per-fold val Dice + test Dice (WT / TC / ET / mean) with MEAN ± STD row |
| `cv_test_per_patient.csv` | Per-patient Dice for every (fold, patient) combination |
| `cv_test_summary.csv` | Mean, std, median, min, max across all test patients and folds |

---

## Early Stopping

Training halts automatically if the **mean validation Dice** (average of WT, TC, ET) does not improve for `--early_stopping_patience` consecutive epochs (default: 100).

```bash
# Tighter patience for faster experiments
python brats_pipeline.py --model unet2d --data_dir ./BraTS2020 \
    --epochs 500 --early_stopping_patience 30
```

The no-improvement streak is **saved in every checkpoint** and restored on resume, so it is not reset if training is interrupted and continued.

The current streak is also written to TensorBoard as `val/epochs_no_improve` so you can track it live.

---

## Output Structure

Each training run creates a timestamped folder inside `--save_dir`:

```
runs/
└── equiunet_20250315_143022/           # one folder per run / fold
    ├── config.json                     # full config used for this run
    ├── run.log                         # console output
    ├── last.pth                        # most recent checkpoint
    ├── best.pth                        # best-Dice checkpoint
    └── tb/                             # TensorBoard event files

# CV mode produces additional files in the root save_dir:
runs/
├── cv_split.json
├── cv_fold_summary.csv
├── cv_test_per_patient.csv
├── cv_test_summary.csv
├── equiunet_20250315_143022_fold01/
├── equiunet_20250315_143501_fold02/
└── ...
```

Launch TensorBoard to monitor all runs simultaneously:

```bash
tensorboard --logdir ./runs
```

---

## Configuration Reference

All options can be set via CLI flags. Defaults are shown below.

### Data

| Flag | Default | Description |
|---|---|---|
| `--data_dir` | `./BraTS2020` | Root directory of the BraTS2020 dataset |
| `--val_ratio` | `0.2` | Validation fraction (single-run mode only) |
| `--patch_size H W D` | `128 128 128` | Spatial size of 3-D patches / 2-D slice spatial size |
| `--num_workers` | `4` | DataLoader worker processes |
| `--cache_rate` | `0.0` | Fraction of dataset to cache in RAM |

### Model

| Flag | Default | Description |
|---|---|---|
| `--model` | `unet3d` | Architecture: `unet2d`, `unet3d`, `attention_unet`, `equiunet`, `diff_unet`, `hvu` |
| `--base_filters` | `32` | Base channel width (doubles at each encoder level) |

### Training

| Flag | Default | Description |
|---|---|---|
| `--epochs` | `200` | Maximum training epochs |
| `--batch_size` | `1` | Batch size (use ≥8 for `unet2d`) |
| `--lr` | `1e-4` | Initial learning rate |
| `--weight_decay` | `1e-5` | AdamW weight decay |
| `--scheduler` | `cosine` | LR schedule: `cosine`, `plateau`, `none` |
| `--amp` | `True` | Enable automatic mixed precision |
| `--seed` | `42` | Global random seed |
| `--checkpoint` | — | Path to `.pth` file to resume from |
| `--save_dir` | `./runs` | Root directory for all outputs |
| `--log_interval` | `10` | Steps between TensorBoard / console logs |
| `--early_stopping_patience` | `100` | Epochs without Dice improvement before stopping |
| `--no_augment` | — | Disable all data augmentation |

### Cross-Validation

| Flag | Default | Description |
|---|---|---|
| `--cv` | `False` | Enable k-fold cross-validation |
| `--n_folds` | `10` | Number of folds |
| `--test_ratio` | `0.10` | Fraction of patients held out as a test set |
