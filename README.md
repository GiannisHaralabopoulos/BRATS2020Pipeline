# Unified BraTS2020 Brain Tumour Segmentation Pipeline

A unified PyTorch pipeline for training, validating, testing, and benchmarking seven brain tumour segmentation architectures on the labelled BraTS2020 cohort.

The pipeline was designed for controlled architecture comparison. It uses a fixed patient-level split, a common multi-class loss, shared GPU augmentation, consistent early stopping, and the same final evaluation procedure across models, while retaining architecture-specific requirements such as 2D versus 3D inputs, internal resizing, iterative diffusion inference, and ensemble test-time augmentation.

## Main features

- Seven 2D and 3D segmentation architectures in one training pipeline
- Fixed patient-level 80% training, 10% validation, and 10% final test split
- Reuses the same saved split across architectures
- Four-class softmax segmentation with background, NCR/NET, oedema, and enhancing tumour
- Combined loss with 0.5 foreground Dice loss and 0.5 categorical cross-entropy
- Patient-level Dice and HD95 for whole tumour, tumour core, and enhancing tumour
- Native PyTorch GPU augmentation
- Automatic VRAM-aware batch-size tuning
- Mixed-precision training
- Memory-mapped preprocessing cache
- Full patient-level computational benchmarking
- Early stopping and best-checkpoint selection
- Patient-level CSV outputs suitable for paired statistical comparisons
- Optional legacy k-fold cross-validation mode
- Windows and RTX 5090-oriented memory and worker safeguards

## Models

| CLI name | Architecture | Dimensionality | Approx. trainable parameters | Key implementation details |
| --- | --- | ---: | ---: | --- |
| `unet2d` | U-Net 2D | 2D | 7.76 M | Four pooling stages, GroupNorm, transposed-convolution decoder |
| `hvu` | HVU / DenseVU-ED | 2D | 36.51 M | U-Net + DenseNet121 feature branch + Vision Transformer branch |
| `deeplabv3plus2d` | DeepLabV3+ | 2D | 40.35 M | ResNet-50 style encoder, ASPP, output stride 16 |
| `diff_unet` | Diff-UNet | 3D | 10.05 M | Separate MRI encoder, START_X diffusion training, 50-step DDIM-style inference |
| `hybridattunet` | HybridAttUNet | 3D | 14.10 M | Residual attention modules and squeeze-excitation decoder |
| `unet3d` | U-Net 3D | 3D | 22.58 M | Conventional volumetric U-Net |
| `deepensemble` | DeepEnsembled U-Net | 3D | 115.78 M total | Five independently trained Henry-style members, deep supervision, 16-way TTA |

The DeepEnsembled U-Net trains its five members sequentially. Peak training VRAM therefore reflects the largest individual member rather than all five models resident simultaneously.

## Dataset

The default experiment uses only the labelled BraTS2020 training cohort. The official BraTS2020 validation cohort is not used because it does not contain ground-truth segmentation masks.

Each patient directory must contain the four MRI modalities and a segmentation mask:

```text
BraTS2020/
└── BraTS20_Training_001/
    ├── BraTS20_Training_001_flair.nii.gz
    ├── BraTS20_Training_001_t1.nii.gz
    ├── BraTS20_Training_001_t1ce.nii.gz
    ├── BraTS20_Training_001_t2.nii.gz
    └── BraTS20_Training_001_seg.nii.gz
```

The pipeline searches recursively for patient folders containing `*_flair.nii*`. If `--data_dir` points to a parent directory containing both training and validation data, the labelled training cohort is auto-detected where possible. You can avoid ambiguity by supplying `--train_data_dir` explicitly.

### MRI preprocessing

The four input modalities are:

1. FLAIR
2. T1
3. T1ce
4. T2

Each modality is independently z-score normalised over non-zero voxels. Background voxels remain zero.

Native BraTS2020 spatial geometry is:

```text
240 x 240 x 155
```

For 3D models, the depth is padded to 160 so that all spatial dimensions are compatible with four downsampling stages:

```text
4 x 240 x 240 x 160
```

The default experiment uses the full native field of view rather than spatial cropping.

### Label mapping

BraTS labels are remapped into four mutually exclusive softmax classes:

| Pipeline class | Meaning | Original BraTS label |
| ---: | --- | ---: |
| 0 | Background | 0 |
| 1 | NCR/NET | 1 |
| 2 | Oedema | 2 |
| 3 | Enhancing tumour | 4 |

The standard overlapping BraTS regions are reconstructed for evaluation:

```text
WT = classes 1 + 2 + 3
TC = classes 1 + 3
ET = class 3
```

## Experimental design

The default training workflow uses one deterministic patient-level split:

```text
80% training
10% validation
10% final held-out test
```

With the 369 labelled BraTS2020 cases, this gives approximately:

```text
295 training patients
37 validation patients
37 final test patients
```

The split is generated using seed `123` and saved to:

```text
runs/fixed_split.json
```

The same split file is reused across architectures when the metadata match.

The validation subset is used for:

- model selection
- early stopping
- best-checkpoint selection

The final test subset is evaluated only after training and model selection are complete.

## Loss function

All architectures use the same multi-class objective:

```text
Loss = 0.5 x DiceLoss + 0.5 x CrossEntropyLoss
```

The Dice component:

- operates on softmax probabilities
- excludes background
- averages over the three foreground classes
- uses smoothing of `1e-5`

Cross-entropy is standard categorical cross-entropy over all four mutually exclusive classes.

For the DeepEnsembled U-Net, the same combined loss is applied to the main prediction and each of the four deep-supervision outputs.

## Data augmentation

Training augmentation is implemented directly in PyTorch and applied on the GPU after batch transfer.

Default augmentation probabilities are:

| Augmentation | Probability | Configuration |
| --- | ---: | --- |
| Random flips | 0.5 per spatial axis | 2D or 3D axis matched |
| Affine transformation | 0.3 | Rotation up to ±10 degrees, scale 0.9 to 1.1 |
| Elastic deformation | 0.2 | 7 control points, maximum displacement 7 voxels |
| Smooth bias field | 0.3 | Multiplicative intensity field |
| Gaussian noise | 0.2 | Standard deviation sampled from 0 to 0.1 |

MRI data use bilinear or trilinear interpolation for spatial transforms. Segmentation labels use nearest-neighbour interpolation.

Disable all training augmentation with:

```bash
python brats_pipeline.py --model unet3d --no_model_prompt --no_augment
```

## Training defaults

Important defaults include:

| Setting | Default |
| --- | --- |
| Optimiser | AdamW |
| Learning rate | `1e-4` |
| Weight decay | `1e-5` |
| Scheduler | Cosine annealing |
| Maximum epochs | 300 |
| Early stopping patience | 30 validation epochs without improvement |
| Validation interval | Every epoch |
| Seed | 123 |
| AMP | Enabled |
| Gradient clipping | Maximum norm 1.0 |
| DataLoader workers | 8 |
| CUDA required | Yes |
| cuDNN benchmark | Enabled |
| TF32 | Enabled |
| `torch.compile` | Disabled by default |

The selected checkpoint is the epoch with the best validation mean Dice.

## Automatic batch-size tuning

Automatic batch-size tuning is enabled by default. It performs real forward, backward, and optimiser probes and selects the largest stable physical batch that satisfies the configured CUDA memory policy.

Important defaults:

```text
Target VRAM fraction: 0.85
Minimum free VRAM headroom: 1 GB
2D starting batch size: 64
3D starting batch size: 1
UNet2D hard cap: 64
Diff-UNet hard cap: 1
DeepEnsemble member hard cap: 1
```

The tuner performs a final forced-augmentation safety check and repeated memory-stability checks. Wall-clock timing variability is diagnostic only and does not cause batch-size rejection.

Selected batch sizes are cached in:

```text
runs/autobatch_cache.json
```

Disable automatic tuning with:

```bash
python brats_pipeline.py \
  --model unet3d \
  --no_model_prompt \
  --no_auto_batch \
  --batch_size 1
```

## Memory-mapped preprocessing cache

By default, NIfTI files are normalised once and converted into compact NumPy memory maps.

Default location:

```text
<data_dir>/.brats_preprocessed_cache/
```

Per patient, the main cache contains:

```text
image_f16.npy
label_u8.npy
foreground_by_z.npy
```

The canonical 3D cache is stored at padded geometry `240 x 240 x 160`.

For supported 2D paths, an optional axial slice-major cache is also created to reduce strided disk reads:

```text
image_axial_f16.npy
label_axial_u8.npy
```

The cache is disk-backed. Worker-local mmap handles and the operating-system file cache are used instead of loading the complete cohort into Python RAM.

Disable preprocessing cache creation with:

```bash
python brats_pipeline.py \
  --model unet3d \
  --no_model_prompt \
  --no_preprocessed_cache
```

## Model-specific behaviour

### UNet2D

`unet2d` processes native axial slices of shape:

```text
4 x 240 x 240
```

The corrected default retains all native axial training slices:

```text
unet2d_skip_empty_ratio = 0.0
```

The final held-out evaluation reconstructs each complete 3D patient before computing patient-level Dice and HD95.

### HVU / DenseVU-ED

`hvu` receives the same 240 x 240 axial slices but internally resizes them to:

```text
256 x 256
```

Its bottleneck combines:

- U-Net local features
- DenseNet121 architectural features
- Vision Transformer global features

The DenseNet branch is initialised without pretrained weights.

### DeepLabV3+

`deeplabv3plus2d` uses:

- a ResNet-50 style encoder
- output stride 16
- atrous spatial pyramid pooling with rates 6, 12, and 18
- low-level feature fusion in the decoder

It is trained from scratch.

### HybridAttUNet

`hybridattunet` accepts the common padded full-volume input but internally resizes it to:

```text
128 x 128 x 128
```

The implementation includes:

- residual bottleneck blocks
- four residual-attention skip modules
- attention depths 1, 2, 3, and 4
- squeeze-excitation recalibration in the decoder

Logits are resized back to the pipeline input geometry before the common loss and evaluation.

### Diff-UNet

`diff_unet` is a four-class adaptation of Diff-UNet.

Training uses:

- a separate 3D MRI image encoder
- one-hot segmentation states mapped to `[-1, 1]`
- random diffusion timesteps from `T = 1000`
- START_X parameterisation
- the same Dice plus categorical cross-entropy objective used by the other architectures

Inference uses deterministic DDIM-style sampling with:

```text
50 denoising steps
```

Because of its full-volume memory requirement, its physical batch size is capped at 1 by default and asynchronous CUDA batch prefetch is disabled.

### DeepEnsembled U-Net

`deepensemble` trains five Henry-style 3D U-Net members sequentially using seeds:

```text
123, 124, 125, 126, 127
```

Each member uses:

- width 48
- GroupNorm
- a dilated pseudo-fifth stage
- trilinear decoder upsampling
- four deep-supervision outputs
- activation checkpointing during training

At inference, probabilities are averaged across all members. With default TTA enabled, each member is evaluated using 16 transformations:

```text
5 members x 16 TTA predictions = 80 predictions per patient
```

Disable DeepEnsemble TTA with:

```bash
python brats_pipeline.py \
  --model deepensemble \
  --no_model_prompt \
  --no_deepensemble_tta
```

## Evaluation metrics

The pipeline reports patient-level metrics for:

- Whole tumour, WT
- Tumour core, TC
- Enhancing tumour, ET

### Dice coefficient

Dice is calculated from the reconstructed region masks.

The reported overall Dice is:

```text
DiceMean = mean(DiceWT, DiceTC, DiceET)
```

### HD95

HD95 is computed from the symmetric set of nearest surface distances and reported in voxel units.

Empty-mask handling is:

```text
Prediction empty and target empty -> 0
Only one mask empty              -> NaN
Both masks non-empty             -> HD95 calculated
```

NaN values are excluded from summary means.

The reported overall HD95 is the mean of the valid WT, TC, and ET HD95 values for each patient.

## Computational metrics

The fixed-split experiment also reports:

| Metric | Meaning |
| --- | --- |
| `Training time (s)` | Total wall-clock training time to the selected stopping point |
| `Trainable Params` | Number of trainable model parameters |
| `GFLOPs` | Complete patient-level inference workload, multiply-add counted as 2 FLOPs |
| `Dice/s` | Mean test Dice divided by inference time in seconds |
| `Dice/M` | Mean test Dice divided by trainable parameters in millions |
| `Inference (ms)` | Mean patient-level forward inference latency after warm-up |
| `Peak VRAM (GB)` | Maximum allocated CUDA memory observed during training |

For 2D architectures, per-slice FLOPs are accumulated over all slices evaluated for the patient.

For Diff-UNet, GFLOPs include all 50 denoising evaluations.

For the DeepEnsembled U-Net, GFLOPs and latency include all ensemble members and all enabled TTA transformations.

## Installation

### Required packages

The core pipeline requires:

```text
numpy
torch
nibabel
scipy
```

Additional packages used by specific features are:

```text
torchvision    # required for HVU / DenseVU-ED
tqdm           # optional live progress bars
tensorboard    # optional TensorBoard logging
psutil         # optional host-memory reporting
```

A typical environment can be prepared with:

```bash
pip install numpy nibabel scipy tqdm tensorboard psutil
```

Install PyTorch and torchvision separately using the build appropriate for your GPU and CUDA environment.

For an RTX 5090 or another Blackwell-class GPU, the PyTorch build must include support for compute capability 12.0 (`sm_120` or `compute_120`). The script performs a CUDA allocation and matrix-multiplication sanity check before dataset preparation begins.

## Usage

### List available models

```bash
python brats_pipeline.py --list_models
```

### Interactive training

Training mode opens a numbered model menu by default:

```bash
python brats_pipeline.py --data_dir /path/to/BraTS2020
```

Even if `--model` is supplied, the interactive menu remains the default unless `--no_model_prompt` is also used.

### Non-interactive training

For scripted runs, supply both `--model` and `--no_model_prompt`:

```bash
python brats_pipeline.py \
  --mode train \
  --model unet3d \
  --no_model_prompt \
  --data_dir /path/to/BraTS2020 \
  --save_dir ./runs
```

Example for UNet2D:

```bash
python brats_pipeline.py \
  --model unet2d \
  --no_model_prompt \
  --data_dir /path/to/BraTS2020 \
  --epochs 300
```

Example for Diff-UNet:

```bash
python brats_pipeline.py \
  --model diff_unet \
  --no_model_prompt \
  --data_dir /path/to/BraTS2020
```

Example for the DeepEnsembled U-Net:

```bash
python brats_pipeline.py \
  --model deepensemble \
  --no_model_prompt \
  --data_dir /path/to/BraTS2020
```

### Explicit labelled training directory

If automatic cohort detection is ambiguous:

```bash
python brats_pipeline.py \
  --model hybridattunet \
  --no_model_prompt \
  --data_dir /path/to/BraTS2020 \
  --train_data_dir /path/to/BraTS2020/BraTS20_Training
```

### Resume training

```bash
python brats_pipeline.py \
  --model unet3d \
  --no_model_prompt \
  --data_dir /path/to/BraTS2020 \
  --checkpoint runs/unet3d_YYYYMMDD_HHMMSS_fixed_80_10_10/last.pth
```

### Standalone evaluation

A checkpoint can be loaded with:

```bash
python brats_pipeline.py \
  --mode eval \
  --model unet3d \
  --data_dir /path/to/BraTS2020 \
  --checkpoint /path/to/best.pth
```

The main fixed-split training workflow already performs the final held-out test evaluation automatically after selecting the best validation checkpoint.

The standalone `Evaluator` path currently constructs a validation split from `--data_dir`, `--val_ratio`, and `--seed`. It should therefore not be confused with the one-time fixed 10% final-test evaluation performed automatically by `FixedSplitRunner`. The standalone evaluator also uses the volumetric dataset path directly, so the automatic fixed-split evaluation is the preferred evaluation route for the 2D architectures.

## Optional cross-validation mode

The file retains an optional k-fold workflow:

```bash
python brats_pipeline.py \
  --model unet3d \
  --no_model_prompt \
  --data_dir /path/to/BraTS2020 \
  --cv \
  --n_folds 10 \
  --test_ratio 0.10
```

This is separate from the default fixed 80/10/10 experiment and is not required for the main workflow.

## Output files

The default `save_dir` is:

```text
./runs
```

### Shared experiment outputs

Across model runs, the pipeline maintains:

```text
runs/
├── fixed_split.json
├── autobatch_cache.json
├── run.log
├── segmentation_metrics.csv
├── efficiency_metrics.csv
├── model_metrics.csv
├── final_test_per_patient_<model>.csv
└── final_test_summary_<model>.csv
```

`segmentation_metrics.csv` contains one row per model with:

```text
Model
DiceCoef
HD95
DiceET
DiceWT
DiceTC
```

`efficiency_metrics.csv` contains:

```text
Model
Batch Size
Training time (s)
Trainable Params
GFLOPs
Dice/s
Dice/M
Inference (ms)
Peak VRAM (GB)
```

`model_metrics.csv` combines both sets of columns.

The per-patient final-test files are intended for paired architecture comparisons and statistical analysis.

### Per-run outputs

Each training run also creates a timestamped directory such as:

```text
runs/unet3d_YYYYMMDD_HHMMSS_fixed_80_10_10/
```

Typical contents include:

```text
config.json
training_history.csv
best.pth
last.pth
fixed_split_experiment_summary.json
tb/
```

For the DeepEnsembled U-Net, each member receives its own run directory and the final ensemble directory also stores:

```text
best.pth
ensemble_members.json
```

## Reproducibility notes

- Default random seed is `123`.
- The fixed patient split is persisted and reused across models.
- The official unlabelled BraTS2020 validation cohort is excluded from the main experiment.
- Validation is performed every epoch by default.
- Early stopping is based on validation mean Dice.
- Final test evaluation occurs only after model selection.
- All models use the same four-class formulation and common combined loss.
- Data augmentation is disabled during validation and testing.
- DeepEnsemble members use consecutive seeds starting from the configured base seed.
- `torch.compile` is disabled by default because repeated compilation and autotuning can consume substantial host memory on Windows.

## Hardware and performance notes

The implementation includes several optimisations intended for large BraTS workloads on CUDA GPUs:

- AMP
- fused AdamW where supported
- TF32 where supported
- cuDNN benchmarking
- channels-last 3D memory format for selected CNNs
- pinned-memory DataLoaders
- asynchronous CUDA prefetch where memory permits
- memory-mapped preprocessing
- patient-grouped 2D batches
- RAM-bounded DataLoader prefetch
- activation checkpointing for DeepEnsemble members
- explicit worker and CUDA cleanup at process exit

CUDA is required by default to prevent accidental multi-hour CPU training. CPU execution must be explicitly enabled with:

```bash
--allow_cpu
```

Full-volume 3D training on CPU is not expected to be practical.

## Important interpretation notes

This is a controlled comparison pipeline, not an attempt to reproduce the individually optimised performance of every source architecture.

Several architectures have been adapted to a common experimental setting:

- four mutually exclusive output classes
- common Dice plus categorical cross-entropy loss
- fixed patient split
- common augmentation framework
- common model selection procedure
- full-volume BraTS geometry for the 3D comparison

Accordingly, results should be interpreted as the behaviour of these implementations under the shared protocol rather than as exact reproductions of the original published systems.

In particular:

- HybridAttUNet is an architecture-faithful PyTorch reimplementation rather than a bit-for-bit reproduction of unpublished source code.
- Diff-UNet is adapted to four-class multi-class segmentation and the common study loss.
- DeepEnsemble retains major Henry et al. architectural and inference elements but uses the study's fixed split and four-class formulation.
- HVU / DenseVU-ED is implemented from the architectural description and uses no pretrained DenseNet weights.

## References represented in the implementation

The source code explicitly draws on or adapts ideas from:

- Ronneberger et al., U-Net
- Çiçek et al., 3D U-Net
- Renugadevi et al., DenseVU-ED / Hybrid Vision U-Net
- Hybrid Attention-Based Residual U-Net
- Henry et al., BraTS2020 deep ensemble
- Xing et al., Diff-UNet
- DeepLabV3+

Consult the associated papers and repositories when using the models for research reporting.

## Licence

No licence is declared by this pipeline file itself. Before redistributing the code or model implementations, ensure that your repository licence is compatible with the licences of any source projects, dependencies, and datasets on which the implementations are based.

BraTS data are distributed separately and are not included in this repository.
