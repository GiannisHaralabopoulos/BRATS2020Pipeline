# Unified BraTS2020 Brain Tumour Segmentation Pipeline

This repository contains the code and archived outputs used for a controlled comparison of seven brain tumour segmentation architectures on the labelled BraTS2020 cohort, together with dedicated HybridAttUNet ablation and pipeline-sensitivity analyses.

This README describes repository commit:

```text
0bced1abbbc92b663092e3d2fbd23d1a4a6b8dac
```

The commit adds the standalone `Ablation.py` and `Sensitivity.py` experiment scripts to the main benchmark pipeline and retains the archived patient-level results, fixed patient split, run summaries, training histories, and locked Python requirements.

## Repository overview

The repository contains three executable experiment scripts:

| File | Purpose |
| --- | --- |
| `brats_pipeline.py` | Main seven-architecture benchmark |
| `Ablation.py` | Controlled HybridAttUNet factorial ablation |
| `Sensitivity.py` | HybridAttUNet pipeline-sensitivity experiments |

The main benchmark compares:

| CLI name | Architecture | Dimensionality | Approx. trainable parameters | Key implementation details |
| --- | --- | ---: | ---: | --- |
| `unet2d` | UNet | 2D | 7.76 M | Conventional four-level U-Net |
| `hvu` | Hybrid UNet / DenseVU-ED | 2D | 36.51 M | U-Net branch with DenseNet121 and Vision Transformer features |
| `deeplabv3plus2d` | DeepLabV3+ | 2D | 40.35 M | ResNet-50 style encoder, ASPP, output stride 16 |
| `diff_unet` | DiffUNet | 3D | 10.05 M | Separate MRI encoder, START_X diffusion training, 50-step DDIM-style inference |
| `hybridattunet` | HybridAttUNet | 3D | 14.10 M | Residual attention modules and squeeze-excitation decoder |
| `unet3d` | 3DUNet | 3D | 22.58 M | Conventional volumetric U-Net |
| `deepensemble` | DeepEnsembled UNet | 3D | 115.78 M total | Five independently trained 3D members with deep supervision and 16-way TTA |

The DeepEnsembled UNet trains its five members sequentially. Peak training VRAM therefore reflects the largest individual member rather than all five models being resident simultaneously.

## Main experimental design

The reported benchmark uses only the labelled BraTS2020 training cohort.

The 369 labelled cases are divided once at patient level into:

```text
295 training patients
37 validation patients
37 final held-out test patients
```

This corresponds to a fixed:

```text
80% training
10% validation
10% final test
```

The identical split is used across all main architectures and secondary experiments. The exact patient allocation is archived at the repository root as:

```text
fixed_split.json
```

The validation subset is used for model selection, early stopping, and best-checkpoint selection. The held-out test subset is evaluated only after model selection.

The official BraTS2020 validation cohort is not used because it does not contain ground-truth segmentation masks.

## Dataset structure

Each labelled patient directory must contain the four MRI modalities and a segmentation mask:

```text
BraTS2020/
└── BraTS20_Training_001/
    ├── BraTS20_Training_001_flair.nii.gz
    ├── BraTS20_Training_001_t1.nii.gz
    ├── BraTS20_Training_001_t1ce.nii.gz
    ├── BraTS20_Training_001_t2.nii.gz
    └── BraTS20_Training_001_seg.nii.gz
```

The pipeline searches recursively for patient folders containing `*_flair.nii*`.

If `--data_dir` points to a parent directory containing several BraTS folders, the labelled training cohort is auto-detected where possible. To remove ambiguity, use `--train_data_dir` explicitly.

## MRI preprocessing

The four input modalities are FLAIR, T1, T1ce, and T2.

Each modality is independently z-score normalised over non-zero voxels. Background voxels remain zero.

Native BraTS2020 geometry is:

```text
240 x 240 x 155
```

For the 3D models, depth is padded to 160 to remain compatible with repeated downsampling:

```text
4 x 240 x 240 x 160
```

No spatial cropping is applied to the common full-volume input.

### Label mapping

BraTS labels are converted into four mutually exclusive softmax classes:

| Pipeline class | Meaning | Original BraTS label |
| ---: | --- | ---: |
| 0 | Background | 0 |
| 1 | NCR/NET | 1 |
| 2 | Oedema | 2 |
| 3 | Enhancing tumour | 4 |

The standard overlapping BraTS evaluation regions are reconstructed as:

```text
WT = classes 1 + 2 + 3
TC = classes 1 + 3
ET = class 3
```

## Common loss

All main architectures use:

```text
Loss = 0.5 x DiceLoss + 0.5 x CrossEntropyLoss
```

The Dice component operates on softmax probabilities, excludes background, averages over the three foreground classes, and uses smoothing of `1e-5`.

Categorical cross-entropy is computed from the four-class logits and integer class labels.

For the DeepEnsembled UNet, the same combined loss is applied to the main prediction and each of the four auxiliary deep-supervision outputs.

## Data augmentation

Training augmentation is implemented using native PyTorch operations and applied on the GPU after batch transfer.

| Augmentation | Probability | Configuration |
| --- | ---: | --- |
| Random flips | 0.5 per available spatial axis | 2D or 3D matched |
| Affine transformation | 0.3 | Rotation up to +/-10 degrees, scale 0.9 to 1.1 |
| Elastic deformation | 0.2 | 7 control points, maximum displacement 7 voxels |
| Smooth bias field | 0.3 | Multiplicative intensity field |
| Gaussian noise | 0.2 | Standard deviation sampled from 0 to 0.1 |

Spatial transforms use bilinear or trilinear interpolation for MRI intensities and nearest-neighbour interpolation for segmentation labels.

Augmentation is applied only during training.

## Training defaults

| Setting | Default |
| --- | --- |
| Optimiser | AdamW |
| Learning rate | `1e-4` |
| Weight decay | `1e-5` |
| Scheduler | Cosine annealing |
| Maximum epochs | 300 |
| Early stopping patience | 30 validation epochs without improvement |
| Validation interval | Every epoch |
| Main seed | 123 |
| AMP | Enabled |
| Gradient clipping | Maximum norm 1.0 |
| DataLoader workers | 8 |
| CUDA required | Yes |
| cuDNN benchmark | Enabled |
| TF32 | Enabled |
| `torch.compile` | Disabled by default |

The checkpoint with the highest validation mean Dice is retained for final evaluation.

## Automatic batch-size tuning

Automatic batch-size tuning is enabled by default. The tuner performs forward, backward, and optimiser probes and selects the largest stable physical batch satisfying the configured CUDA memory policy.

Important defaults include:

```text
Target VRAM fraction: 0.85
Minimum free VRAM headroom: 1 GB
2D starting batch size: 64
3D starting batch size: 1
UNet2D hard cap: 64
DiffUNet hard cap: 1
DeepEnsemble member hard cap: 1
```

Selected batch sizes can be cached in:

```text
runs/autobatch_cache.json
```

To disable automatic tuning:

```bash
python brats_pipeline.py   --model unet3d   --no_model_prompt   --no_auto_batch   --batch_size 1
```

## Memory-mapped preprocessing cache

By default, NIfTI files are normalised once and converted into compact NumPy memory maps.

Default location:

```text
<data_dir>/.brats_preprocessed_cache/
```

Per patient, the main cache includes:

```text
image_f16.npy
label_u8.npy
foreground_by_z.npy
```

For supported 2D paths, an axial slice-major cache can also be created:

```text
image_axial_f16.npy
label_axial_u8.npy
```

The cache is disk-backed and avoids loading the complete cohort into Python RAM.

## Model-specific behaviour

### UNet

`unet2d` operates on native axial slices of shape:

```text
4 x 240 x 240
```

The final configuration retains all axial training slices, including tumour-free slices.

During validation and testing, slice predictions are reassembled in their original axial order to reconstruct each complete patient volume before Dice and HD95 are calculated.

### Hybrid UNet / DenseVU-ED

`hvu` receives 240 x 240 axial slices and internally resizes them to:

```text
256 x 256
```

Its bottleneck combines local U-Net features, DenseNet121 architectural features, and Vision Transformer features. The DenseNet branch is not initialised with pretrained weights.

During training, approximately 90% of tumour-free slices are excluded by the retained architecture-specific sampling procedure.

### DeepLabV3+

`deeplabv3plus2d` uses a ResNet-50 style encoder, output stride 16, atrous spatial pyramid pooling with rates 6, 12, and 18, and low-level feature fusion in the decoder. It is trained from scratch.

During training, approximately 90% of tumour-free slices are excluded by the retained sampling procedure.

### HybridAttUNet

`hybridattunet` accepts the padded full-volume input and internally resizes it to:

```text
128 x 128 x 128
```

The implementation contains residual blocks, four residual-attention skip modules, and squeeze-excitation recalibration in the decoder.

Logits are resized back to the common pipeline geometry before the common loss and evaluation procedure.

### DiffUNet

`diff_unet` is a four-class adaptation of DiffUNet to the common experimental framework.

Training uses a separate 3D MRI encoder, one-hot segmentation states mapped to `[-1, 1]`, diffusion timesteps sampled from `T = 1000`, START_X parameterisation, and the same Dice plus categorical cross-entropy objective used by the other models.

Inference uses deterministic DDIM-style sampling with:

```text
50 denoising steps
```

MRI conditioning features are computed once and reused across the denoising sequence.

The implementation should be interpreted as the study-specific four-class adaptation evaluated in this repository rather than as an exact reproduction of the original published system.

### 3DUNet

`unet3d` is a conventional full-volume 3D U-Net operating on the padded 240 x 240 x 160 volume.

### DeepEnsembled UNet

`deepensemble` trains five independently initialised 3D members sequentially using seeds:

```text
123, 124, 125, 126, 127
```

Each member uses deep supervision during optimisation.

At inference, each member is evaluated with 16 test-time transformations and probabilities are averaged across transformations and members:

```text
5 members x 16 TTA predictions = 80 model evaluations per patient
```

No SWA weight averaging or warm-restart ensemble procedure is used in the reported implementation.

## Evaluation metrics

All final segmentation metrics are calculated at patient level.

### Dice

Dice is reported for WT, TC, and ET.

The overall patient-level mean is:

```text
DiceMean = (DiceWT + DiceTC + DiceET) / 3
```

The implementation uses smoothing of `1e-5`.

```text
Prediction empty and target empty -> Dice = 1
Only one mask empty              -> Dice approaches 0
```

### HD95

HD95 is the symmetric 95th percentile of the pooled nearest-surface distances in both directions.

The implementation uses SciPy Euclidean distance transforms and reports distances in voxel units. Because the BraTS2020 images are at 1 mm isotropic resolution, one voxel corresponds to 1 mm in the reported dataset geometry.

```text
Prediction empty and target empty -> 0
Only one mask empty              -> NaN
Both masks non-empty             -> symmetric HD95 calculated
```

NaN values are excluded from the corresponding HD95 summary mean.

For the 2D architectures, final Dice and HD95 are not calculated slice by slice. All axial predictions are first reassembled into the complete 3D patient volume.

## Computational metrics

| Metric | Definition |
| --- | --- |
| `Training time (s)` | Total wall-clock training time to the selected stopping point, including validation |
| `Trainable Params` | Number of trainable parameters |
| `GFLOPs` | Estimated complete patient-level forward inference workload |
| `Dice/s` | Mean patient Dice divided by mean patient inference time in seconds |
| `Dice/M` | Mean patient Dice divided by trainable parameters in millions |
| `Inference (ms)` | Mean patient-level model forward latency after warm-up |
| `Peak VRAM (GB)` | Maximum PyTorch allocated CUDA memory observed during training |

### GFLOPs implementation

GFLOPs are estimated using custom PyTorch runtime forward hooks with profiling batch size 1.

The implementation counts:

```text
Conv2d
Conv3d
ConvTranspose2d
ConvTranspose3d
Linear
MultiheadAttention
```

A multiply and an addition are counted as two FLOPs. Bias additions are included.

The estimate covers forward inference only. Backward propagation is not included.

Operations outside the explicitly hooked module types, including activation functions, normalisation, pooling, and softmax, are not included.

For 2D architectures, one axial input is profiled and the operation count is scaled to complete patient-level inference.

For 3D architectures, one complete padded patient volume is profiled.

DiffUNet GFLOPs include the complete 50-step denoising procedure.

DeepEnsembled UNet GFLOPs are scaled across all five members and all 16 TTA transformations per member.

### Inference timing

Inference timing uses CUDA events around the model forward operation.

The benchmark performs three untimed warm-up evaluations for the single models. The DeepEnsembled UNet uses one complete untimed ensemble warm-up evaluation.

Input transfer to the GPU occurs before the timed forward operation. CUDA is synchronised before elapsed time is read.

The timed interval excludes data loading, preprocessing, host-to-device transfer, prediction transfer to CPU, 2D volumetric reconstruction, Dice calculation, HD95 calculation, and file writing.

For 2D models, forward times of all slice batches are accumulated and converted to mean patient-level latency.

Peak VRAM uses `torch.cuda.max_memory_allocated()` and should be interpreted as peak PyTorch allocated training memory, not as the minimum physical GPU capacity required for execution.

## HybridAttUNet ablation study

`Ablation.py` contains the controlled factorial ablation used to isolate the contributions of attention and residual connections.

| CLI name | Attention | Residual blocks | Description |
| --- | --- | --- | --- |
| `hybridattunet` | Yes | Yes | Full reference HybridAttUNet |
| `hybridattunet_no_attention` | No | Yes | Attention mechanisms removed |
| `hybridattunet_no_residual` | Yes | No | Residual shortcuts removed from convolutional blocks |
| `hybridattunet_no_attention_no_residual` | No | No | Attention and residual shortcuts removed |

Run all four configurations sequentially on the same fixed patient split:

```bash
python Ablation.py   --run_all_hybrid_ablation   --data_dir /path/to/BraTS2020   --save_dir ./runs/ablation
```

Combined metrics are written to:

```text
hybridattunet_ablation_metrics.csv
```

An individual configuration can be run explicitly, for example:

```bash
python Ablation.py   --model hybridattunet_no_attention   --no_model_prompt   --data_dir /path/to/BraTS2020   --save_dir ./runs/ablation_no_attention
```

### Archived ablation outputs

The reported ablation outputs are under:

```text
hybridattunet/ablation/
```

This includes:

```text
hybridattunet_ablation_metrics.csv
final_test_per_patient_hybridattunet_no_attention.csv
final_test_per_patient_hybridattunet_no_residual.csv
final_test_per_patient_hybridattunet_no_attention_no_residual.csv
final_test_summary_hybridattunet_no_attention.csv
final_test_summary_hybridattunet_no_residual.csv
final_test_summary_hybridattunet_no_attention_no_residual.csv
```

Run-specific subdirectories are also archived for:

```text
no_attention/
no_residual/
no_attention_no_residual/
```

The full reference HybridAttUNet run is archived in the parent `hybridattunet/` directory and in `results/`.

## HybridAttUNet sensitivity study

`Sensitivity.py` defines three executable pipeline-sensitivity conditions:

| CLI name | Spatial condition | Augmentation |
| --- | --- | --- |
| `hybridatt_lowres` | 96^3 effective spatial information resampled to the unchanged 128^3 working grid | On |
| `hybridatt_highres` | 160^3 HybridAttUNet internal working grid | On |
| `hybridatt_ref_noaug` | Reference 128^3 internal grid | Off |

The low-resolution condition preserves the reference network topology. The incoming volume is first resampled to 96^3, removing spatial information, then resampled to the unchanged 128^3 working grid before entering the reference architecture.

Run all three executable sensitivity conditions sequentially:

```bash
python Sensitivity.py   --run_all_sensitivity   --data_dir /path/to/BraTS2020   --save_dir ./runs/sensitivity
```

The script writes:

```text
hybridattunet_sensitivity_manifest.json
hybridattunet_sensitivity_metrics.csv
```

An individual condition can be run explicitly, for example:

```bash
python Sensitivity.py   --model hybridatt_ref_noaug   --no_model_prompt   --data_dir /path/to/BraTS2020   --save_dir ./runs/sensitivity_noaug
```

### Archived sensitivity outputs

The archived manuscript sensitivity outputs are under:

```text
hybridattunet/sensitivity/
```

The patient-level and summary files present in this commit are:

```text
final_test_per_patient_hybridatt_lowres.csv
final_test_summary_hybridatt_lowres.csv
final_test_per_patient_hybridatt_ref_noaug.csv
final_test_summary_hybridatt_ref_noaug.csv
hybridattunet_sensitivity_metrics.csv
```

Run-specific folders are archived as:

```text
low_res/
no_augmentation/
```

Important: `Sensitivity.py` supports `hybridatt_highres`, but commit `0bced1abbbc92b663092e3d2fbd23d1a4a6b8dac` does not contain an archived high-resolution patient-level or summary result. The archived manuscript sensitivity evidence in this commit consists of the low-resolution and no-augmentation conditions together with the reference HybridAttUNet result.

## Installation

The repository contains:

```text
requirements-lock.txt
```

Install the recorded Python packages with:

```bash
pip install -r requirements-lock.txt
```

The lock file includes:

```text
torch==2.13.0+cu132
torchvision==0.28.0+cu132
```

The pipeline requires a CUDA-capable PyTorch environment by default. CPU execution must be explicitly enabled with `--allow_cpu`, but full-volume 3D training on CPU is not expected to be practical.

## Main benchmark usage

List the available main models:

```bash
python brats_pipeline.py --list_models
```

Interactive training:

```bash
python brats_pipeline.py --data_dir /path/to/BraTS2020
```

For scripted training, supply both `--model` and `--no_model_prompt`:

```bash
python brats_pipeline.py   --mode train   --model unet3d   --no_model_prompt   --data_dir /path/to/BraTS2020   --save_dir ./runs
```

DiffUNet example:

```bash
python brats_pipeline.py   --model diff_unet   --no_model_prompt   --data_dir /path/to/BraTS2020
```

DeepEnsembled UNet example:

```bash
python brats_pipeline.py   --model deepensemble   --no_model_prompt   --data_dir /path/to/BraTS2020
```

### Resume training

```bash
python brats_pipeline.py   --model unet3d   --no_model_prompt   --data_dir /path/to/BraTS2020   --checkpoint runs/unet3d_YYYYMMDD_HHMMSS_fixed_80_10_10/last.pth
```

### Standalone evaluation

```bash
python brats_pipeline.py   --mode eval   --model unet3d   --data_dir /path/to/BraTS2020   --checkpoint /path/to/best.pth
```

The preferred manuscript-reproduction path is the fixed-split training workflow, which performs the held-out test evaluation automatically after checkpoint selection.

## Optional legacy cross-validation mode

The main pipeline retains an optional k-fold workflow:

```bash
python brats_pipeline.py   --model unet3d   --no_model_prompt   --data_dir /path/to/BraTS2020   --cv   --n_folds 10   --test_ratio 0.10
```

This mode is separate from the fixed 80/10/10 experiment and was not used for the reported manuscript results.

The `--run_all_hybrid_ablation` and `--run_all_sensitivity` workflows are designed for the fixed-split experiment and reject legacy cross-validation mode.

## Archived reproducibility material

At commit `0bced1abbbc92b663092e3d2fbd23d1a4a6b8dac`, the repository root contains:

```text
README.md
brats_pipeline.py
Ablation.py
Sensitivity.py
fixed_split.json
requirements-lock.txt
model_metrics.csv
efficiency_metrics.csv
results/
unet/
hybridunet/
deeplabsv3/
diff_unet/
hybridattunet/
unet3D/
deepensemble/
```

### Main patient-level results

The `results/` directory contains held-out patient-level results for all seven main architectures:

```text
final_test_per_patient_unet2d.csv
final_test_per_patient_hvu.csv
final_test_per_patient_deeplabv3plus2d.csv
final_test_per_patient_diff_unet.csv
final_test_per_patient_hybridattunet.csv
final_test_per_patient_unet3d.csv
final_test_per_patient_deepensemble.csv
```

Corresponding aggregate files are stored as:

```text
final_test_summary_<model>.csv
```

These patient-level files contain the Dice and HD95 values used for the reported descriptive summaries and paired statistical comparisons.

### Main run artefacts

Where archived, architecture directories contain structured run evidence such as:

```text
config.json
training_history.csv
run_metrics.json
fixed_split_experiment_summary.json
```

`config.json` records the resolved run configuration.

`training_history.csv` contains epoch-level training and validation metrics together with timing, memory, and GPU telemetry recorded during the run.

`run_metrics.json` records run-level quantities such as training duration, best validation epoch, completed epochs, peak allocated VRAM, validation performance, and stopping reason.

`fixed_split_experiment_summary.json` records fixed-split patient counts, checkpoint information, validation performance, and held-out test performance.

For the DeepEnsembled UNet, equivalent material is archived for the five independently trained members.

Final Hybrid UNet / HVU patient-level and summary results are present in `results/`, but a corresponding run-specific HVU archive directory is not present in this commit.

## Output files created by new runs

The default output directory is:

```text
./runs
```

Typical shared outputs include:

```text
fixed_split.json
autobatch_cache.json
run.log
segmentation_metrics.csv
efficiency_metrics.csv
model_metrics.csv
final_test_per_patient_<model>.csv
final_test_summary_<model>.csv
```

Typical per-run outputs include:

```text
config.json
training_history.csv
best.pth
last.pth
fixed_split_experiment_summary.json
tb/
```

Model checkpoints and TensorBoard event files generated locally are not necessarily included in the repository snapshot.

## Reproducibility notes

- The reported primary experiments use one fixed 80/10/10 patient-level split rather than cross-validation.
- The exact split is archived as `fixed_split.json`.
- The default primary seed is 123.
- DeepEnsembled UNet members use seeds 123 to 127.
- Validation is performed after every epoch.
- Early stopping patience is 30 epochs.
- The held-out test subset is evaluated only after model selection.
- The official unlabelled BraTS2020 validation cohort is not used.
- All main models use the same four-class formulation and combined Dice plus categorical cross-entropy loss.
- Training augmentation is applied only to the training subset.
- Patient-level outputs are available for paired statistical analysis.
- `requirements-lock.txt` records the Python package versions available in this snapshot.
- `Ablation.py` and `Sensitivity.py` were added in commit `0bced1abbbc92b663092e3d2fbd23d1a4a6b8dac`.
- Optional cross-validation functionality remains in the source but was not used for the reported manuscript results.
- `torch.compile` is disabled by default.

## Hardware and interpretation

The reported computational measurements were obtained on a single NVIDIA RTX 5090 with 32 GB VRAM.

Training time, inference latency, and memory utilisation depend on model structure, tensor geometry, software implementation, CUDA kernels, and accelerator characteristics. The observed computational ordering should therefore be interpreted as specific to the reported benchmark configuration rather than assumed to transfer unchanged to other hardware.

The repository does not provide direct profiling of memory bandwidth, arithmetic intensity, or kernel occupancy. The computational comparison is based on measured training time, forward inference latency, peak allocated GPU memory, parameter count, and estimated forward GFLOPs.

The repository also does not establish deployment performance on commodity GPUs, CPUs, or other resource-constrained clinical hardware.

## Scope of the implementations

This repository is intended for a controlled within-pipeline comparison. It is not an attempt to reproduce the individually optimised training pipelines of every source architecture.

Several architectures are adapted to the common experimental setting, including the common four-class output formulation, common loss, fixed patient split, common augmentation framework, common model-selection procedure, and full-volume BraTS geometry for the 3D comparison.

Accordingly, results should be interpreted as the behaviour of these implementations under the shared protocol.

In particular:

- HybridAttUNet is a PyTorch implementation based on the reported architecture rather than a bit-for-bit reproduction of unavailable source code.
- DiffUNet is adapted to four-class segmentation and the common study loss and uses 50-step deterministic DDIM-style inference.
- DeepEnsembled UNet uses the study's independently trained members, deep supervision, fixed split, and TTA inference procedure.
- Hybrid UNet / DenseVU-ED is implemented from the architectural description and does not use pretrained DenseNet weights.

## References represented in the implementation

The source code draws on or adapts ideas from:

- Ronneberger et al., U-Net
- Cicek et al., 3D U-Net
- Renugadevi et al., DenseVU-ED / Hybrid Vision U-Net
- Khan et al., Hybrid Attention-Based Residual U-Net
- Henry et al., BraTS2020 deep ensemble
- Xing et al., DiffUNet
- Chen et al., DeepLabV3+

Consult the corresponding publications when describing architectural provenance.

## Licence

No repository licence is declared in this commit. Before redistributing the code or model implementations, ensure that use is compatible with the licences of the underlying dependencies, source projects, and BraTS data.

BraTS data are distributed separately and are not included in this repository.
