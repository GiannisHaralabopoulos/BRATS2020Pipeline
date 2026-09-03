# BraTS2020 Unified Segmentation Pipeline

**Python script:** `brats_pipeline.py`  
**SHA-256:** `193c594f254d5cdfc1035acb03017138b808b45014ec29ffcd6a002e0761c1f7`

## Overview

This script implements a unified PyTorch pipeline for training and evaluating seven 2D and 3D brain tumour segmentation architectures on the labelled BraTS 2020 training cohort.

The default experimental design uses a single fixed patient-level 80/10/10 split of the 369 labelled BraTS 2020 subjects:

- 80% training, approximately 295 patients
- 10% validation, approximately 37 patients
- 10% final held-out testing, approximately 37 patients

The split is saved to `runs/fixed_split.json` and reused across architectures so that every model is trained, validated and tested on the same subjects. The official BraTS 2020 validation cohort is not used because ground-truth segmentation labels are unavailable.

The pipeline is designed for CUDA training and has been optimised for a 32 GB NVIDIA RTX 5090 and a 64 GB RAM Windows workstation.

---

## Models

The interactive model menu is ordered by dimensionality and trainable parameter count.

### 2D models

1. **UNet 2D**, approximately 7.85 M trainable parameters
2. **HVU 2D DenseVU-ED**, approximately 36.51 M trainable parameters
3. **DeepLabV3+ 2D**, approximately 40.35 M trainable parameters

### 3D models

4. **Diff-UNet 3D**, approximately 10.99 M trainable parameters
5. **HybridAttUnet 3D**, approximately 14.10 M trainable parameters
6. **UNet 3D**, approximately 22.58 M trainable parameters
7. **DeepEnsemble 3D**, approximately 34.71 M trainable parameters

The internal model keys are:

```text
unet2d
hvu
deeplabv3plus2d
diff_unet
hybridattunet
unet3d
deepensemble
```

---

## BraTS 2020 Inputs

The script expects patient directories containing the four MRI modalities:

```text
*_flair.nii.gz
*_t1.nii.gz
*_t1ce.nii.gz
*_t2.nii.gz
```

and a segmentation mask.

The four MRI channels are:

1. FLAIR
2. T1
3. T1ce
4. T2

### Native geometry

BraTS 2020 volumes use the native spatial geometry:

```text
240 x 240 x 155
```

with four MRI modalities.

### 2D inputs

The 2D models operate on individual axial slices:

```text
4 x 240 x 240
```

The common data input therefore corresponds to `240 x 240 x 4` when written with channels last.

The HVU DenseVU-ED model internally resizes each slice to `256 x 256` for its forward pass and returns its output to `240 x 240`.

### 3D inputs

The native depth of 155 is padded to 160 before the 3D networks are applied, giving a model-ready tensor equivalent to:

```text
4 x 240 x 240 x 160
```

or `240 x 240 x 160 x 4` when written with modalities last.

No spatial cropping is applied to the full-volume 3D inputs.

---

## Preprocessing

### MRI normalisation

Each MRI modality is independently z-score normalised over its non-zero voxels:

```text
x' = (x - mean_nonzero) / std_nonzero
```

Background voxels remain zero after normalisation.

### Multi-class label mapping

BraTS labels are remapped to a contiguous, mutually exclusive four-class representation:

| BraTS label | Internal class | Meaning |
|---|---:|---|
| 0 | 0 | Background |
| 1 | 1 | NCR/NET |
| 2 | 2 | Oedema |
| 4 | 3 | Enhancing tumour |

The code also maps an existing label value of 3 to the enhancing tumour class for compatibility with already remapped datasets.

The three standard BraTS evaluation regions are reconstructed as:

- **WT:** classes 1, 2 and 3
- **TC:** classes 1 and 3
- **ET:** class 3

---

## Preprocessed Cache and Data Loading

Normalised data can be cached as memory-mapped NumPy arrays to reduce repeated NIfTI decompression and preprocessing.

The cached formats are:

```text
MRI: float16
label: uint8
```

Runtime image tensors are converted back to `float32` on the GPU before augmentation and model execution.

The pipeline uses:

- 8 DataLoader workers
- persistent workers
- pinned-memory loading
- worker-local memory-map handle caches
- a RAM-bounded prefetch queue
- a slice-major cache for 2D axial access
- pre-padded 3D cached volumes

The 2D slice-major cache stores slices contiguously in depth-first form to avoid strided disk reads.

---

## Training Data Augmentation

Augmentation is applied **on-the-fly on the GPU** after a training batch has been transferred to CUDA and before the model forward pass.

No augmented duplicate is added to the dataset. Each training sample is transformed probabilistically each time it is encountered.

The augmentation order is:

1. random spatial flips
2. affine transformation and/or elastic deformation
3. multiplicative bias field
4. Gaussian noise

### Random flips

Applied independently along each available spatial axis:

```text
p = 0.5 per axis
```

The identical flip is applied to the MRI image and segmentation label.

### Affine augmentation

```text
p = 0.3
rotation: -10 to +10 degrees
scaling: 0.9 to 1.1 independently by spatial axis
```

For 2D models, the transform is 2D. For 3D models, independent rotations and scaling factors are sampled across the three spatial axes.

### Elastic deformation

```text
p = 0.2
control points = 7
maximum displacement = 7 voxels
```

Affine and elastic transforms are sampled independently. If both are selected, they are combined into one sampling grid.

MRI images use linear interpolation and labels use nearest-neighbour interpolation.

### Intensity bias field

```text
p = 0.3
bias strength = 0.3
```

A smooth coarse random field is interpolated to the full image dimensions, normalised, and converted to a multiplicative intensity field.

### Gaussian noise

```text
p = 0.2
sigma ~ Uniform(0, 0.1)
```

Noise is added directly to the image tensor.

Augmentation is disabled during validation and final testing.

---

## Loss Function

All models use the same multi-class composite loss:

```text
L = 0.5 * DiceLoss + 0.5 * CrossEntropyLoss
```

### Dice loss

Dice is calculated from softmax probabilities and averaged across the three foreground classes only:

- NCR/NET
- oedema
- enhancing tumour

The background class is excluded from the Dice term.

The smoothing constant is:

```text
epsilon = 1e-5
```

### Categorical cross-entropy

PyTorch `nn.CrossEntropyLoss()` is used without class-specific weights.

Cross-entropy includes all four classes, including the background.

---

## Optimisation

Default optimisation settings are:

```text
Optimizer: AdamW
Initial learning rate: 1e-4
Weight decay: 1e-5
Scheduler: cosine annealing
Maximum epochs: 300
Early-stopping patience: 30 epochs
Validation frequency: every epoch
Random seed: 123
Gradient accumulation: none
```

One physical batch produces one optimiser update.

Gradient clipping is applied with:

```text
max_norm = 1.0
```

Automatic mixed precision is enabled.

Fused AdamW is used where supported.

---

## Automatic Batch-Size Selection

Batch size is model dependent and is selected automatically according to available GPU memory.

The default target is:

```text
85% of total VRAM
```

with an additional emergency headroom floor of approximately 1 GB.

Default starting batch sizes are:

```text
2D: 64
3D: 1
```

The tuner uses real forward, backward and optimiser probes.

### Search behaviour

- UNet 2D can grow more aggressively.
- Heavy 2D models use additive batch-size increments.
- 3D models increase conservatively.
- CUDA OOM conditions are caught and treated as failed probes rather than terminating the experiment.
- If batch size 1 physically fits but exceeds the preferred 85% VRAM target, batch size 1 is accepted because no smaller physical batch exists.
- Once this minimum-batch condition is reached, the tuner does not probe batch size 2.
- The final candidate can be checked using forced worst-case GPU augmentation.
- Stable batch sizes are stored in `autobatch_cache.json`.
- A valid cached batch size is reused directly without additional probing.

---

## Diff-UNet Memory Handling

Diff-UNet receives several model-specific memory-management optimisations that do not change its architecture, input geometry, loss or augmentation distribution.

### CUDA prefetch

Asynchronous host-to-device batch prefetch is disabled for Diff-UNet so that the next full 3D batch is not simultaneously resident on the GPU.

CPU-side DataLoader prefetch remains active.

### Augmentation memory

Large 3D augmentation tensors are reused or updated in place where safe, and temporary sampling grids and displacement tensors are deleted immediately after use.

### Epoch loss accumulation

The training loop uses one running loss accumulator rather than retaining a detached CUDA loss tensor for every batch.

### CUDA timing

One CUDA event pair is used for epoch-level GPU timing instead of retaining an event pair for every batch.

### CUDA allocator cache

For Diff-UNet only:

```python
torch.cuda.empty_cache()
```

is called every 10 training batches to return unused cached CUDA blocks to the driver.

This does not remove live model tensors, gradients or optimiser state, but it can introduce a small training-time overhead.

---

## HybridAttUnet Batch-Size-1 Handling

HybridAttUnet can reach a deepest feature tensor of:

```text
[1, 512, 1, 1, 1]
```

when the physical training batch size is 1.

Standard `BatchNorm3d` cannot estimate batch statistics from a single value per channel. The implementation therefore uses a batch-size-1-safe BatchNorm fallback inside HybridAttUnet.

When more than one value per channel is available, normal BatchNorm behaviour is used. In the degenerate `1 x 1 x 1` case, the layer uses its stored running statistics.

---

## Learning-Rate Scheduling and Early Stopping

The default scheduler is cosine annealing.

Validation is performed after every epoch.

The checkpoint with the highest mean validation Dice is stored as the best model.

Training stops if the validation Dice does not improve for:

```text
30 consecutive epochs
```

The final held-out test set is evaluated only after the best checkpoint has been selected.

---

## Segmentation Metrics

Final segmentation metrics are calculated at the patient level.

For the 2D models, all axial predictions are reconstructed into a complete 3D patient volume before final metrics are calculated.

### Dice

The pipeline reports:

- DiceCoef, mean of WT, TC and ET Dice
- DiceWT
- DiceTC
- DiceET

### HD95

The implementation computes a symmetric pooled bidirectional 95th-percentile surface Hausdorff distance.

For masks `X` and `Y`, nearest-surface distances are calculated in both directions, pooled, and the 95th percentile is taken.

The implementation uses:

- binary surface extraction
- Euclidean distance transforms
- a tight union bounding box for computational efficiency
- CPU thread parallelism where appropriate

HD95 is currently calculated in voxel units.

If both masks are empty, HD95 is 0. If only one mask is empty, HD95 is reported as NaN.

---

## Efficiency Metrics

The aggregate efficiency table contains:

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

### Training time

Training time is the total wall-clock time required to train the model to its selected stopping point, including validation performed during training.

### Trainable parameters

Only parameters with `requires_grad=True` are counted.

### GFLOPs

GFLOPs represent forward inference complexity per complete patient.

- 2D models: per-slice complexity accumulated across the axial slices required for a full patient volume
- 3D models: complete volumetric forward inference
- Diff-UNet: includes its iterative inference procedure

Multiply-add operations are counted as two floating-point operations.

### Dice/s

```text
Dice/s = mean DiceCoef / inference time in seconds per patient
```

### Dice/M

```text
Dice/M = mean DiceCoef / trainable parameters in millions
```

### Inference latency

Inference latency is the mean model forward-pass latency per patient after warm-up.

It excludes:

- disk I/O
- preprocessing
- data loading
- Dice calculation
- HD95 calculation

CUDA timing events are used for GPU latency measurement.

### Peak VRAM

Peak VRAM is measured during training using:

```python
torch.cuda.max_memory_allocated()
```

The peak-memory counter is reset at the beginning of each training run.

The value represents total peak allocated GPU memory for the training process and selected batch size, not memory per sample.

---

## Output Files

The default output directory is:

```text
./runs
```

The pipeline creates model-specific run directories containing logs, configuration, checkpoints and training history.

Important outputs include:

```text
fixed_split.json
autobatch_cache.json
segmentation_metrics.csv
efficiency_metrics.csv
model_metrics.csv
```

Per-model final-test outputs include:

```text
final_test_per_patient_<model>.csv
final_test_summary_<model>.csv
```

Each run directory can also contain:

```text
best.pth
last.pth
config.json
run_metrics.json
fixed_split_experiment_summary.json
run.log
```

TensorBoard logs are generated when TensorBoard is installed.

---

## Checkpoint Behaviour

`last.pth` stores the most recent checkpoint.

`best.pth` stores the checkpoint with the highest validation Dice.

Checkpoint state includes:

- epoch
- model name
- model state dictionary
- optimiser state
- Dice score
- early-stopping counter
- best validation metrics
- full configuration

---

## Resource Cleanup

The pipeline includes explicit cleanup to minimise lingering RAM and VRAM after training or script termination.

Cleanup includes:

- closing TensorBoard writers
- shutting down persistent DataLoader workers
- clearing worker and dataset memory-map caches
- deleting model, optimiser, scheduler, scaler and augmentation objects when no longer required
- Python garbage collection
- CUDA synchronisation
- `torch.cuda.empty_cache()`
- `torch.cuda.ipc_collect()`
- termination of surviving multiprocessing children
- logging shutdown

Training resources are released before the final test model is loaded.

A process-wide cleanup is also attempted when execution finishes or exits through an exception.

Once the Python interpreter exits, the operating system reclaims all remaining RAM and VRAM owned by the process.

---

## CUDA Runtime Configuration

The script sets:

```text
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

before importing PyTorch to reduce CUDA allocator fragmentation.

It also enables, where available:

- cuDNN benchmarking
- TF32
- channels-last-3D memory format for selected 3D CNNs
- fused AdamW
- automatic mixed precision

`torch.compile` is disabled by default because of host-RAM pressure observed under Windows.

CUDA is required by default. The script does not silently fall back to CPU execution.

---

## Dependencies

Core dependencies include:

```text
Python
PyTorch
NumPy
NiBabel
SciPy
```

The HVU DenseVU-ED implementation additionally requires:

```text
torchvision
```

Optional packages include:

```text
tqdm
tensorboard
```

`nvidia-smi` is used when available for GPU telemetry such as utilisation, temperature, clock speed and power.

---

## Typical Usage

### Interactive training

```bash
python brats_pipeline_5090.py --data_dir /path/to/BraTS2020
```

The script presents the seven-model selection menu.

### List models

```bash
python brats_pipeline_5090.py --list_models
```

### Scripted model selection

```bash
python brats_pipeline_5090.py \
    --model unet2d \
    --data_dir /path/to/BraTS2020
```

Depending on the current prompt-first CLI settings, the explicit no-prompt option may be required for fully unattended execution.

### Evaluate a checkpoint

```bash
python brats_pipeline_5090.py \
    --mode eval \
    --model diff_unet \
    --checkpoint /path/to/best.pth \
    --data_dir /path/to/BraTS2020
```

---

## Reproducibility Notes

The default random seed is:

```text
123
```

The fixed patient split is persisted to disk and reused across models.

A fixed seed improves reproducibility but does not guarantee bit-for-bit identical CUDA results because some GPU kernels can remain nondeterministic.

For manuscript reproducibility, the final repository commit hash, PyTorch version, CUDA version, cuDNN version and GPU driver version should be recorded after the final experimental code has been frozen.

---

## Default Experimental Configuration

| Setting | Default |
|---|---|
| Data split | Fixed patient-level 80/10/10 |
| Validation | Every epoch |
| Maximum epochs | 300 |
| Early stopping | 30 epochs without Dice improvement |
| Loss | 0.5 Dice + 0.5 categorical cross-entropy |
| Optimiser | AdamW |
| Learning rate | `1e-4` |
| Weight decay | `1e-5` |
| Scheduler | Cosine annealing |
| AMP | Enabled |
| Gradient accumulation | None |
| Workers | 8 |
| Auto batch | Enabled |
| VRAM target | 85% |
| Seed | 123 |
| CUDA required | Yes |
| `torch.compile` | Disabled by default |
| Official BraTS validation cohort | Not used |

---

## Important Methodological Notes

1. The experiment is **not ten-fold cross-validation** by default. The current design is one fixed 80/10/10 patient-level split.
2. Batch size is **model dependent**, not fixed at four.
3. The 3D networks receive depth-padded `240 x 240 x 160` volumes.
4. Final 2D segmentation metrics are calculated volumetrically after reconstructing complete patient volumes.
5. Peak VRAM refers to **peak allocated training memory**, not Task Manager or `nvidia-smi` reserved-memory readings.
6. Inference latency refers to **model forward-pass latency per patient**, excluding I/O and metric computation.
7. Diff-UNet uses memory-management changes that do not alter its architecture or augmentation distribution.
8. HybridAttUnet includes a batch-size-1-safe normalisation fallback at its deepest `1 x 1 x 1` feature map.
9. The code still contains optional cross-validation support for compatibility, but `cv=False` is the default and is not part of the current primary experiment.
