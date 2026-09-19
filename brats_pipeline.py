"""
Unified Brain Tumor Segmentation Pipeline
==========================================
Consolidates training and evaluation from:
  - HVU-ED paper         (2-D DenseVU-ED: DenseNet121 + ViT + U-Net)
  - HybridAttUnet 3D       (3-D Hybrid Attention-Based Residual U-Net, HA-RUnet)
  - Henry et al. 2020    (DeepEnsemble 3D, deep-supervised 3-D U-Net ensemble, BraTS 2020)
  - Diff-UNet            (Diffusion-embedded UNet, PyTorch, 3D)
  - PyTorch-UNet         (Vanilla 3D UNet, PyTorch)

Dataset: BraTS2020 - 4 MRI modalities (T1, T1ce, T2, FLAIR), 3 tumour classes
  - WT  = Whole Tumour      (labels 1+2+4)
  - TC  = Tumour Core       (labels 1+4)
  - ET  = Enhancing Tumour  (label 4)

Usage
-----
  # Train with a specific model:
  python brats_pipeline_5090_all_models_io.py --model deepensemble --data_dir /data/BraTS2020 --epochs 200

  # Evaluate a saved checkpoint:
  python brats_pipeline_5090_all_models_io.py --mode eval --model diff_unet --checkpoint runs/exp1/best.pth

  # List all available models:
  python brats_pipeline_5090_all_models_io.py --list_models
"""

import os
# Reduce CUDA allocator fragmentation during large, shape-stable training runs.
# This must be set before importing torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
import argparse
import logging

# Make Windows console logging Unicode-safe where supported.
for _stream in (getattr(sys, "stdout", None), getattr(sys, "stderr", None)):
    try:
        if _stream is not None and hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import time
import json
import csv
import gc
import multiprocessing as mp
import inspect
import subprocess
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    class SummaryWriter:
        """No-op fallback when the optional tensorboard package is unavailable."""
        def __init__(self, *args, **kwargs):
            pass
        def add_scalar(self, *args, **kwargs):
            pass
        def close(self):
            pass

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

try:
    import nibabel as nib
except ImportError:
    raise ImportError("nibabel is required: pip install nibabel")


# ─────────────────────────────────────────────────────────────────────────────
# RESOURCE CLEANUP
# ─────────────────────────────────────────────────────────────────────────────

def _shutdown_data_loader(loader, logger=None) -> None:
    """Stop persistent DataLoader workers without waiting for process exit."""
    if loader is None:
        return
    try:
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            shutdown = getattr(iterator, "_shutdown_workers", None)
            if callable(shutdown):
                shutdown()
            try:
                loader._iterator = None
            except Exception:
                pass
    except Exception as exc:
        try:
            if logger is not None:
                logger.debug(f"DataLoader worker shutdown warning: {exc}")
        except Exception:
            pass


def _clear_dataset_runtime_cache(dataset) -> None:
    """Drop parent-process mmap/cache handles retained by a Dataset."""
    if dataset is None:
        return
    for name in ("_mmap_lru", "_slice_mmap_lru", "_cache"):
        cache = getattr(dataset, name, None)
        try:
            if cache is not None and hasattr(cache, "clear"):
                cache.clear()
        except Exception:
            pass
    nested = getattr(dataset, "_vol_ds", None)
    if nested is not None and nested is not dataset:
        _clear_dataset_runtime_cache(nested)


def _safe_cuda_release(device=None) -> None:
    """Best-effort CUDA cleanup, including after a prior CUDA error."""
    if not torch.cuda.is_available():
        return
    try:
        if device is not None:
            torch.cuda.synchronize(device)
        else:
            torch.cuda.synchronize()
    except Exception:
        pass
    gc.collect()
    for action in (torch.cuda.empty_cache, getattr(torch.cuda, "ipc_collect", None), torch.cuda.empty_cache):
        try:
            if callable(action):
                action()
        except Exception:
            pass


def _terminate_remaining_children(timeout_s: float = 0.75) -> int:
    """Terminate any multiprocessing children that survived loader shutdown."""
    try:
        children = list(mp.active_children())
    except Exception:
        return 0
    for child in children:
        try:
            child.join(timeout=timeout_s)
        except Exception:
            pass
    for child in children:
        try:
            if child.is_alive():
                child.terminate()
        except Exception:
            pass
    for child in children:
        try:
            child.join(timeout=timeout_s)
        except Exception:
            pass
    return len(children)


def shutdown_process_resources(logger=None) -> None:
    """Release workers, RAM handles and CUDA caches before interpreter exit."""
    try:
        if logger is not None:
            logger.info("Final cleanup: releasing workers, RAM handles and CUDA memory...")
    except Exception:
        pass

    gc.collect()
    child_count = _terminate_remaining_children()
    gc.collect()
    _safe_cuda_release()

    try:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            msg = (
                f"Final CUDA cleanup complete: allocated={allocated:.3f} GB, "
                f"reserved={reserved:.3f} GB; child processes handled={child_count}."
            )
        else:
            msg = f"Final cleanup complete; child processes handled={child_count}."
        if logger is not None:
            logger.info(msg)
        else:
            print(msg, flush=True)
    except Exception:
        pass

    try:
        logging.shutdown()
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CFG = {
    # Data
    "data_dir":        "./BraTS2020",
    "train_data_dir":  None,          # labelled BraTS training cohort; auto-detected under data_dir when possible
    "test_data_dir":   None,          # deprecated/ignored in the default 80/10/10 experiment
    "train_csv":       None,          # optional explicit split CSV
    "val_ratio":       0.10,          # 10% of labelled cohort reserved for early stopping / checkpoint selection
    "fixed_test_ratio": 0.10,         # 10% of labelled cohort reserved for final test only
    "fixed_split_file": None,         # default: <save_dir>/fixed_split.json; reused across all model architectures
    "external_holdout": False,        # official unlabelled validation cohort is not used
    "patch_size":      (240, 240, 155),   # full native BraTS2020 geometry (H, W, D); model sees 4 modalities
    "num_workers":     8,             # use 8 DataLoader workers for every model
    "num_workers_2d":  8,             # retained for CLI compatibility; common default is 8
    "num_workers_3d":  8,             # retained for CLI compatibility; common default is 8
    "cache_rate":      -1.0,          # <0 = auto-size RAM cache from system_ram_gb
    "system_ram_gb":   64.0,          # machine RAM available for auto cache sizing
    "ram_cache_fraction": 0.70,       # target up to 70% of RAM across loader workers
    "cache_compress":  True,          # cache MRI as float16 and labels as uint8
    "prefetch_factor": 2,             # fallback; model-specific factors below take precedence
    "prefetch_factor_2d": 4,          # deeper 2-D queue; RAM cap still limits actual prefetched batches
    "prefetch_factor_3d": 2,          # two full-volume batches queued per worker
    "prefetch_ram_gb": 8.0,           # hard cap across queued pinned batches
    "mmap_lru_patients": 64,          # lightweight worker-local mmap handles; 2-D slice-major cache avoids strided reads
    "patient_grouped_batches": True,  # 2-D training batches preserve patient locality while shuffling
    "slice_major_cache_2d": True,     # store axial slices as (D,C,H,W) so every 2-D slice is contiguous on disk
    "cuda_prefetch": True,            # overlap pinned-memory H2D transfer with GPU compute for models that can afford it
    "diff_unet_cuda_prefetch": False, # Diff-UNet: keep only the current full-volume batch resident on CUDA
    "deepensemble_cuda_prefetch": False, # avoid a second full-volume CUDA batch while training ensemble members
    "hd95_workers": 4,                # parallel CPU workers for validation/test surface distances
    "profile_pipeline": True,          # report data-wait vs GPU-compute time each epoch
    "gpu_telemetry": True,             # log NVIDIA temperature/clock/power/utilisation each epoch when nvidia-smi is available
    "gpu_slowdown_warn_ratio": 1.50,   # warn if GPU training time rises 50% above early-epoch baseline
    "persistent_workers": True,       # keep workers alive between epochs

    # Model
    "model":           None,          # selected interactively unless --model is supplied
    "in_channels":     4,
    "num_classes":     4,             # 0=background, 1=NCR/NET, 2=edema, 3=ET (mutually exclusive)
    "base_filters":    32,
    "unet2d_norm_groups": 8,
    "unet2d_max_batch_size": 64,
    "unet2d_skip_empty_ratio": 0.0,  # retain all native axial slices for UNet2D

    # Training
    "mode":            "train",
    "epochs":          300,
    "batch_size":      1,             # starting batch size for 3-D models
    "batch_size_2d":   64,            # starting batch size for 2-D auto-tuning
    "auto_batch_size": True,           # probe the GPU and select the largest stable training batch
    "batch_vram_fraction": 0.85,       # target up to 85% of total VRAM after accounting for prefetch overlap
    "batch_vram_headroom_gb": 1.0,      # emergency free-memory floor in addition to the 85% VRAM target
    "batch_size_step": 16,             # final 2-D search granularity
    "max_batch_size_2d": 4096,         # high ceiling; tuner stops at the 85% VRAM target
    "max_batch_size_3d": 32,           # high ceiling for conventional 3-D models
    "diff_unet_max_batch_size": 1,     # corrected full-volume Diff-UNet: never probe batch 2 on Windows
    "autobatch_disk_cache": True,      # persist tuned batch sizes as a starting point across runs
    "autobatch_revalidate_cache": False, # cached batch is authoritative; reuse immediately without probing
    "autobatch_cache_version": 9,      # memory-only repeated stability criterion
    "autobatch_cache_file": None,      # default: <save_dir>/autobatch_cache.json
    "autobatch_final_aug_check": True, # run expensive worst-case GPU augmentation only on final candidate
    "autobatch_stability_steps": 3,       # consecutive full-augmentation probes before accepting a new batch size
    "autobatch_stability_slowdown": 1.50, # legacy/unused: timing no longer affects batch selection
    "lr":              1e-4,
    "weight_decay":    1e-5,
    "scheduler":       "cosine",      # cosine | plateau | none
    "amp":             True,
    "seed":            123,
    "checkpoint":      None,          # path to resume from
    "save_dir":        "./runs",
    "log_interval":    10,            # steps between console/tb logs
    "live_refresh_steps": 10,         # avoid forcing a GPU sync for tqdm loss text every batch
    "live_console":    True,          # live tqdm progress bars during train/validation
    "val_interval":    1,             # full validation every epoch
    "early_stopping_patience": 30,   # epochs without Dice improvement before stopping

    # RTX 5090 / CUDA performance
    "compile_model":   False,         # OFF by default on Windows: Inductor can consume large committed host memory
    "compile_mode":    "default",       # compile once after batch tuning; avoids huge max-autotune host-RAM use
    "fused_adamw":     True,
    "channels_last_3d": True,
    "cudnn_benchmark": True,
    "allow_tf32":      True,
    "inference_warmup": 3,           # untimed warm-up forwards before inference timing
    "require_cuda":    True,          # never silently fall back to CPU on the RTX 5090 machine

    # Preprocessed memory-mapped cache
    "preprocessed_cache": True,       # normalise/decompress NIfTI once, then reuse .npy memmaps
    "preprocessed_cache_dir": None,   # default: <data_dir>/.brats_preprocessed_cache
    "prewarm_cache":   False,         # let Windows page cache grow naturally; prewarming ~30 GB caused paging

    # Diff-UNet specific
    "diffusion_steps": 1000,
    "diffusion_infer_steps": 50,

    # DeepEnsemble specific, adapted from Henry et al. (arXiv:2011.01045)
    "deepensemble_members": 5,
    "deepensemble_width": 48,
    "deepensemble_tta": True,
    "deepensemble_deep_supervision": True,
    "deepensemble_norm_groups": 16,
    "deepensemble_activation_checkpointing": True,
    "deepensemble_checkpoint_losses": True,
    "deepensemble_max_batch_size": 1,
    "deepensemble_empty_cache_interval": 10,

    # Cross-validation
    "cv":              False,   # enable k-fold CV
    "n_folds":         10,      # number of folds
    "test_ratio":      0.10,    # fraction held out as final test set (never seen during CV)

    # Augmentation (native PyTorch, applied on the GPU after batch transfer)
    "augment":         True,
    "flip_prob":       0.5,    # random axis flips
    "affine_prob":     0.3,    # random affine: rotation + geometric scaling
    "elastic_prob":    0.2,    # random GPU elastic deformation (2-D/3-D, dimension-matched)
    "noise_prob":      0.2,    # additive Gaussian noise
    "intensity_prob":  0.3,    # smooth GPU intensity inhomogeneity (bias field)
}


def build_config(args: argparse.Namespace) -> dict:
    cfg = dict(DEFAULT_CFG)
    for k, v in vars(args).items():
        if v is not None:
            cfg[k] = v
    if bool(getattr(args, "no_deepensemble_tta", False)):
        cfg["deepensemble_tta"] = False
    cfg.pop("no_deepensemble_tta", None)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def configure_torch_runtime(cfg: dict, logger: Optional[logging.Logger] = None):
    """Enable safe CUDA runtime optimisations for fixed-shape 3-D training."""
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", True))
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.get("allow_tf32", True))
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = bool(cfg.get("allow_tf32", True))
    if cfg.get("allow_tf32", True):
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    if logger is not None:
        logger.info(
            f"CUDA optimisations: cudnn_benchmark={torch.backends.cudnn.benchmark}, "
            f"TF32={bool(cfg.get('allow_tf32', True))}"
        )


def query_nvidia_smi() -> Dict[str, float]:
    """Best-effort NVIDIA telemetry. Returns an empty dict if nvidia-smi is unavailable."""
    fields = [
        "temperature.gpu", "utilization.gpu", "clocks.sm",
        "power.draw", "memory.used", "memory.total",
    ]
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=" + ",".join(fields),
                "--format=csv,noheader,nounits",
                "-i", "0",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).strip().splitlines()[0]
        vals = [x.strip() for x in out.split(",")]
        keys = [
            "gpu_temp_c", "gpu_util_pct", "gpu_sm_clock_mhz",
            "gpu_power_w", "gpu_mem_used_mb", "gpu_mem_total_mb",
        ]
        result = {}
        for k, v in zip(keys, vals):
            try:
                result[k] = float(v)
            except (TypeError, ValueError):
                pass
        return result
    except Exception:
        return {}


def verify_cuda_runtime(cfg: dict, logger: logging.Logger) -> torch.device:
    """Fail fast when GPU training was expected but CUDA is unavailable."""
    logger.info(
        f"PyTorch={torch.__version__} | PyTorch CUDA runtime={torch.version.cuda} | "
        f"cuda_available={torch.cuda.is_available()}"
    )

    if not torch.cuda.is_available():
        msg = (
            "CUDA is not available to PyTorch. Training would therefore run on the CPU, "
            "which explains both ~0 GB VRAM and extremely long epochs. Install a CUDA-enabled "
            "PyTorch build compatible with the RTX 5090/Blackwell GPU and restart Python."
        )
        if cfg.get("require_cuda", True):
            raise RuntimeError(msg)
        logger.warning(msg + " Continuing only because CPU fallback was explicitly allowed.")
        return torch.device("cpu")

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    capability = torch.cuda.get_device_capability(device)
    total_gb = props.total_memory / 1024**3
    arch_list = torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else []
    logger.info(
        f"CUDA device: {props.name} | compute capability={capability[0]}.{capability[1]} | "
        f"VRAM={total_gb:.1f} GB | compiled arches={arch_list}"
    )

    # RTX 5090 is Blackwell (sm_120). Older CUDA/PyTorch wheels may detect a
    # CUDA device but not contain kernels/PTX capable of executing on it.
    if capability >= (12, 0) and arch_list:
        has_blackwell = any(a in {"sm_120", "compute_120"} for a in arch_list)
        if not has_blackwell:
            raise RuntimeError(
                "The GPU is Blackwell-class (compute capability 12.0), but this PyTorch "
                f"build does not advertise sm_120/compute_120 support: {arch_list}. "
                "Install a current Blackwell-capable CUDA PyTorch build."
            )

    # Force a real CUDA allocation and kernel now, before a long dataset setup.
    try:
        x = torch.ones((1024, 1024), device=device)
        y = x @ x
        torch.cuda.synchronize(device)
        del x, y
        torch.cuda.empty_cache()
    except Exception as exc:
        raise RuntimeError(
            "PyTorch can see the GPU but failed a CUDA compute sanity check. "
            f"Underlying error: {exc}"
        ) from exc

    logger.info("CUDA sanity check passed: GPU allocation and matrix multiplication succeeded.")
    return device


def estimate_cached_patient_bytes(cfg: dict) -> int:
    """Approximate bytes per cached native BraTS patient."""
    voxels = int(np.prod(FULL_GEOMETRY)) if "FULL_GEOMETRY" in globals() else 240 * 240 * 155
    if cfg.get("cache_compress", True):
        # Four float16 MRI modalities plus one uint8 label map.
        return voxels * (int(cfg.get("in_channels", 4)) * 2 + 1)
    # Four float32 MRI modalities plus one float32 label map.
    return voxels * (int(cfg.get("in_channels", 4)) * 4 + 4)


def resolve_cache_rate(cfg: dict, n_patients: int, logger: logging.Logger) -> float:
    """Auto-size per-dataset RAM caching while accounting for worker duplication."""
    requested = float(cfg.get("cache_rate", -1.0))
    if requested >= 0:
        return min(1.0, requested)

    workers = max(1, resolve_num_workers(cfg))
    ram_gb = max(1.0, float(cfg.get("system_ram_gb", 64.0)))
    fraction = float(np.clip(cfg.get("ram_cache_fraction", 0.70), 0.10, 0.90))
    bytes_per_patient = estimate_cached_patient_bytes(cfg)
    target_bytes = ram_gb * fraction * (1024 ** 3)

    # Train and validation worker pools both persist.  Their dataset sizes sum
    # to n_patients, so multiplying by worker count estimates total duplicated cache.
    denom = max(1, workers * n_patients * bytes_per_patient)
    rate = float(np.clip(target_bytes / denom, 0.0, 1.0))
    estimated_gb = rate * workers * n_patients * bytes_per_patient / (1024 ** 3)
    logger.info(
        f"RAM cache auto-size: {ram_gb:.0f} GB system RAM, target={fraction:.0%}, "
        f"workers={workers}, cache_rate={rate:.3f}, estimated cache={estimated_gb:.1f} GB"
    )
    return rate


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying eager module when torch.compile wrapped the model."""
    return getattr(model, "_orig_mod", model)


def is_diffusion_model(model: nn.Module) -> bool:
    return isinstance(unwrap_model(model), DiffUNet) if "DiffUNet" in globals() else False


def amp_context(device: torch.device, enabled: bool):
    """Modern autocast helper with CPU-safe disabling."""
    return torch.amp.autocast(device_type=device.type, enabled=bool(enabled and device.type == "cuda"))


def move_image_to_device(image: torch.Tensor, device: torch.device, cfg: dict) -> torch.Tensor:
    # Keep cached MRI data as float16 on the CPU to reduce host RAM, collation
    # and PCIe source traffic, but always promote to float32 as part of the
    # device copy. GPU augmentation runs before autocast, and several PyTorch
    # spatial/grid operations expect float32-compatible tensors. AMP then
    # chooses the efficient lower precision inside the model forward pass.
    kwargs = {
        "non_blocking": device.type == "cuda",
        "dtype": torch.float32,
    }
    if (
        device.type == "cuda"
        and cfg.get("channels_last_3d", True)
        and image.dim() == 5
        and cfg.get("model", "").lower() in {"unet3d", "hybridattunet", "deepensemble"}
    ):
        kwargs["memory_format"] = torch.channels_last_3d
    return image.to(device, **kwargs)


def resolve_num_workers(cfg: dict) -> int:
    """Resolve DataLoader workers. Default is 8 for every model."""
    requested = int(cfg.get("num_workers", -1))
    if requested >= 0:
        return requested
    is2d = str(cfg.get("model", "")).lower() in {"unet2d", "deeplabv3plus2d"}
    requested_model = int(cfg.get("num_workers_2d" if is2d else "num_workers_3d", 8))
    cpu = os.cpu_count() or 8
    # Never consume every logical CPU. Windows, Python, SciPy HD95 and the
    # CUDA feeder all need some headroom.
    return max(1, min(requested_model, max(1, cpu - 2)))


def get_host_memory_status() -> tuple:
    """Return (total_gb, available_gb, used_percent) without requiring psutil."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.total / 1024**3, vm.available / 1024**3, float(vm.percent)
    except Exception:
        # On systems without psutil we still know the configured machine RAM.
        return float(DEFAULT_CFG.get("system_ram_gb", 64.0)), float("nan"), float("nan")


def _loader_worker_init(worker_id: int):
    """Keep DataLoader worker CPU threading predictable on Windows."""
    torch.set_num_threads(1)
    try:
        import SimpleITK as sitk
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    except Exception:
        pass


def make_data_loader(
    dataset,
    cfg: dict,
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool = False,
    batch_sampler=None,
):
    """
    Create a model-aware, RAM-bounded DataLoader. 2-D training may provide a
    patient-grouped batch sampler so consecutive 2-D samples reuse the same mmap
    handles and nearby file pages.
    """
    n_workers = resolve_num_workers(cfg)
    kwargs = dict(
        dataset=dataset,
        num_workers=n_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_loader_worker_init if n_workers > 0 else None,
    )
    if batch_sampler is not None:
        kwargs["batch_sampler"] = batch_sampler
    else:
        kwargs.update(
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    if n_workers > 0:
        kwargs["persistent_workers"] = bool(cfg.get("persistent_workers", True))
        is2d = str(cfg.get("model", "")).lower() in {"unet2d", "deeplabv3plus2d"}
        requested_pf = max(1, int(cfg.get(
            "prefetch_factor_2d" if is2d else "prefetch_factor_3d",
            cfg.get("prefetch_factor", 2),
        )))
        prefetch_gb = max(0.5, float(cfg.get("prefetch_ram_gb", 8.0)))
        try:
            sample = dataset[0]
            tensors = [x for x in sample if torch.is_tensor(x)]
            bytes_per_sample = sum(x.numel() * x.element_size() for x in tensors)
            bytes_per_batch = max(1, bytes_per_sample * int(batch_size))
            ram_cap = int(prefetch_gb * 1024**3)
            safe_pf = max(1, ram_cap // max(1, n_workers * bytes_per_batch))
            kwargs["prefetch_factor"] = int(min(requested_pf, safe_pf))
        except Exception:
            kwargs["prefetch_factor"] = requested_pf

        # If supported by this PyTorch version, allow whichever worker finishes
        # first to feed the shuffled training pipeline. Validation remains
        # deterministic because shuffle=False and no batch sampler is used.
        try:
            if "in_order" in inspect.signature(DataLoader).parameters:
                kwargs["in_order"] = False if (shuffle or batch_sampler is not None) else True
        except Exception:
            pass
    return DataLoader(**kwargs)


def cuda_prefetch_enabled(cfg: dict) -> bool:
    """Return whether asynchronous H2D CUDA prefetch is enabled for this model.

    Diff-UNet is intentionally excluded by default because even batch size 1 is
    a very large full-volume training step. Keeping the next volume resident on
    CUDA at the same time adds memory pressure without changing the model.
    """
    enabled = bool(cfg.get("cuda_prefetch", True))
    model_name = str(cfg.get("model", "")).lower()
    if model_name == "diff_unet":
        enabled = enabled and bool(cfg.get("diff_unet_cuda_prefetch", False))
    elif model_name == "deepensemble":
        enabled = enabled and bool(cfg.get("deepensemble_cuda_prefetch", False))
    return enabled


class CUDAPrefetcher:
    """Overlap pinned-memory host-to-device transfer with model execution."""
    def __init__(self, loader, device: torch.device, cfg: dict):
        self.loader = loader
        self.device = device
        self.cfg = cfg
        self.enabled = bool(device.type == "cuda" and cuda_prefetch_enabled(cfg))

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        if not self.enabled:
            yield from self.loader
            return

        stream = torch.cuda.Stream(device=self.device)
        it = iter(self.loader)

        def preload():
            try:
                image, label, pids = next(it)
            except StopIteration:
                return None
            with torch.cuda.stream(stream):
                image = move_image_to_device(image, self.device, self.cfg)
                label = label.to(self.device, non_blocking=True)
            return image, label, pids

        nxt = preload()
        while nxt is not None:
            torch.cuda.current_stream(self.device).wait_stream(stream)
            current = nxt
            # Tell PyTorch's caching allocator that these tensors are consumed
            # on the current stream after having been created on the prefetch stream.
            current[0].record_stream(torch.cuda.current_stream(self.device))
            current[1].record_stream(torch.cuda.current_stream(self.device))
            nxt = preload()
            yield current


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(save_dir: str) -> logging.Logger:
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger("brats_pipeline")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(os.path.join(save_dir, "run.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# DATASET  (BraTS2020 NIfTI loader)
# ─────────────────────────────────────────────────────────────────────────────

MODALITIES = ["flair", "t1", "t1ce", "t2"]
SEG_FILE   = "seg"

# Native BraTS2020 volume geometry: 240 x 240 x 155 spatial, 4 MRI modalities.
FULL_GEOMETRY = (240, 240, 155)

# Every 3-D U-Net in this file down-samples four times (MaxPool3d(2)), so any
# spatial dimension fed to a model must be divisible by 2**4 = 16.  Crops are
# therefore padded up to the next multiple of this value - e.g. the native
# depth 155 is padded to 160 so full-geometry volumes pass through the models.
PATCH_STRIDE_MULTIPLE = 16


def find_patient_dirs(data_dir: str) -> List[Path]:
    """
    Discovers patient directories under data_dir.
    Expects each subfolder to contain *_flair.nii.gz, *_t1.nii.gz, etc.
    Compatible with the Kaggle BraTS2020 directory layout used by all 5 repos.
    """
    root = Path(data_dir)
    patients = sorted([
        d for d in root.rglob("*")
        if d.is_dir() and any(d.glob("*_flair.nii*"))
    ])
    return patients


def load_nii(path: Path) -> np.ndarray:
    return nib.load(str(path)).get_fdata(dtype=np.float32)


def normalise_volume(vol: np.ndarray) -> np.ndarray:
    """Z-score normalisation over non-zero voxels (per modality)."""
    mask = vol > 0
    if mask.sum() == 0:
        return vol
    mean = vol[mask].mean()
    std  = vol[mask].std() + 1e-8
    vol = (vol - mean) / std
    vol[~mask] = 0.0
    return vol


def build_label_map(seg: np.ndarray) -> np.ndarray:
    """
    BraTS labels -> a single mutually-exclusive integer class map.

    Remapping (labels are not contiguous in BraTS: there is no label 3):
        0 -> 0  background
        1 -> 1  NCR / NET   (necrotic & non-enhancing tumour core)
        2 -> 2  ED          (peritumoural oedema)
        4 -> 3  ET          (enhancing tumour)
    Some pre-processed copies already store ET as 3, so 3 is also mapped to ET.

    Returns shape (1, H, W, D) float32 holding class indices {0,1,2,3}.  A
    singleton channel is kept so the crop / flip / slice code (which is
    channel-first) works unchanged; padding with 0 is background, as desired.
    """
    cls = np.zeros(seg.shape, dtype=np.float32)
    cls[seg == 1] = 1
    cls[seg == 2] = 2
    cls[(seg == 4) | (seg == 3)] = 3
    return cls[None]  # (1, H, W, D)


def resolve_preprocessed_cache_dir(cfg: dict) -> Optional[Path]:
    """Location for normalised, decompressed NumPy cache files."""
    if not cfg.get("preprocessed_cache", True):
        return None
    configured = cfg.get("preprocessed_cache_dir")
    if configured:
        return Path(configured)
    return Path(cfg["data_dir"]) / ".brats_preprocessed_cache"


def _patient_cache_paths(cache_root: Path, patient_dir: Path) -> Tuple[Path, Path]:
    patient_cache = cache_root / patient_dir.name
    return patient_cache / "image_f16.npy", patient_cache / "label_u8.npy"


def _patient_slice_cache_paths(cache_root: Path, patient_dir: Path) -> Tuple[Path, Path]:
    """Axial slice-major cache used only by UNet2D.

    The normal 3-D cache is (C,H,W,D), which makes image[..., z] a heavily
    strided read across the whole file.  The 2-D cache is (D,C,H,W), so one
    axial slice is a single contiguous block and can be wrapped by PyTorch
    without a CPU-side contiguous copy.
    """
    patient_cache = cache_root / patient_dir.name
    return patient_cache / "image_axial_f16.npy", patient_cache / "label_axial_u8.npy"


def prepare_preprocessed_cache(
    patient_dirs: List[Path],
    cfg: dict,
    logger: logging.Logger,
) -> Optional[Path]:
    """
    Decompress and normalise every BraTS case once, then store compact padded
    NumPy arrays. The cache is padded once to 240 x 240 x 160, so every 3-D
    architecture can mmap a model-ready tensor without performing a ~75 MB
    np.pad/copy for every patient in every epoch. UNet2D simply reads the first
    155 axial slices from the same cache.
    """
    cache_root = resolve_preprocessed_cache_dir(cfg)
    if cache_root is None:
        return None
    cache_root.mkdir(parents=True, exist_ok=True)

    target_spatial = (
        FULL_GEOMETRY[0],
        FULL_GEOMETRY[1],
        int(np.ceil(FULL_GEOMETRY[2] / PATCH_STRIDE_MULTIPLE) * PATCH_STRIDE_MULTIPLE),
    )

    def cache_ok(pdir: Path) -> bool:
        image_path, label_path = _patient_cache_paths(cache_root, pdir)
        if not image_path.exists() or not label_path.exists():
            return False
        try:
            im = np.load(image_path, mmap_mode="r", allow_pickle=False)
            lb = np.load(label_path, mmap_mode="r", allow_pickle=False)
            ok = tuple(im.shape[1:]) == target_spatial and tuple(lb.shape[1:]) == target_spatial
            del im, lb
            return bool(ok)
        except Exception:
            return False

    needs_work = [pdir for pdir in patient_dirs if not cache_ok(pdir)]

    if needs_work:
        iterator = needs_work
        if tqdm is not None:
            iterator = tqdm(
                needs_work, desc="Preparing padded MRI cache", unit="patient",
                dynamic_ncols=True, file=sys.stdout
            )
        for pdir in iterator:
            patient_cache = cache_root / pdir.name
            patient_cache.mkdir(parents=True, exist_ok=True)
            image_path, label_path = _patient_cache_paths(cache_root, pdir)

            # Upgrade an existing native-depth cache without re-reading NIfTI.
            upgraded = False
            if image_path.exists() and label_path.exists():
                try:
                    old_image = np.load(image_path, mmap_mode="r", allow_pickle=False)
                    old_label = np.load(label_path, mmap_mode="r", allow_pickle=False)
                    if (
                        tuple(old_image.shape[1:]) == FULL_GEOMETRY
                        and tuple(old_label.shape[1:]) == FULL_GEOMETRY
                    ):
                        pad_d = target_spatial[2] - FULL_GEOMETRY[2]
                        image = np.pad(
                            np.asarray(old_image),
                            ((0, 0), (0, 0), (0, 0), (0, pad_d)),
                            mode="constant",
                        ).astype(np.float16, copy=False)
                        label = np.pad(
                            np.asarray(old_label),
                            ((0, 0), (0, 0), (0, 0), (0, pad_d)),
                            mode="constant",
                        ).astype(np.uint8, copy=False)
                        del old_image, old_label
                        upgraded = True
                    else:
                        del old_image, old_label
                except Exception:
                    upgraded = False

            if not upgraded:
                vols = []
                for mod in MODALITIES:
                    candidates = sorted(pdir.glob(f"*_{mod}.nii*"))
                    if not candidates:
                        raise FileNotFoundError(f"Missing {mod} for patient {pdir.name}")
                    vols.append(normalise_volume(load_nii(candidates[0])))
                native_image = np.stack(vols, axis=0).astype(np.float16, copy=False)

                seg_candidates = sorted(pdir.glob("*_seg.nii*"))
                if seg_candidates:
                    native_label = build_label_map(load_nii(seg_candidates[0])).astype(np.uint8, copy=False)
                else:
                    native_label = np.zeros((1,) + native_image.shape[1:], dtype=np.uint8)

                pad_d = target_spatial[2] - native_image.shape[-1]
                image = np.pad(
                    native_image, ((0, 0), (0, 0), (0, 0), (0, max(0, pad_d))), mode="constant"
                )
                label = np.pad(
                    native_label, ((0, 0), (0, 0), (0, 0), (0, max(0, pad_d))), mode="constant"
                )

            tmp_image = image_path.with_name(image_path.name + ".tmp.npy")
            tmp_label = label_path.with_name(label_path.name + ".tmp.npy")
            np.save(tmp_image, image, allow_pickle=False)
            np.save(tmp_label, label, allow_pickle=False)
            os.replace(tmp_image, image_path)
            os.replace(tmp_label, label_path)

            # Foreground metadata is intentionally native-depth only.
            fg_meta = patient_cache / "foreground_by_z.npy"
            native_fg = np.any(label[0, :, :, :FULL_GEOMETRY[2]] > 0, axis=(0, 1))
            np.save(fg_meta, native_fg.astype(np.uint8), allow_pickle=False)
            del image, label

        logger.info(
            f"Prepared/upgraded {len(needs_work)} padded patient cache(s) in {cache_root}"
        )
    else:
        logger.info(f"Padded preprocessed MRI cache already complete: {cache_root}")

    return cache_root


def prepare_slice_major_2d_cache(
    patient_dirs: List[Path],
    cache_root: Optional[Path],
    logger: logging.Logger,
) -> None:
    """Create a one-time axial slice-major cache for the 2-D models.

    The source cache remains the canonical padded 3-D representation used by
    all 3-D models.  This secondary cache duplicates only the native 155 axial
    slices on disk, trading disk capacity for much lower 2-D page-fault and
    CPU-copy overhead.  Arrays are written one slice at a time so preparation
    never materialises the complete cohort in RAM.
    """
    if cache_root is None:
        return

    native_d = FULL_GEOMETRY[2]
    expected_image = (native_d, 4, FULL_GEOMETRY[0], FULL_GEOMETRY[1])
    expected_label = (native_d, 1, FULL_GEOMETRY[0], FULL_GEOMETRY[1])

    def valid(path: Path, shape: tuple, dtype) -> bool:
        if not path.exists():
            return False
        try:
            mm = np.load(path, mmap_mode="r", allow_pickle=False)
            ok = tuple(mm.shape) == tuple(shape) and mm.dtype == np.dtype(dtype)
            del mm
            return bool(ok)
        except Exception:
            return False

    pending = []
    for pdir in patient_dirs:
        img_z, lbl_z = _patient_slice_cache_paths(cache_root, pdir)
        if not (valid(img_z, expected_image, np.float16) and valid(lbl_z, expected_label, np.uint8)):
            pending.append(pdir)

    if not pending:
        logger.info(f"2-D axial slice-major cache already complete: {cache_root}")
        return

    logger.info(
        f"Preparing contiguous axial cache for the 2-D models ({len(pending)} patient(s)). "
        "This is a one-time disk conversion; later epochs avoid strided full-volume reads."
    )
    iterator = pending
    if tqdm is not None:
        iterator = tqdm(pending, desc="Preparing 2-D axial cache", unit="patient",
                        dynamic_ncols=True, file=sys.stdout)

    for pdir in iterator:
        src_img_path, src_lbl_path = _patient_cache_paths(cache_root, pdir)
        if not src_img_path.exists() or not src_lbl_path.exists():
            raise RuntimeError(f"Missing canonical preprocessed cache for {pdir.name}")
        dst_img_path, dst_lbl_path = _patient_slice_cache_paths(cache_root, pdir)
        dst_img_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_img = dst_img_path.with_name(dst_img_path.name + ".tmp.npy")
        tmp_lbl = dst_lbl_path.with_name(dst_lbl_path.name + ".tmp.npy")

        src_img = np.load(src_img_path, mmap_mode="r", allow_pickle=False)
        src_lbl = np.load(src_lbl_path, mmap_mode="r", allow_pickle=False)
        out_img = np.lib.format.open_memmap(
            tmp_img, mode="w+", dtype=np.float16, shape=expected_image
        )
        out_lbl = np.lib.format.open_memmap(
            tmp_lbl, mode="w+", dtype=np.uint8, shape=expected_label
        )
        for z in range(native_d):
            out_img[z] = src_img[:, :, :, z]
            out_lbl[z] = src_lbl[:, :, :, z]
        out_img.flush(); out_lbl.flush()
        del out_img, out_lbl, src_img, src_lbl
        os.replace(tmp_img, dst_img_path)
        os.replace(tmp_lbl, dst_lbl_path)

    logger.info("2-D axial slice-major cache preparation complete.")


def prewarm_preprocessed_cache(
    patient_dirs: List[Path],
    cache_root: Optional[Path],
    logger: logging.Logger,
):
    """Sequentially read cached files so the OS can use available RAM as file cache."""
    if cache_root is None:
        return
    logger.info(
        "Prewarming the preprocessed cache into the operating-system file cache. "
        "On a 64 GB machine this allows the OS to retain much of the ~30 GB BraTS cache in RAM."
    )
    iterator = patient_dirs
    if tqdm is not None:
        iterator = tqdm(
            patient_dirs, desc="Prewarming RAM cache", unit="patient",
            dynamic_ncols=True, file=sys.stdout
        )
    checksum = 0
    block = 16 * 1024 * 1024
    for pdir in iterator:
        for path in _patient_cache_paths(cache_root, pdir):
            if not path.exists():
                continue
            with open(path, "rb", buffering=0) as f:
                while True:
                    chunk = f.read(block)
                    if not chunk:
                        break
                    checksum ^= chunk[0]
    logger.info("Preprocessed cache prewarm complete.")
    return checksum


class GPUBatchAugmenter:
    """
    Native PyTorch augmentation executed after a batch has been copied to CUDA.

    This replaces the previous CPU augmentation backend completely. Spatial transformations are applied
    consistently to MRI and segmentation tensors. MRI data use bilinear or
    trilinear interpolation, while class-index labels use nearest-neighbour
    interpolation so labels remain in {0, 1, 2, 3}.

    Augmentations preserved from the previous pipeline:
      * random axis flips
      * random rotation + geometric scaling
      * smooth elastic deformation (dimension-matched in 2-D and 3-D)
      * smooth multiplicative intensity bias field
      * additive Gaussian noise
    """

    def __init__(self, cfg: dict, is_2d: bool):
        self.is_2d = bool(is_2d)
        self.flip_prob = float(cfg.get("flip_prob", 0.5))
        self.affine_prob = float(cfg.get("affine_prob", 0.3))
        self.elastic_prob = float(cfg.get("elastic_prob", 0.2))
        self.noise_prob = float(cfg.get("noise_prob", 0.2))
        self.intensity_prob = float(cfg.get("intensity_prob", 0.3))
        self.max_rotation_deg = 10.0
        self.scale_min = 0.9
        self.scale_max = 1.1
        self.max_elastic_displacement = 7.0
        self.elastic_control_points = 7
        self.bias_strength = 0.3

    @staticmethod
    def _mask(batch: int, prob: float, device: torch.device, force: bool = False):
        if prob <= 0:
            return torch.zeros(batch, dtype=torch.bool, device=device)
        if force:
            return torch.ones(batch, dtype=torch.bool, device=device)
        return torch.rand(batch, device=device) < prob

    def _random_flips(self, image: torch.Tensor, label: torch.Tensor, force: bool):
        """Flip selected samples independently along each spatial tensor axis."""
        batch = image.shape[0]
        # image dims are B,C,H,W for 2-D and B,C,H,W,D for 3-D.
        spatial_dims = (2, 3) if self.is_2d else (2, 3, 4)
        for dim in spatial_dims:
            mask = self._mask(batch, self.flip_prob, image.device, force)
            if bool(mask.any()):
                image[mask] = torch.flip(image[mask], dims=[dim])
                label[mask] = torch.flip(label[mask], dims=[dim])
        return image, label

    def _spatial_2d(self, image: torch.Tensor, label: torch.Tensor, force: bool):
        """Apply affine and elastic deformation in one 2-D grid_sample operation.

        This mirrors the 3-D spatial augmentation policy: the same affine and
        elastic probabilities, rotation range, scale range, control-point count
        and maximum displacement are used, adapted only to 2-D geometry.
        """
        batch = image.shape[0]
        affine_mask = self._mask(batch, self.affine_prob, image.device, force)
        elastic_mask = self._mask(batch, self.elastic_prob, image.device, force)
        use_mask = affine_mask | elastic_mask
        n = int(use_mask.sum().item())
        if n == 0:
            return image, label

        img = image[use_mask]
        lbl = label[use_mask]
        device, dtype = img.device, img.dtype
        local_affine = affine_mask[use_mask]
        local_elastic = elastic_mask[use_mask]

        theta = torch.zeros(n, 2, 3, device=device, dtype=dtype)
        theta[:, 0, 0] = 1
        theta[:, 1, 1] = 1

        na = int(local_affine.sum().item())
        if na:
            angle = (torch.rand(na, device=device, dtype=dtype) * 2 - 1) * self.max_rotation_deg
            angle = torch.deg2rad(angle)
            c, ss = torch.cos(angle), torch.sin(angle)
            scales = torch.empty(na, 2, device=device, dtype=dtype).uniform_(
                self.scale_min, self.scale_max
            )
            inv_x, inv_y = 1.0 / scales[:, 0], 1.0 / scales[:, 1]
            theta[local_affine, 0, 0] = c * inv_x
            theta[local_affine, 0, 1] = -ss * inv_y
            theta[local_affine, 1, 0] = ss * inv_x
            theta[local_affine, 1, 1] = c * inv_y

        grid = F.affine_grid(theta, img.shape, align_corners=False)

        ne = int(local_elastic.sum().item())
        if ne:
            cp = self.elastic_control_points
            disp_low = torch.zeros(n, 2, cp, cp, device=device, dtype=dtype)
            disp_low[local_elastic] = torch.randn(
                ne, 2, cp, cp, device=device, dtype=dtype
            )
            disp = F.interpolate(
                disp_low, size=img.shape[2:], mode="bilinear", align_corners=False
            )
            h, w = img.shape[2:]
            norm = torch.tensor(
                [
                    2.0 * self.max_elastic_displacement / max(w - 1, 1),
                    2.0 * self.max_elastic_displacement / max(h - 1, 1),
                ],
                device=device,
                dtype=dtype,
            ).view(1, 2, 1, 1)
            disp = torch.tanh(disp) * norm
            grid = grid + disp.permute(0, 2, 3, 1)

        img_aug = F.grid_sample(
            img, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        lbl_aug = F.grid_sample(
            lbl.float(), grid, mode="nearest", padding_mode="zeros", align_corners=False
        )
        image[use_mask] = img_aug
        label[use_mask] = lbl_aug.to(label.dtype)
        return image, label

    @staticmethod
    def _rotation_matrices_3d(rx, ry, rz):
        """Vectorised Rz @ Ry @ Rx rotation matrices."""
        n, device, dtype = rx.numel(), rx.device, rx.dtype
        cx, sx = torch.cos(rx), torch.sin(rx)
        cy, sy = torch.cos(ry), torch.sin(ry)
        cz, sz = torch.cos(rz), torch.sin(rz)

        rxm = torch.zeros(n, 3, 3, device=device, dtype=dtype)
        rym = torch.zeros_like(rxm)
        rzm = torch.zeros_like(rxm)
        rxm[:, 0, 0] = 1
        rxm[:, 1, 1] = cx
        rxm[:, 1, 2] = -sx
        rxm[:, 2, 1] = sx
        rxm[:, 2, 2] = cx
        rym[:, 1, 1] = 1
        rym[:, 0, 0] = cy
        rym[:, 0, 2] = sy
        rym[:, 2, 0] = -sy
        rym[:, 2, 2] = cy
        rzm[:, 2, 2] = 1
        rzm[:, 0, 0] = cz
        rzm[:, 0, 1] = -sz
        rzm[:, 1, 0] = sz
        rzm[:, 1, 1] = cz
        return torch.bmm(rzm, torch.bmm(rym, rxm))

    def _spatial_3d(self, image: torch.Tensor, label: torch.Tensor, force: bool):
        """Apply affine and elastic deformation in one grid_sample operation.

        The full-batch path is memory-conscious for large 3-D inputs: when all
        samples are transformed (always true for a forced batch-size-1 probe),
        it avoids boolean-index copies of the complete MRI volume and label.
        Large grid/displacement intermediates are updated in place and released
        immediately after use. The augmentation distribution is unchanged.
        """
        batch = image.shape[0]
        affine_mask = self._mask(batch, self.affine_prob, image.device, force)
        elastic_mask = self._mask(batch, self.elastic_prob, image.device, force)
        use_mask = affine_mask | elastic_mask
        n = int(use_mask.sum().item())
        if n == 0:
            return image, label

        all_selected = (n == batch)
        if all_selected:
            img = image
            lbl = label
            local_affine = affine_mask
            local_elastic = elastic_mask
        else:
            img = image[use_mask]
            lbl = label[use_mask]
            local_affine = affine_mask[use_mask]
            local_elastic = elastic_mask[use_mask]

        device, dtype = img.device, img.dtype
        theta = torch.zeros(n, 3, 4, device=device, dtype=dtype)
        theta[:, 0, 0] = 1
        theta[:, 1, 1] = 1
        theta[:, 2, 2] = 1

        na = int(local_affine.sum().item())
        if na:
            angles = (torch.rand(na, 3, device=device, dtype=dtype) * 2 - 1)
            angles = torch.deg2rad(angles * self.max_rotation_deg)
            rot = self._rotation_matrices_3d(angles[:, 0], angles[:, 1], angles[:, 2])
            scales = torch.empty(na, 3, device=device, dtype=dtype).uniform_(
                self.scale_min, self.scale_max
            )
            inv_scale = torch.diag_embed(1.0 / scales)
            theta[local_affine, :, :3] = torch.bmm(rot, inv_scale)
            del angles, rot, scales, inv_scale

        grid = F.affine_grid(theta, img.shape, align_corners=False)
        del theta

        ne = int(local_elastic.sum().item())
        if ne:
            cp = self.elastic_control_points
            disp_low = torch.zeros(n, 3, cp, cp, cp, device=device, dtype=dtype)
            disp_low[local_elastic] = torch.randn(
                ne, 3, cp, cp, cp, device=device, dtype=dtype
            )
            disp = F.interpolate(
                disp_low, size=img.shape[2:], mode="trilinear", align_corners=False
            )
            del disp_low

            d0, d1, d2 = img.shape[2:]
            norm = torch.tensor(
                [
                    2.0 * self.max_elastic_displacement / max(d2 - 1, 1),
                    2.0 * self.max_elastic_displacement / max(d1 - 1, 1),
                    2.0 * self.max_elastic_displacement / max(d0 - 1, 1),
                ],
                device=device,
                dtype=dtype,
            ).view(1, 3, 1, 1, 1)

            # Avoid allocating a second full-resolution displacement tensor and
            # a second full-resolution sampling grid.
            disp.tanh_()
            disp.mul_(norm)
            grid.add_(disp.permute(0, 2, 3, 4, 1))
            del disp, norm

        img_aug = F.grid_sample(
            img, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        lbl_float = lbl.float()
        lbl_aug = F.grid_sample(
            lbl_float, grid, mode="nearest", padding_mode="zeros", align_corners=False
        )
        del lbl_float, grid

        if all_selected:
            # Copy back into the existing batch buffers so the model sees only
            # one resident augmented batch after this function returns.
            image.copy_(img_aug)
            label.copy_(lbl_aug)
        else:
            image[use_mask] = img_aug
            label[use_mask] = lbl_aug.to(label.dtype)

        del img_aug, lbl_aug, img, lbl, local_affine, local_elastic
        return image, label

    def _bias_field(self, image: torch.Tensor, force: bool):
        batch = image.shape[0]
        mask = self._mask(batch, self.intensity_prob, image.device, force)
        n = int(mask.sum().item())
        if n == 0:
            return image

        all_selected = (n == batch)
        img = image if all_selected else image[mask]
        if self.is_2d:
            field = torch.randn(n, 1, 4, 4, device=img.device, dtype=img.dtype)
            field = F.interpolate(field, size=img.shape[2:], mode="bilinear", align_corners=False)
            reduce_dims = (2, 3)
        else:
            field = torch.randn(n, 1, 4, 4, 4, device=img.device, dtype=img.dtype)
            field = F.interpolate(field, size=img.shape[2:], mode="trilinear", align_corners=False)
            reduce_dims = (2, 3, 4)

        mean = field.mean(dim=reduce_dims, keepdim=True)
        field.sub_(mean)
        del mean
        std_field = field.std(dim=reduce_dims, keepdim=True).clamp_min_(1e-6)
        field.div_(std_field)
        del std_field

        strength_shape = (n, 1) + (1,) * (image.dim() - 2)
        strength = torch.empty(strength_shape, device=img.device, dtype=img.dtype).uniform_(
            -self.bias_strength, self.bias_strength
        )

        # Reuse the interpolated field as the multiplicative bias tensor.
        field.clamp_(-2.0, 2.0)
        field.mul_(strength)
        field.exp_()
        del strength

        if all_selected:
            image.mul_(field)
        else:
            image[mask] = img * field

        del field, img
        return image

    def _gaussian_noise(self, image: torch.Tensor, force: bool):
        batch = image.shape[0]
        mask = self._mask(batch, self.noise_prob, image.device, force)
        n = int(mask.sum().item())
        if n == 0:
            return image

        all_selected = (n == batch)
        img = image if all_selected else image[mask]
        std_shape = (n, 1) + (1,) * (image.dim() - 2)
        std = torch.empty(std_shape, device=img.device, dtype=img.dtype).uniform_(0.0, 0.1)
        noise = torch.randn_like(img)
        noise.mul_(std)
        del std

        if all_selected:
            image.add_(noise)
        else:
            image[mask] = img + noise

        del noise, img
        return image

    @torch.no_grad()
    def __call__(self, image: torch.Tensor, label: torch.Tensor, force: bool = False):
        image, label = self._random_flips(image, label, force)
        if self.is_2d:
            image, label = self._spatial_2d(image, label, force)
        else:
            image, label = self._spatial_3d(image, label, force)
        image = self._bias_field(image, force)
        image = self._gaussian_noise(image, force)
        return image, label


class BraTS2020Dataset(Dataset):
    """
    Unified dataset compatible with all five repo conventions.
    Returns:
        image  - (4, H, W, D) float32 tensor
        label  - (1, H, W, D) float32 tensor of integer class indices {0,1,2,3}
        pid    - patient folder name string
    """

    def __init__(
        self,
        patient_dirs: List[Path],
        patch_size: Tuple[int, int, int] = FULL_GEOMETRY,
        augment: bool = False,
        flip_prob: float = 0.5,
        affine_prob: float = 0.3,
        elastic_prob: float = 0.2,
        noise_prob: float = 0.2,
        intensity_prob: float = 0.3,
        cache_rate: float = 0.0,
        cache_compress: bool = True,
        preprocessed_cache_dir: Optional[Path] = None,
        mmap_lru_patients: int = 32,
        pad_multiple: int = PATCH_STRIDE_MULTIPLE,
    ):
        self.patients    = patient_dirs
        self.patch_size  = patch_size
        self.augment     = augment  # augmentation is applied later on the GPU
        self.pad_multiple = pad_multiple
        self._cache: Dict[int, tuple] = {}
        self._cache_limit = int(len(patient_dirs) * cache_rate)
        self.cache_compress = cache_compress
        self.preprocessed_cache_dir = Path(preprocessed_cache_dir) if preprocessed_cache_dir else None
        self.mmap_lru_patients = max(0, int(mmap_lru_patients))
        self._mmap_lru = OrderedDict()

    def __len__(self) -> int:
        return len(self.patients)

    def __getstate__(self):
        """Keep DataLoader worker spawning lightweight on Windows.

        numpy.memmap objects must not be carried from the parent process into
        spawned workers. Workers reopen the required files lazily instead.
        """
        state = self.__dict__.copy()
        state["_cache"] = {}
        state["_mmap_lru"] = OrderedDict()
        return state

    def _load(self, idx: int) -> tuple:
        if idx in self._cache:
            return self._cache[idx]
        pdir = self.patients[idx]
        pid  = pdir.name

        # Fast path: preprocessed .npy files are memory-mapped. Each spawned
        # worker keeps only a small LRU of open patient handles. This removes
        # repeated np.load/open calls without ever serialising hundreds of live
        # memmaps from the Windows parent process.
        if self.preprocessed_cache_dir is not None:
            if idx in self._mmap_lru:
                result = self._mmap_lru.pop(idx)
                self._mmap_lru[idx] = result
                return result
            image_path, label_path = _patient_cache_paths(self.preprocessed_cache_dir, pdir)
            if image_path.exists() and label_path.exists():
                # Copy-on-write mmap: remains disk-backed and does not modify the
                # cached .npy files, but exposes a writable NumPy view so
                # torch.from_numpy() can safely wrap full 3-D contiguous arrays.
                image = np.load(image_path, mmap_mode="c", allow_pickle=False)
                label = np.load(label_path, mmap_mode="c", allow_pickle=False)
                result = (image, label, pid)
                if self.mmap_lru_patients > 0:
                    self._mmap_lru[idx] = result
                    while len(self._mmap_lru) > self.mmap_lru_patients:
                        self._mmap_lru.popitem(last=False)
                return result

        # Fallback path for users who disable the preprocessed cache.
        vols = []
        for mod in MODALITIES:
            candidates = sorted(pdir.glob(f"*_{mod}.nii*"))
            if not candidates:
                raise FileNotFoundError(f"Missing {mod} for patient {pid}")
            vols.append(normalise_volume(load_nii(candidates[0])))
        image = np.stack(vols, axis=0)  # (4, H, W, D)

        seg_candidates = sorted(pdir.glob(f"*_seg.nii*"))
        if not seg_candidates:
            label = np.zeros((1,) + image.shape[1:], dtype=np.float32)
        else:
            label = build_label_map(load_nii(seg_candidates[0]))

        result = (image, label, pid)
        if idx < self._cache_limit:
            if self.cache_compress:
                result = (image.astype(np.float16), label.astype(np.uint8), pid)
            self._cache[idx] = result
        return result

    @staticmethod
    def _round_up(value: int, multiple: int) -> int:
        """Smallest multiple of ``multiple`` that is >= ``value`` (identity if multiple <= 1)."""
        if multiple <= 1:
            return int(value)
        return int(np.ceil(value / multiple) * multiple)

    def _foreground_biased_crop(self, image: np.ndarray, label: np.ndarray):
        """
        Foreground-biased patch extraction.

        A crop of size ``self.patch_size`` (H, W, D) is taken from the full
        volume and is *centred on a randomly chosen foreground voxel* - a random
        voxel belonging to any tumour class (class index > 0).  When a volume
        contains no foreground the crop location falls back to a uniform-random position.

        With ``patch_size`` at the full native geometry (240, 240, 155) the crop
        spans the whole brain; with a smaller ``patch_size`` it yields
        tumour-centred sub-volumes.

        Finally the tensors are zero-padded (trailing side) so every spatial
        dimension is a multiple of ``self.pad_multiple`` (default 16).  The 3-D
        U-Nets here down-sample four times, so their input must be divisible by
        16 - e.g. the native depth 155 is padded to 160.  Padding to a fixed
        target also keeps every sample the same shape, which batching requires.
        """
        ph, pw, pd = self.patch_size
        _, h, w, d = image.shape

        # The preprocessed cache stores the native BraTS volume already padded
        # to depth 160. For the default full-volume experiment, return that mmap
        # directly. No foreground scan, crop, np.pad or full-volume copy is needed.
        padded_depth = self._round_up(FULL_GEOMETRY[2], self.pad_multiple)
        if (ph, pw, pd) == FULL_GEOMETRY and (h, w, d) == (FULL_GEOMETRY[0], FULL_GEOMETRY[1], padded_depth):
            return image, label

        # Largest valid top-left corner that keeps the crop inside the volume.
        sh = max(0, h - ph)
        sw = max(0, w - pw)
        sd = max(0, d - pd)

        # Fast path for the default full-volume BraTS geometry. There is only
        # one possible crop origin, so scanning ~9 million voxels to find a
        # random foreground centre is pure overhead and is skipped entirely.
        if sh == 0 and sw == 0 and sd == 0:
            image = image[:, :ph, :pw, :pd]
            label = label[:, :ph, :pw, :pd]
            target = tuple(self._round_up(p, self.pad_multiple) for p in self.patch_size)

            if image.shape[1:] != target:
                pads = [(0, 0)] + [(0, max(0, t - ss))
                                   for t, ss in zip(target, image.shape[1:])]
                image = np.pad(image, pads)
            if label.shape[1:] != target:
                pads = [(0, 0)] + [(0, max(0, t - ss))
                                   for t, ss in zip(target, label.shape[1:])]
                label = np.pad(label, pads)
            return image, label

        fg = np.argwhere(label.sum(0) > 0)              # foreground voxel coords (N, 3)
        if len(fg) > 0:
            centre = fg[np.random.randint(len(fg))]      # pick one foreground voxel ...
            x = int(np.clip(centre[0] - ph // 2, 0, sh)) # ... and centre the crop on it,
            y = int(np.clip(centre[1] - pw // 2, 0, sw)) #     clipped to stay in-bounds
            z = int(np.clip(centre[2] - pd // 2, 0, sd))
        else:
            x = np.random.randint(0, sh + 1)
            y = np.random.randint(0, sw + 1)
            z = np.random.randint(0, sd + 1)

        image = image[:, x:x + ph, y:y + pw, z:z + pd]
        label = label[:, x:x + ph, y:y + pw, z:z + pd]

        # Pad each spatial dim up to the next network-friendly multiple.
        target = tuple(self._round_up(p, self.pad_multiple) for p in self.patch_size)

        def pad_to(arr, tgt_spatial):
            pads = [(0, 0)] + [(0, max(0, t - s))
                               for t, s in zip(tgt_spatial, arr.shape[1:])]
            return np.pad(arr, pads)

        image = pad_to(image, target)
        label = pad_to(label, target)
        return image, label


    @staticmethod
    def _torch_safe_array(arr: np.ndarray) -> np.ndarray:
        """Return a C-contiguous, writable NumPy view for torch.from_numpy.

        Runtime cache files are opened with mmap_mode="c" (copy-on-write), so
        the normal fast path requires no copy. This guard only copies if a
        future/fallback source is non-contiguous or genuinely read-only.
        """
        arr = np.asarray(arr)
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        if not arr.flags.writeable:
            arr = arr.copy()
        return arr

    def __getitem__(self, idx: int):
        image, label, pid = self._load(idx)
        image, label = self._foreground_biased_crop(image, label)
        # Keep compact cache dtypes on the CPU. DataLoader collation, pinned
        # memory and PCIe transfer are therefore roughly half the size of the
        # previous float32 pipeline. CombinedLoss converts labels to long on GPU.
        return (
            torch.from_numpy(self._torch_safe_array(image)),
            torch.from_numpy(self._torch_safe_array(label)),
            pid,
        )


def make_splits(patient_dirs: List[Path], val_ratio: float, seed: int):
    """Simple single train/val split (used when CV is disabled)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(patient_dirs))
    n_val = max(1, int(len(patient_dirs) * val_ratio))
    val_idx   = perm[:n_val]
    train_idx = perm[n_val:]
    return [patient_dirs[i] for i in train_idx], [patient_dirs[i] for i in val_idx]



def _normalised_path_string(path: Path) -> str:
    try:
        return str(path.resolve())
    except Exception:
        return str(path)


def _looks_like_training_root(path: Path) -> bool:
    name = path.name.lower().replace("-", "_")
    return ("training" in name) or name in {"train", "training_data", "trainingdata"}


def resolve_labelled_training_root(cfg: dict) -> Path:
    """Resolve only the labelled BraTS training cohort.

    The official BraTS validation cohort is deliberately ignored because it has
    no ground-truth segmentation masks and is not part of the 80/10/10 design.
    """
    explicit = cfg.get("train_data_dir")
    if explicit:
        root = Path(explicit)
        patients = find_patient_dirs(str(root))
        if not patients:
            raise RuntimeError(f"No BraTS patients found in --train_data_dir: {root}")
        return root

    root = Path(cfg["data_dir"])
    if not root.exists():
        raise RuntimeError(f"Dataset root does not exist: {root}")

    # If data_dir itself is the labelled training folder, use it directly.
    direct = find_patient_dirs(str(root))
    direct_with_seg = [p for p in direct if any(p.glob("*_seg.nii*"))]
    if direct and len(direct_with_seg) == len(direct):
        return root

    children = [p for p in root.iterdir() if p.is_dir()]
    candidates = [
        p for p in children
        if _looks_like_training_root(p)
        and find_patient_dirs(str(p))
    ]

    if not candidates:
        grandchildren = []
        for child in children:
            try:
                grandchildren.extend([p for p in child.iterdir() if p.is_dir()])
            except OSError:
                pass
        candidates = [
            p for p in grandchildren
            if _looks_like_training_root(p)
            and find_patient_dirs(str(p))
        ]

    labelled = []
    for candidate in candidates:
        pts = find_patient_dirs(str(candidate))
        if pts and all(any(p.glob("*_seg.nii*")) for p in pts):
            labelled.append(candidate)

    if len(labelled) != 1:
        raise RuntimeError(
            "Could not uniquely identify the labelled BraTS training cohort. "
            "Specify it explicitly with --train_data_dir.\n"
            f"Detected labelled training candidates: {[str(p) for p in labelled]}"
        )
    return labelled[0]


def require_segmentation_masks(patient_dirs: List[Path], split_name: str) -> None:
    """Fail loudly if a metric-bearing split has no ground-truth masks."""
    missing = [p for p in patient_dirs if not any(p.glob("*_seg.nii*"))]
    if missing:
        examples = ", ".join(p.name for p in missing[:5])
        raise RuntimeError(
            f"{split_name} contains {len(missing)} patient(s) without *_seg.nii* ground-truth masks "
            f"(examples: {examples}). The 80/10/10 experiment requires labelled cases only."
        )


def make_train_val_test_split(
    patient_dirs: List[Path],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Create a deterministic patient-level train/validation/test split.

    Validation and test counts are rounded to the nearest patient so 369 cases
    become 295 train, 37 validation and 37 final test cases.
    """
    n = len(patient_dirs)
    if n < 3:
        raise RuntimeError("At least three labelled patients are required for an 80/10/10 split.")
    if val_ratio <= 0 or test_ratio <= 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and fixed_test_ratio must be > 0 and sum to < 1.")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n).tolist()
    n_val = max(1, int(round(n * val_ratio)))
    n_test = max(1, int(round(n * test_ratio)))
    if n_val + n_test >= n:
        raise RuntimeError("Validation and test splits leave no training patients.")

    test_idx = perm[:n_test]
    val_idx = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]
    return (
        [patient_dirs[i] for i in train_idx],
        [patient_dirs[i] for i in val_idx],
        [patient_dirs[i] for i in test_idx],
    )


def make_or_load_fixed_80_10_10_split(
    labelled_patients: List[Path],
    cfg: dict,
    logger: logging.Logger,
) -> Tuple[List[Path], List[Path], List[Path], Path]:
    """Create/reuse one fixed patient-level 80/10/10 split for all models."""
    save_dir = Path(cfg["save_dir"])
    configured = cfg.get("fixed_split_file")
    split_path = Path(configured) if configured else save_dir / "fixed_split.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)

    by_name = {p.name: p for p in labelled_patients}
    metadata = {
        "design": "fixed_80_train_10_validation_10_test",
        "seed": int(cfg["seed"]),
        "val_ratio": float(cfg["val_ratio"]),
        "test_ratio": float(cfg["fixed_test_ratio"]),
        "labelled_count": len(labelled_patients),
        "train_data_dir": _normalised_path_string(Path(cfg["train_data_dir"])),
    }

    if split_path.exists():
        try:
            saved = json.loads(split_path.read_text(encoding="utf-8"))
            saved_meta = saved.get("metadata", {})
            same_design = all(saved_meta.get(k) == v for k, v in metadata.items())
            train_names = saved.get("train_patients", [])
            val_names = saved.get("val_patients", [])
            test_names = saved.get("test_patients", [])
            all_names = train_names + val_names + test_names
            names_valid = (
                all(n in by_name for n in all_names)
                and len(all_names) == len(set(all_names))
                and set(all_names) == set(by_name)
            )
            if same_design and names_valid:
                logger.info(f"Reusing fixed 80/10/10 patient split -> {split_path}")
                return (
                    [by_name[n] for n in train_names],
                    [by_name[n] for n in val_names],
                    [by_name[n] for n in test_names],
                    split_path,
                )
        except Exception as exc:
            logger.warning(f"Could not reuse {split_path}: {exc}. Creating a new fixed split.")

    train_pts, val_pts, test_pts = make_train_val_test_split(
        labelled_patients,
        float(cfg["val_ratio"]),
        float(cfg["fixed_test_ratio"]),
        int(cfg["seed"]),
    )
    payload = {
        "design": "fixed_80_train_10_validation_10_test",
        "metadata": metadata,
        "train_patients": [p.name for p in train_pts],
        "val_patients": [p.name for p in val_pts],
        "test_patients": [p.name for p in test_pts],
    }
    split_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(f"Created fixed 80/10/10 patient split -> {split_path}")
    return train_pts, val_pts, test_pts, split_path


def make_cv_splits(
    patient_dirs: List[Path],
    n_folds: int,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Path], List[Tuple[List[Path], List[Path]]]]:
    """
    Partition patients into:
      * a held-out test set  (``test_ratio`` of the full pool, e.g. 10 %)
      * k stratified folds   (over the remaining 90 %)

    Returns
    -------
    test_patients : List[Path]
        Patients that are NEVER used during cross-validation.
    folds : List[Tuple[List[Path], List[Path]]]
        List of (train_patients, val_patients) for each fold.
        len(folds) == n_folds.
    """
    rng  = np.random.default_rng(seed)
    perm = rng.permutation(len(patient_dirs)).tolist()

    # ── 1. carve out the held-out test set ──────────────────────────────────
    n_test       = max(1, int(len(patient_dirs) * test_ratio))
    test_idx     = perm[:n_test]
    trainval_idx = perm[n_test:]
    test_patients = [patient_dirs[i] for i in test_idx]

    # ── 2. k-fold split over the remaining patients ──────────────────────────
    n_tv   = len(trainval_idx)
    fold_size = n_tv // n_folds   # patients per validation fold
    folds: List[Tuple[List[Path], List[Path]]] = []

    for k in range(n_folds):
        val_start = k * fold_size
        val_end   = val_start + fold_size if k < n_folds - 1 else n_tv  # last fold gets remainder
        val_idx_k   = trainval_idx[val_start:val_end]
        train_idx_k = trainval_idx[:val_start] + trainval_idx[val_end:]
        folds.append((
            [patient_dirs[i] for i in train_idx_k],
            [patient_dirs[i] for i in val_idx_k],
        ))

    return test_patients, folds


# ─────────────────────────────────────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """
    Multi-class soft Dice on softmax probabilities, averaged over the foreground
    classes (background excluded by default).

    Accepts raw logits of shape (B, C, *spatial) and an integer target of shape
    (B, *spatial) or (B, 1, *spatial).
    """
    def __init__(self, smooth: float = 1e-5, ignore_background: bool = True):
        super().__init__()
        self.smooth = smooth
        self.ignore_background = ignore_background

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        target = target.long()
        if target.dim() == logits.dim():          # (B, 1, *spatial) -> (B, *spatial)
            target = target.squeeze(1)

        probs   = torch.softmax(logits, dim=1)                       # (B, C, *spatial)
        true_1h = F.one_hot(target, num_classes).movedim(-1, 1).float()  # (B, C, *spatial)

        probs   = probs.flatten(2)      # (B, C, N)
        true_1h = true_1h.flatten(2)
        if self.ignore_background:
            probs   = probs[:, 1:]
            true_1h = true_1h[:, 1:]

        num = 2 * (probs * true_1h).sum(-1) + self.smooth
        den = probs.sum(-1) + true_1h.sum(-1) + self.smooth
        return (1 - num / den).mean()


class CombinedLoss(nn.Module):
    """
    Combined multi-class loss: a * Dice + (1 - a) * categorical cross-entropy.

    Cross-entropy is computed with ``nn.CrossEntropyLoss`` (log-softmax applied
    internally), so this is categorical CE with softmax over the class channel.
    """
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha              # a = 0.5
        self.dice  = DiceLoss()
        self.ce    = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.long()
        target_idx = target.squeeze(1) if target.dim() == logits.dim() else target
        ce_loss   = self.ce(logits, target_idx)     # softmax categorical cross-entropy
        dice_loss = self.dice(logits, target_idx)
        return self.alpha * dice_loss + (1 - self.alpha) * ce_loss


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

# Multi-class label convention (mutually exclusive), remapped from BraTS:
#   0 = background, 1 = NCR/NET, 2 = ED (oedema), 3 = ET (enhancing tumour)
NUM_CLASSES  = 4
REGION_NAMES = ["WT", "TC", "ET"]


def region_masks(cls_map: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Reconstruct the standard overlapping BraTS regions from a class-index map.
        WT (whole tumour) = classes {1, 2, 3}
        TC (tumour core)  = classes {1, 3}
        ET (enhancing)    = class   {3}
    Returns a dict of boolean masks.
    """
    return {
        "WT": cls_map > 0,
        "TC": (cls_map == 1) | (cls_map == 3),
        "ET": cls_map == 3,
    }


def dice_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
    num = 2 * (pred * target).sum() + smooth
    den = pred.sum() + target.sum() + smooth
    return float(num / den)


def safe_nanmean(values) -> float:
    """Mean that returns NaN cleanly when every value is NaN."""
    arr = np.asarray(values, dtype=np.float64)
    valid = arr[~np.isnan(arr)]
    return float(valid.mean()) if valid.size else float("nan")


def hausdorff95(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Exact symmetric HD95 in voxel units, computed on the tight union bounding
    box of the two masks. Cropping is mathematically equivalent here because
    every queried surface point and every possible nearest opposite surface
    remain inside the union box, while the EDT workload can shrink by orders
    of magnitude for small tumours.
    """
    try:
        from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure
    except ImportError:
        return float("nan")

    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)
    pred_any = bool(pred.any())
    target_any = bool(target.any())
    if not pred_any and not target_any:
        return 0.0
    if not pred_any or not target_any:
        return float("nan")

    union = pred | target
    coords = np.where(union)
    crop = tuple(
        slice(max(0, int(c.min()) - 1), min(pred.shape[i], int(c.max()) + 2))
        for i, c in enumerate(coords)
    )
    pred = pred[crop]
    target = target[crop]

    structure = generate_binary_structure(pred.ndim, 1)
    pred_surface = pred ^ binary_erosion(pred, structure=structure, border_value=0)
    target_surface = target ^ binary_erosion(target, structure=structure, border_value=0)

    dt_target = distance_transform_edt(~target_surface)
    d_pred_to_target = dt_target[pred_surface]
    del dt_target

    dt_pred = distance_transform_edt(~pred_surface)
    d_target_to_pred = dt_pred[target_surface]
    del dt_pred

    distances = np.concatenate((d_pred_to_target, d_target_to_pred))
    return float(np.percentile(distances, 95)) if distances.size else float("nan")


def metrics_from_class_maps(pred_cls: np.ndarray, target_cls: np.ndarray) -> Dict[str, float]:
    """Compute BraTS Dice and true HD95 for WT, TC and ET from class maps."""
    pr = region_masks(pred_cls)
    gt = region_masks(target_cls)
    out: Dict[str, float] = {}
    for name in REGION_NAMES:
        out[f"dice_{name}"] = dice_score(
            pr[name].astype(np.float32), gt[name].astype(np.float32)
        )
        out[f"hd95_{name}"] = hausdorff95(pr[name], gt[name])
    out["dice_mean"] = float(np.mean([out[f"dice_{n}"] for n in REGION_NAMES]))
    out["hd95_mean"] = safe_nanmean([out[f"hd95_{n}"] for n in REGION_NAMES])
    return out


def evaluate_batch(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Multi-class evaluation returning both Dice and true HD95 for WT, TC and ET.

    For 3-D models the values are volumetric. For the 2-D model this helper is
    slice-wise during validation; held-out test evaluation reconstructs each
    complete 3-D patient before computing the final Dice and HD95 metrics.
    """
    pred_cls = logits.detach().argmax(dim=1).cpu().numpy()
    tgt = targets.detach().cpu().numpy()
    if tgt.ndim == pred_cls.ndim + 1:
        tgt = tgt[:, 0]
    tgt = tgt.astype(np.int64)

    per: Dict[str, List[float]] = {}
    for name in REGION_NAMES:
        per[f"dice_{name}"] = []
        per[f"hd95_{name}"] = []

    for b in range(pred_cls.shape[0]):
        m = metrics_from_class_maps(pred_cls[b], tgt[b])
        for name in REGION_NAMES:
            per[f"dice_{name}"].append(m[f"dice_{name}"])
            per[f"hd95_{name}"].append(m[f"hd95_{name}"])

    results: Dict[str, float] = {}
    for name in REGION_NAMES:
        results[f"dice_{name}"] = float(np.mean(per[f"dice_{name}"]))
        results[f"hd95_{name}"] = safe_nanmean(per[f"hd95_{name}"])
    results["dice_mean"] = float(np.mean([results[f"dice_{n}"] for n in REGION_NAMES]))
    results["hd95_mean"] = safe_nanmean([results[f"hd95_{n}"] for n in REGION_NAMES])
    return results


def count_trainable_params(model: nn.Module) -> int:
    """Number of trainable parameters, unwrapping torch.compile when needed."""
    return int(sum(p.numel() for p in unwrap_model(model).parameters() if p.requires_grad))


def estimate_gflops(model: nn.Module, sample: torch.Tensor, cfg: dict) -> float:
    """
    Estimate forward-pass GFLOPs with runtime shape hooks. Multiply-add is
    counted as two FLOPs. Conv2d/3d, transposed convolutions, Linear and
    MultiheadAttention are included. The returned value is for one supplied
    input sample. For Diff-UNet this naturally includes the separate image encoder and all
    50 iterative START_X denoising steps executed by ``model(sample)``.
    """
    eager = unwrap_model(model)
    total_flops = 0.0
    handles = []

    def conv_hook(mod, inputs, output):
        nonlocal total_flops
        out = output if torch.is_tensor(output) else output[0]
        if not torch.is_tensor(out):
            return
        out_elements = out.numel()
        kernel_ops = int(np.prod(mod.kernel_size)) * (mod.in_channels // mod.groups)
        total_flops += float(out_elements * kernel_ops * 2)
        if mod.bias is not None:
            total_flops += float(out_elements)

    def linear_hook(mod, inputs, output):
        nonlocal total_flops
        out = output if torch.is_tensor(output) else output[0]
        if not torch.is_tensor(out):
            return
        total_flops += float(out.numel() * mod.in_features * 2)
        if mod.bias is not None:
            total_flops += float(out.numel())

    def mha_hook(mod, inputs, output):
        nonlocal total_flops
        if not inputs:
            return
        q = inputs[0]
        if not torch.is_tensor(q) or q.dim() != 3:
            return
        if getattr(mod, "batch_first", False):
            B, N, E = q.shape
        else:
            N, B, E = q.shape
        H = mod.num_heads
        head_dim = E // H
        # Q, K, V projections + output projection.
        total_flops += float(4 * 2 * B * N * E * E)
        # QK^T and attention-weighted V.
        total_flops += float(2 * 2 * B * H * N * N * head_dim)

    for module in eager.modules():
        if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, nn.MultiheadAttention):
            handles.append(module.register_forward_hook(mha_hook))

    was_training = eager.training
    eager.eval()
    try:
        with torch.inference_mode(), amp_context(sample.device, cfg.get("amp", True)):
            eager(sample)
    finally:
        for h in handles:
            h.remove()
        eager.train(was_training)

    return float(total_flops / 1e9)


# ─────────────────────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────────────────────

# ── Shared building blocks ───────────────────────────────────────────────────

class ConvBnRelu(nn.Sequential):
    def __init__(self, in_c, out_c, kernel=3, stride=1, padding=1):
        super().__init__(
            nn.Conv3d(in_c, out_c, kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
        )


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBnRelu(channels, channels),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


# ── 1. Vanilla 3D UNet  (mtancak / base) ────────────────────────────────────

class UNet3D(nn.Module):
    """
    Standard 3D U-Net.
    Reference: Çiçek et al., 2016 (https://arxiv.org/abs/1606.06650)
    Matches mtancak/PyTorch-UNet-Brain-Cancer-Segmentation architecture.
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 3,
                 base_filters: int = 32):
        super().__init__()
        f = base_filters

        def enc(ic, oc):
            return nn.Sequential(ConvBnRelu(ic, oc), ConvBnRelu(oc, oc))

        self.enc1 = enc(in_channels, f)
        self.enc2 = enc(f,    f*2)
        self.enc3 = enc(f*2,  f*4)
        self.enc4 = enc(f*4,  f*8)
        self.bottleneck = enc(f*8, f*16)

        self.pool = nn.MaxPool3d(2)

        self.up4   = nn.ConvTranspose3d(f*16, f*8, 2, stride=2)
        self.dec4  = enc(f*16, f*8)
        self.up3   = nn.ConvTranspose3d(f*8,  f*4, 2, stride=2)
        self.dec3  = enc(f*8,  f*4)
        self.up2   = nn.ConvTranspose3d(f*4,  f*2, 2, stride=2)
        self.dec2  = enc(f*4,  f*2)
        self.up1   = nn.ConvTranspose3d(f*2,  f,   2, stride=2)
        self.dec1  = enc(f*2,  f)
        self.head  = nn.Conv3d(f, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


# ── 2. HybridAttUnet 3D  (HA-RUnet) ────────────────────────────────────────────────

class SafeBatchNorm3d(nn.BatchNorm3d):
    """
    BatchNorm3d with a batch-size-1 safeguard for a degenerate 1x1x1 feature map.

    Standard BatchNorm3d is used whenever more than one value per channel is
    available.  If B*D*H*W == 1 during training, batch statistics are undefined,
    so the layer falls back to its running statistics for that forward pass.
    This is required by the deepest HybridAttUnet attention branch when the
    physical training batch size is 1.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values_per_channel = x.numel() // x.shape[1]
        if self.training and values_per_channel <= 1:
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                momentum=0.0,
                eps=self.eps,
            )
        return super().forward(x)

class HARResidualBlock3D(nn.Module):
    """
    Pre-activation 3-D residual bottleneck block used by the HA-RUnet
    reimplementation.

    The architecture uses residual blocks with identity mapping and three
    BN -> ReLU -> convolution sub-blocks.  The paper does not publish the
    authors' source code or every kernel/channel detail, so this PyTorch
    implementation follows the residual-attention bottleneck convention
    (1x1x1, 3x3x3, 1x1x1) while preserving the reported three-stage
    pre-activation residual mechanism.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        mid_channels = max(out_channels // 4, 8)

        self.bn1 = SafeBatchNorm3d(in_channels)
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn2 = SafeBatchNorm3d(mid_channels)
        self.conv2 = nn.Conv3d(
            mid_channels, mid_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn3 = SafeBatchNorm3d(mid_channels)
        self.conv3 = nn.Conv3d(mid_channels, out_channels, kernel_size=1, bias=False)

        self.identity = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.identity(x)
        y = self.conv1(F.relu(self.bn1(x), inplace=True))
        y = self.conv2(F.relu(self.bn2(y), inplace=True))
        y = self.conv3(F.relu(self.bn3(y), inplace=True))
        return identity + y


class HARSoftMaskBranch3D(nn.Module):
    """
    Encoder-decoder soft-mask branch of the residual-attention module.

    D controls branch depth.  The four skip-level attention modules use
    D = 1, 2, 3 and 4 respectively, matching the HA-RUnet design.
    """
    def __init__(self, channels: int, depth: int, r: int = 1):
        super().__init__()
        self.depth = int(depth)
        self.down_blocks = nn.ModuleList([
            nn.Sequential(*[HARResidualBlock3D(channels, channels) for _ in range(r)])
            for _ in range(self.depth)
        ])
        self.bottom = nn.Sequential(
            HARResidualBlock3D(channels, channels),
            HARResidualBlock3D(channels, channels),
        )
        self.up_blocks = nn.ModuleList([
            nn.Sequential(*[HARResidualBlock3D(channels, channels) for _ in range(r)])
            for _ in range(self.depth)
        ])

        # The paper specifies two convolution layers followed by sigmoid at
        # the end of the soft-mask branch.
        self.mask_head = nn.Sequential(
            SafeBatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=1, bias=True),
            SafeBatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        skips = []
        for block in self.down_blocks:
            y = F.max_pool3d(y, kernel_size=2, stride=2)
            y = block(y)
            skips.append(y)

        y = self.bottom(y)

        for block, skip in zip(self.up_blocks, reversed(skips)):
            y = F.interpolate(
                y, size=skip.shape[2:], mode='trilinear', align_corners=False
            )
            y = y + skip
            y = block(y)

        if y.shape[2:] != x.shape[2:]:
            y = F.interpolate(
                y, size=x.shape[2:], mode='trilinear', align_corners=False
            )
        return self.mask_head(y)


class HARResidualAttention3D(nn.Module):
    """
    Residual-attention module used by HybridAttUnet 3D, following:
        Y_A(I) = (1 + S(I)) * F(I)

    p = 1 preprocessing residual block, t = 2 trunk residual blocks and
    r = 1 residual block per soft-mask encoder/decoder level, as depicted in
    the paper's residual-attention figure.
    """
    def __init__(self, channels: int, depth: int, p: int = 1,
                 t: int = 2, r: int = 1):
        super().__init__()
        self.pre = nn.Sequential(*[
            HARResidualBlock3D(channels, channels) for _ in range(p)
        ])
        self.trunk = nn.Sequential(*[
            HARResidualBlock3D(channels, channels) for _ in range(t)
        ])
        self.soft_mask = HARSoftMaskBranch3D(channels, depth=depth, r=r)
        self.post = HARResidualBlock3D(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        trunk = self.trunk(x)
        mask = self.soft_mask(x)
        attended = (1.0 + mask) * trunk
        return self.post(attended)


class HARSqueezeExcitation3D(nn.Module):
    """3-D squeeze-excitation channel recalibration used in the decoder."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Conv3d(channels, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv3d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.adaptive_avg_pool3d(x, output_size=1)
        weights = F.relu(self.fc1(weights), inplace=True)
        weights = torch.sigmoid(self.fc2(weights))
        return x * weights


class HybridAttUnet3D(nn.Module):
    """
    HybridAttUnet 3D, an implementation of the Hybrid Attention-Based Residual U-Net (HA-RUnet), adapted from:

       (2023), A Hybrid Attention-Based Residual Unet for
      Semantic Segmentation of Brain Tumor, Computers, Materials & Continua,
      76(1), 647-664. DOI: 10.32604/cmc.2023.039188.

    Architecture retained from the paper:
      * 3-D residual U-Net backbone
      * four residual-attention modules on encoder-decoder skip pathways
      * residual-attention depths D = 1, 2, 3, 4 from shallow to deep
      * squeeze-excitation recalibration in each decoder stage
      * four stacked MRI modalities
      * model-specific 128x128x128 working volume

    For fair integration with this pipeline, the network returns logits at the
    input spatial resolution.  Thus a 240x240x160 pipeline tensor is internally
    resized to 128^3, processed by HA-RUnet, then the logits are resized back
    before the common Dice + categorical cross-entropy loss and metrics.

    The paper reports 13,253,348 trainable parameters for the TensorFlow/Keras
    implementation.  Because the paper does not publish all low-level layer
    specifications/source code, this is an architecture-faithful PyTorch
    reimplementation rather than a claim of bit-for-bit source equivalence.
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 4,
                 base_filters: int = 32, internal_size: int = 128):
        super().__init__()
        # The HA-RUnet design uses a 32-channel stem followed by
        # 64/128/256/512 residual feature levels.  Keep that layout fixed;
        # base_filters is accepted for compatibility with the unified builder.
        _ = base_filters
        self.internal_size = int(internal_size)

        self.stem = nn.Conv3d(
            in_channels, 32, kernel_size=3, padding=1, bias=True
        )

        self.enc1 = HARResidualBlock3D(32, 64)
        self.enc2 = HARResidualBlock3D(64, 128)
        self.enc3 = HARResidualBlock3D(128, 256)
        self.enc4 = HARResidualBlock3D(256, 512)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Attention module number increases with depth in the paper, hence
        # D = 1,2,3,4 for the four skip pathways.
        self.att1 = HARResidualAttention3D(64, depth=1)
        self.att2 = HARResidualAttention3D(128, depth=2)
        self.att3 = HARResidualAttention3D(256, depth=3)
        self.att4 = HARResidualAttention3D(512, depth=4)

        self.bottleneck = HARResidualBlock3D(512, 512)

        self.dec4 = HARResidualBlock3D(512 + 512, 512)
        self.se4 = HARSqueezeExcitation3D(512)
        self.dec3 = HARResidualBlock3D(512 + 256, 256)
        self.se3 = HARSqueezeExcitation3D(256)
        self.dec2 = HARResidualBlock3D(256 + 128, 128)
        self.se2 = HARSqueezeExcitation3D(128)
        self.dec1 = HARResidualBlock3D(128 + 64, 64)
        self.se1 = HARSqueezeExcitation3D(64)

        self.head = nn.Conv3d(
            64, out_channels, kernel_size=3, padding=1, bias=True
        )

    @staticmethod
    def _upsample_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x, size=ref.shape[2:], mode='trilinear', align_corners=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape[2:]
        working_shape = (self.internal_size,) * 3
        if tuple(original_shape) != working_shape:
            x = F.interpolate(
                x, size=working_shape, mode='trilinear', align_corners=False
            )

        x0 = self.stem(x)
        e1 = self.enc1(x0)                    # 128^3, 64 ch
        e2 = self.enc2(self.pool(e1))         # 64^3, 128 ch
        e3 = self.enc3(self.pool(e2))         # 32^3, 256 ch
        e4 = self.enc4(self.pool(e3))         # 16^3, 512 ch
        b = self.bottleneck(self.pool(e4))    # 8^3, 512 ch

        # Residual-attention skip features.
        s1 = self.att1(e1)
        s2 = self.att2(e2)
        s3 = self.att3(e3)
        s4 = self.att4(e4)

        d4 = self._upsample_to(b, s4)
        d4 = self.se4(self.dec4(torch.cat([d4, s4], dim=1)))

        d3 = self._upsample_to(d4, s3)
        d3 = self.se3(self.dec3(torch.cat([d3, s3], dim=1)))

        d2 = self._upsample_to(d3, s2)
        d2 = self.se2(self.dec2(torch.cat([d2, s2], dim=1)))

        d1 = self._upsample_to(d2, s1)
        d1 = self.se1(self.dec1(torch.cat([d1, s1], dim=1)))

        logits = self.head(d1)
        if tuple(logits.shape[2:]) != tuple(original_shape):
            logits = F.interpolate(
                logits, size=original_shape, mode='trilinear', align_corners=False
            )
        return logits


# ── 3. DeepEnsemble 3D  (Henry et al., BraTS 2020) ──────────────────────────

class HenryGroupNorm3d(nn.GroupNorm):
    """GroupNorm used by the Henry et al. Pipeline A / open_brats2020 default."""
    def __init__(self, channels: int, max_groups: int = 16):
        groups = min(int(max_groups), int(channels))
        while groups > 1 and channels % groups != 0:
            groups -= 1
        super().__init__(groups, channels)


class HenryConvNormRelu(nn.Module):
    """3x3x3 convolution followed by GroupNorm and ReLU."""
    def __init__(self, in_c: int, out_c: int, groups: int = 16,
                 dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Conv3d(
            in_c, out_c, kernel_size=3, stride=1,
            padding=dilation, dilation=dilation, bias=False
        )
        self.norm = HenryGroupNorm3d(out_c, groups)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.act(self.norm(self.conv(x))))


class HenryUBlock(nn.Module):
    """Two-convolution U-Net stage matching the reference EquiUnet."""
    def __init__(self, in_c: int, mid_c: int, out_c: int,
                 groups: int = 16, dilation=(1, 1), dropout: float = 0.0):
        super().__init__()
        self.c1 = HenryConvNormRelu(
            in_c, mid_c, groups=groups, dilation=int(dilation[0]), dropout=dropout
        )
        self.c2 = HenryConvNormRelu(
            mid_c, out_c, groups=groups, dilation=int(dilation[1]), dropout=dropout
        )

    def forward(self, x):
        return self.c2(self.c1(x))


class HenryEquiUNet3D(nn.Module):
    """
    Paper-based 3-D U-Net backbone adapted from Henry et al. (2020),
    arXiv:2011.01045, and the accompanying open_brats2020 EquiUnet code.

    Architectural elements retained from the paper:
      * four encoder stages, width 48 then doubling after each pooling step
      * two 3x3x3 convolutions per encoder stage
      * GroupNorm + ReLU
      * pseudo-fifth stage made of two dilation-2 convolutions without
        additional spatial downsampling
      * concatenation of the dilated block with encoder stage 4
      * trilinear decoder upsampling and concatenated skip connections
      * four deep-supervision heads

    Study-specific adaptation:
      * four mutually exclusive classes and the common multi-class CombinedLoss
        are retained for comparability with the other six architectures.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        width: int = 48,
        norm_groups: int = 16,
        deep_supervision: bool = True,
        dropout: float = 0.0,
        activation_checkpointing: bool = True,
    ):
        super().__init__()
        w = int(width)
        f0, f1, f2, f3 = w, w * 2, w * 4, w * 8

        self.deep_supervision = bool(deep_supervision)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder1 = HenryUBlock(in_channels, f0, f0, norm_groups, dropout=dropout)
        self.encoder2 = HenryUBlock(f0, f1, f1, norm_groups, dropout=dropout)
        self.encoder3 = HenryUBlock(f1, f2, f2, norm_groups, dropout=dropout)
        self.encoder4 = HenryUBlock(f2, f3, f3, norm_groups, dropout=dropout)

        # Pseudo-fifth stage, exactly the paper's dilation trick.
        self.bottom = HenryUBlock(
            f3, f3, f3, norm_groups, dilation=(2, 2), dropout=dropout
        )
        self.bottom_2 = HenryConvNormRelu(
            f3 * 2, f2, groups=norm_groups, dilation=1, dropout=dropout
        )

        self.decoder3 = HenryUBlock(
            f2 * 2, f2, f1, norm_groups, dropout=dropout
        )
        self.decoder2 = HenryUBlock(
            f1 * 2, f1, f0, norm_groups, dropout=dropout
        )
        self.decoder1 = HenryUBlock(
            f0 * 2, f0, f0, norm_groups, dropout=dropout
        )

        self.outconv = nn.Conv3d(f0, out_channels, kernel_size=1, bias=True)

        if self.deep_supervision:
            self.deep_bottom = nn.Conv3d(f3, out_channels, kernel_size=1)
            self.deep_bottom2 = nn.Conv3d(f2, out_channels, kernel_size=1)
            self.deep3 = nn.Conv3d(f1, out_channels, kernel_size=1)
            self.deep2 = nn.Conv3d(f0, out_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.InstanceNorm3d, nn.BatchNorm3d)):
                if getattr(m, "weight", None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _upsample_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x, size=ref.shape[2:], mode="trilinear", align_corners=True
        )

    @staticmethod
    def _upsample_to_input(x: torch.Tensor, spatial_shape) -> torch.Tensor:
        if tuple(x.shape[2:]) == tuple(spatial_shape):
            return x
        return F.interpolate(
            x, size=spatial_shape, mode="trilinear", align_corners=True
        )

    def _run_block(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        Run a Henry U-Net block with activation checkpointing during training.
        The block computation is unchanged; internal activations are recomputed
        during backward instead of being retained from the forward pass.
        """
        if (
            self.training
            and self.activation_checkpointing
            and torch.is_grad_enabled()
        ):
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(self, x: torch.Tensor):
        down1 = self._run_block(self.encoder1, x)
        down2 = self._run_block(self.encoder2, self.pool(down1))
        down3 = self._run_block(self.encoder3, self.pool(down2))
        down4 = self._run_block(self.encoder4, self.pool(down3))

        bottom = self._run_block(self.bottom, down4)
        bottom_2 = self._run_block(
            self.bottom_2, torch.cat([down4, bottom], dim=1)
        )

        up3 = self._upsample_like(bottom_2, down3)
        up3 = self._run_block(
            self.decoder3, torch.cat([down3, up3], dim=1)
        )

        up2 = self._upsample_like(up3, down2)
        up2 = self._run_block(
            self.decoder2, torch.cat([down2, up2], dim=1)
        )

        up1 = self._upsample_like(up2, down1)
        up1 = self._run_block(
            self.decoder1, torch.cat([down1, up1], dim=1)
        )

        logits = self.outconv(up1)

        if self.training and self.deep_supervision:
            # Keep aux logits at native resolution. The loss helper upsamples
            # one head at a time, giving the same loss values without holding
            # four full-resolution auxiliary logits at once.
            aux_native = [
                self.deep_bottom(bottom),
                self.deep_bottom2(bottom_2),
                self.deep3(up3),
                self.deep2(up2),
            ]
            return logits, aux_native

        return logits


def _deepensemble_tta_spec():
    """Identity plus 15 flip/rotation transforms, matching open_brats2020."""
    specs = [(None, 0)]
    for flip in (2, 3, 4, None):
        for rot in (1, 2, 3, 0):
            if flip is None and rot == 0:
                continue
            specs.append((flip, rot))
    return specs


def _apply_deepensemble_tta(x: torch.Tensor, flip, rot):
    y = x
    if flip is not None:
        y = torch.flip(y, dims=(int(flip),))
    if rot:
        # Our tensors are (B,C,H,W,D). Rotate in the axial H/W plane.
        y = torch.rot90(y, int(rot), dims=(2, 3))
    return y


def _revert_deepensemble_tta(x: torch.Tensor, flip, rot):
    # Forward transform is R(F(x)); inverse is F(R^{-1}(x)).
    y = x
    if rot:
        y = torch.rot90(y, -int(rot), dims=(2, 3))
    if flip is not None:
        y = torch.flip(y, dims=(int(flip),))
    return y


class DeepEnsemble3D(nn.Module):
    """
    Inference wrapper for independently trained Henry-style members.

    Member probabilities are averaged. By default each member is evaluated
    with the 16-way TTA scheme used by the reference repository.
    """
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        width: int = 48,
        member_count: int = 5,
        tta: bool = True,
        norm_groups: int = 16,
        deep_supervision: bool = True,
    ):
        super().__init__()
        self.member_count = int(member_count)
        self.tta = bool(tta)
        self.members = nn.ModuleList([
            HenryEquiUNet3D(
                in_channels=in_channels,
                out_channels=out_channels,
                width=width,
                norm_groups=norm_groups,
                deep_supervision=deep_supervision,
                activation_checkpointing=False,
            )
            for _ in range(self.member_count)
        ])

    def load_member_state_dicts(self, states):
        if len(states) != len(self.members):
            raise ValueError(
                f"Checkpoint has {len(states)} members, expected {len(self.members)}."
            )
        for member, state in zip(self.members, states):
            member.load_state_dict(state)

    def forward(self, x: torch.Tensor):
        total = None
        count = 0
        specs = _deepensemble_tta_spec() if self.tta else [(None, 0)]

        for member in self.members:
            member.eval()
            for flip, rot in specs:
                tx = _apply_deepensemble_tta(x, flip, rot)
                logits = member(tx)
                probs = torch.softmax(logits, dim=1)
                probs = _revert_deepensemble_tta(probs, flip, rot)
                total = probs if total is None else total + probs
                count += 1

        mean_probs = total / float(max(1, count))
        return torch.log(mean_probs.clamp_min(1e-7))


def _deep_supervised_output_and_loss(
    pred,
    label,
    criterion,
    checkpoint_losses: bool = True,
):
    """
    Return main logits and the unweighted main + four auxiliary losses.

    Auxiliary predictions are upsampled one at a time. During training, the
    criterion calculations can also be checkpointed so the large Dice/CE
    intermediates are recomputed during backward rather than kept in VRAM.
    The loss definition and numerical forward values are unchanged.
    """
    if not (
        isinstance(pred, tuple)
        and len(pred) == 2
        and isinstance(pred[1], (list, tuple))
    ):
        return pred, criterion(pred, label)

    main, aux_native = pred

    def _main_loss(logits, target):
        return criterion(logits, target)

    if checkpoint_losses and torch.is_grad_enabled():
        loss = checkpoint(_main_loss, main, label, use_reentrant=False)
    else:
        loss = criterion(main, label)

    for aux_logits_native in aux_native:
        def _aux_loss(aux_logits, target):
            # Segmentation targets are stored as (B, 1, H, W, D), whereas
            # auxiliary logits are (B, C, h, w, d).  F.interpolate expects
            # only the three spatial output dimensions, so always take the
            # final three target dimensions rather than target.shape[1:].
            target_spatial = tuple(target.shape[-3:])
            if tuple(aux_logits.shape[2:]) != target_spatial:
                aux_logits = F.interpolate(
                    aux_logits,
                    size=target_spatial,
                    mode="trilinear",
                    align_corners=True,
                )
            return criterion(aux_logits, target)

        if checkpoint_losses and torch.is_grad_enabled():
            aux_loss = checkpoint(
                _aux_loss, aux_logits_native, label, use_reentrant=False
            )
        else:
            aux_loss = _aux_loss(aux_logits_native, label)

        loss = loss + aux_loss

    return main, loss


# ── 4. Diff-UNet  (ge-xing/Diff-UNet, PyTorch) ──────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding used by the diffusion denoiser."""
    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 1:
            t = t.reshape(-1)
        half = self.dim // 2
        if half < 2:
            raise ValueError("Diffusion timestep embedding dimension must be >= 4.")
        scale = np.log(10000.0) / float(half - 1)
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32) * (-scale)
        )
        emb = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb


def _diff_norm(channels: int) -> nn.Module:
    """Instance normalisation used by the published Diff-UNet building blocks."""
    return nn.InstanceNorm3d(channels, affine=True)


class DiffTwoConv(nn.Module):
    """Two 3-D convolutions with InstanceNorm and LeakyReLU."""
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, 3, padding=1, bias=True)
        self.norm1 = _diff_norm(out_c)
        self.act1 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.conv2 = nn.Conv3d(out_c, out_c, 3, padding=1, bias=True)
        self.norm2 = _diff_norm(out_c)
        self.act2 = nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        return x


class DiffTimeTwoConv(nn.Module):
    """Diff-UNet two-convolution block with additive timestep conditioning."""
    def __init__(self, in_c: int, out_c: int, time_dim: int = 512):
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, 3, padding=1, bias=True)
        self.norm1 = _diff_norm(out_c)
        self.act1 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.time_proj = nn.Linear(time_dim, out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, 3, padding=1, bias=True)
        self.norm2 = _diff_norm(out_c)
        self.act2 = nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.norm1(self.conv1(x)))
        x = x + self.time_proj(F.silu(temb))[:, :, None, None, None]
        x = self.act2(self.norm2(self.conv2(x)))
        return x


class DiffImageEncoder(nn.Module):
    """
    Separate multiscale MRI encoder used to condition the diffusion denoiser.

    This mirrors the image-embedding branch in the reference BraTS2020
    Diff-UNet implementation. The common pipeline base-filter setting controls
    its width so the architecture can still be compared under the same
    full-volume experimental protocol.
    """
    def __init__(self, in_channels: int = 4, base_filters: int = 32):
        super().__init__()
        f = int(base_filters)
        self.pool = nn.MaxPool3d(2)
        self.enc0 = DiffTwoConv(in_channels, f)
        self.enc1 = DiffTwoConv(f, f)
        self.enc2 = DiffTwoConv(f, f * 2)
        self.enc3 = DiffTwoConv(f * 2, f * 4)
        self.enc4 = DiffTwoConv(f * 4, f * 8)

    def forward(self, image: torch.Tensor) -> List[torch.Tensor]:
        x0 = self.enc0(image)
        x1 = self.enc1(self.pool(x0))
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        return [x0, x1, x2, x3, x4]


class DiffDenoiser3D(nn.Module):
    """
    Time-conditioned U-Net denoiser with additive multiscale MRI embeddings.

    The noisy segmentation state and MRI modalities are concatenated at the
    denoiser input, while the separate image encoder contributes features at
    each encoder resolution, matching the conditioning pattern of Diff-UNet.
    """
    def __init__(self, image_channels: int, seg_channels: int,
                 base_filters: int = 32, t_embed_dim: int = 128,
                 time_dim: int = 512):
        super().__init__()
        f = int(base_filters)
        self.pool = nn.MaxPool3d(2)

        self.time_embedding = SinusoidalTimeEmbedding(t_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(t_embed_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.enc0 = DiffTimeTwoConv(image_channels + seg_channels, f, time_dim)
        self.enc1 = DiffTimeTwoConv(f, f, time_dim)
        self.enc2 = DiffTimeTwoConv(f, f * 2, time_dim)
        self.enc3 = DiffTimeTwoConv(f * 2, f * 4, time_dim)
        self.enc4 = DiffTimeTwoConv(f * 4, f * 8, time_dim)

        self.up4 = nn.ConvTranspose3d(f * 8, f * 4, 2, stride=2)
        self.dec4 = DiffTimeTwoConv(f * 8, f * 4, time_dim)
        self.up3 = nn.ConvTranspose3d(f * 4, f * 2, 2, stride=2)
        self.dec3 = DiffTimeTwoConv(f * 4, f * 2, time_dim)
        self.up2 = nn.ConvTranspose3d(f * 2, f, 2, stride=2)
        self.dec2 = DiffTimeTwoConv(f * 2, f, time_dim)
        # The final upsampling keeps f channels, as in the reference BasicUNetDe.
        self.up1 = nn.ConvTranspose3d(f, f, 2, stride=2)
        self.dec1 = DiffTimeTwoConv(f * 2, f, time_dim)
        self.head = nn.Conv3d(f, seg_channels, 1)

    @staticmethod
    def _match_shape(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Pad/crop a decoder tensor to exactly match its skip connection."""
        target = ref.shape[2:]
        if x.shape[2:] == target:
            return x
        # Interpolation is only a shape safeguard for odd pooled dimensions.
        return F.interpolate(x, size=target, mode="trilinear", align_corners=False)

    def forward(self, xt: torch.Tensor, image: torch.Tensor, t: torch.Tensor,
                embeddings: List[torch.Tensor]) -> torch.Tensor:
        temb = self.time_mlp(self.time_embedding(t))
        x = torch.cat([image, xt], dim=1)

        x0 = self.enc0(x, temb) + embeddings[0]
        x1 = self.enc1(self.pool(x0), temb) + embeddings[1]
        x2 = self.enc2(self.pool(x1), temb) + embeddings[2]
        x3 = self.enc3(self.pool(x2), temb) + embeddings[3]
        x4 = self.enc4(self.pool(x3), temb) + embeddings[4]

        u4 = self._match_shape(self.up4(x4), x3)
        u4 = self.dec4(torch.cat([x3, u4], dim=1), temb)
        u3 = self._match_shape(self.up3(u4), x2)
        u3 = self.dec3(torch.cat([x2, u3], dim=1), temb)
        u2 = self._match_shape(self.up2(u3), x1)
        u2 = self.dec2(torch.cat([x1, u2], dim=1), temb)
        u1 = self._match_shape(self.up1(u2), x0)
        u1 = self.dec1(torch.cat([x0, u1], dim=1), temb)
        return self.head(u1)


class DiffUNet(nn.Module):
    """
    Multi-class adaptation of Diff-UNet for the common BraTS2020 experiment.

    Reference implementation: ge-xing/Diff-UNet, BraTS2020 branch.

    Key diffusion behaviour retained from Diff-UNet:
      * a separate 3-D MRI encoder supplies multiscale image embeddings;
      * the segmentation itself is diffused over T=1000 timesteps;
      * the denoiser uses START_X parameterisation, predicting the clean
        segmentation state rather than diffusion noise;
      * inference uses deterministic DDIM-style sampling with 50 timesteps.

    Adaptations required by the common experiment:
      * four mutually exclusive classes are used instead of three overlapping
        WT/TC/ET channels;
      * the denoiser emits four class logits and is optimised with the same
        0.5 Dice + 0.5 categorical cross-entropy loss as every other model;
      * full 240x240x160 inputs are retained rather than 96x96x96 training crops.

    For diffusion, the one-hot segmentation is mapped from {0,1} to [-1,1].
    Predicted class logits are converted back to the same diffusion state via
    x0 = 2*softmax(logits)-1 for each DDIM update.
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 4,
                 base_filters: int = 32, T: int = 1000,
                 infer_steps: int = 50):
        super().__init__()
        self.T = int(T)
        self.infer_steps = int(infer_steps)
        self.out_channels = int(out_channels)

        self.embed_model = DiffImageEncoder(in_channels, base_filters)
        self.model = DiffDenoiser3D(
            image_channels=in_channels,
            seg_channels=out_channels,
            base_filters=base_filters,
        )

        # Linear DDPM schedule used by the reference implementation at T=1000.
        betas = torch.linspace(1e-4, 0.02, self.T, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_abar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_1mabar", torch.sqrt(1.0 - alpha_bar))

    def _target_to_x0(self, target: torch.Tensor) -> torch.Tensor:
        """Convert integer class labels to a four-channel diffusion state in [-1,1]."""
        if target.dim() == 5 and target.shape[1] == 1:
            target = target[:, 0]
        if target.dim() != 4:
            raise ValueError(
                f"Diff-UNet target must be (B,D,H,W) or (B,1,D,H,W); got {tuple(target.shape)}"
            )
        one_hot = F.one_hot(target.long(), self.out_channels).movedim(-1, 1).float()
        return one_hot.mul(2.0).sub(1.0)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward diffusion q(x_t | x_0)."""
        noise = torch.randn_like(x0)
        sa = self.sqrt_abar[t][:, None, None, None, None]
        sma = self.sqrt_1mabar[t][:, None, None, None, None]
        return sa * x0 + sma * noise

    def _predict_logits(self, xt: torch.Tensor, image: torch.Tensor,
                        t: torch.Tensor,
                        embeddings: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        if embeddings is None:
            embeddings = self.embed_model(image)
        return self.model(xt, image, t, embeddings)

    def forward(self, image: torch.Tensor,
                target: Optional[torch.Tensor] = None,
                t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Training: ``model(image, target)`` returns predicted clean-segmentation
        logits at a randomly sampled diffusion timestep.

        Evaluation: ``model(image)`` runs 50-step deterministic DDIM sampling
        and returns final four-class logits for the common argmax evaluation.
        """
        if target is not None:
            if t is None:
                t = torch.randint(0, self.T, (image.shape[0],), device=image.device)
            x0 = self._target_to_x0(target)
            xt = self.q_sample(x0, t)
            embeddings = self.embed_model(image)
            return self._predict_logits(xt, image, t, embeddings)

        return self._ddim_sample(image, steps=self.infer_steps)

    @torch.no_grad()
    def _ddim_sample(self, image: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
        """
        Deterministic START_X DDIM-style sampling.

        The denoiser predicts class logits. Softmax converts those logits to
        class probabilities, which are mapped to [-1,1] to obtain the clean
        diffusion state x_0 used in the DDIM update.
        """
        steps = int(self.infer_steps if steps is None else steps)
        if steps < 1:
            raise ValueError("Diff-UNet inference requires at least one DDIM step.")

        B = image.shape[0]
        dev = image.device
        embeddings = self.embed_model(image)
        xt = torch.randn(
            B, self.out_channels, *image.shape[2:],
            device=dev, dtype=image.dtype,
        )

        # Equivalent to the evenly spaced 50-step respacing used by the
        # reference BraTS2020 implementation when T=1000.
        step_indices = torch.linspace(
            self.T - 1, 0, steps, device=dev, dtype=torch.float32
        ).round().long()

        final_logits = None
        eps = 1e-8
        for i, ti in enumerate(step_indices):
            t_batch = ti.expand(B)
            logits = self._predict_logits(xt, image, t_batch, embeddings)
            final_logits = logits

            # START_X prediction in the diffusion state space.
            x0_pred = torch.softmax(logits.float(), dim=1)
            x0_pred = x0_pred.mul(2.0).sub(1.0).to(dtype=xt.dtype)

            if i == len(step_indices) - 1:
                break

            abar_t = self.alpha_bar[ti].to(dtype=xt.dtype)
            ti_prev = step_indices[i + 1]
            abar_prev = self.alpha_bar[ti_prev].to(dtype=xt.dtype)

            # Recover epsilon from x_t and the predicted x_0, then perform the
            # deterministic eta=0 DDIM update to the previous respaced step.
            eps_pred = (xt - torch.sqrt(abar_t) * x0_pred) / (
                torch.sqrt(1.0 - abar_t) + eps
            )
            xt = (
                torch.sqrt(abar_prev) * x0_pred
                + torch.sqrt(1.0 - abar_prev) * eps_pred
            )

        if final_logits is None:
            raise RuntimeError("Diff-UNet DDIM sampler produced no prediction.")
        return final_logits


# ── 5. HVU 2D / DenseVU-ED ──────────────────────────────────────────────────
# Paper-aligned 2-D implementation based on:
# Renugadevi et al. (2025), Scientific Reports 15:23742.
# The paper describes a 256x256 2-D HVU-ED segmenter that fuses:
#   (1) U-Net encoder features,
#   (2) Vision Transformer global features using non-overlapping 16x16 patches,
#   (3) features from a CNN architecture such as DenseNet121,
# at the U-Net bottleneck. DenseVU-ED was the best-performing BraTS2020
# segmentation variant in the paper. The paper states that architectural
# knowledge, rather than pretrained weights, is transferred, so weights=None
# is used for the DenseNet branch.

class PatchEmbed2D(nn.Module):
    """Non-overlapping 16x16 ViT patch embedding used by the published HVU."""
    def __init__(self, in_channels: int, embed_dim: int = 256, patch_size: int = 16):
        super().__init__()
        self.patch_size = int(patch_size)
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x: torch.Tensor):
        x = self.proj(x)                          # (B,E,16,16) for 256x256 input
        B, E, H, W = x.shape
        return x.flatten(2).transpose(1, 2), (H, W)


class HVUTransformerBlock2D(nn.Module):
    """ViT encoder block for the 2-D HVU global-context branch."""
    def __init__(self, dim: int, heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.norm1(x)
        # HVU does not use attention weights. need_weights=False permits
        # memory-efficient scaled-dot-product attention on supported GPUs.
        h, _ = self.attn(qkv, qkv, qkv, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class DenseNet121EarlyFeatures2D(nn.Module):
    """
    DenseNet121 architectural feature branch used by DenseVU-ED.

    The paper reports using the first five DenseNet121 feature layers and
    explicitly states that pretrained weights are not transferred. torchvision
    is imported lazily so the other models do not depend on it at runtime.
    """
    def __init__(self, in_channels: int = 4):
        super().__init__()
        try:
            from torchvision.models import densenet121
        except Exception as exc:
            raise ImportError(
                "Model 6 (2-D DenseVU-ED HVU) requires torchvision. "
                "Install a torchvision build compatible with your PyTorch/CUDA installation."
            ) from exc

        dense = densenet121(weights=None)
        # BraTS supplies four MRI modalities rather than RGB.
        dense.features.conv0 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # torchvision DenseNet121 features are ordered as:
        # conv0, norm0, relu0, pool0, denseblock1, transition1, ...
        # Retain the first five top-level feature modules as described by the paper.
        first_five = list(dense.features._modules.items())[:5]
        self.features = nn.Sequential(OrderedDict(first_five))
        self.out_channels = 256  # DenseNet121 denseblock1 output width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class HVUNet(nn.Module):
    """
    2-D DenseVU-ED style Hybrid Vision U-Net.

    Paper-aligned features:
      * 2-D MRI slices resized internally to 256x256.
      * U-Net encoder-decoder with 3x3 convolutions, ReLU and 2x pooling.
      * ViT branch uses non-overlapping 16x16 patches and learned positional
        embeddings for global context.
      * DenseNet121 architectural features, with no pretrained weights.
      * U-Net, DenseNet and ViT representations are fused at the 16x16
        bottleneck before decoding.

    The surrounding experiment deliberately retains the common pipeline used
    by every other architecture: the same four MRI modalities, 80/10/10
    patient split, GPU augmentation policy, CombinedLoss, optimiser, scheduler,
    early stopping and final Dice/HD95 evaluation. The network output is resized
    back to the incoming slice geometry so patient reconstruction and metrics
    remain directly comparable with the other models.
    """
    PAPER_INPUT_SIZE = (256, 256)

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        base_filters: int = 32,       # accepted for registry compatibility
        embed_dim: int = 256,
        num_heads: int = 8,
        depth: int = 4,
        patch_size: int = 16,
    ):
        super().__init__()

        # The published HVU-ED is a classic-width 2-D U-Net. Use f=64 here
        # independently of the generic 3-D base_filters setting so Model 6 is
        # a paper-aligned architecture rather than a width-scaled 3-D port.
        f = 64

        def double_conv(ic: int, oc: int):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, padding=1, bias=False),
                nn.BatchNorm2d(oc),
                nn.ReLU(inplace=True),
                nn.Conv2d(oc, oc, 3, padding=1, bias=False),
                nn.BatchNorm2d(oc),
                nn.ReLU(inplace=True),
            )

        # U-Net branch. At 256x256, four pool operations produce a 16x16 bottleneck.
        self.enc1 = double_conv(in_channels, f)       # 256x256, 64
        self.enc2 = double_conv(f, f * 2)             # 128x128, 128
        self.enc3 = double_conv(f * 2, f * 4)         # 64x64, 256
        self.enc4 = double_conv(f * 4, f * 8)         # 32x32, 512
        self.pool = nn.MaxPool2d(2)
        self.unet_bottleneck = double_conv(f * 8, f * 16)  # 16x16, 1024

        # DenseNet121 architectural branch, no pretrained weights.
        self.dense_branch = DenseNet121EarlyFeatures2D(in_channels=in_channels)
        self.dense_pool = nn.AdaptiveAvgPool2d((16, 16))
        self.dense_proj = nn.Sequential(
            nn.Conv2d(self.dense_branch.out_channels, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        # Vision Transformer branch directly from the same multimodal 2-D MRI slice.
        self.patch_embed = PatchEmbed2D(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )
        n_tokens = (self.PAPER_INPUT_SIZE[0] // patch_size) * (self.PAPER_INPUT_SIZE[1] // patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.transformer = nn.Sequential(*[
            HVUTransformerBlock2D(embed_dim, num_heads)
            for _ in range(depth)
        ])
        self.vit_norm = nn.LayerNorm(embed_dim)

        # Bottleneck fusion. All three branches share the paper's 16x16 spatial
        # resolution before concatenation. A 1x1 projection restores the U-Net
        # bottleneck width without adding an unnecessary expensive 3x3 fusion.
        fusion_in = f * 16 + embed_dim + embed_dim
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_in, f * 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(f * 16),
            nn.ReLU(inplace=True),
        )

        # U-Net decoder with skip connections.
        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, 2, stride=2)
        self.dec4 = double_conv(f * 16, f * 8)
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, 2, stride=2)
        self.dec3 = double_conv(f * 8, f * 4)
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
        self.dec2 = double_conv(f * 4, f * 2)
        self.up1 = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
        self.dec1 = double_conv(f * 2, f)
        self.head = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_size = x.shape[-2:]

        # The paper uses 256x256 2-D images. Keep augmentation/data handling
        # identical to the other 2-D models, then resize only inside Model 6.
        if original_size != self.PAPER_INPUT_SIZE:
            x_model = F.interpolate(
                x,
                size=self.PAPER_INPUT_SIZE,
                mode="bilinear",
                align_corners=False,
            )
        else:
            x_model = x

        # U-Net local feature path.
        e1 = self.enc1(x_model)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        u = self.unet_bottleneck(self.pool(e4))

        # DenseNet architectural feature path.
        dense = self.dense_branch(x_model)
        dense = self.dense_pool(dense)
        dense = self.dense_proj(dense)

        # ViT global-context path.
        tokens, spatial = self.patch_embed(x_model)
        tokens = tokens + self.pos_embed[:, :tokens.shape[1]]
        tokens = self.transformer(tokens)
        tokens = self.vit_norm(tokens)
        h, w = spatial
        vit = tokens.transpose(1, 2).reshape(tokens.shape[0], -1, h, w)
        if vit.shape[-2:] != (16, 16):
            vit = F.interpolate(vit, size=(16, 16), mode="bilinear", align_corners=False)

        # Published HVU-ED fusion concept: local U-Net + CNN + global ViT at bottleneck.
        b = self.fusion(torch.cat([u, dense, vit], dim=1))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        out = self.head(d1)

        if out.shape[-2:] != original_size:
            out = F.interpolate(out, size=original_size, mode="bilinear", align_corners=False)
        return out


# ── 6. Corrected 2D U-Net ────────────────────────────────────────────────────

class ConvNormRelu2D(nn.Module):
    """
    3x3 convolution + GroupNorm + ReLU.

    GroupNorm is used instead of BatchNorm so optimisation is not tied to
    slice-batch composition or to adjacent slices from the same patient.
    """
    def __init__(
        self,
        in_c: int,
        out_c: int,
        groups: int = 8,
        kernel: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        g = min(int(groups), int(out_c))
        while g > 1 and out_c % g != 0:
            g -= 1
        self.conv = nn.Conv2d(
            in_c, out_c, kernel_size=kernel, stride=stride,
            padding=padding, bias=False
        )
        self.norm = nn.GroupNorm(g, out_c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class UNet2D(nn.Module):
    """
    Padded 2-D U-Net baseline based on Ronneberger et al. (2015).

    Input:  (B, 4, 240, 240)
    Output: (B, 4, 240, 240)

    The encoder uses four pooling stages and doubles channels after each stage.
    The decoder uses learned 2x2 transposed convolutions to halve channel count
    before concatenation with the corresponding encoder feature map, followed
    by two 3x3 convolutions.

    The base width remains 32 to preserve the approximately 7.8 M parameter
    scale used by the existing benchmark. This is intentionally not widened
    to the original paper's 64-channel base because that would increase the
    model to roughly 31 M parameters and materially change the efficiency
    comparison.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        base_filters: int = 32,
        norm_groups: int = 8,
    ):
        super().__init__()
        f = int(base_filters)
        g = int(norm_groups)

        def double_conv(ic, oc):
            return nn.Sequential(
                ConvNormRelu2D(ic, oc, groups=g),
                ConvNormRelu2D(oc, oc, groups=g),
            )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        self.enc1 = double_conv(in_channels, f)
        self.enc2 = double_conv(f, f * 2)
        self.enc3 = double_conv(f * 2, f * 4)
        self.enc4 = double_conv(f * 4, f * 8)

        # Bottleneck
        self.bottleneck = double_conv(f * 8, f * 16)

        # Canonical learned up-convolutions halve channels before skip concat.
        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.dec4 = double_conv(f * 16, f * 8)

        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = double_conv(f * 8, f * 4)

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = double_conv(f * 4, f * 2)

        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = double_conv(f * 2, f)

        self.head = nn.Conv2d(f, out_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _match(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        # 240x240 is exactly divisible by 16, so normally no resize is needed.
        # Keep this guard for robustness if a different input geometry is used.
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(
                x, size=ref.shape[-2:], mode="bilinear", align_corners=False
            )
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        u4 = self._match(self.up4(b), e4)
        d4 = self.dec4(torch.cat([u4, e4], dim=1))

        u3 = self._match(self.up3(d4), e3)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self._match(self.up2(d3), e2)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self._match(self.up1(d2), e1)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        return self.head(d1)


# ---- 7. DeepLabV3+ 2D --------------------------------------------------------

class ResNetBottleneck2D(nn.Module):
    """ResNet bottleneck block used by the DeepLabV3+ 2D backbone."""
    expansion = 4

    def __init__(self, in_channels: int, planes: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        out_channels = planes * self.expansion
        self.conv1 = nn.Conv2d(in_channels, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=dilation,
            dilation=dilation, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


class ASPPConv2D(nn.Sequential):
    """One atrous branch in the DeepLabV3+ ASPP module."""
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        kernel_size = 1 if dilation == 1 else 3
        padding = 0 if dilation == 1 else dilation
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size,
                padding=padding, dilation=dilation, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPPPooling2D(nn.Module):
    """Global-context branch for ASPP."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        # No BatchNorm after global 1x1 pooling. This remains valid even if
        # automatic batch tuning selects batch size 1.
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        y = self.proj(self.pool(x))
        return F.interpolate(y, size=size, mode="bilinear", align_corners=False)


class ASPP2D(nn.Module):
    """Atrous Spatial Pyramid Pooling used by DeepLabV3+."""
    def __init__(self, in_channels: int, out_channels: int = 256, rates=(6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList([
            ASPPConv2D(in_channels, out_channels, dilation=1),
            *[ASPPConv2D(in_channels, out_channels, dilation=r) for r in rates],
            ASPPPooling2D(in_channels, out_channels),
        ])
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * len(self.branches), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class DeepLabV3Plus2D(nn.Module):
    """
    DeepLabV3+ 2D with a ResNet-50 style encoder, output stride 16.

    The model accepts the same four BraTS MRI modalities as UNet2D and emits
    the same four mutually exclusive segmentation logits. It is intentionally
    trained from scratch so that it follows the same data, augmentation, loss,
    optimiser, scheduler, early-stopping and evaluation procedures as every
    other architecture in this pipeline.

    Low-level features are taken at output stride 4. High-level features are
    processed by ASPP with rates 6, 12 and 18, then fused in the DeepLabV3+
    decoder before bilinear upsampling back to the input resolution.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        base_filters: int = 32,  # accepted for common registry compatibility
    ):
        super().__init__()
        del base_filters
        self.inplanes = 64

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # ResNet-50 stage depths: 3, 4, 6, 3. The final stage uses dilation
        # instead of another spatial downsampling, yielding output stride 16.
        self.layer1 = self._make_layer(64, blocks=3, stride=1, dilation=1)   # 256, OS=4
        self.layer2 = self._make_layer(128, blocks=4, stride=2, dilation=1)  # 512, OS=8
        self.layer3 = self._make_layer(256, blocks=6, stride=2, dilation=1)  # 1024, OS=16
        self.layer4 = self._make_layer(512, blocks=3, stride=1, dilation=2)  # 2048, OS=16

        self.aspp = ASPP2D(2048, out_channels=256, rates=(6, 12, 18))
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, out_channels, kernel_size=1),
        )

        self._init_weights()

    def _make_layer(self, planes: int, blocks: int, stride: int, dilation: int) -> nn.Sequential:
        layers = [ResNetBottleneck2D(self.inplanes, planes, stride=stride, dilation=dilation)]
        self.inplanes = planes * ResNetBottleneck2D.expansion
        for _ in range(1, blocks):
            layers.append(ResNetBottleneck2D(self.inplanes, planes, stride=1, dilation=dilation))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        x = self.stem(x)
        low = self.layer1(x)
        x = self.layer2(low)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.aspp(x)
        low = self.low_level_proj(low)
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)
        x = self.decoder(torch.cat([x, low], dim=1))
        return F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)


# ── 2-D Slice Dataset ─────────────────────────────────────────────────────────

class BraTS2020SliceDataset(Dataset):
    """
    Thin wrapper around BraTS2020Dataset that decomposes each 3-D volume
    into individual axial slices for use with UNet2D.

    Returns:
        image  - (4, H, W) float32  (one axial slice, all modalities)
        label  - (1, H, W) float32  (integer class indices {0,1,2,3})
        pid    - "<patient_id>_z<slice_index>" string
    """

    def __init__(
        self,
        patient_dirs: List[Path],
        patch_size: Tuple[int, int, int] = FULL_GEOMETRY,
        augment: bool = False,
        flip_prob: float = 0.5,
        affine_prob: float = 0.3,
        noise_prob: float = 0.2,
        intensity_prob: float = 0.3,
        skip_empty_ratio: float = 0.9,
        cache_rate: float = 0.0,
        cache_compress: bool = True,
        preprocessed_cache_dir: Optional[Path] = None,
        mmap_lru_patients: int = 32,
    ):
        """
        skip_empty_ratio: fraction of all-background slices to randomly drop
                          during dataset construction (keeps training balanced).
        """
        # Reuse the 3-D loader for I/O and normalisation. Build the slice
        # index without filling a parent-process RAM cache, which would be
        # duplicated when DataLoader workers are spawned on Windows.
        self._vol_ds = BraTS2020Dataset(
            patient_dirs, patch_size, augment=False, cache_rate=0.0,
            cache_compress=cache_compress,
            preprocessed_cache_dir=preprocessed_cache_dir,
            mmap_lru_patients=mmap_lru_patients,
        )
        self.augment = augment  # augmentation is applied later on the GPU
        self.skip_empty_ratio = skip_empty_ratio
        self._slice_mmap_lru = OrderedDict()
        self._slice_mmap_lru_limit = max(0, int(mmap_lru_patients))
        self._slice_cache_root = Path(preprocessed_cache_dir) if preprocessed_cache_dir else None

        # Build a compact slice index. Foreground-per-slice metadata is cached
        # as a tiny file beside each patient's mmap arrays so subsequent folds
        # do not scan every 3-D segmentation again. Crucially, no patient mmap
        # is retained in the parent Dataset while this index is built.
        self._index: List[Tuple[int, int]] = []
        rng = np.random.default_rng(0)
        for vi, pdir in enumerate(patient_dirs):
            has_fg_by_z = None
            if self._vol_ds.preprocessed_cache_dir is not None:
                patient_cache = self._vol_ds.preprocessed_cache_dir / pdir.name
                fg_meta = patient_cache / "foreground_by_z.npy"
                if fg_meta.exists():
                    has_fg_by_z = np.load(fg_meta, allow_pickle=False).astype(bool, copy=False)
                else:
                    _, label_path = _patient_cache_paths(self._vol_ds.preprocessed_cache_dir, pdir)
                    if label_path.exists():
                        label_mm = np.load(label_path, mmap_mode="r", allow_pickle=False)
                        has_fg_by_z = np.any(np.asarray(label_mm[0]) > 0, axis=(0, 1))
                        np.save(fg_meta, has_fg_by_z.astype(np.uint8), allow_pickle=False)
                        del label_mm

            if has_fg_by_z is None:
                _, label, _ = self._vol_ds._load(vi)
                has_fg_by_z = np.any(np.asarray(label[0]) > 0, axis=(0, 1))

            for z, has_fg in enumerate(has_fg_by_z[:FULL_GEOMETRY[2]]):
                if not bool(has_fg) and rng.random() < skip_empty_ratio:
                    continue
                self._index.append((vi, z))

        # The parent process must stay free of live patient mmap handles before
        # DataLoader workers are spawned. Worker-local caching can be enabled
        # independently afterwards if cache_rate > 0.
        self._vol_ds._cache.clear()
        self._vol_ds._cache_limit = int(len(patient_dirs) * cache_rate)

    def __getstate__(self):
        state = self.__dict__.copy()
        # BraTS2020Dataset.__getstate__ also clears its cache when pickled, but
        # clear it explicitly here so nested worker spawning remains safe.
        if "_vol_ds" in state:
            state["_vol_ds"]._cache = {}
            state["_vol_ds"]._mmap_lru = OrderedDict()
        state["_slice_mmap_lru"] = OrderedDict()
        return state

    def __len__(self) -> int:
        return len(self._index)


    def _load_slice_major_patient(self, vi: int):
        """Open/reuse one patient's contiguous axial cache inside the worker."""
        if self._slice_cache_root is None:
            return None
        if vi in self._slice_mmap_lru:
            value = self._slice_mmap_lru.pop(vi)
            self._slice_mmap_lru[vi] = value
            return value
        pdir = self._vol_ds.patients[vi]
        img_path, lbl_path = _patient_slice_cache_paths(self._slice_cache_root, pdir)
        if not img_path.exists() or not lbl_path.exists():
            return None
        # Copy-on-write mode is still disk-backed, but PyTorch sees writable,
        # C-contiguous slice views and can wrap them without an extra copy.
        image_z = np.load(img_path, mmap_mode="c", allow_pickle=False)
        label_z = np.load(lbl_path, mmap_mode="c", allow_pickle=False)
        value = (image_z, label_z, pdir.name)
        if self._slice_mmap_lru_limit > 0:
            self._slice_mmap_lru[vi] = value
            while len(self._slice_mmap_lru) > self._slice_mmap_lru_limit:
                self._slice_mmap_lru.popitem(last=False)
        return value

    def __getitem__(self, idx: int):
        vi, z = self._index[idx]

        # Fast path for the 2-D models: each axial slice is contiguous on disk.
        slice_patient = self._load_slice_major_patient(vi)
        if slice_patient is not None:
            image_z, label_z, pid = slice_patient
            img_sl = image_z[z]        # (4,H,W), C-contiguous float16 view
            lbl_sl = label_z[z]        # (1,H,W), C-contiguous uint8 view
            return (
                torch.from_numpy(img_sl),
                torch.from_numpy(lbl_sl),
                f"{pid}_z{z:03d}",
            )

        # Fallback when the optional slice-major cache is disabled/missing.
        image, label, pid = self._vol_ds._load(vi)
        img_sl = np.asarray(image[:, :, :, z])
        lbl_sl = np.asarray(label[:, :, :, z])
        return (
            torch.from_numpy(self._vol_ds._torch_safe_array(img_sl)),
            torch.from_numpy(self._vol_ds._torch_safe_array(lbl_sl)),
            f"{pid}_z{z:03d}",
        )


class PatientGroupedBatchSampler(Sampler):
    """
    Shuffle patients and slices every epoch while building each 2-D batch from
    as few patient files as possible. This preserves stochastic training but
    turns thousands of tiny random mmap opens into locality-friendly reads.
    """
    def __init__(self, dataset: BraTS2020SliceDataset, batch_size: int, *, drop_last: bool, seed: int):
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        by_patient = defaultdict(list)
        for dataset_idx, (vol_idx, _z) in enumerate(dataset._index):
            by_patient[int(vol_idx)].append(dataset_idx)
        self.by_patient = dict(by_patient)
        self.total = sum(len(v) for v in self.by_patient.values())

    def __len__(self):
        if self.drop_last:
            return self.total // self.batch_size
        return (self.total + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        patient_ids = list(self.by_patient)
        rng.shuffle(patient_ids)
        pending = []
        for pid in patient_ids:
            indices = list(self.by_patient[pid])
            rng.shuffle(indices)
            pending.extend(indices)
            while len(pending) >= self.batch_size:
                yield pending[:self.batch_size]
                pending = pending[self.batch_size:]
        if pending and not self.drop_last:
            yield pending


# ── Registry ──────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    # 2-D models, ordered by trainable parameter count (smallest -> largest)
    "unet2d":          UNet2D,
    "hvu":             HVUNet,
    "deeplabv3plus2d": DeepLabV3Plus2D,

    # 3-D models, ordered by trainable parameter count (smallest -> largest)
    "diff_unet":       DiffUNet,
    "hybridattunet":        HybridAttUnet3D,
    "unet3d":          UNet3D,
    "deepensemble":        DeepEnsemble3D,
}

IS_2D_MODEL = {"unet2d", "hvu", "deeplabv3plus2d"}


def build_model(cfg: dict) -> nn.Module:
    name = cfg["model"].lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    kwargs = dict(
        in_channels=cfg["in_channels"],
        out_channels=cfg["num_classes"],       # one logit per class (incl. background) for softmax
        base_filters=cfg["base_filters"],
    )
    if name == "unet2d":
        kwargs["norm_groups"] = int(cfg.get("unet2d_norm_groups", 8))
    if name == "diff_unet":
        kwargs["T"] = cfg["diffusion_steps"]
        kwargs["infer_steps"] = cfg.get("diffusion_infer_steps", 50)
    if name == "deepensemble":
        kwargs.pop("base_filters")
        width = int(cfg.get("deepensemble_width", 48))
        norm_groups = int(cfg.get("deepensemble_norm_groups", 16))
        deep_sup = bool(cfg.get("deepensemble_deep_supervision", True))
        if cfg.get("_deepensemble_training_member", False):
            return HenryEquiUNet3D(
                in_channels=cfg["in_channels"],
                out_channels=cfg["num_classes"],
                width=width,
                norm_groups=norm_groups,
                deep_supervision=deep_sup,
                activation_checkpointing=bool(
                    cfg.get("deepensemble_activation_checkpointing", True)
                ),
            )
        return DeepEnsemble3D(
            in_channels=cfg["in_channels"],
            out_channels=cfg["num_classes"],
            width=width,
            member_count=int(cfg.get("deepensemble_members", 5)),
            tta=bool(cfg.get("deepensemble_tta", True)),
            norm_groups=norm_groups,
            deep_supervision=deep_sup,
        )
    # UNet2D uses 2-D convolutions - no extra kwargs needed
    return MODEL_REGISTRY[name](**kwargs)


def is_2d_model(cfg: dict) -> bool:
    return cfg["model"].lower() in IS_2D_MODEL


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(
        self,
        cfg: dict,
        logger: logging.Logger,
        train_pts: Optional[List[Path]] = None,
        val_pts:   Optional[List[Path]] = None,
        run_suffix: str = "",
    ):
        self.cfg        = cfg
        self.logger     = logger
        self.run_suffix = run_suffix
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if cfg.get("require_cuda", True) and self.device.type != "cuda":
            raise RuntimeError(
                "CUDA is required for this run but is unavailable. Refusing to silently train on CPU."
            )

        torch.manual_seed(cfg["seed"])
        np.random.seed(cfg["seed"])

        # ── data ──
        if train_pts is None or val_pts is None:
            # Stand-alone (non-CV) mode: derive split from config
            all_patients = find_patient_dirs(cfg["data_dir"])
            if not all_patients:
                raise RuntimeError(f"No BraTS patient dirs found in {cfg['data_dir']}")
            train_pts, val_pts = make_splits(all_patients, cfg["val_ratio"], cfg["seed"])
        logger.info(f"Patients -> train: {len(train_pts)}, val: {len(val_pts)}")
        logger.info(f"DataLoader workers: {resolve_num_workers(cfg)} (CPU threads detected: {os.cpu_count() or 'unknown'})")
        total_ram, avail_ram, used_ram = get_host_memory_status()
        if np.isfinite(avail_ram):
            logger.info(
                f"Host RAM at loader setup: total={total_ram:.1f} GB | "
                f"available={avail_ram:.1f} GB | used={used_ram:.1f}% | "
                f"prefetch budget={cfg.get('prefetch_ram_gb', 4.0):.1f} GB"
            )
        logger.info(
            "RAM-safe mmap mode: preprocessed files stay on disk and Windows is allowed "
            "to cache only actively used pages. Full cache prewarming is disabled by default."
        )

        cache_root = resolve_preprocessed_cache_dir(cfg)
        if cache_root is not None:
            # Memory-mapped .npy files plus the OS page cache replace duplicated
            # per-worker Python caches. This is both faster and much safer on Windows.
            cache_rate = 0.0
            logger.info(
                f"Using memory-mapped preprocessed cache: {cache_root}. "
                "Per-worker Python RAM caching disabled; the OS can use spare RAM as shared file cache."
            )
        else:
            cache_rate = resolve_cache_rate(cfg, len(train_pts) + len(val_pts), logger)
        self.resolved_cache_rate = cache_rate

        if is_2d_model(cfg):
            # UNet2D works on axial slices - use the slice dataset
            logger.info("2-D mode: building per-slice datasets (axial slices)")
            _slice_setup_t0 = time.perf_counter()
            logger.info("Building training slice index...")
            unet2d_skip_empty = (
                float(cfg.get("unet2d_skip_empty_ratio", 0.0))
                if str(cfg.get("model", "")).lower() == "unet2d"
                else 0.9
            )
            if str(cfg.get("model", "")).lower() == "unet2d":
                logger.info(
                    "UNet2D training slice policy: retaining all native axial "
                    f"slices (skip_empty_ratio={unet2d_skip_empty:.1f})."
                )

            self.train_ds = BraTS2020SliceDataset(
                train_pts, cfg["patch_size"], augment=cfg["augment"],
                flip_prob=cfg["flip_prob"], affine_prob=cfg["affine_prob"],
                noise_prob=cfg["noise_prob"], intensity_prob=cfg["intensity_prob"],
                skip_empty_ratio=unet2d_skip_empty,
                cache_rate=cache_rate, cache_compress=cfg.get("cache_compress", True),
                preprocessed_cache_dir=cache_root,
                mmap_lru_patients=cfg.get("mmap_lru_patients", 32),
            )
            logger.info(f"Training slice index ready in {time.perf_counter() - _slice_setup_t0:.1f}s; building validation slice index...")
            _val_slice_t0 = time.perf_counter()
            self.val_ds = BraTS2020SliceDataset(
                val_pts, cfg["patch_size"], augment=False,
                skip_empty_ratio=0.0,      # keep all slices so 3-D Dice/HD95 reconstruction is exact
                cache_rate=cache_rate, cache_compress=cfg.get("cache_compress", True),
                preprocessed_cache_dir=cache_root,
                mmap_lru_patients=cfg.get("mmap_lru_patients", 32),
            )
            logger.info(f"Validation slice index ready in {time.perf_counter() - _val_slice_t0:.1f}s")
            logger.info(f"Slices -> train: {len(self.train_ds)}, val: {len(self.val_ds)}")
            logger.info("Parent Dataset contains no retained patient mmap cache; safe to spawn Windows DataLoader workers.")
        else:
            self.train_ds = BraTS2020Dataset(
                train_pts, cfg["patch_size"], augment=cfg["augment"],
                flip_prob=cfg["flip_prob"], affine_prob=cfg["affine_prob"],
                elastic_prob=cfg["elastic_prob"], noise_prob=cfg["noise_prob"],
                intensity_prob=cfg["intensity_prob"], cache_rate=cache_rate,
                cache_compress=cfg.get("cache_compress", True),
                preprocessed_cache_dir=cache_root,
                mmap_lru_patients=cfg.get("mmap_lru_patients", 32),
            )
            self.val_ds = BraTS2020Dataset(
                val_pts, cfg["patch_size"], augment=False, cache_rate=cache_rate,
                cache_compress=cfg.get("cache_compress", True),
                preprocessed_cache_dir=cache_root,
                mmap_lru_patients=cfg.get("mmap_lru_patients", 32),
            )

        # DataLoaders are created after the model has been placed on the GPU.
        # This lets the automatic batch-size tuner run a real forward/backward
        # probe before worker processes start prefetching large batches.

        # ── model ──
        configure_torch_runtime(cfg, logger)
        self.model = build_model(cfg).to(self.device)
        if (
            self.device.type == "cuda"
            and cfg.get("channels_last_3d", True)
            and not is_2d_model(cfg)
            and cfg["model"].lower() in {"unet3d", "hybridattunet", "deepensemble"}
        ):
            self.model = self.model.to(memory_format=torch.channels_last_3d)
            logger.info("3-D channels-last memory format enabled.")

        param_device = next(self.model.parameters()).device
        logger.info(f"Model parameter device after .to(...): {param_device}")
        if self.device.type == "cuda" and param_device.type != "cuda":
            raise RuntimeError(f"Model failed to move to CUDA; parameters are on {param_device}.")

        self.trainable_params = count_trainable_params(self.model)
        self.params_m = self.trainable_params / 1e6
        logger.info(
            f"Model: {cfg['model']}  |  Trainable params: {self.trainable_params:,} "
            f"({self.params_m:.3f} M)"
        )

        # IMPORTANT: do not torch.compile before automatic batch-size probing.
        # The tuner deliberately tests several batch shapes; compiling first can
        # trigger a separate Inductor/autotune graph for each shape and consume
        # enormous amounts of host committed memory on Windows. We tune the eager
        # model first, then compile exactly once at the selected batch size.

        # ── loss + GPU augmentation ──
        self.criterion = CombinedLoss()
        self.gpu_augmenter = (
            GPUBatchAugmenter(cfg, is_2d=is_2d_model(cfg))
            if cfg.get("augment", True) else None
        )
        if self.gpu_augmenter is not None:
            logger.info(
                "Augmentation backend: native PyTorch on CUDA "
                "(flip, affine rotation/scaling, elastic deformation, noise, bias field). "
                "No external augmentation library is used."
            )
            self.logger.info(
                "Augmentation policy is shared across all models: "
                f"flip={cfg['flip_prob']}, affine={cfg['affine_prob']} "
                f"(rotation +/-10 deg, scale 0.9-1.1), elastic={cfg['elastic_prob']} "
                f"(max displacement 7 voxels), intensity={cfg['intensity_prob']}, "
                f"noise={cfg['noise_prob']}."
            )

        if cfg.get("model", "").lower() == "diff_unet" and not cuda_prefetch_enabled(cfg):
            logger.info(
                "Diff-UNet low-VRAM mode: asynchronous CUDA batch prefetch is disabled. "
                "Only the current full-volume batch is resident on CUDA; DataLoader "
                "worker prefetch remains CPU-side."
            )
        if cfg.get("model", "").lower() == "deepensemble" and not cuda_prefetch_enabled(cfg):
            logger.info(
                "DeepEnsemble member low-VRAM mode: asynchronous CUDA batch prefetch "
                "is disabled while each paper-style full-volume member is trained."
            )
            logger.info(
                "DeepEnsemble low-VRAM settings: activation checkpointing="
                f"{bool(cfg.get('deepensemble_activation_checkpointing', True))}, "
                "checkpointed deep-supervision losses="
                f"{bool(cfg.get('deepensemble_checkpoint_losses', True))}, "
                "physical batch cap="
                f"{int(cfg.get('deepensemble_max_batch_size', 1))}, "
                "CUDA cache interval="
                f"{int(cfg.get('deepensemble_empty_cache_interval', 10))} batches."
            )

        # ── automatic batch-size selection ──
        requested_batch = (
            int(cfg.get("batch_size_2d", 64)) if is_2d_model(cfg) else int(cfg["batch_size"])
        )
        self.effective_batch_size = self._autotune_batch_size(requested_batch)
        logger.info(f"Effective training batch size: {self.effective_batch_size}")

        # Compile only AFTER the batch tuner has finished, so Inductor sees a
        # stable training shape instead of every probe batch. The default compile
        # mode is intentionally conservative on host RAM; users can still disable
        # compilation is optional and disabled by default; enable it explicitly with --compile.
        if cfg.get("compile_model", False) and self.device.type == "cuda" and cfg["model"].lower() != "diff_unet":
            try:
                if hasattr(torch, "_dynamo"):
                    torch._dynamo.config.suppress_errors = True
                compile_mode = cfg.get("compile_mode", "default")
                self.model = torch.compile(
                    self.model,
                    mode=compile_mode,
                    dynamic=True,
                )
                logger.info(
                    f"torch.compile enabled AFTER batch tuning (mode={compile_mode}). "
                    "This avoids compiling every auto-batch probe shape."
                )
            except Exception as exc:
                logger.warning(f"torch.compile unavailable, continuing in eager mode: {exc}")

        # Keep validation at the training batch for 2-D, where no gradients means
        # it will comfortably fit. Full-volume 3-D validation remains batch 1.
        train_batch_sampler = None
        if is_2d_model(cfg) and cfg.get("patient_grouped_batches", True):
            train_batch_sampler = PatientGroupedBatchSampler(
                self.train_ds, self.effective_batch_size, drop_last=True, seed=cfg["seed"]
            )
            logger.info(
                "2-D batch sampler: patient-grouped stochastic batches enabled "
                "to maximise mmap/page-cache locality."
            )

        self.train_loader = make_data_loader(
            self.train_ds, cfg, batch_size=self.effective_batch_size,
            shuffle=(train_batch_sampler is None), drop_last=True,
            batch_sampler=train_batch_sampler,
        )
        self.val_loader = make_data_loader(
            self.val_ds, cfg,
            batch_size=self.effective_batch_size if is_2d_model(cfg) else 1,
            shuffle=False,
        )

        # ── optimiser & scheduler ──
        opt_kwargs = dict(lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        if cfg.get("fused_adamw", True) and self.device.type == "cuda":
            opt_kwargs["fused"] = True
        try:
            self.opt = AdamW(self.model.parameters(), **opt_kwargs)
        except (TypeError, RuntimeError) as exc:
            opt_kwargs.pop("fused", None)
            logger.warning(f"Fused AdamW unavailable, using standard AdamW: {exc}")
            self.opt = AdamW(self.model.parameters(), **opt_kwargs)
        if cfg["scheduler"] == "cosine":
            self.sched = CosineAnnealingLR(self.opt, T_max=cfg["epochs"])
        elif cfg["scheduler"] == "plateau":
            self.sched = ReduceLROnPlateau(self.opt, patience=10, factor=0.5)
        else:
            self.sched = None

        # ── AMP scaler ──
        amp_enabled = bool(cfg.get("amp", True) and self.device.type == "cuda")
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        except (AttributeError, TypeError):
            self.scaler = GradScaler(enabled=amp_enabled)

        # ── checkpointing & logging ──
        tag      = f"_{run_suffix}" if run_suffix else ""
        run_name = f"{cfg['model']}_{datetime.now():%Y%m%d_%H%M%S}{tag}"
        self.run_dir = Path(cfg["save_dir"]) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.writer    = SummaryWriter(self.run_dir / "tb")
        self.best_dice   = 0.0
        self.best_val_metrics: Optional[Dict[str, float]] = None
        self.epochs_no_improve = 0
        self.start_ep    = 1
        self.best_epoch: Optional[int] = None
        self.training_time_s = 0.0
        self.peak_vram_gb = 0.0
        self.stop_reason = "max_epochs"
        self.epochs_completed = 0
        self._first_batch_verified = False
        self._gpu_epoch_times: List[float] = []
        self.history_path = self.run_dir / "training_history.csv"
        with open(self.history_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "epoch", "train_loss", "val_loss",
                "dice_WT", "dice_TC", "dice_ET", "dice_mean",
                "hd95_WT", "hd95_TC", "hd95_ET", "hd95_mean",
                "learning_rate", "epoch_time_s", "data_wait_s", "gpu_step_s",
                "validation_s", "samples_per_s", "peak_vram_gb",
                "gpu_allocated_gb", "gpu_reserved_gb", "gpu_temp_c",
                "gpu_util_pct", "gpu_sm_clock_mhz", "gpu_power_w",
                "gpu_mem_used_mb", "gpu_mem_total_mb"
            ])
            writer.writeheader()

        # ── resume ──
        if cfg.get("checkpoint"):
            self._load_checkpoint(cfg["checkpoint"])

        # Save config
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(cfg, f, indent=2, default=str)

    # ── automatic batch-size tuner ─────────────────────────────────────────────

    def _autotune_batch_size(self, requested_batch: int) -> int:
        """
        Find the largest stable batch under the VRAM target without making
        Diff-UNet spend minutes doing repeated worst-case augmentation probes.

        Search probes include the real forward/backward pass, AdamW state and
        the simultaneously resident CUDA-prefetched next batch.  Expensive GPU
        augmentation is omitted during the search and checked only once on the
        final candidate (with automatic back-off if necessary).

        The successful batch size is persisted to disk and reused on later runs
        when the GPU/model/input/memory-policy signature is unchanged.
        """
        cfg = self.cfg
        if self.device.type != "cuda" or not cfg.get("auto_batch_size", True):
            return max(1, int(requested_batch))

        is2d = is_2d_model(cfg)
        # The small 2-D U-Net can safely use exponential batch growth, but the
        # much larger HVU / DeepLabV3+ networks can cross the WDDM
        # oversubscription boundary with a single 2x jump.  Treat those as
        # heavy 2-D models and advance in modest additive steps instead.
        heavy_2d = is2d and cfg.get("model") in {"hvu", "deeplabv3plus2d"}
        max_batch = int(cfg.get("max_batch_size_2d", 4096) if is2d
                        else cfg.get("max_batch_size_3d", 32))

        # The corrected full-volume Diff-UNet can terminate the Windows process
        # at native CUDA/WDDM level when probing batch 2, before PyTorch can
        # raise a catchable CUDA OOM. Keep its physical batch ceiling at 1 by
        # default. This does not change the model architecture or per-batch math.
        if cfg.get("model") == "unet2d":
            # Do not let the small U-Net expand to an enormous physical batch.
            # With a fixed 1e-4 LR, batches around 200-300 substantially reduce
            # optimiser updates per epoch and can undertrain minority tumour
            # classes. Keep the common study LR and cap the physical batch at 64.
            unet2d_cap = max(1, int(cfg.get("unet2d_max_batch_size", 64)))
            max_batch = min(max_batch, unet2d_cap)
        elif cfg.get("model") == "diff_unet":
            diff_cap = max(1, int(cfg.get("diff_unet_max_batch_size", 1)))
            max_batch = min(max_batch, diff_cap)
        elif cfg.get("model") == "deepensemble":
            ensemble_cap = max(1, int(cfg.get("deepensemble_max_batch_size", 1)))
            max_batch = min(max_batch, ensemble_cap)

        max_batch = max(1, min(max_batch, len(self.train_ds)))
        start_batch = max(1, min(int(requested_batch), max_batch))
        step = max(1, int(cfg.get("batch_size_step", 16))) if is2d else 1
        target_fraction = float(np.clip(cfg.get("batch_vram_fraction", 0.85), 0.50, 0.97))
        props = torch.cuda.get_device_properties(self.device)
        total_bytes = int(props.total_memory)
        target_bytes = int(total_bytes * target_fraction)
        min_free = int(float(cfg.get("batch_vram_headroom_gb", 1.0)) * 1024**3)

        # Only Diff-UNet changed architecture/training memory behaviour in the
        # START_X correction. Preserve authoritative cached batch sizes for the
        # other six models while forcing Diff-UNet to obtain a fresh value.
        diff_revision = (
            f"|unet2d_v3|gn{int(cfg.get('unet2d_norm_groups', 8))}"
            f"|cap{int(cfg.get('unet2d_max_batch_size', 64))}"
            f"|emptydrop{float(cfg.get('unet2d_skip_empty_ratio', 0.0)):.3f}"
            if cfg.get("model") == "unet2d" else ""
        )
        if cfg.get("model") == "diff_unet":
            diff_revision += (
                f"|diffstartx3|dcap{int(cfg.get('diff_unet_max_batch_size', 1))}"
            )
        if cfg.get("model") == "deepensemble":
            diff_revision += (
                f"|henrydeepsup3|w{int(cfg.get('deepensemble_width', 48))}"
                f"|g{int(cfg.get('deepensemble_norm_groups', 16))}"
                f"|ckpt{int(bool(cfg.get('deepensemble_activation_checkpointing', True)))}"
                f"|lossc{int(bool(cfg.get('deepensemble_checkpoint_losses', True)))}"
                f"|cap{int(cfg.get('deepensemble_max_batch_size', 1))}"
            )
        runtime_key = (
            f"v{int(cfg.get('autobatch_cache_version', 4))}|"
            f"{cfg.get('model')}|{tuple(cfg.get('patch_size', ())) }|"
            f"f{cfg.get('base_filters')}|c{cfg.get('num_classes')}|"
            f"amp{int(bool(cfg.get('amp', True)))}|pf{int(cuda_prefetch_enabled(cfg))}|"
            f"aug{int(bool(cfg.get('augment', True)))}|"
            f"flip{float(cfg.get('flip_prob', 0.5)):.3f}|"
            f"aff{float(cfg.get('affine_prob', 0.3)):.3f}|"
            f"ela{float(cfg.get('elastic_prob', 0.2)):.3f}|"
            f"noise{float(cfg.get('noise_prob', 0.2)):.3f}|"
            f"bias{float(cfg.get('intensity_prob', 0.3)):.3f}|"
            f"target{target_fraction:.3f}|head{min_free}|gpu{props.name}|vram{total_bytes}"
            f"{diff_revision}"
        )
        resolved = cfg.setdefault("_autotuned_batch_sizes", {})
        if runtime_key in resolved:
            batch = int(resolved[runtime_key])
            self.logger.info(f"Auto batch: reusing in-process tuned batch size {batch}.")
            return batch

        # Persistent cross-run cache. This is especially valuable for Diff-UNet,
        # whose full-volume training probe is much more expensive than the other
        # architectures. The key contains all settings that materially affect
        # batch-memory requirements.
        cache_path = None
        persistent_cache = {}
        if cfg.get("autobatch_disk_cache", True):
            configured = cfg.get("autobatch_cache_file")
            cache_path = Path(configured) if configured else Path(cfg.get("save_dir", "./runs")) / "autobatch_cache.json"
            try:
                if cache_path.exists():
                    with open(cache_path, "r", encoding="utf-8") as f:
                        persistent_cache = json.load(f)
                cached = int(persistent_cache.get(runtime_key, 0))
                if 1 <= cached <= max_batch:
                    # A valid cached batch size is authoritative. Once it is read,
                    # reuse it immediately and do not launch any additional probes
                    # or alternative batch-size combinations.
                    resolved[runtime_key] = cached
                    cfg["resolved_batch_size"] = cached
                    self.logger.info(
                        f"Auto batch: loaded cached batch size {cached} from {cache_path}. "
                        "Reusing it directly; no further batch-size probes will be run."
                    )
                    return cached
            except Exception as exc:
                self.logger.warning(f"Could not read auto-batch cache {cache_path}: {exc}")
                persistent_cache = {}

        self.logger.info(
            f"Auto batch tuning: start={start_batch}, max={max_batch}, "
            f"VRAM target={target_fraction:.0%} of {total_bytes / 1024**3:.1f} GB; "
            f"minimum free headroom={min_free / 1024**3:.1f} GB."
        )
        if cfg.get("model") == "unet2d":
            self.logger.info(
                "UNet2D optimisation-safe auto-batch mode: physical batch size is "
                f"capped at {max_batch} so the fixed 1e-4 learning rate is not paired "
                "with the previous very-large-batch regime."
            )
        if is_diffusion_model(self.model):
            self.logger.info(
                "Diff-UNet safe auto-batch mode: physical batch size is capped at "
                f"{max_batch}. Probe 2 will not be launched. Batch 1 still receives "
                "the real training probe and full-augmentation safety checks."
            )
        if heavy_2d:
            self.logger.info(
                f"Heavy 2-D safe tuning enabled for {cfg.get('model')}: "
                f"batch size will increase by {step} at a time rather than doubling."
            )

        sample_image, sample_label, _ = self.train_ds[0]
        sample_image = move_image_to_device(sample_image.unsqueeze(0), self.device, cfg)
        sample_label = sample_label.unsqueeze(0).to(self.device, non_blocking=False)

        base_model = unwrap_model(self.model)
        initial_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
        probe_opt_kwargs = dict(lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        if cfg.get("fused_adamw", True):
            probe_opt_kwargs["fused"] = True
        try:
            probe_opt = AdamW(self.model.parameters(), **probe_opt_kwargs)
        except (TypeError, RuntimeError):
            probe_opt_kwargs.pop("fused", None)
            probe_opt = AdamW(self.model.parameters(), **probe_opt_kwargs)

        def _is_cuda_oom(exc: BaseException) -> bool:
            """Return True for CUDA OOMs, including Windows AcceleratorError variants."""
            out_of_memory_cls = getattr(torch, "OutOfMemoryError", None)
            if out_of_memory_cls is not None and isinstance(exc, out_of_memory_cls):
                return True
            text = str(exc).lower()
            oom_markers = (
                "out of memory",
                "cudaerrormemoryallocation",
                "memory allocation",
                "cuda error: out of memory",
                "cudaerror_memory_allocation",
            )
            return any(marker in text for marker in oom_markers)

        def _safe_cuda_cleanup():
            """Best-effort cleanup after a failed probe without re-raising the same OOM."""
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception as cleanup_exc:
                if not _is_cuda_oom(cleanup_exc):
                    self.logger.debug(f"CUDA cleanup warning after auto-batch probe: {cleanup_exc}")

        def probe(batch: int, include_augmentation: bool = False, clear_cache: bool = True):
            """
            Run one real training probe.

            Any CUDA memory-allocation failure is converted into a normal failed
            probe result.  The caller can then fall back to the last known-good
            batch instead of terminating the entire experiment.
            """
            image = label = pred = loss = next_image = next_label = None
            try:
                if clear_cache:
                    _safe_cuda_cleanup()
                torch.cuda.reset_peak_memory_stats(self.device)
                probe_opt.zero_grad(set_to_none=True)

                image = sample_image.expand(batch, *sample_image.shape[1:]).contiguous()
                label = sample_label.expand(batch, *sample_label.shape[1:]).contiguous()

                # Match the real training pipeline. Diff-UNet disables CUDA
                # prefetch by default, so its probe must not reserve a second
                # full-volume CUDA batch.
                if cuda_prefetch_enabled(cfg):
                    next_image = sample_image.expand(batch, *sample_image.shape[1:]).contiguous()
                    next_label = sample_label.expand(batch, *sample_label.shape[1:]).contiguous()

                if include_augmentation and self.gpu_augmenter is not None:
                    image, label = self.gpu_augmenter(image, label, force=True)

                self.model.train()
                with amp_context(self.device, cfg.get("amp", True)):
                    if is_diffusion_model(self.model):
                        # Correct Diff-UNet START_X training path. The model
                        # internally maps the integer class target to a one-hot
                        # diffusion state in [-1,1] and predicts clean-segmentation
                        # logits. Optimisation uses the same common Dice + CE
                        # criterion as the other architectures.
                        pred = self.model(image, label)
                        loss = self.criterion(pred, label)
                    else:
                        pred = self.model(image)
                        if cfg.get("model", "").lower() == "deepensemble":
                            pred, loss = _deep_supervised_output_and_loss(
                                pred,
                                label,
                                self.criterion,
                                checkpoint_losses=bool(
                                    cfg.get("deepensemble_checkpoint_losses", True)
                                ),
                            )
                        else:
                            loss = self.criterion(pred, label)

                loss.backward()
                probe_opt.step()
                torch.cuda.synchronize(self.device)

                peak_alloc = int(torch.cuda.max_memory_allocated(self.device))
                peak_reserved = int(torch.cuda.max_memory_reserved(self.device))
                free_bytes, _ = torch.cuda.mem_get_info(self.device)
                ok = peak_reserved <= target_bytes and int(free_bytes) >= min_free
                return ok, peak_reserved, peak_alloc, int(free_bytes)

            except Exception as exc:
                if not _is_cuda_oom(exc):
                    raise

                # CUDA may surface a memory allocation failure asynchronously.
                # Never call an unguarded CUDA API here, because that can raise
                # a second AcceleratorError and mask the failed-probe result.
                try:
                    peak_reserved = int(torch.cuda.max_memory_reserved(self.device))
                except Exception:
                    peak_reserved = 0

                self.logger.warning(
                    f"Auto batch probe {batch} hit CUDA OOM; backing off to the "
                    "last stable batch instead of aborting the run."
                )
                return False, peak_reserved, f"{type(exc).__name__}: {exc}", 0

            finally:
                try:
                    probe_opt.zero_grad(set_to_none=True)
                except Exception:
                    pass
                del image, label, pred, loss, next_image, next_label
                if clear_cache:
                    _safe_cuda_cleanup()

        def log_probe(batch, result, suffix=""):
            ok, peak, extra, free_b = result
            alloc_txt = f", allocated={extra / 1024**3:.2f} GB" if isinstance(extra, (int, float)) else ""
            free_txt = f", free={free_b / 1024**3:.2f} GB" if free_b else ""
            self.logger.info(
                f"Auto batch probe {batch}{suffix}: reserved={peak / 1024**3:.2f} GB"
                f"{alloc_txt}{free_txt} ({'OK' if ok else 'too high/OOM'})"
            )

        last_good = 0
        first_bad = None
        successful_peaks = {}
        # True when batch size 1 physically fits but is already above the
        # preferred VRAM target. In that case there is no valid smaller batch,
        # so lock the tuner to batch 1 and never probe a larger batch.
        minimum_batch_locked = False

        # Find a valid starting point without expensive augmentation.
        b = start_batch
        while b >= 1:
            self.logger.info(f"Auto batch: starting probe {b}...")
            result = probe(b, include_augmentation=False)
            log_probe(b, result)
            if result[0]:
                last_good = b
                successful_peaks[b] = int(result[1])
                break
            first_bad = b
            if b == 1:
                break
            b = max(1, b // 2)

        if last_good == 0:
            # Distinguish a genuine CUDA OOM from a batch that physically fits
            # but exceeds the preferred VRAM target.  For a full-volume model
            # such as HVU there may be no smaller batch than 1, so rejecting a
            # physically valid batch solely because it uses >90% VRAM would make
            # the model impossible to run.
            self.logger.info(
                "Auto batch: batch 1 did not satisfy the preferred VRAM policy; "
                "checking whether it physically fits on the GPU."
            )
            physical = probe(1, include_augmentation=False)
            # A numeric third field means the CUDA step completed; a string is
            # returned only for an OOM/memory-allocation exception.
            physically_fit = isinstance(physical[2], (int, float))
            if physically_fit:
                last_good = 1
                successful_peaks[1] = int(physical[1])
                minimum_batch_locked = True
                self.logger.warning(
                    f"Batch size 1 physically fits but exceeds the preferred "
                    f"{target_fraction:.0%} VRAM policy. Locking batch size 1 "
                    "because no smaller training batch exists; no larger batch "
                    "probes will be attempted."
                )
            else:
                base_model.load_state_dict(initial_state)
                del initial_state, probe_opt, sample_image, sample_label
                torch.cuda.empty_cache()
                raise RuntimeError(
                    "Batch size 1 genuinely OOMed during the training probe. "
                    "The selected full-volume model cannot fit at the current "
                    "patch/model size even with the minimum batch size."
                )

        # Model-aware search. The small UNet2D can use exponential growth.
        # Heavy 2-D models (HVU and DeepLabV3+) advance in `step` increments,
        # while full-volume 3-D models advance one sample at a time. This avoids
        # a dangerous jump such as HVU 64 -> 128 or Diff-UNet 2 -> 4, which can
        # push Windows/WDDM into shared-GPU-memory thrashing rather than raising
        # a clean CUDA OOM.
        b = last_good
        while (not minimum_batch_locked) and b < max_batch:
            if is2d and not heavy_2d:
                candidate = min(max_batch, b * 2)
            elif heavy_2d:
                candidate = min(max_batch, b + step)
            else:
                candidate = b + 1

            # For heavy 2-D and all 3-D models, use the measured memory slope to
            # stop before launching a probe that cannot plausibly fit.
            if heavy_2d or not is2d:
                peak_now = successful_peaks.get(b)
                if peak_now is not None:
                    if peak_now >= int(target_bytes * 0.92):
                        self.logger.info(
                            f"Auto batch predictive stop at {b}: current peak "
                            f"{peak_now / 1024**3:.2f} GB is already >=92% of the "
                            f"{target_bytes / 1024**3:.2f} GB tuning target."
                        )
                        break

                    prev_batches = sorted(k for k in successful_peaks if k < b)
                    if prev_batches:
                        prev_b = prev_batches[-1]
                        prev_peak = successful_peaks[prev_b]
                        slope = max(0.0, (peak_now - prev_peak) / max(1, b - prev_b))
                        projected = peak_now + slope * (candidate - b)
                        if slope > 0 and projected >= int(target_bytes * 0.98):
                            self.logger.info(
                                f"Auto batch predictive stop at {b}: batch {candidate} "
                                f"is projected to require ~{projected / 1024**3:.2f} GB "
                                f"vs target {target_bytes / 1024**3:.2f} GB."
                            )
                            break

            self.logger.info(f"Auto batch: starting probe {candidate}...")
            result = probe(candidate, include_augmentation=False)
            log_probe(candidate, result)
            if result[0]:
                last_good = candidate
                successful_peaks[candidate] = int(result[1])
                b = candidate
                if candidate == max_batch:
                    break
            else:
                first_bad = candidate
                break

        # Binary refinement is useful only for the wide 2-D search. The 3-D
        # search is intentionally sequential to avoid WDDM oversubscription.
        if is2d and not heavy_2d and first_bad is not None and first_bad > last_good + step:
            lo, hi = last_good, first_bad
            while hi - lo > step:
                mid = ((lo + hi) // (2 * step)) * step
                if mid <= lo:
                    mid = lo + step
                if mid >= hi:
                    break
                self.logger.info(f"Auto batch: starting refinement probe {mid}...")
                result = probe(mid, include_augmentation=False)
                log_probe(mid, result)
                if result[0]:
                    lo = mid
                    last_good = mid
                else:
                    hi = mid

        # One expensive final safety check with every GPU augmentation forced on.
        # If it fails, back off rather than repeating the entire search.
        if cfg.get("autobatch_final_aug_check", True) and self.gpu_augmenter is not None:
            candidate = last_good
            while candidate >= 1:
                self.logger.info(
                    f"Auto batch: starting final full-augmentation safety check at batch {candidate}..."
                )
                result = probe(candidate, include_augmentation=True)
                log_probe(candidate, result, suffix=" [full augmentation check]")
                if result[0]:
                    last_good = candidate
                    break

                # If batch 1 completes physically but merely exceeds the target,
                # accept it with a warning. If it genuinely OOMs with full
                # augmentation, fail explicitly rather than entering training.
                if candidate == 1:
                    physically_fit = isinstance(result[2], (int, float))
                    if physically_fit:
                        last_good = 1
                        self.logger.warning(
                            "Batch size 1 passed the full-augmentation training "
                            "step but exceeds the preferred VRAM target. Using "
                            "batch 1 because it is the minimum possible batch."
                        )
                        break
                    base_model.load_state_dict(initial_state)
                    del initial_state, probe_opt, sample_image, sample_label
                    self.model.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    raise RuntimeError(
                        "Batch size 1 genuinely OOMed during the full-augmentation "
                        "safety check. Reduce the model/input memory requirement."
                    )

                candidate = max(1, candidate - step)

        # Repeated memory-stability check. A candidate must complete several
        # consecutive real augmented training steps without relying on
        # torch.cuda.empty_cache() between steps. Batch selection is based only
        # on CUDA memory safety, never on wall-clock timing variation.
        #
        # Timing is deliberately diagnostic only. CUDA kernel warm-up,
        # activation-checkpoint recomputation, Windows/WDDM scheduling, clock
        # changes and allocator behaviour can make consecutive step times vary
        # substantially even when the batch is completely safe.
        stability_steps = max(1, int(cfg.get("autobatch_stability_steps", 3)))
        if stability_steps > 1 and self.gpu_augmenter is not None:
            candidate = int(last_good)
            while candidate >= 1:
                self.logger.info(
                    f"Auto batch: memory-stability testing batch {candidate} for "
                    f"{stability_steps} consecutive full-augmentation steps..."
                )
                torch.cuda.empty_cache()
                times = []
                stable = True

                for rep in range(stability_steps):
                    t_rep = time.perf_counter()
                    result = probe(
                        candidate,
                        include_augmentation=True,
                        clear_cache=False,
                    )
                    dt_rep = time.perf_counter() - t_rep
                    times.append(dt_rep)

                    log_probe(
                        candidate,
                        result,
                        suffix=f" [memory stability {rep+1}/{stability_steps}]",
                    )

                    # probe()[0] already requires:
                    #   1. the real forward/backward/optimiser step completed,
                    #   2. peak reserved VRAM stayed within the target, and
                    #   3. the configured minimum free-VRAM headroom remained.
                    # Any CUDA OOM is also converted to result[0] == False.
                    if not result[0]:
                        stable = False
                        break

                torch.cuda.empty_cache()

                if stable:
                    last_good = candidate
                    self.logger.info(
                        f"Auto batch memory stability passed at {candidate}: "
                        f"{stability_steps} consecutive augmented steps completed "
                        "within the CUDA memory policy. "
                        f"Diagnostic step times={[round(x, 2) for x in times]} s."
                    )
                    break

                if candidate == 1:
                    # If batch 1 reaches this point it failed the preferred
                    # repeated memory policy. Check whether it still physically
                    # completes, because no smaller physical batch exists.
                    physical = probe(
                        1,
                        include_augmentation=True,
                        clear_cache=True,
                    )
                    physically_fit = isinstance(physical[2], (int, float))
                    if physically_fit:
                        last_good = 1
                        self.logger.warning(
                            "Batch size 1 physically completes repeated augmented "
                            "training but exceeds the preferred VRAM/headroom policy. "
                            "Using batch 1 because no smaller physical batch exists."
                        )
                        break

                    base_model.load_state_dict(initial_state)
                    del initial_state, probe_opt, sample_image, sample_label
                    self.model.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    raise RuntimeError(
                        "Batch size 1 genuinely OOMed during the repeated "
                        "full-augmentation memory-stability check."
                    )

                self.logger.warning(
                    f"Auto batch {candidate} failed the repeated CUDA memory "
                    "stability check; backing off to a smaller batch."
                )
                candidate = max(1, candidate - step)

        base_model.load_state_dict(initial_state)
        del initial_state, probe_opt, sample_image, sample_label
        self.model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        resolved[runtime_key] = int(last_good)
        cfg["resolved_batch_size"] = int(last_good)

        if cache_path is not None:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                persistent_cache[runtime_key] = int(last_good)
                tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(persistent_cache, f, indent=2)
                os.replace(tmp_path, cache_path)
            except Exception as exc:
                self.logger.warning(f"Could not persist auto-batch cache {cache_path}: {exc}")

        self.logger.info(
            f"Auto batch selected {last_good} after current-run VRAM/stability checks. "
            "This value will be reused directly on the next matching run."
        )
        return int(last_good)


    # ── training step ─────────────────────────────────────────────────────────

    def _train_step(self, image: torch.Tensor, label: torch.Tensor) -> float:
        self.model.train()
        image = move_image_to_device(image, self.device, self.cfg)
        label = label.to(self.device, non_blocking=self.device.type == "cuda")
        if not self._first_batch_verified:
            param_device = next(self.model.parameters()).device
            if self.device.type == "cuda":
                if not image.is_cuda or not label.is_cuda or param_device.type != "cuda":
                    raise RuntimeError(
                        f"CUDA device verification failed: image={image.device}, "
                        f"label={label.device}, model={param_device}"
                    )
                torch.cuda.synchronize(self.device)
                allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                reserved = torch.cuda.memory_reserved(self.device) / 1024**3
                self.logger.info(
                    f"First training batch verified on GPU: image={tuple(image.shape)} | "
                    f"allocated={allocated:.2f} GB | reserved={reserved:.2f} GB"
                )
            self._first_batch_verified = True
        if self.gpu_augmenter is not None:
            image, label = self.gpu_augmenter(image, label)
        self.opt.zero_grad(set_to_none=True)
        with amp_context(self.device, self.cfg["amp"]):
            if is_diffusion_model(self.model):
                # START_X Diff-UNet: diffuse the one-hot segmentation state
                # internally, predict clean four-class logits, and optimise
                # with the same 0.5 Dice + 0.5 categorical CE loss used by
                # every other architecture in the experiment.
                pred = self.model(image, label)
                loss = self.criterion(pred, label)
            else:
                pred = self.model(image)
                if self.cfg.get("model", "").lower() == "deepensemble":
                    pred, loss = _deep_supervised_output_and_loss(
                        pred,
                        label,
                        self.criterion,
                        checkpoint_losses=bool(
                            self.cfg.get("deepensemble_checkpoint_losses", True)
                        ),
                    )
                else:
                    loss = self.criterion(pred, label)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.opt)
        self.scaler.update()
        # Do NOT call loss.item() here. .item() synchronises CUDA and would
        # serialize every training batch, preventing overlap with the CUDA
        # prefetch stream and DataLoader workers. The caller samples the loss
        # only at the configured logging interval and once at epoch end.
        return loss.detach()

    # ── validation ────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def _validate(self) -> Dict[str, float]:
        self.model.eval()
        losses: List[float] = []
        metric_workers = max(1, int(self.cfg.get("hd95_workers", 4)))

        source = CUDAPrefetcher(self.val_loader, self.device, self.cfg)
        val_iter = source
        if self.cfg.get("live_console", True) and tqdm is not None:
            val_iter = tqdm(
                source,
                total=len(self.val_loader),
                desc="Validating",
                unit="batch",
                leave=False,
                dynamic_ncols=True,
                file=sys.stdout,
            )

        # UNet2D: reconstruct complete patient volumes, then calculate exact
        # volumetric Dice/HD95 in parallel across patients.
        if is_2d_model(self.cfg):
            depth = FULL_GEOMETRY[2]
            patient_pred: Dict[str, np.ndarray] = {}
            patient_true: Dict[str, np.ndarray] = {}

            for image, label, pids in val_iter:
                image = move_image_to_device(image, self.device, self.cfg)
                label_gpu = label.to(self.device, non_blocking=True)
                with amp_context(self.device, self.cfg["amp"]):
                    pred = self.model(image)
                    if self.cfg.get("model", "").lower() == "deepensemble":
                        pred, loss = _deep_supervised_output_and_loss(
                            pred,
                            label_gpu,
                            self.criterion,
                            checkpoint_losses=False,
                        )
                    else:
                        loss = self.criterion(pred, label_gpu)
                losses.append(loss.item())

                pred_cls = pred.argmax(1).detach().cpu().numpy().astype(np.uint8, copy=False)
                true_cls = label.detach().cpu().numpy()
                if true_cls.ndim == pred_cls.ndim + 1:
                    true_cls = true_cls[:, 0]
                true_cls = true_cls.astype(np.uint8, copy=False)

                for b, raw_pid in enumerate(pids):
                    pid, z_text = raw_pid.rsplit("_z", 1)
                    z = int(z_text)
                    if pid not in patient_pred:
                        h, w = pred_cls[b].shape
                        patient_pred[pid] = np.zeros((h, w, depth), dtype=np.uint8)
                        patient_true[pid] = np.zeros((h, w, depth), dtype=np.uint8)
                    patient_pred[pid][:, :, z] = pred_cls[b]
                    patient_true[pid][:, :, z] = true_cls[b]

                if tqdm is not None and hasattr(val_iter, "set_postfix"):
                    val_iter.set_postfix(loss=f"{np.mean(losses):.4f}", refresh=False)

            patient_ids = list(patient_pred)
            pairs = [(patient_pred[pid], patient_true[pid]) for pid in patient_ids]
            if metric_workers > 1 and len(pairs) > 1:
                with ThreadPoolExecutor(max_workers=min(metric_workers, len(pairs))) as pool:
                    metrics = list(pool.map(lambda pair: metrics_from_class_maps(pair[0], pair[1]), pairs))
            else:
                metrics = [metrics_from_class_maps(a, b) for a, b in pairs]

            out = {"loss": float(np.mean(losses))}
            for name in REGION_NAMES:
                out[f"dice_{name}"] = float(np.mean([m[f"dice_{name}"] for m in metrics]))
                out[f"hd95_{name}"] = safe_nanmean([m[f"hd95_{name}"] for m in metrics])
            out["dice_mean"] = float(np.mean([out[f"dice_{n}"] for n in REGION_NAMES]))
            out["hd95_mean"] = safe_nanmean([out[f"hd95_{n}"] for n in REGION_NAMES])
            return out

        # 3-D architectures: submit each completed patient to the CPU metric
        # pool immediately. HD95 therefore runs concurrently with subsequent
        # GPU inference instead of stalling the accelerator between patients.
        metric_futures = []
        with ThreadPoolExecutor(max_workers=metric_workers) as pool:
            for image, label, _ in val_iter:
                image = move_image_to_device(image, self.device, self.cfg)
                label_gpu = label.to(self.device, non_blocking=self.device.type == "cuda")
                with amp_context(self.device, self.cfg["amp"]):
                    pred = self.model(image)
                    if self.cfg.get("model", "").lower() == "deepensemble":
                        pred, loss = _deep_supervised_output_and_loss(
                            pred,
                            label_gpu,
                            self.criterion,
                            checkpoint_losses=False,
                        )
                    else:
                        loss = self.criterion(pred, label_gpu)
                losses.append(loss.item())

                pred_cls = pred.argmax(1).detach().cpu().numpy().astype(np.uint8, copy=False)
                tgt = label.detach().cpu().numpy()
                if tgt.ndim == pred_cls.ndim + 1:
                    tgt = tgt[:, 0]
                tgt = tgt.astype(np.uint8, copy=False)

                for b in range(pred_cls.shape[0]):
                    metric_futures.append(
                        pool.submit(metrics_from_class_maps, pred_cls[b], tgt[b])
                    )

                if tqdm is not None and hasattr(val_iter, "set_postfix"):
                    val_iter.set_postfix(
                        loss=f"{np.mean(losses):.4f}",
                        metrics=len(metric_futures),
                        refresh=False,
                    )

            metrics = [f.result() for f in metric_futures]

        out: Dict[str, float] = {"loss": float(np.mean(losses))}
        for name in REGION_NAMES:
            out[f"dice_{name}"] = float(np.mean([m[f"dice_{name}"] for m in metrics]))
            out[f"hd95_{name}"] = safe_nanmean([m[f"hd95_{name}"] for m in metrics])
        out["dice_mean"] = float(np.mean([out[f"dice_{n}"] for n in REGION_NAMES]))
        out["hd95_mean"] = safe_nanmean([out[f"hd95_{n}"] for n in REGION_NAMES])
        return out

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self):
        logger = self.logger
        cfg    = self.cfg
        use_live = bool(cfg.get("live_console", True) and tqdm is not None)

        # Human-readable prefix, especially useful during k-fold CV.
        if self.run_suffix.startswith("fold"):
            try:
                fold_num = int(self.run_suffix.replace("fold", ""))
                run_label = f"Fold {fold_num}/{cfg.get('n_folds', '?')}"
            except ValueError:
                run_label = self.run_suffix
        else:
            run_label = "Training"

        if cfg.get("live_console", True) and tqdm is None:
            logger.warning(
                "Live console requested but tqdm is not installed. "
                "Install it with `pip install tqdm`; falling back to periodic logs."
            )

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        training_started = time.perf_counter()

        for epoch in range(self.start_ep, cfg["epochs"] + 1):
            epoch_started = time.perf_counter()
            # Keep only a single running loss accumulator. Retaining one detached
            # CUDA loss tensor per batch is unnecessary and can gradually increase
            # live allocations during long 3-D epochs.
            if self.device.type == "cuda":
                epoch_loss_sum = torch.zeros((), device=self.device, dtype=torch.float32)
            else:
                epoch_loss_sum = 0.0
            epoch_loss_count = 0
            val_m: Optional[Dict[str, float]] = None
            data_wait_s = 0.0
            gpu_step_s = 0.0
            validation_s = 0.0

            train_source = CUDAPrefetcher(self.train_loader, self.device, cfg)
            train_iter = train_source
            if use_live:
                train_iter = tqdm(
                    train_source,
                    total=len(self.train_loader),
                    desc=f"{run_label} | Epoch {epoch}/{cfg['epochs']}",
                    unit="batch",
                    leave=True,
                    dynamic_ncols=True,
                    file=sys.stdout,
                )

            last_batch_end = time.perf_counter()
            # Use one CUDA event pair for the training section of the epoch rather
            # than retaining two CUDA event objects for every batch until epoch end.
            # The pair is recorded around the complete sequence of training steps.
            if self.device.type == "cuda":
                epoch_gpu_start_evt = torch.cuda.Event(enable_timing=True)
                epoch_gpu_end_evt = torch.cuda.Event(enable_timing=True)
                epoch_gpu_timing_started = False
            else:
                epoch_gpu_start_evt = None
                epoch_gpu_end_evt = None
                epoch_gpu_timing_started = False
            cpu_gpu_step_s = 0.0
            last_loss_value = float("nan")
            running_loss_sum = 0.0
            running_loss_count = 0
            refresh_steps = max(1, int(cfg.get("live_refresh_steps", cfg.get("log_interval", 10))))

            for step, (image, label, _) in enumerate(train_iter, 1):
                batch_ready = time.perf_counter()
                data_wait_s += max(0.0, batch_ready - last_batch_end)

                if self.device.type == "cuda":
                    if not epoch_gpu_timing_started:
                        epoch_gpu_start_evt.record()
                        epoch_gpu_timing_started = True
                    loss_tensor = self._train_step(image, label)
                    # Exact epoch-loss accumulation stays on CUDA, so there is no
                    # per-batch device synchronisation and no list of CUDA tensors.
                    epoch_loss_sum.add_(loss_tensor.float())
                else:
                    step_started = time.perf_counter()
                    loss_tensor = self._train_step(image, label)
                    cpu_gpu_step_s += time.perf_counter() - step_started
                    epoch_loss_sum += float(loss_tensor.item())
                epoch_loss_count += 1

                if (
                    cfg.get("model", "").lower() == "deepensemble"
                    and self.device.type == "cuda"
                    and step % max(
                        1, int(cfg.get("deepensemble_empty_cache_interval", 10))
                    ) == 0
                ):
                    # Release only unused cached blocks. Live model tensors,
                    # gradients and optimiser state are untouched.
                    torch.cuda.empty_cache()

                global_step = (epoch - 1) * len(self.train_loader) + step

                # Reading a CUDA scalar with .item() is a synchronisation point.
                # Do it only periodically, never on every batch.
                should_sample_loss = (
                    step == 1
                    or step % max(1, int(cfg.get("log_interval", 10))) == 0
                    or step == len(self.train_loader)
                    or (use_live and step % refresh_steps == 0)
                )
                if should_sample_loss:
                    last_loss_value = float(loss_tensor.item())
                    running_loss_sum += last_loss_value
                    running_loss_count += 1

                if step % max(1, int(cfg.get("log_interval", 10))) == 0:
                    self.writer.add_scalar("train/loss_step", last_loss_value, global_step)
                    if not use_live:
                        logger.info(
                            f"Ep {epoch}/{cfg['epochs']} | step {step} | loss {last_loss_value:.4f}"
                        )

                if use_live and (step == 1 or step % refresh_steps == 0 or step == len(self.train_loader)):
                    sampled_avg = running_loss_sum / max(1, running_loss_count)
                    postfix = {
                        "loss": f"{last_loss_value:.4f}",
                        "avg~": f"{sampled_avg:.4f}",
                        "lr":   f"{self.opt.param_groups[0]['lr']:.2e}",
                        "best": f"{self.best_dice:.4f}",
                    }
                    if self.device.type == "cuda":
                        # This is an instantaneous post-step allocation, not the
                        # peak VRAM metric. Peak VRAM is recorded separately.
                        postfix["VRAM-now"] = f"{torch.cuda.memory_allocated(self.device) / 1024**3:.1f}G"
                    train_iter.set_postfix(postfix, refresh=False)

                # The loss has already been accumulated and, when required,
                # sampled for logging. Drop the per-batch tensor immediately.
                del loss_tensor

                # Diff-UNet uses very large transient 3-D tensors. Periodically
                # return unused cached blocks to the CUDA driver so the allocator
                # does not progressively reserve essentially all 32 GB of VRAM.
                # This does not alter live tensors, gradients, optimiser state or
                # model outputs. The trade-off is a small allocator overhead.
                if (
                    self.device.type == "cuda"
                    and str(cfg.get("model", "")).lower() == "diff_unet"
                    and step % 10 == 0
                ):
                    torch.cuda.empty_cache()

                last_batch_end = time.perf_counter()

            # Record one end event after the final optimiser update and perform a
            # single synchronisation for the epoch-level CUDA timing measurement.
            if self.device.type == "cuda":
                if epoch_gpu_timing_started:
                    epoch_gpu_end_evt.record()
                    torch.cuda.synchronize(self.device)
                    gpu_step_s = epoch_gpu_start_evt.elapsed_time(epoch_gpu_end_evt) / 1000.0
                else:
                    gpu_step_s = 0.0
                del epoch_gpu_start_evt, epoch_gpu_end_evt
            else:
                gpu_step_s = cpu_gpu_step_s

            # A single device-to-host reduction for the exact epoch mean loss.
            if epoch_loss_count > 0:
                if self.device.type == "cuda":
                    mean_loss = float((epoch_loss_sum / epoch_loss_count).item())
                else:
                    mean_loss = float(epoch_loss_sum / epoch_loss_count)
            else:
                mean_loss = float("nan")
            del epoch_loss_sum

            # Capture GPU state immediately after training, before validation changes
            # utilisation or memory pressure.
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                train_gpu_allocated_gb = torch.cuda.memory_allocated(self.device) / 1024**3
                train_gpu_reserved_gb = torch.cuda.memory_reserved(self.device) / 1024**3
                train_telemetry = query_nvidia_smi() if cfg.get("gpu_telemetry", True) else {}
            else:
                train_gpu_allocated_gb = 0.0
                train_gpu_reserved_gb = 0.0
                train_telemetry = {}

            self._gpu_epoch_times.append(float(gpu_step_s))
            if len(self._gpu_epoch_times) >= 4:
                baseline = float(np.median(self._gpu_epoch_times[:3]))
                ratio = gpu_step_s / max(baseline, 1e-9)
                warn_ratio = float(cfg.get("gpu_slowdown_warn_ratio", 1.50))
                if ratio >= warn_ratio:
                    telemetry_txt = ""
                    if train_telemetry:
                        telemetry_txt = (
                            f" temp={train_telemetry.get('gpu_temp_c', float('nan')):.0f}C"
                            f" clock={train_telemetry.get('gpu_sm_clock_mhz', float('nan')):.0f}MHz"
                            f" power={train_telemetry.get('gpu_power_w', float('nan')):.0f}W"
                            f" util={train_telemetry.get('gpu_util_pct', float('nan')):.0f}%"
                        )
                    logger.warning(
                        f"GPU training slowdown detected: epoch GPU time is {ratio:.2f}x "
                        f"the median of epochs 1-3. allocated={train_gpu_allocated_gb:.2f} GB "
                        f"reserved={train_gpu_reserved_gb:.2f} GB.{telemetry_txt}"
                    )

            # Full validation runs every epoch by default.
            val_interval = max(1, int(cfg.get("val_interval", 1)))
            should_validate = (epoch == 1 or epoch % val_interval == 0 or epoch == cfg["epochs"])

            self.writer.add_scalar("train/loss_epoch", mean_loss, epoch)

            if should_validate:
                validation_started = time.perf_counter()
                val_m = self._validate()
                validation_s = time.perf_counter() - validation_started
                mean_dice = float(val_m["dice_mean"])
                mean_hd95 = float(val_m["hd95_mean"])

                if isinstance(self.sched, ReduceLROnPlateau):
                    self.sched.step(val_m["loss"])

                is_best = mean_dice > self.best_dice
                if is_best:
                    self.best_dice = mean_dice
                    self.best_val_metrics = dict(val_m)
                    self.best_epoch = epoch
                    self.epochs_no_improve = 0
                else:
                    self.epochs_no_improve += val_interval

                self._save_checkpoint(epoch, mean_dice, is_best)
                self.writer.add_scalar("val/epochs_no_improve", self.epochs_no_improve, epoch)
                for k, v in val_m.items():
                    self.writer.add_scalar(f"val/{k}", v, epoch)

                logger.info(
                    f"[{run_label} | Ep {epoch:03d}] "
                    f"train_loss={mean_loss:.4f}  val_loss={val_m['loss']:.4f}  "
                    f"Dice WT={val_m['dice_WT']:.4f} TC={val_m['dice_TC']:.4f} "
                    f"ET={val_m['dice_ET']:.4f} mean={mean_dice:.4f}  "
                    f"HD95 WT={val_m['hd95_WT']:.2f} TC={val_m['hd95_TC']:.2f} "
                    f"ET={val_m['hd95_ET']:.2f} mean={mean_hd95:.2f}"
                )
            else:
                logger.info(
                    f"[{run_label} | Ep {epoch:03d}] train_loss={mean_loss:.4f}  "
                    f"validation=skipped (interval={val_interval})"
                )

            # Per-epoch schedulers still step every epoch.
            if self.sched and not isinstance(self.sched, ReduceLROnPlateau):
                self.sched.step()
            self.writer.add_scalar("train/lr", self.opt.param_groups[0]["lr"], epoch)

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                current_peak_vram = torch.cuda.max_memory_allocated(self.device) / 1024**3
                self.peak_vram_gb = max(self.peak_vram_gb, float(current_peak_vram))
            else:
                current_peak_vram = 0.0

            epoch_time_s = time.perf_counter() - epoch_started
            self.epochs_completed = epoch
            self.writer.add_scalar("system/epoch_time_s", epoch_time_s, epoch)
            self.writer.add_scalar("system/peak_vram_gb", current_peak_vram, epoch)

            history_row = {
                "epoch": epoch,
                "train_loss": mean_loss,
                "val_loss": "" if val_m is None else val_m["loss"],
                "dice_WT": "" if val_m is None else val_m["dice_WT"],
                "dice_TC": "" if val_m is None else val_m["dice_TC"],
                "dice_ET": "" if val_m is None else val_m["dice_ET"],
                "dice_mean": "" if val_m is None else val_m["dice_mean"],
                "hd95_WT": "" if val_m is None else val_m["hd95_WT"],
                "hd95_TC": "" if val_m is None else val_m["hd95_TC"],
                "hd95_ET": "" if val_m is None else val_m["hd95_ET"],
                "hd95_mean": "" if val_m is None else val_m["hd95_mean"],
                "learning_rate": self.opt.param_groups[0]["lr"],
                "epoch_time_s": epoch_time_s,
                "data_wait_s": data_wait_s,
                "gpu_step_s": gpu_step_s,
                "validation_s": validation_s,
                "samples_per_s": (len(self.train_ds) / max(1e-9, data_wait_s + gpu_step_s)),
                "peak_vram_gb": current_peak_vram,
                "gpu_allocated_gb": train_gpu_allocated_gb,
                "gpu_reserved_gb": train_gpu_reserved_gb,
                "gpu_temp_c": train_telemetry.get("gpu_temp_c", ""),
                "gpu_util_pct": train_telemetry.get("gpu_util_pct", ""),
                "gpu_sm_clock_mhz": train_telemetry.get("gpu_sm_clock_mhz", ""),
                "gpu_power_w": train_telemetry.get("gpu_power_w", ""),
                "gpu_mem_used_mb": train_telemetry.get("gpu_mem_used_mb", ""),
                "gpu_mem_total_mb": train_telemetry.get("gpu_mem_total_mb", ""),
            }
            with open(self.history_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=history_row.keys()).writerow(history_row)

            if cfg.get("profile_pipeline", True):
                train_profile_total = max(1e-9, data_wait_s + gpu_step_s)
                wait_pct = 100.0 * data_wait_s / train_profile_total
                gpu_pct = 100.0 * gpu_step_s / train_profile_total
                logger.info(
                    f"[{run_label} | Ep {epoch:03d}] pipeline: "
                    f"data_wait={data_wait_s:.1f}s ({wait_pct:.1f}%) | "
                    f"GPU_train={gpu_step_s:.1f}s ({gpu_pct:.1f}%) | "
                    f"validation={validation_s:.1f}s | "
                    f"throughput={len(self.train_ds)/train_profile_total:.2f} samples/s"
                )
                if self.device.type == "cuda":
                    t = train_telemetry
                    logger.info(
                        f"[{run_label} | Ep {epoch:03d}] GPU state after training: "
                        f"allocated={train_gpu_allocated_gb:.2f} GB | "
                        f"reserved={train_gpu_reserved_gb:.2f} GB"
                        + (
                            f" | temp={t.get('gpu_temp_c', float('nan')):.0f}C "
                            f"util={t.get('gpu_util_pct', float('nan')):.0f}% "
                            f"clock={t.get('gpu_sm_clock_mhz', float('nan')):.0f}MHz "
                            f"power={t.get('gpu_power_w', float('nan')):.0f}W "
                            f"driver_mem={t.get('gpu_mem_used_mb', float('nan')):.0f}/"
                            f"{t.get('gpu_mem_total_mb', float('nan')):.0f} MB"
                            if t else ""
                        )
                    )
            logger.info(
                f"[{run_label} | Ep {epoch:03d}] lr={self.opt.param_groups[0]['lr']:.2e}  "
                f"time={epoch_time_s:.1f}s  peak_VRAM={current_peak_vram:.2f} GB"
            )

            # Release unused cached CUDA blocks once per completed epoch.
            # This is intentionally not done per batch, because per-batch cache clearing
            # would reduce throughput.  It helps keep long-run reserved VRAM stable.
            if self.device.type == "cuda":
                gc.collect()
                torch.cuda.empty_cache()

            if should_validate and self.epochs_no_improve >= cfg["early_stopping_patience"]:
                self.stop_reason = "early_stopping"
                logger.info(
                    f"Early stopping triggered: no Dice improvement for approximately "
                    f"{self.epochs_no_improve} epochs (patience={cfg['early_stopping_patience']}, "
                    f"best={self.best_dice:.4f})."
                )
                break

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            self.peak_vram_gb = max(
                self.peak_vram_gb,
                float(torch.cuda.max_memory_allocated(self.device) / 1024**3),
            )
        self.training_time_s = float(time.perf_counter() - training_started)

        run_metrics = {
            "model": cfg["model"],
            "training_time_s": self.training_time_s,
            "trainable_params": self.trainable_params,
            "trainable_params_m": self.params_m,
            "peak_vram_gb": self.peak_vram_gb,
            "best_epoch": self.best_epoch,
            "best_dice": self.best_dice,
            "best_val_metrics": self.best_val_metrics,
            "epochs_completed": self.epochs_completed,
            "stop_reason": self.stop_reason,
        }
        with open(self.run_dir / "run_metrics.json", "w") as f:
            json.dump(run_metrics, f, indent=2, default=str)

        logger.info(
            f"Training complete. Best mean Dice={self.best_dice:.4f} | "
            f"training_time={self.training_time_s:.1f}s | "
            f"params={self.params_m:.3f}M | peak_VRAM={self.peak_vram_gb:.2f}GB"
        )
        self.writer.close()

    def release_runtime_resources(self) -> None:
        """Release heavy training resources while retaining run metadata."""
        if getattr(self, "_runtime_resources_released", False):
            return
        self._runtime_resources_released = True

        try:
            self.release_runtime_resources()
        except Exception:
            pass

        for name in ("train_loader", "val_loader"):
            _shutdown_data_loader(getattr(self, name, None), self.logger)
        for name in ("train_ds", "val_ds"):
            _clear_dataset_runtime_cache(getattr(self, name, None))

        # Keep run_dir, metrics, batch size and other lightweight metadata, but
        # remove all large CPU/GPU objects before final-test evaluation starts.
        for name in (
            "train_loader", "val_loader", "train_ds", "val_ds",
            "model", "opt", "sched", "criterion", "scaler", "gpu_augmenter",
        ):
            try:
                if hasattr(self, name):
                    setattr(self, name, None)
            except Exception:
                pass

        gc.collect()
        _safe_cuda_release(getattr(self, "device", None))

        try:
            if self.device.type == "cuda":
                self.logger.info(
                    "Training resources released before final evaluation: "
                    f"CUDA allocated={torch.cuda.memory_allocated(self.device)/1024**3:.3f} GB, "
                    f"reserved={torch.cuda.memory_reserved(self.device)/1024**3:.3f} GB."
                )
        except Exception:
            pass

    # ── checkpoint helpers ────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, dice: float, is_best: bool):
        state = {
            "epoch":             epoch,
            "model":             self.cfg["model"],
            "state_dict":        unwrap_model(self.model).state_dict(),
            "optimizer":         self.opt.state_dict(),
            "dice":              dice,
            "epochs_no_improve": self.epochs_no_improve,
            "best_val_metrics":  self.best_val_metrics,
            "cfg":               self.cfg,
        }
        path = self.run_dir / "last.pth"
        torch.save(state, path)
        if is_best:
            torch.save(state, self.run_dir / "best.pth")
            self.logger.info(f"  New best Dice {dice:.4f} saved.")

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        unwrap_model(self.model).load_state_dict(ckpt["state_dict"])
        self.opt.load_state_dict(ckpt["optimizer"])
        self.start_ep          = ckpt["epoch"] + 1
        self.best_dice         = ckpt.get("dice", 0.0)
        self.epochs_no_improve = ckpt.get("epochs_no_improve", 0)
        self.best_val_metrics  = ckpt.get("best_val_metrics")
        self.logger.info(
            f"Resumed from {path} "
            f"(ep {ckpt['epoch']}, dice {self.best_dice:.4f}, "
            f"no-improve streak {self.epochs_no_improve})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class Evaluator:
    """Full evaluation on a given split: Dice + HD95 per region."""

    def __init__(self, cfg: dict, logger: logging.Logger):
        self.cfg    = cfg
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not cfg.get("checkpoint"):
            raise ValueError("--checkpoint must be provided for evaluation mode.")

        is_deepensemble = cfg.get("model", "").lower() == "deepensemble"
        ckpt = torch.load(
            cfg["checkpoint"],
            map_location="cpu" if is_deepensemble else self.device
        )
        saved_cfg = ckpt.get("cfg", cfg)
        saved_cfg.update({k: cfg[k] for k in ["data_dir", "val_ratio", "seed", "num_workers"]})

        self.model = build_model(saved_cfg)
        if is_deepensemble and "ensemble_member_state_dicts" in ckpt:
            self.model.load_member_state_dicts(ckpt["ensemble_member_state_dicts"])
        else:
            self.model.load_state_dict(ckpt["state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info(f"Loaded checkpoint: {cfg['checkpoint']}")

        all_pts  = find_patient_dirs(cfg["data_dir"])
        _, val_pts = make_splits(all_pts, cfg["val_ratio"], cfg["seed"])
        self.ds  = BraTS2020Dataset(
            val_pts, cfg["patch_size"], augment=False, cache_rate=0.0,
            preprocessed_cache_dir=resolve_preprocessed_cache_dir(cfg),
            mmap_lru_patients=cfg.get("mmap_lru_patients", 32),
        )
        self.loader = make_data_loader(self.ds, cfg, batch_size=1, shuffle=False)
        self.results_path = Path(cfg["save_dir"]) / "eval_results.csv"
        Path(cfg["save_dir"]).mkdir(parents=True, exist_ok=True)

    def run(self):
        logger  = self.logger
        rows    = []
        agg     = {f"dice_{n}": [] for n in REGION_NAMES}
        agg.update({f"hd95_{n}": [] for n in REGION_NAMES})

        for image, label, pid in self.loader:
            image = move_image_to_device(image, self.device, self.cfg)
            with torch.inference_mode(), amp_context(self.device, self.cfg.get("amp", True)):
                if isinstance(self.model, DiffUNet):
                    pred = self.model(image)
                else:
                    pred = self.model(image)
            pred_cls  = pred.argmax(1).cpu().numpy()[0]      # (H, W, D) predicted classes
            label_cls = label.cpu().numpy()[0, 0]            # (H, W, D) ground-truth classes
            pred_r    = region_masks(pred_cls)
            label_r   = region_masks(label_cls)

            row = {"patient": pid[0]}
            for name in REGION_NAMES:
                d = dice_score(pred_r[name].astype(np.float32),
                               label_r[name].astype(np.float32))
                h = hausdorff95(pred_r[name], label_r[name])
                row[f"dice_{name}"] = round(d, 4)
                row[f"hd95_{name}"] = round(h, 4)
                agg[f"dice_{name}"].append(d)
                agg[f"hd95_{name}"].append(h)
            rows.append(row)
            logger.info(f"  {pid[0]:30s} | WT={row['dice_WT']:.4f}  TC={row['dice_TC']:.4f}  ET={row['dice_ET']:.4f}")

        # Summary
        summary = {k: round(float(np.nanmean(v)), 4) for k, v in agg.items()}
        rows.append({"patient": "MEAN", **summary})
        logger.info("\n=== SUMMARY ===")
        for name in REGION_NAMES:
            logger.info(f"  {name}  Dice={summary[f'dice_{name}']:.4f}  HD95={summary[f'hd95_{name}']:.4f}")

        # Write CSV
        with open(self.results_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"\nResults saved to {self.results_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIXED 80/10/10 EXPERIMENT RUNNER
# ─────────────────────────────────────────────────────────────────────────────


class FixedSplitRunner:
    """Single-fit 80/10/10 experiment using only the labelled BraTS cohort.

    The labelled training cohort is split once at patient level into 80% training,
    10% validation and 10% final test. Validation controls early stopping and best
    checkpoint selection. The final test patients are evaluated only once after
    training and never participate in fitting or model selection.
    """

    def __init__(self, cfg: dict, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self.save_dir = Path(cfg["save_dir"])
        self.save_dir.mkdir(parents=True, exist_ok=True)

        train_root = resolve_labelled_training_root(cfg)
        cfg["train_data_dir"] = str(train_root)
        cfg["test_data_dir"] = None

        self.labelled_patients = find_patient_dirs(str(train_root))
        if not self.labelled_patients:
            raise RuntimeError(f"No labelled BraTS patients found in {train_root}")
        require_segmentation_masks(self.labelled_patients, "Labelled BraTS cohort")

        self.train_patients, self.val_patients, self.test_patients, self.split_path = (
            make_or_load_fixed_80_10_10_split(self.labelled_patients, cfg, logger)
        )
        total = len(self.labelled_patients)
        logger.info(
            f"Fixed 80/10/10 design: labelled={total} | "
            f"train={len(self.train_patients)} ({100.0 * len(self.train_patients) / total:.1f}%) | "
            f"validation={len(self.val_patients)} ({100.0 * len(self.val_patients) / total:.1f}%) | "
            f"final_test={len(self.test_patients)} ({100.0 * len(self.test_patients) / total:.1f}%)"
        )
        logger.info(f"Labelled cohort root: {train_root}")
        logger.info("Official BraTS validation cohort is ignored and is not scanned, cached, trained on or evaluated.")
        logger.info("Final-test patients are not used for training, augmentation fitting, early stopping, LR scheduling or checkpoint selection.")

    @staticmethod
    def _upsert_model_row(path: Path, row: Dict[str, object]):
        rows = []
        if path.exists():
            try:
                with open(path, "r", newline="", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
            except Exception:
                rows = []
        model_name = str(row["Model"])
        rows = [r for r in rows if str(r.get("Model", "")) != model_name]
        rows.append({k: row[k] for k in row})
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            for existing in rows:
                writer.writerow({k: existing.get(k, "") for k in row})

    def _train_deepensemble_members(self):
        """Train five paper-based ensemble members sequentially on the fixed split."""
        cfg = self.cfg
        logger = self.logger
        member_count = max(1, int(cfg.get("deepensemble_members", 5)))
        base_seed = int(cfg.get("seed", 123))

        logger.info(
            f"DeepEnsemble: training {member_count} independently initialised "
            "Henry-style members sequentially on the same fixed train/validation split."
        )
        logger.info(
            "Paper-based elements: width 48, GroupNorm, dilated pseudo-fifth stage, "
            "trilinear decoder, four deep-supervision heads, ensemble probability "
            "averaging and 16-way TTA. The study's four-class loss/labels and fixed "
            "80/10/10 split are retained for comparability."
        )

        member_states = []
        member_meta = []
        total_training_s = 0.0
        peak_vram_gb = 0.0
        batch_sizes = []

        ensemble_run_dir = (
            self.save_dir
            / f"deepensemble_{datetime.now():%Y%m%d_%H%M%S}_fixed_80_10_10_ensemble"
        )
        ensemble_run_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(member_count):
            member_cfg = dict(cfg)
            member_cfg["_deepensemble_training_member"] = True
            member_cfg["deepensemble_member_index"] = idx
            member_cfg["seed"] = base_seed + idx
            member_cfg["checkpoint"] = None

            logger.info("")
            logger.info("=" * 72)
            logger.info(
                f"DeepEnsemble member {idx + 1}/{member_count}, seed={member_cfg['seed']}"
            )
            logger.info("=" * 72)

            trainer = Trainer(
                member_cfg,
                logger,
                train_pts=self.train_patients,
                val_pts=self.val_patients,
                run_suffix=f"fixed_80_10_10_member{idx + 1:02d}",
            )
            trainer.run()

            best_member = trainer.run_dir / "best.pth"
            if not best_member.exists():
                raise RuntimeError(
                    f"DeepEnsemble member {idx + 1} did not create best.pth"
                )

            ckpt = torch.load(best_member, map_location="cpu")
            member_states.append(ckpt["state_dict"])
            member_meta.append({
                "member": idx + 1,
                "seed": member_cfg["seed"],
                "best_epoch": trainer.best_epoch,
                "best_validation_dice": trainer.best_dice,
                "training_time_s": trainer.training_time_s,
                "peak_vram_gb": trainer.peak_vram_gb,
                "batch_size": trainer.effective_batch_size,
                "checkpoint": str(best_member),
            })

            total_training_s += float(trainer.training_time_s)
            peak_vram_gb = max(peak_vram_gb, float(trainer.peak_vram_gb))
            batch_sizes.append(int(trainer.effective_batch_size))

            trainer.release_runtime_resources()
            del trainer, ckpt
            gc.collect()
            _safe_cuda_release()

        ensemble_ckpt = ensemble_run_dir / "best.pth"
        torch.save({
            "model": "deepensemble",
            "cfg": cfg,
            "ensemble_member_state_dicts": member_states,
            "ensemble_members": member_count,
            "member_metadata": member_meta,
            "paper_reference": "Henry et al., arXiv:2011.01045",
        }, ensemble_ckpt)

        if len(set(batch_sizes)) > 1:
            logger.warning(
                f"DeepEnsemble member batch sizes differed: {batch_sizes}. "
                "The efficiency table reports the smallest physical member batch."
            )

        summary = SimpleNamespace(
            run_dir=ensemble_run_dir,
            training_time_s=float(total_training_s),
            peak_vram_gb=float(peak_vram_gb),
            best_epoch=[m["best_epoch"] for m in member_meta],
            stop_reason="ensemble_members_completed",
            best_dice=float(np.mean([m["best_validation_dice"] for m in member_meta])),
            effective_batch_size=int(min(batch_sizes) if batch_sizes else 1),
            batch_size=int(min(batch_sizes) if batch_sizes else 1),
        )

        with open(ensemble_run_dir / "ensemble_members.json", "w", encoding="utf-8") as f:
            json.dump(member_meta, f, indent=2, default=str)

        logger.info(
            f"DeepEnsemble members complete: total training={total_training_s:.1f}s | "
            f"peak member VRAM={peak_vram_gb:.2f}GB | checkpoint={ensemble_ckpt}"
        )
        return summary, ensemble_ckpt

    def run(self):
        cfg = self.cfg
        logger = self.logger

        if cfg.get("model", "").lower() == "deepensemble":
            trainer, best_ckpt = self._train_deepensemble_members()
        else:
            trainer = Trainer(
                cfg,
                logger,
                train_pts=self.train_patients,
                val_pts=self.val_patients,
                run_suffix="fixed_80_10_10",
            )
            trainer.run()

            best_ckpt = trainer.run_dir / "best.pth"
            if not best_ckpt.exists():
                raise RuntimeError(f"Best checkpoint was not created: {best_ckpt}")

        logger.info("Training/selection complete. Beginning one-time evaluation on the fixed 10% final test split.")

        # Reuse the thoroughly tested benchmark/evaluation implementation from
        # CrossValidationRunner without constructing CV folds.
        evaluator = object.__new__(CrossValidationRunner)
        evaluator.cfg = cfg
        evaluator.logger = logger
        evaluator.save_dir = self.save_dir
        evaluator.test_patients = self.test_patients
        evaluator.static_gflops_per_case = None
        test_rows, benchmark = CrossValidationRunner._eval_checkpoint_on_test(
            evaluator, best_ckpt, fold_idx=0
        )
        if not test_rows:
            raise RuntimeError("Final 10% test evaluation produced no patient results.")

        # Persist patient-level results for paired architecture comparisons.
        per_patient_path = self.save_dir / f"final_test_per_patient_{cfg['model']}.csv"
        clean_rows = []
        for row in test_rows:
            clean = dict(row)
            clean.pop("fold", None)
            clean_rows.append(clean)
        with open(per_patient_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(clean_rows[0].keys()))
            writer.writeheader()
            writer.writerows(clean_rows)
        logger.info(f"Final-test per-patient results -> {per_patient_path}")

        def mean_metric(key: str) -> float:
            vals = np.asarray([float(r[key]) for r in clean_rows], dtype=np.float64)
            vals = vals[~np.isnan(vals)]
            return float(vals.mean()) if vals.size else float("nan")

        dice_wt = mean_metric("dice_WT")
        dice_tc = mean_metric("dice_TC")
        dice_et = mean_metric("dice_ET")
        dice_mean = mean_metric("dice_mean")
        hd95_wt = mean_metric("hd95_WT")
        hd95_tc = mean_metric("hd95_TC")
        hd95_et = mean_metric("hd95_ET")
        hd95_mean = mean_metric("hd95_mean")

        # Full descriptive test summary.
        summary_rows = []
        for metric in [
            "dice_WT", "dice_TC", "dice_ET", "dice_mean",
            "hd95_WT", "hd95_TC", "hd95_ET", "hd95_mean",
        ]:
            arr = np.asarray([float(r[metric]) for r in clean_rows], dtype=np.float64)
            valid = arr[~np.isnan(arr)]
            summary_rows.append({
                "metric": metric,
                "mean": round(float(valid.mean()), 4) if valid.size else float("nan"),
                "std": round(float(valid.std()), 4) if valid.size else float("nan"),
                "median": round(float(np.median(valid)), 4) if valid.size else float("nan"),
                "min": round(float(valid.min()), 4) if valid.size else float("nan"),
                "max": round(float(valid.max()), 4) if valid.size else float("nan"),
                "n_valid": int(valid.size),
            })
        summary_path = self.save_dir / f"final_test_summary_{cfg['model']}.csv"
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        logger.info(f"Final-test summary -> {summary_path}")

        inference_ms = float(benchmark["inference_ms"])
        inference_s = inference_ms / 1000.0
        params = int(benchmark["trainable_params"])
        params_m = float(benchmark["params_m"])
        gflops = float(benchmark["gflops"])

        segmentation_row = {
            "Model": cfg["model"],
            "DiceCoef": round(dice_mean, 4),
            "HD95": round(hd95_mean, 4) if not np.isnan(hd95_mean) else float("nan"),
            "DiceET": round(dice_et, 4),
            "DiceWT": round(dice_wt, 4),
            "DiceTC": round(dice_tc, 4),
        }
        efficiency_row = {
            "Model": cfg["model"],
            "Batch Size": int(getattr(trainer, "batch_size", cfg.get("resolved_batch_size", cfg.get("batch_size", 0)))),
            "Training time (s)": round(float(trainer.training_time_s), 3),
            "Trainable Params": params,
            # Forward inference FLOPs per complete patient.  For 2-D models the
            # per-slice profile is multiplied by the number of slices actually
            # evaluated for each patient; 3-D models use one full-volume pass.
            "GFLOPs": round(gflops, 4),
            "Dice/s": round(dice_mean / inference_s, 6) if inference_s > 0 else float("nan"),
            "Dice/M": round(dice_mean / params_m, 6) if params_m > 0 else float("nan"),
            "Inference (ms)": round(inference_ms, 4),
            "Peak VRAM (GB)": round(float(trainer.peak_vram_gb), 4),
        }

        segmentation_path = self.save_dir / "segmentation_metrics.csv"
        efficiency_path = self.save_dir / "efficiency_metrics.csv"
        combined_path = self.save_dir / "model_metrics.csv"
        self._upsert_model_row(segmentation_path, segmentation_row)
        self._upsert_model_row(efficiency_path, efficiency_row)
        combined_row = {**segmentation_row, **{k: v for k, v in efficiency_row.items() if k != "Model"}}
        self._upsert_model_row(combined_path, combined_row)

        run_summary = {
            "model": cfg["model"],
            "design": "80% train / 10% validation / 10% final test from labelled BraTS cohort",
            "train_n": len(self.train_patients),
            "validation_n": len(self.val_patients),
            "final_test_n": len(self.test_patients),
            "best_epoch": trainer.best_epoch,
            "stop_reason": trainer.stop_reason,
            "best_validation_dice": trainer.best_dice,
            "deepensemble_members": (
                int(cfg.get("deepensemble_members", 5))
                if cfg.get("model", "").lower() == "deepensemble" else 1
            ),
            "deepensemble_tta": (
                bool(cfg.get("deepensemble_tta", True))
                if cfg.get("model", "").lower() == "deepensemble" else False
            ),
            "final_test_dice": dice_mean,
            "final_test_hd95": hd95_mean,
            "gflops_definition": "forward inference GFLOPs per complete patient (multiply-add = 2 FLOPs)",
            "split_file": str(self.split_path),
            "best_checkpoint": str(best_ckpt),
        }
        with open(trainer.run_dir / "fixed_split_experiment_summary.json", "w", encoding="utf-8") as f:
            json.dump(run_summary, f, indent=2, default=str)

        logger.info("=== FINAL 10% TEST PERFORMANCE ===")
        logger.info(
            f"Model={segmentation_row['Model']} | DiceCoef={segmentation_row['DiceCoef']:.4f} | "
            f"HD95={segmentation_row['HD95']:.4f} | DiceET={segmentation_row['DiceET']:.4f} | "
            f"DiceWT={segmentation_row['DiceWT']:.4f} | DiceTC={segmentation_row['DiceTC']:.4f}"
        )
        logger.info("=== EFFICIENCY ===")
        logger.info(
            f"Training={efficiency_row['Training time (s)']:.1f}s | "
            f"Params={efficiency_row['Trainable Params']:,} | GFLOPs={efficiency_row['GFLOPs']:.3f} | "
            f"Dice/s={efficiency_row['Dice/s']:.4f} | Dice/M={efficiency_row['Dice/M']:.4f} | "
            f"Inference={efficiency_row['Inference (ms)']:.2f}ms | "
            f"Peak VRAM={efficiency_row['Peak VRAM (GB)']:.2f}GB"
        )


class CrossValidationRunner:
    """
    Orchestrates k-fold cross-validation with a held-out test set.

    Workflow
    --------
    1. Hold out ``test_ratio`` (default 10 %) of patients -> never touched
       during CV.
    2. Split the remaining patients into ``n_folds`` folds.
    3. For each fold: train a fresh model, save best checkpoint, record
       per-fold validation Dice.
    4. After all folds: evaluate every best checkpoint on the held-out test
       set and report aggregated statistics.
    5. Write three CSVs to ``save_dir``:
         cv_fold_summary.csv      - per-fold val Dice (WT/TC/ET + mean)
         cv_test_per_patient.csv  - per-patient test Dice for each fold model
         cv_test_summary.csv      - mean +/- std across fold models on test set
    """

    def __init__(self, cfg: dict, logger: logging.Logger):
        self.cfg    = cfg
        self.logger = logger
        self.save_dir = Path(cfg["save_dir"])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.static_gflops_per_case: Optional[float] = None

        all_patients = find_patient_dirs(cfg["data_dir"])
        if not all_patients:
            raise RuntimeError(f"No BraTS patient dirs found in {cfg['data_dir']}")

        self.test_patients, self.folds = make_cv_splits(
            all_patients,
            n_folds    = cfg["n_folds"],
            test_ratio = cfg["test_ratio"],
            seed       = cfg["seed"],
        )
        logger.info(
            f"CV setup: {len(all_patients)} total patients | "
            f"{len(self.test_patients)} held-out test | "
            f"{len(self.folds)} folds of "
            f"~{len(self.folds[0][0])} train / ~{len(self.folds[0][1])} val"
        )

        # Persist the split so results are reproducible
        split_info = {
            "test_patients": [str(p) for p in self.test_patients],
            "folds": [
                {"train": [str(p) for p in tr], "val": [str(p) for p in va]}
                for tr, va in self.folds
            ],
        }
        with open(self.save_dir / "cv_split.json", "w") as f:
            json.dump(split_info, f, indent=2)

    # ── per-fold evaluation on the test set ──────────────────────────────────

    def _eval_checkpoint_on_test(
        self,
        ckpt_path: Path,
        fold_idx: int,
    ) -> Tuple[List[Dict], Dict[str, float]]:
        """
        Evaluate a best checkpoint on the held-out test set and benchmark
        inference. Timing covers model forward execution only, excluding data
        loading, Dice/HD95 calculation and CSV writing.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_deepensemble = self.cfg.get("model", "").lower() == "deepensemble"
        ckpt = torch.load(
            str(ckpt_path),
            map_location="cpu" if is_deepensemble else device
        )
        model = build_model(self.cfg)
        if is_deepensemble and "ensemble_member_state_dicts" in ckpt:
            model.load_member_state_dicts(ckpt["ensemble_member_state_dicts"])
        else:
            model.load_state_dict(ckpt["state_dict"])
        model = model.to(device)
        if (
            device.type == "cuda"
            and self.cfg.get("channels_last_3d", True)
            and not is_2d_model(self.cfg)
            and self.cfg["model"].lower() in {"unet3d", "hybridattunet", "deepensemble"}
        ):
            model = model.to(memory_format=torch.channels_last_3d)
        model.eval()

        if is_2d_model(self.cfg):
            ds = BraTS2020SliceDataset(
                self.test_patients, self.cfg["patch_size"],
                augment=False, skip_empty_ratio=0.0,
                preprocessed_cache_dir=resolve_preprocessed_cache_dir(self.cfg),
                mmap_lru_patients=self.cfg.get("mmap_lru_patients", 32),
            )
            batch_size = int(self.cfg.get("batch_size_2d", 64))
        else:
            ds = BraTS2020Dataset(
                self.test_patients, self.cfg["patch_size"], augment=False,
                preprocessed_cache_dir=resolve_preprocessed_cache_dir(self.cfg),
                mmap_lru_patients=self.cfg.get("mmap_lru_patients", 32),
            )
            batch_size = 1

        loader = make_data_loader(ds, self.cfg, batch_size=batch_size, shuffle=False)
        loader_iter = iter(loader)
        try:
            first_batch = next(loader_iter)
        except StopIteration:
            return [], {
                "inference_ms": float("nan"),
                "gflops": float("nan"),
                "trainable_params": count_trainable_params(model),
                "params_m": count_trainable_params(model) / 1e6,
            }

        # Profile GFLOPs once per complete CV experiment. For UNet2D the hook
        # measures one slice, then scales to the average number of test slices
        # in a complete patient volume so architectures remain comparable.
        first_image_for_profile = move_image_to_device(first_batch[0][:1], device, self.cfg)
        if self.static_gflops_per_case is None:
            if is_deepensemble:
                # Count one Henry member once, then scale by the number of
                # independently trained members and TTA predictions. This is
                # mathematically equivalent to profiling the complete wrapper
                # but avoids running 80 full-volume forwards just for FLOP hooks.
                profile_member = HenryEquiUNet3D(
                    in_channels=self.cfg["in_channels"],
                    out_channels=self.cfg["num_classes"],
                    width=int(self.cfg.get("deepensemble_width", 48)),
                    norm_groups=int(self.cfg.get("deepensemble_norm_groups", 16)),
                    deep_supervision=False,
                    activation_checkpointing=False,
                ).to(device)
                profile_member.eval()
                gflops_one = estimate_gflops(
                    profile_member, first_image_for_profile, self.cfg
                )
                member_count = int(self.cfg.get("deepensemble_members", 5))
                tta_count = 16 if bool(self.cfg.get("deepensemble_tta", True)) else 1
                self.static_gflops_per_case = (
                    gflops_one * member_count * tta_count
                )
                del profile_member
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            else:
                gflops_per_input = estimate_gflops(model, first_image_for_profile, self.cfg)
                if is_2d_model(self.cfg):
                    avg_slices_per_case = len(ds) / max(1, len(self.test_patients))
                    self.static_gflops_per_case = gflops_per_input * avg_slices_per_case
                else:
                    self.static_gflops_per_case = gflops_per_input
            self.logger.info(
                f"Model complexity: {self.static_gflops_per_case:.3f} GFLOPs per patient inference"
            )

        # Warm up CUDA kernels. These forwards are deliberately excluded from
        # the inference-time measurement.
        warmup_image = move_image_to_device(first_batch[0], device, self.cfg)
        warmup_steps = max(0, int(self.cfg.get("inference_warmup", 3)))
        if is_deepensemble:
            warmup_steps = min(warmup_steps, 1)
        with torch.inference_mode():
            for _ in range(warmup_steps):
                with amp_context(device, self.cfg.get("amp", True)):
                    model(warmup_image)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        rows: List[Dict] = []
        # 2-D models must be reconstructed into complete volumes before Dice
        # and HD95 are computed. Values are uint8 to keep RAM usage modest.
        volume_slices: Dict[str, List[Tuple[int, np.ndarray, np.ndarray]]] = {}
        total_inference_ms = 0.0

        def process_batch(batch):
            nonlocal total_inference_ms
            image, label, pids = batch
            image = move_image_to_device(image, device, self.cfg)

            with torch.inference_mode():
                if device.type == "cuda":
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    with amp_context(device, self.cfg.get("amp", True)):
                        pred = model(image)
                    end_event.record()
                    torch.cuda.synchronize(device)
                    total_inference_ms += float(start_event.elapsed_time(end_event))
                else:
                    t0 = time.perf_counter()
                    with amp_context(device, self.cfg.get("amp", True)):
                        pred = model(image)
                    total_inference_ms += float((time.perf_counter() - t0) * 1000.0)

            pred_cls = pred.argmax(1).cpu().numpy().astype(np.uint8)
            label_cls = label.cpu().numpy()
            if label_cls.ndim == pred_cls.ndim + 1:
                label_cls = label_cls[:, 0]
            label_cls = label_cls.astype(np.uint8)

            if is_2d_model(self.cfg):
                for b, raw_pid in enumerate(pids):
                    pid, z_text = raw_pid.rsplit("_z", 1)
                    z = int(z_text)
                    volume_slices.setdefault(pid, []).append(
                        (z, pred_cls[b], label_cls[b])
                    )
            else:
                for b, pid in enumerate(pids):
                    m = metrics_from_class_maps(pred_cls[b], label_cls[b])
                    row = {"fold": fold_idx, "patient": pid}
                    for name in REGION_NAMES:
                        row[f"dice_{name}"] = round(m[f"dice_{name}"], 4)
                        row[f"hd95_{name}"] = (
                            round(m[f"hd95_{name}"], 4)
                            if not np.isnan(m[f"hd95_{name}"]) else float("nan")
                        )
                    row["dice_mean"] = round(m["dice_mean"], 4)
                    row["hd95_mean"] = (
                        round(m["hd95_mean"], 4) if not np.isnan(m["hd95_mean"]) else float("nan")
                    )
                    rows.append(row)

        process_batch(first_batch)
        for batch in loader_iter:
            process_batch(batch)

        if is_2d_model(self.cfg):
            for pid, slices in volume_slices.items():
                slices.sort(key=lambda x: x[0])
                pred_volume = np.stack([x[1] for x in slices], axis=-1)
                label_volume = np.stack([x[2] for x in slices], axis=-1)
                m = metrics_from_class_maps(pred_volume, label_volume)
                row = {"fold": fold_idx, "patient": pid}
                for name in REGION_NAMES:
                    row[f"dice_{name}"] = round(m[f"dice_{name}"], 4)
                    row[f"hd95_{name}"] = (
                        round(m[f"hd95_{name}"], 4)
                        if not np.isnan(m[f"hd95_{name}"]) else float("nan")
                    )
                row["dice_mean"] = round(m["dice_mean"], 4)
                row["hd95_mean"] = (
                    round(m["hd95_mean"], 4) if not np.isnan(m["hd95_mean"]) else float("nan")
                )
                rows.append(row)

        n_cases = max(1, len(rows))
        inference_ms = total_inference_ms / n_cases
        params = count_trainable_params(model)
        benchmark = {
            "inference_ms": float(inference_ms),
            "gflops": float(self.static_gflops_per_case),
            "trainable_params": int(params),
            "params_m": float(params / 1e6),
        }
        self.logger.info(
            f"Fold {fold_idx} inference benchmark: {inference_ms:.2f} ms/patient | "
            f"{self.static_gflops_per_case:.3f} GFLOPs | {params/1e6:.3f} M params"
        )
        return rows, benchmark

    # ── main CV loop ──────────────────────────────────────────────────────────

    def run(self):
        logger        = self.logger
        cfg           = self.cfg
        fold_summary  = []   # one row per fold
        all_test_rows = []   # one row per (fold, patient)

        for k, (train_pts, val_pts) in enumerate(self.folds, start=1):
            logger.info(f"\n{'='*60}")
            logger.info(f"  FOLD {k} / {cfg['n_folds']}")
            logger.info(f"{'='*60}")

            # Fresh trainer for this fold
            trainer = Trainer(
                cfg,
                logger,
                train_pts  = train_pts,
                val_pts    = val_pts,
                run_suffix = f"fold{k:02d}",
            )
            trainer.run()

            # Record validation, training-resource and efficiency metrics for the fold.
            fold_row = {
                "fold": k,
                "best_epoch": trainer.best_epoch if trainer.best_epoch is not None else "",
                "best_dice": round(trainer.best_dice, 4),
                "training_time_s": round(trainer.training_time_s, 3),
                "batch_size": int(trainer.effective_batch_size),
                "trainable_params": trainer.trainable_params,
                "params_m": round(trainer.params_m, 6),
                "peak_vram_gb": round(trainer.peak_vram_gb, 4),
                "stop_reason": trainer.stop_reason,
            }
            best_ckpt = trainer.run_dir / "best.pth"
            if best_ckpt.exists():
                best_val = trainer.best_val_metrics or {}
                for name in REGION_NAMES:
                    fold_row[f"val_dice_{name}"] = round(float(best_val.get(f"dice_{name}", 0.0)), 4)
                    hd = float(best_val.get(f"hd95_{name}", float("nan")))
                    fold_row[f"val_hd95_{name}"] = round(hd, 4) if not np.isnan(hd) else float("nan")
                fold_row["val_dice_mean"] = round(float(best_val.get("dice_mean", trainer.best_dice)), 4)
                val_hd95_mean = float(best_val.get("hd95_mean", float("nan")))
                fold_row["val_hd95_mean"] = (
                    round(val_hd95_mean, 4) if not np.isnan(val_hd95_mean) else float("nan")
                )

                # Evaluate on held-out test set.
                logger.info(f"  Evaluating fold {k} best checkpoint on held-out test set ...")
                test_rows, benchmark = self._eval_checkpoint_on_test(best_ckpt, fold_idx=k)
                all_test_rows.extend(test_rows)
                for name in REGION_NAMES:
                    fold_row[f"test_dice_{name}"] = round(
                        float(np.mean([r[f"dice_{name}"] for r in test_rows])), 4
                    )
                    fold_row[f"test_hd95_{name}"] = round(
                        safe_nanmean([r[f"hd95_{name}"] for r in test_rows]), 4
                    )
                fold_row["test_dice_mean"] = round(
                    float(np.mean([r["dice_mean"] for r in test_rows])), 4
                )
                fold_row["test_hd95_mean"] = round(
                    safe_nanmean([r["hd95_mean"] for r in test_rows]), 4
                )
                fold_row["gflops"] = round(benchmark["gflops"], 4)
                fold_row["inference_ms"] = round(benchmark["inference_ms"], 4)

                inference_s = benchmark["inference_ms"] / 1000.0
                fold_row["dice_per_s"] = (
                    round(fold_row["test_dice_mean"] / inference_s, 6)
                    if inference_s > 0 else float("nan")
                )
                fold_row["dice_per_m"] = (
                    round(fold_row["test_dice_mean"] / trainer.params_m, 6)
                    if trainer.params_m > 0 else float("nan")
                )

                logger.info(
                    f"  Fold {k} | test Dice={fold_row['test_dice_mean']:.4f} | "
                    f"HD95={fold_row['test_hd95_mean']:.2f} | "
                    f"train={fold_row['training_time_s']:.1f}s | "
                    f"inference={fold_row['inference_ms']:.2f}ms | "
                    f"VRAM={fold_row['peak_vram_gb']:.2f}GB"
                )

            fold_summary.append(fold_row)

        # ── aggregate and write CSVs ──────────────────────────────────────────
        self._write_cv_summaries(fold_summary, all_test_rows)

    def _write_cv_summaries(
        self,
        fold_summary:  List[Dict],
        all_test_rows: List[Dict],
    ):
        logger = self.logger

        # 1. Per-fold val + test summary
        summary_path = self.save_dir / "cv_fold_summary.csv"
        if fold_summary:
            with open(summary_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fold_summary[0].keys())
                writer.writeheader()
                writer.writerows(fold_summary)

            # Append a MEAN +/- STD row
            numeric_keys = [k for k in fold_summary[0] if k != "fold"]
            stats_row = {"fold": "MEAN+/-STD"}
            for k in numeric_keys:
                vals = [r[k] for r in fold_summary if isinstance(r.get(k), (int, float))]
                if vals:
                    arr = np.asarray(vals, dtype=np.float64)
                    valid = arr[~np.isnan(arr)]
                    if valid.size:
                        stats_row[k] = f"{valid.mean():.4f}+/-{valid.std():.4f}"
            with open(summary_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fold_summary[0].keys()).writerow(stats_row)

        logger.info(f"Fold summary -> {summary_path}")

        # 2. Per-patient test results
        if all_test_rows:
            test_pp_path = self.save_dir / "cv_test_per_patient.csv"
            with open(test_pp_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_test_rows[0].keys())
                writer.writeheader()
                writer.writerows(all_test_rows)
            logger.info(f"Per-patient test results -> {test_pp_path}")

            # 3. Test set aggregate summary: Dice and true HD95.
            test_summary_path = self.save_dir / "cv_test_summary.csv"
            agg: Dict[str, List[float]] = {}
            for name in REGION_NAMES:
                agg[f"dice_{name}"] = []
                agg[f"hd95_{name}"] = []
            agg["dice_mean"] = []
            agg["hd95_mean"] = []

            for row in all_test_rows:
                for metric in agg:
                    agg[metric].append(row[metric])

            summary_rows = []
            for metric, vals in agg.items():
                arr = np.asarray(vals, dtype=np.float64)
                valid = arr[~np.isnan(arr)]
                if valid.size:
                    summary_rows.append({
                        "metric": metric,
                        "mean": round(float(valid.mean()), 4),
                        "std": round(float(valid.std()), 4),
                        "median": round(float(np.median(valid)), 4),
                        "min": round(float(valid.min()), 4),
                        "max": round(float(valid.max()), 4),
                        "n_valid": int(valid.size),
                    })
                else:
                    summary_rows.append({
                        "metric": metric, "mean": float("nan"), "std": float("nan"),
                        "median": float("nan"), "min": float("nan"), "max": float("nan"),
                        "n_valid": 0,
                    })
            with open(test_summary_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
                writer.writeheader()
                writer.writerows(summary_rows)

            logger.info(f"Test set summary -> {test_summary_path}")

            # 4. One model-level efficiency row for easy architecture comparison.
            valid_folds = [r for r in fold_summary if isinstance(r.get("test_dice_mean"), (int, float))]
            if valid_folds:
                mean_test_dice = float(np.mean([r["test_dice_mean"] for r in valid_folds]))
                mean_test_hd95 = safe_nanmean([r.get("test_hd95_mean", float("nan")) for r in valid_folds])
                mean_inference_ms = float(np.mean([r["inference_ms"] for r in valid_folds]))
                params = int(valid_folds[0]["trainable_params"])
                params_m = float(valid_folds[0]["params_m"])
                gflops = float(np.mean([r["gflops"] for r in valid_folds]))
                total_training_time_s = float(np.sum([r["training_time_s"] for r in valid_folds]))
                mean_training_time_s = float(np.mean([r["training_time_s"] for r in valid_folds]))
                peak_vram_gb = float(np.max([r["peak_vram_gb"] for r in valid_folds]))
                inference_s = mean_inference_ms / 1000.0

                model_row = {
                    "model": self.cfg["model"],
                    "folds_completed": len(valid_folds),
                    "dice_mean": round(mean_test_dice, 4),
                    "hd95_mean": round(mean_test_hd95, 4) if not np.isnan(mean_test_hd95) else float("nan"),
                    "training_time_s_total": round(total_training_time_s, 3),
                    "training_time_s_mean_fold": round(mean_training_time_s, 3),
                    "batch_size": int(valid_folds[0].get("batch_size", self.cfg.get("resolved_batch_size", 0))),
                    "trainable_params": params,
                    "params_m": round(params_m, 6),
                    "gflops": round(gflops, 4),
                    "dice_per_s": round(mean_test_dice / inference_s, 6) if inference_s > 0 else float("nan"),
                    "dice_per_m": round(mean_test_dice / params_m, 6) if params_m > 0 else float("nan"),
                    "inference_ms": round(mean_inference_ms, 4),
                    "peak_vram_gb": round(peak_vram_gb, 4),
                }
                model_summary_path = self.save_dir / "cv_model_summary.csv"
                with open(model_summary_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=model_row.keys())
                    writer.writeheader()
                    writer.writerow(model_row)
                logger.info(f"Model efficiency summary -> {model_summary_path}")

                # Publication-ready segmentation table requested for architecture comparison.
                mean_dice_et = float(np.mean([r["test_dice_ET"] for r in valid_folds]))
                mean_dice_wt = float(np.mean([r["test_dice_WT"] for r in valid_folds]))
                mean_dice_tc = float(np.mean([r["test_dice_TC"] for r in valid_folds]))

                segmentation_row = {
                    "Model": self.cfg["model"],
                    "DiceCoef": round(mean_test_dice, 4),
                    "HD95": round(mean_test_hd95, 4) if not np.isnan(mean_test_hd95) else float("nan"),
                    "DiceET": round(mean_dice_et, 4),
                    "DiceWT": round(mean_dice_wt, 4),
                    "DiceTC": round(mean_dice_tc, 4),
                }

                efficiency_row = {
                    "Model": self.cfg["model"],
                    "Batch Size": int(valid_folds[0].get("batch_size", self.cfg.get("resolved_batch_size", self.cfg.get("batch_size", 0)))),
                    "Training time (s)": round(total_training_time_s, 3),
                    "Trainable Params": params,
                    # Forward inference FLOPs per complete patient.
                    "GFLOPs": round(gflops, 4),
                    "Dice/s": round(mean_test_dice / inference_s, 6) if inference_s > 0 else float("nan"),
                    "Dice/M": round(mean_test_dice / params_m, 6) if params_m > 0 else float("nan"),
                    "Inference (ms)": round(mean_inference_ms, 4),
                    "Peak VRAM (GB)": round(peak_vram_gb, 4),
                }

                def upsert_model_row(path: Path, row: Dict[str, object]):
                    """Insert or replace one model row while preserving results from other models."""
                    rows = []
                    if path.exists():
                        try:
                            with open(path, "r", newline="", encoding="utf-8") as f:
                                rows = list(csv.DictReader(f))
                        except Exception:
                            rows = []

                    # Replace an earlier run of the same architecture instead of creating duplicates.
                    model_name = str(row["Model"])
                    rows = [r for r in rows if str(r.get("Model", "")) != model_name]
                    rows.append({k: row[k] for k in row.keys()})

                    with open(path, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                        writer.writeheader()
                        for existing in rows:
                            writer.writerow({k: existing.get(k, "") for k in row.keys()})

                segmentation_path = self.save_dir / "segmentation_metrics.csv"
                efficiency_path = self.save_dir / "efficiency_metrics.csv"
                upsert_model_row(segmentation_path, segmentation_row)
                upsert_model_row(efficiency_path, efficiency_row)

                # Keep a combined compatibility file as well, but make it complete.
                combined_row = {**segmentation_row, **{k: v for k, v in efficiency_row.items() if k != "Model"}}
                combined_path = self.save_dir / "model_metrics.csv"
                upsert_model_row(combined_path, combined_row)

                logger.info(f"Segmentation metrics -> {segmentation_path}")
                logger.info(f"Efficiency metrics -> {efficiency_path}")
                logger.info(f"Combined model metrics -> {combined_path}")

                logger.info("\n=== SEGMENTATION PERFORMANCE ===")
                logger.info(
                    f"Model={segmentation_row['Model']} | DiceCoef={segmentation_row['DiceCoef']:.4f} | "
                    f"HD95={segmentation_row['HD95']:.4f} | DiceET={segmentation_row['DiceET']:.4f} | "
                    f"DiceWT={segmentation_row['DiceWT']:.4f} | DiceTC={segmentation_row['DiceTC']:.4f}"
                )
                logger.info("=== EFFICIENCY ===")
                logger.info(
                    f"Model={efficiency_row['Model']} | Training={efficiency_row['Training time (s)']:.1f}s | "
                    f"Params={efficiency_row['Trainable Params']:,} | GFLOPs={efficiency_row['GFLOPs']:.3f} | "
                    f"Dice/s={efficiency_row['Dice/s']:.4f} | Dice/M={efficiency_row['Dice/M']:.4f} | "
                    f"Inference={efficiency_row['Inference (ms)']:.2f}ms | "
                    f"Peak VRAM={efficiency_row['Peak VRAM (GB)']:.2f}GB"
                )

            logger.info("\n=== CROSS-VALIDATION COMPLETE ===")
            for row in summary_rows:
                logger.info(
                    f"  {row['metric']:12s}  mean={row['mean']:.4f}  "
                    f"std={row['std']:.4f}  median={row['median']:.4f}"
                )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

MODEL_MENU = [
    # 2-D models, ranked by trainable parameter count
    ("unet2d",          "UNet 2D (~7.76M params, corrected)"),
    ("hvu",             "HVU 2D DenseVU-ED (~36.51M params)"),
    ("deeplabv3plus2d", "DeepLabV3+ 2D (~40.35M params)"),

    # 3-D models, ranked by trainable parameter count
    ("diff_unet",       "Diff-UNet 3D (~10.05M params)"),
    ("hybridattunet",        "HybridAttUnet 3D (~14.10M params)"),
    ("unet3d",          "UNet 3D (~22.58M params)"),
    ("deepensemble",        "DeepEnsemble 3D (5 x ~23.16M = ~115.78M params)"),
]


def select_model_interactively() -> str:
    """Require an explicit numbered model selection when --model is omitted."""
    print("\nSelect a model:")
    for idx, (_, label) in enumerate(MODEL_MENU, start=1):
        print(f"  {idx}. {label}")

    while True:
        try:
            choice = input(
                f"\nEnter model number 1-{len(MODEL_MENU)}: "
            ).strip()
        except KeyboardInterrupt:
            print("\nModel selection cancelled.")
            raise SystemExit(130)
        except EOFError:
            raise RuntimeError(
                "No interactive input is available. Supply a model explicitly with "
                "--model (for example: --model unet3d)."
            )

        if choice.isdigit():
            selected = int(choice)
            if 1 <= selected <= len(MODEL_MENU):
                key, label = MODEL_MENU[selected - 1]
                print(f"Selected: {label}\n")
                return key

        if choice == "":
            print("No default model is selected. Please choose a number.")
        else:
            print(f"Invalid selection. Please enter a number from 1 to {len(MODEL_MENU)}.")


def parse_args():
    p = argparse.ArgumentParser(
        description="Unified BraTS2020 Brain Tumour Segmentation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode",       default="train",  choices=["train", "eval"])
    p.add_argument("--model",      default=None,
                   choices=list(MODEL_REGISTRY),
                   help="Model architecture. Interactive training still prompts by default; use --no_model_prompt to bypass it.")
    p.add_argument("--no_model_prompt", action="store_true",
                   help="Skip the interactive model menu. Requires --model in training mode.")
    p.add_argument("--list_models", action="store_true",
                   help="Print available models and exit")

    # Data
    p.add_argument("--data_dir",   default=None, help="Root containing the labelled BraTS training cohort (or its parent folder)")
    p.add_argument("--train_data_dir", default=None,
                   help="Labelled BraTS training cohort root. If omitted, auto-detected under --data_dir")
    p.add_argument("--test_data_dir", default=None,
                   help="Deprecated and ignored. The official unlabelled validation cohort is not used.")
    p.add_argument("--val_ratio",  type=float, default=None,
                   help="Fraction of the labelled cohort reserved for validation (default 0.10)")
    p.add_argument("--fixed_test_ratio", type=float, default=None,
                   help="Fraction of the labelled cohort reserved for final testing (default 0.10)")
    p.add_argument("--fixed_split_file", default=None,
                   help="JSON file storing the fixed 80/10/10 split reused across models")
    p.add_argument("--patch_size", type=int, nargs=3, default=None,
                   metavar=("H", "W", "D"),
                   help="Model input crop size H W D. Default: full BraTS "
                        "geometry 240 240 155. Each dim is padded up to a "
                        "multiple of 16 for the 3-D U-Nets (e.g. 155 -> 160). "
                        "A smaller size yields tumour-centred sub-volumes.")
    p.add_argument("--num_workers",type=int, default=None,
                   help="DataLoader workers for all models (default 8); negative uses model-specific fallback")
    p.add_argument("--num_workers_2d", type=int, default=None,
                   help="2-D model worker fallback if --num_workers is negative (default 8)")
    p.add_argument("--num_workers_3d", type=int, default=None,
                   help="3-D model worker fallback if --num_workers is negative (default 8)")
    p.add_argument("--cache_rate", type=float, default=None,
                   help="RAM cache fraction per dataset; negative uses automatic sizing")
    p.add_argument("--system_ram_gb", type=float, default=None,
                   help="System RAM used for automatic cache sizing (default 64 GB)")
    p.add_argument("--ram_cache_fraction", type=float, default=None,
                   help="Fraction of system RAM targeted by all worker caches (default 0.70)")
    p.add_argument("--prefetch_factor", type=int, default=None)
    p.add_argument("--prefetch_ram_gb", type=float, default=None,
                   help="Maximum RAM used by prefetched pinned batches across workers (default 8 GB)")
    p.add_argument("--mmap_lru_patients", type=int, default=None,
                   help="Open patient mmap handles retained inside each worker (default 32)")
    p.add_argument("--hd95_workers", type=int, default=None,
                   help="CPU threads used for parallel HD95 calculation (default 4)")
    p.add_argument("--no_patient_grouped_batches", action="store_true",
                   help="Disable patient-local batch construction for the 2-D models")
    p.add_argument("--no_cuda_prefetch", action="store_true",
                   help="Disable asynchronous pinned-memory CUDA prefetching")

    # Model
    p.add_argument("--base_filters", type=int, default=None)
    p.add_argument("--unet2d_max_batch_size", type=int, default=None,
                   help="Hard physical batch-size ceiling for UNet2D (default 64)")
    p.add_argument("--unet2d_norm_groups", type=int, default=None,
                   help="Number of GroupNorm groups used by UNet2D (default 8)")
    p.add_argument(
        "--unet2d_skip_empty_ratio",
        type=float,
        default=None,
        help=(
            "Fraction of tumour-free axial training slices discarded by UNet2D. "
            "Corrected default is 0.0, meaning all native slices are retained."
        ),
    )
    p.add_argument("--deepensemble_members", type=int, default=None,
                   help="Number of independently trained DeepEnsemble members (default 5)")
    p.add_argument("--deepensemble_width", type=int, default=None,
                   help="Henry-style base feature width (paper default 48)")
    p.add_argument("--deepensemble_max_batch_size", type=int, default=None,
                   help="Hard physical batch-size ceiling per DeepEnsemble member (default 1)")
    p.add_argument("--deepensemble_empty_cache_interval", type=int, default=None,
                   help="Release unused CUDA cache every N DeepEnsemble training batches (default 10)")
    p.add_argument("--no_deepensemble_tta", action="store_true",
                   help="Disable paper-style 16-way DeepEnsemble test-time augmentation")

    # Training
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--batch_size", type=int,   default=None,
                   help="Batch size for 3-D models (default 1)")
    p.add_argument("--batch_size_2d", type=int, default=None,
                   help="Starting batch size for 2-D model auto-tuning (default 64)")
    p.add_argument("--no_auto_batch", action="store_true",
                   help="Disable automatic VRAM batch-size tuning and use the configured batch size exactly")
    p.add_argument("--batch_vram_fraction", type=float, default=None,
                   help="Fraction of total VRAM the auto batch tuner may reserve (default 0.90)")
    p.add_argument("--batch_vram_headroom_gb", type=float, default=None,
                   help="Minimum free VRAM required after an auto-batch probe (default 3 GB)")
    p.add_argument("--max_batch_size_2d", type=int, default=None,
                   help="Maximum 2-D batch size considered by the auto tuner (default 1048)")
    p.add_argument("--max_batch_size_3d", type=int, default=None,
                   help="Maximum 3-D batch size considered by the auto tuner")
    p.add_argument("--diff_unet_max_batch_size", type=int, default=None,
                   help="Hard physical batch-size ceiling for Diff-UNet (default 1; prevents unsafe batch-2 probes)")
    p.add_argument("--batch_size_step", type=int, default=None,
                   help="2-D batch-size refinement step (default 16)")
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--weight_decay",type=float,default=None)
    p.add_argument("--scheduler",  default=None, choices=["cosine","plateau","none"])
    p.add_argument("--amp",        action="store_true", default=None)
    p.add_argument("--seed",       type=int, default=None)
    p.add_argument("--checkpoint", default=None, help="Path to .pth checkpoint")
    p.add_argument("--save_dir",   default=None, help="Directory for run outputs")
    p.add_argument("--log_interval", type=int, default=None)
    p.add_argument("--live_refresh_steps", type=int, default=None,
                   help="Refresh live loss text every N batches; larger values reduce CUDA sync overhead")
    p.add_argument("--val_interval", type=int, default=None,
                   help="Run full validation every N epochs (default 1)")
    p.add_argument("--early_stopping_patience", type=int, default=None,
                   help="Stop training after approximately N epochs without Dice improvement (default 30)")
    p.add_argument("--no_augment", action="store_true")
    p.add_argument(
        "--no_live_console", action="store_true",
        help="Disable live training/validation progress bars and use periodic logs only",
    )
    p.add_argument("--compile", action="store_true",
                   help="Enable torch.compile after auto batch tuning (off by default for host-RAM safety)")
    p.add_argument("--no_compile", action="store_true",
                   help="Disable torch.compile")
    p.add_argument("--no_fused_adamw", action="store_true",
                   help="Disable fused CUDA AdamW")
    p.add_argument("--no_channels_last_3d", action="store_true",
                   help="Disable channels-last 3-D memory format")
    p.add_argument("--no_persistent_workers", action="store_true",
                   help="Restart DataLoader workers every epoch")
    p.add_argument("--no_tf32", action="store_true", help="Disable TF32 CUDA kernels")
    p.add_argument("--no_cudnn_benchmark", action="store_true",
                   help="Disable cuDNN fixed-shape autotuning")
    p.add_argument("--no_cache_compress", action="store_true",
                   help="Keep cached MRI volumes in float32 instead of float16")
    p.add_argument("--allow_cpu", action="store_true",
                   help="Explicitly allow CPU training. By default CUDA is required to prevent silent CPU fallback.")
    p.add_argument("--no_preprocessed_cache", action="store_true",
                   help="Disable the normalised memory-mapped .npy cache")
    p.add_argument("--no_prewarm_cache", action="store_true",
                   help="Do not sequentially prewarm the preprocessed cache into the OS RAM file cache")
    p.add_argument("--no_profile_pipeline", action="store_true",
                   help="Disable per-epoch data-wait/GPU-compute timing diagnostics")

    # Cross-validation
    p.add_argument("--cv",         action="store_true",
                   help="Run k-fold cross-validation instead of a single train run")
    p.add_argument("--n_folds",    type=int, default=None,
                   help="Number of CV folds (default 10)")
    p.add_argument("--test_ratio", type=float, default=None,
                   help="Fraction of patients held out as a test set, never used during CV (default 0.10)")

    return p.parse_args()


def main():
    args = parse_args()

    if args.list_models:
        print("\nAvailable models:")
        descriptions = {
            # 2-D models, ordered by trainable parameter count
            "unet2d":          "UNet 2D (~7.76M params) - corrected padded 2-D U-Net on axial slices",
            "hvu":             "HVU 2D DenseVU-ED (~36.51M params) - DenseNet121 + ViT + U-Net",
            "deeplabv3plus2d": "DeepLabV3+ 2D (~40.35M params) - ResNet-50 style encoder + ASPP",
            # 3-D models, ordered by trainable parameter count
            "diff_unet":       "Diff-UNet 3D (~10.05M params) - diffusion-embedded U-Net",
            "hybridattunet":        "HybridAttUnet 3D (~14.10M params) - residual attention + squeeze-excitation",
            "unet3d":          "UNet 3D (~22.58M params) - vanilla volumetric U-Net",
            "deepensemble":        "DeepEnsemble 3D - 5-member Henry-style deep-supervised ensemble with TTA",
        }
        for k, v in descriptions.items():
            print(f"  {k:<18}  {v}")
        sys.exit(0)

    # Interactive training is prompt-first by design.  This deliberately does
    # not silently accept a model value that may have been left in an IDE run
    # configuration.  To bypass the menu for scripted runs, explicitly use
    # both --no_model_prompt and --model <name>.
    if args.mode == "train":
        if args.no_model_prompt:
            if args.model is None:
                raise SystemExit(
                    "--no_model_prompt requires an explicit --model, for example "
                    "--model unet3d --no_model_prompt"
                )
        else:
            if args.model is not None:
                print(
                    f"\nA command-line model ('{args.model}') was supplied, but the "
                    "interactive model menu is the default. Please choose the model below."
                )
            args.model = select_model_interactively()

    cfg = build_config(args)
    if args.no_augment:
        cfg["augment"] = False
    if args.no_live_console:
        cfg["live_console"] = False
    if getattr(args, "compile", False):
        cfg["compile_model"] = True
    if args.no_compile:
        cfg["compile_model"] = False
    if args.no_auto_batch:
        cfg["auto_batch_size"] = False
    if args.no_fused_adamw:
        cfg["fused_adamw"] = False
    if args.no_channels_last_3d:
        cfg["channels_last_3d"] = False
    if args.no_persistent_workers:
        cfg["persistent_workers"] = False
    if args.no_tf32:
        cfg["allow_tf32"] = False
    if args.no_cudnn_benchmark:
        cfg["cudnn_benchmark"] = False
    if args.no_cache_compress:
        cfg["cache_compress"] = False
    if args.allow_cpu:
        cfg["require_cuda"] = False
    if args.no_preprocessed_cache:
        cfg["preprocessed_cache"] = False
    if args.no_prewarm_cache:
        cfg["prewarm_cache"] = False
    if args.no_profile_pipeline:
        cfg["profile_pipeline"] = False
    if args.no_patient_grouped_batches:
        cfg["patient_grouped_batches"] = False
    if args.no_cuda_prefetch:
        cfg["cuda_prefetch"] = False
    if isinstance(cfg["patch_size"], list):
        cfg["patch_size"] = tuple(cfg["patch_size"])

    if not 0.0 <= float(cfg.get("unet2d_skip_empty_ratio", 0.0)) <= 1.0:
        raise ValueError(
            "unet2d_skip_empty_ratio must be between 0.0 and 1.0."
        )

    run_dir = Path(cfg["save_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(run_dir))

    device = verify_cuda_runtime(cfg, logger)
    logger.info(f"Mode: {cfg['mode']} | Model: {cfg['model']} | Device: {device}")

    # Resolve only the labelled BraTS cohort. The official unlabelled validation
    # dataset is deliberately ignored in the default experiment.
    if cfg["mode"] == "train":
        train_root = resolve_labelled_training_root(cfg)
        cfg["train_data_dir"] = str(train_root)
        cfg["test_data_dir"] = None
        # Force every training workflow, including optional legacy CV, to the
        # labelled training cohort only. This prevents recursive discovery from
        # ever pulling in the official unlabelled BraTS validation cases.
        cfg["data_dir"] = str(train_root)
        all_patients_for_cache = find_patient_dirs(str(train_root))
        require_segmentation_masks(all_patients_for_cache, "Labelled BraTS cohort")
    else:
        all_patients_for_cache = find_patient_dirs(cfg["data_dir"])

    # Build a compact normalised cache once for the labelled cohort only.
    if cfg.get("preprocessed_cache", True):
        if not all_patients_for_cache:
            raise RuntimeError(f"No BraTS patient dirs found in {cfg['data_dir']}")
        cache_root = prepare_preprocessed_cache(all_patients_for_cache, cfg, logger)
        if cache_root is not None:
            cfg["preprocessed_cache_dir"] = str(cache_root)
            if str(cfg.get("model", "")).lower() in {"unet2d", "deeplabv3plus2d"} and cfg.get("slice_major_cache_2d", True):
                prepare_slice_major_2d_cache(all_patients_for_cache, cache_root, logger)
            if cfg.get("prewarm_cache", False):
                prewarm_preprocessed_cache(all_patients_for_cache, cache_root, logger)

    if cfg.get("cv"):
        CrossValidationRunner(cfg, logger).run()
    elif cfg["mode"] == "train":
        FixedSplitRunner(cfg, logger).run()
    else:
        Evaluator(cfg, logger).run()


if __name__ == "__main__":
    # Always clean up, on normal completion, Ctrl+C, or any raised exception.
    # Once this finally block ends, the Python process exits and Windows also
    # reclaims any remaining process RAM and VRAM.
    try:
        main()
    finally:
        try:
            _cleanup_logger = logging.getLogger("brats_pipeline")
        except Exception:
            _cleanup_logger = None
        shutdown_process_resources(_cleanup_logger)
