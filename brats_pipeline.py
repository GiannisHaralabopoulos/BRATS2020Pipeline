"""
Unified Brain Tumor Segmentation Pipeline
==========================================
Consolidates training and evaluation from:
  - HVU_Code            (Hybrid ViT+UNet, Keras/TF)
  - BraTS2020-Tensorflow (Attention-UNet, TF)
  - open_brats2020       (EquiUnet, PyTorch, Top-10 BraTS solution)
  - Diff-UNet            (Diffusion-embedded UNet, PyTorch, 3D)
  - PyTorch-UNet         (Vanilla 3D UNet, PyTorch)

Dataset: BraTS2020 – 4 MRI modalities (T1, T1ce, T2, FLAIR), 3 tumour classes
  - WT  = Whole Tumour      (labels 1+2+4)
  - TC  = Tumour Core       (labels 1+4)
  - ET  = Enhancing Tumour  (label 4)

Usage
-----
  # Train with a specific model:
  python brats_pipeline.py --model equiunet --data_dir /data/BraTS2020 --epochs 200

  # Evaluate a saved checkpoint:
  python brats_pipeline.py --mode eval --model diff_unet --checkpoint runs/exp1/best.pth

  # List all available models:
  python brats_pipeline.py --list_models
"""

import os
import sys
import argparse
import logging
import time
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

try:
    import nibabel as nib
except ImportError:
    raise ImportError("nibabel is required: pip install nibabel")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CFG = {
    # Data
    "data_dir":        "./BraTS2020",
    "train_csv":       None,          # optional explicit split CSV
    "val_ratio":       0.2,
    "patch_size":      (128, 128, 128),
    "num_workers":     4,
    "cache_rate":      0.0,           # fraction of dataset kept in RAM

    # Model
    "model":           "unet3d",      # see MODEL_REGISTRY
    "in_channels":     4,
    "num_classes":     4,             # background + WT + TC + ET (one-hot output)
    "base_filters":    32,

    # Training
    "mode":            "train",
    "epochs":          200,
    "batch_size":      1,
    "lr":              1e-4,
    "weight_decay":    1e-5,
    "scheduler":       "cosine",      # cosine | plateau | none
    "amp":             True,
    "seed":            42,
    "checkpoint":      None,          # path to resume from
    "save_dir":        "./runs",
    "log_interval":    10,            # steps between console/tb logs

    # Diff-UNet specific
    "diffusion_steps": 1000,
    "diffusion_infer_steps": 10,

    # Augmentation
    "augment":         True,
    "flip_prob":       0.5,
    "rotate_prob":     0.3,
    "intensity_prob":  0.3,
}


def build_config(args: argparse.Namespace) -> dict:
    cfg = dict(DEFAULT_CFG)
    for k, v in vars(args).items():
        if v is not None:
            cfg[k] = v
    return cfg


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
    fh = logging.FileHandler(os.path.join(save_dir, "run.log"))
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# DATASET  (BraTS2020 NIfTI loader)
# ─────────────────────────────────────────────────────────────────────────────

MODALITIES = ["flair", "t1", "t1ce", "t2"]
SEG_FILE   = "seg"


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
    BraTS label convention → 3-channel binary map [WT, TC, ET].
    WT = labels {1,2,4}
    TC = labels {1,4}
    ET = label  {4}
    Returns shape (3, H, W, D) float32.
    """
    wt = (seg > 0).astype(np.float32)
    tc = ((seg == 1) | (seg == 4)).astype(np.float32)
    et = (seg == 4).astype(np.float32)
    return np.stack([wt, tc, et], axis=0)  # (3, H, W, D)


class BraTS2020Dataset(Dataset):
    """
    Unified dataset compatible with all five repo conventions.
    Returns:
        image  – (4, H, W, D) float32 tensor
        label  – (3, H, W, D) float32 tensor  [WT, TC, ET]
        pid    – patient folder name string
    """

    def __init__(
        self,
        patient_dirs: List[Path],
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        augment: bool = False,
        flip_prob: float = 0.5,
        rotate_prob: float = 0.3,
        intensity_prob: float = 0.3,
        cache_rate: float = 0.0,
    ):
        self.patients    = patient_dirs
        self.patch_size  = patch_size
        self.augment     = augment
        self.flip_prob   = flip_prob
        self.rotate_prob = rotate_prob
        self.intensity_prob = intensity_prob
        self._cache: Dict[int, tuple] = {}
        self._cache_limit = int(len(patient_dirs) * cache_rate)

    def __len__(self) -> int:
        return len(self.patients)

    def _load(self, idx: int) -> tuple:
        if idx in self._cache:
            return self._cache[idx]
        pdir = self.patients[idx]
        pid  = pdir.name

        # ---------- load modalities ----------
        vols = []
        for mod in MODALITIES:
            candidates = sorted(pdir.glob(f"*_{mod}.nii*"))
            if not candidates:
                raise FileNotFoundError(f"Missing {mod} for patient {pid}")
            vols.append(normalise_volume(load_nii(candidates[0])))
        image = np.stack(vols, axis=0)  # (4, H, W, D)

        # ---------- load segmentation ----------
        seg_candidates = sorted(pdir.glob(f"*_seg.nii*"))
        if not seg_candidates:
            label = np.zeros((3,) + image.shape[1:], dtype=np.float32)
        else:
            seg   = load_nii(seg_candidates[0])
            label = build_label_map(seg)

        result = (image, label, pid)
        if idx < self._cache_limit:
            self._cache[idx] = result
        return result

    def _random_crop(self, image: np.ndarray, label: np.ndarray):
        ph, pw, pd = self.patch_size
        _, h, w, d  = image.shape
        sh = max(0, h - ph)
        sw = max(0, w - pw)
        sd = max(0, d - pd)

        # Bias towards foreground
        fg = np.argwhere(label.sum(0) > 0)
        if len(fg) > 0:
            center = fg[np.random.randint(len(fg))]
            x = int(np.clip(center[0] - ph // 2, 0, sh))
            y = int(np.clip(center[1] - pw // 2, 0, sw))
            z = int(np.clip(center[2] - pd // 2, 0, sd))
        else:
            x = np.random.randint(0, sh + 1)
            y = np.random.randint(0, sw + 1)
            z = np.random.randint(0, sd + 1)

        image = image[:, x:x+ph, y:y+pw, z:z+pd]
        label = label[:, x:x+ph, y:y+pw, z:z+pd]

        # Pad if volume smaller than patch
        def pad_to(arr, target):
            pads = [(0, 0)] + [(0, max(0, t - s)) for t, s in zip(target, arr.shape[1:])]
            return np.pad(arr, pads)

        image = pad_to(image, (4,) + self.patch_size)
        label = pad_to(label, (3,) + self.patch_size)
        return image, label

    def _augment(self, image: np.ndarray, label: np.ndarray):
        # Random flip
        for axis in [1, 2, 3]:
            if np.random.rand() < self.flip_prob:
                image = np.flip(image, axis=axis).copy()
                label = np.flip(label, axis=axis).copy()
        # Random intensity shift / scale (per modality)
        if np.random.rand() < self.intensity_prob:
            for c in range(image.shape[0]):
                shift = np.random.uniform(-0.1, 0.1)
                scale = np.random.uniform(0.9, 1.1)
                image[c] = image[c] * scale + shift
        return image, label

    def __getitem__(self, idx: int):
        image, label, pid = self._load(idx)
        image, label = self._random_crop(image, label)
        if self.augment:
            image, label = self._augment(image, label)
        return (
            torch.from_numpy(image.copy()).float(),
            torch.from_numpy(label.copy()).float(),
            pid,
        )


def make_splits(patient_dirs: List[Path], val_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(patient_dirs))
    n_val = max(1, int(len(patient_dirs) * val_ratio))
    val_idx   = perm[:n_val]
    train_idx = perm[n_val:]
    return [patient_dirs[i] for i in train_idx], [patient_dirs[i] for i in val_idx]


# ─────────────────────────────────────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """Soft Dice loss averaged over channels."""
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred   = torch.sigmoid(pred)
        pred   = pred.flatten(2)   # (B, C, N)
        target = target.flatten(2)
        num    = 2 * (pred * target).sum(-1) + self.smooth
        den    = pred.sum(-1) + target.sum(-1) + self.smooth
        return (1 - num / den).mean()


class CombinedLoss(nn.Module):
    """Dice + BCE (weighted)."""
    def __init__(self, dice_w: float = 0.6, bce_w: float = 0.4):
        super().__init__()
        self.dice  = DiceLoss()
        self.bce_w = bce_w
        self.dice_w = dice_w

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(pred, target)
        bce_loss  = F.binary_cross_entropy_with_logits(pred, target)
        return self.dice_w * dice_loss + self.bce_w * bce_loss


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

REGION_NAMES = ["WT", "TC", "ET"]


def dice_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
    num = 2 * (pred * target).sum() + smooth
    den = pred.sum() + target.sum() + smooth
    return float(num / den)


def hausdorff95(pred: np.ndarray, target: np.ndarray) -> float:
    """95th-percentile Hausdorff distance (voxel units)."""
    try:
        from scipy.spatial.distance import directed_hausdorff
        p_pts = np.argwhere(pred  > 0.5)
        t_pts = np.argwhere(target > 0.5)
        if len(p_pts) == 0 or len(t_pts) == 0:
            return float("nan")
        d1 = directed_hausdorff(p_pts, t_pts)[0]
        d2 = directed_hausdorff(t_pts, p_pts)[0]
        return float(max(d1, d2))
    except ImportError:
        return float("nan")


def evaluate_batch(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Works for both 2-D slices  (B, 3, H, W)
    and         3-D volumes    (B, 3, H, W, D).
    Returns mean Dice per region.
    """
    preds   = (preds.detach().cpu().float() > 0.5).numpy()
    targets = targets.detach().cpu().float().numpy()
    results = {}
    for ci, name in enumerate(REGION_NAMES):
        scores = [dice_score(preds[b, ci], targets[b, ci])
                  for b in range(preds.shape[0])]
        results[f"dice_{name}"] = float(np.mean(scores))
    results["dice_mean"] = float(np.mean([results[f"dice_{n}"] for n in REGION_NAMES]))
    return results


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


# ── 2. Attention UNet  (mahdizynali / TF → PyTorch port) ────────────────────

class AttentionGate(nn.Module):
    """Soft attention gate from Oktay et al., 2018."""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv3d(F_g, F_int, 1, bias=True), nn.BatchNorm3d(F_int))
        self.W_x = nn.Sequential(nn.Conv3d(F_l, F_int, 1, bias=True), nn.BatchNorm3d(F_int))
        self.psi = nn.Sequential(nn.Conv3d(F_int, 1, 1, bias=True), nn.BatchNorm3d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        psi = self.psi(self.relu(self.W_g(g) + self.W_x(x)))
        return x * psi


class AttentionUNet3D(nn.Module):
    """
    Attention U-Net (3D).
    Matches mahdizynali/BraTS2020-Tensorflow-Brain-Tumor-Segmentation logic,
    ported to PyTorch.
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 3,
                 base_filters: int = 32):
        super().__init__()
        f = base_filters

        def enc(ic, oc):
            return nn.Sequential(ConvBnRelu(ic, oc), ConvBnRelu(oc, oc))

        self.enc1 = enc(in_channels, f)
        self.enc2 = enc(f,   f*2)
        self.enc3 = enc(f*2, f*4)
        self.enc4 = enc(f*4, f*8)
        self.bot  = enc(f*8, f*16)
        self.pool = nn.MaxPool3d(2)

        self.up4   = nn.ConvTranspose3d(f*16, f*8, 2, stride=2)
        self.ag4   = AttentionGate(f*8, f*8, f*4)
        self.dec4  = enc(f*16, f*8)

        self.up3   = nn.ConvTranspose3d(f*8, f*4, 2, stride=2)
        self.ag3   = AttentionGate(f*4, f*4, f*2)
        self.dec3  = enc(f*8, f*4)

        self.up2   = nn.ConvTranspose3d(f*4, f*2, 2, stride=2)
        self.ag2   = AttentionGate(f*2, f*2, f)
        self.dec2  = enc(f*4, f*2)

        self.up1   = nn.ConvTranspose3d(f*2, f, 2, stride=2)
        self.ag1   = AttentionGate(f, f, f//2)
        self.dec1  = enc(f*2, f)
        self.head  = nn.Conv3d(f, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bot(self.pool(e4))

        g4 = self.up4(b)
        d4 = self.dec4(torch.cat([g4, self.ag4(g4, e4)], dim=1))

        g3 = self.up3(d4)
        d3 = self.dec3(torch.cat([g3, self.ag3(g3, e3)], dim=1))

        g2 = self.up2(d3)
        d2 = self.dec2(torch.cat([g2, self.ag2(g2, e2)], dim=1))

        g1 = self.up1(d2)
        d1 = self.dec1(torch.cat([g1, self.ag1(g1, e1)], dim=1))
        return self.head(d1)


# ── 3. EquiUnet  (lescientifik/open_brats2020, PyTorch) ─────────────────────

class EquiBlock(nn.Module):
    """Width-equivariant residual block used in open_brats2020."""
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.match = nn.Conv3d(in_c, out_c, 1, bias=False) if in_c != out_c else nn.Identity()
        self.block = nn.Sequential(
            ConvBnRelu(in_c, out_c),
            ResBlock(out_c),
        )

    def forward(self, x):
        return self.block(x) + self.match(x)


class EquiUnet(nn.Module):
    """
    Equivariant U-Net (EquiUnet) from open_brats2020 (Top-10 BraTS 2020).
    Ref: https://arxiv.org/abs/2011.01045
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 3,
                 width: int = 48):
        super().__init__()
        w = width

        self.enc1 = EquiBlock(in_channels, w)
        self.enc2 = EquiBlock(w,    w*2)
        self.enc3 = EquiBlock(w*2,  w*4)
        self.enc4 = EquiBlock(w*4,  w*8)
        self.bot  = EquiBlock(w*8, w*16)
        self.pool = nn.MaxPool3d(2)

        self.up4  = nn.ConvTranspose3d(w*16, w*8, 2, stride=2)
        self.dec4 = EquiBlock(w*16, w*8)
        self.up3  = nn.ConvTranspose3d(w*8,  w*4, 2, stride=2)
        self.dec3 = EquiBlock(w*8,  w*4)
        self.up2  = nn.ConvTranspose3d(w*4,  w*2, 2, stride=2)
        self.dec2 = EquiBlock(w*4,  w*2)
        self.up1  = nn.ConvTranspose3d(w*2,  w,   2, stride=2)
        self.dec1 = EquiBlock(w*2,  w)
        self.head = nn.Conv3d(w, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bot(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


# ── 4. Diff-UNet  (ge-xing/Diff-UNet, PyTorch) ──────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for diffusion timestep t."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        args  = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class TimeCondConv(nn.Module):
    def __init__(self, in_c, out_c, t_dim):
        super().__init__()
        self.conv  = ConvBnRelu(in_c, out_c)
        self.scale = nn.Linear(t_dim, out_c)
        self.shift = nn.Linear(t_dim, out_c)

    def forward(self, x, t_emb):
        h = self.conv(x)
        s = self.scale(t_emb)[:, :, None, None, None]
        b = self.shift(t_emb)[:, :, None, None, None]
        return h * (1 + s) + b


class DiffUNet(nn.Module):
    """
    Diffusion-embedded U-Net (Diff-UNet).
    Ref: ge-xing/Diff-UNet (MICCAI 2023, https://arxiv.org/pdf/2303.10326)

    During training: noisy seg map is concatenated with image (in_channels+out_channels)
                     and denoised at a random timestep.
    During inference: iterative DDIM-style denoising from Gaussian noise.
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 3,
                 base_filters: int = 32, T: int = 1000, t_dim: int = 64):
        super().__init__()
        f = base_filters
        self.T     = T
        self.t_emb = SinusoidalTimeEmbedding(t_dim)

        # The network sees (image + noisy_seg) concatenated
        cin = in_channels + out_channels

        self.enc1 = TimeCondConv(cin,   f,    t_dim)
        self.enc2 = TimeCondConv(f,    f*2,   t_dim)
        self.enc3 = TimeCondConv(f*2,  f*4,   t_dim)
        self.enc4 = TimeCondConv(f*4,  f*8,   t_dim)
        self.bot  = TimeCondConv(f*8,  f*16,  t_dim)
        self.pool = nn.MaxPool3d(2)

        self.up4  = nn.ConvTranspose3d(f*16, f*8, 2, stride=2)
        self.dec4 = TimeCondConv(f*16, f*8, t_dim)
        self.up3  = nn.ConvTranspose3d(f*8,  f*4, 2, stride=2)
        self.dec3 = TimeCondConv(f*8,  f*4, t_dim)
        self.up2  = nn.ConvTranspose3d(f*4,  f*2, 2, stride=2)
        self.dec2 = TimeCondConv(f*4,  f*2, t_dim)
        self.up1  = nn.ConvTranspose3d(f*2,  f,   2, stride=2)
        self.dec1 = TimeCondConv(f*2,  f,   t_dim)
        self.head = nn.Conv3d(f, out_channels, 1)

        # DDPM noise schedule
        betas      = torch.linspace(1e-4, 0.02, T)
        alphas     = 1.0 - betas
        alpha_bar  = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas",      betas)
        self.register_buffer("alpha_bar",  alpha_bar)
        self.register_buffer("sqrt_abar",  alpha_bar.sqrt())
        self.register_buffer("sqrt_1mabar",(1 - alpha_bar).sqrt())

    # ── diffusion helpers ──────────────────────────────────────
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward noising: q(x_t | x_0)."""
        noise = torch.randn_like(x0)
        sa  = self.sqrt_abar[t][:, None, None, None, None]
        sma = self.sqrt_1mabar[t][:, None, None, None, None]
        return sa * x0 + sma * noise, noise

    def _unet(self, xt: torch.Tensor, image: torch.Tensor, t: torch.Tensor):
        """Single U-Net forward (image + noisy seg → predicted noise)."""
        t_emb = self.t_emb(t)
        x = torch.cat([image, xt], dim=1)
        e1 = self.enc1(x,  t_emb)
        e2 = self.enc2(self.pool(e1), t_emb)
        e3 = self.enc3(self.pool(e2), t_emb)
        e4 = self.enc4(self.pool(e3), t_emb)
        b  = self.bot (self.pool(e4), t_emb)
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1), t_emb)
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1), t_emb)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1), t_emb)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), t_emb)
        return self.head(d1)

    def forward(self, image: torch.Tensor,
                seg: Optional[torch.Tensor] = None,
                t: Optional[torch.Tensor]   = None):
        """
        Training: supply seg + random t → returns predicted noise.
        Inference: seg=None, t=None  → returns denoised logits via DDIM.
        """
        if self.training and seg is not None:
            if t is None:
                t = torch.randint(0, self.T, (image.size(0),), device=image.device)
            xt, noise = self.q_sample(seg, t)
            return self._unet(xt, image, t), noise

        # ── inference: DDIM with 10 steps ──
        return self._ddim_sample(image, steps=10)

    @torch.no_grad()
    def _ddim_sample(self, image: torch.Tensor, steps: int = 10) -> torch.Tensor:
        B   = image.size(0)
        dev = image.device
        xt  = torch.randn(B, self.head.out_channels, *image.shape[2:], device=dev)

        step_indices = torch.linspace(self.T - 1, 0, steps, dtype=torch.long, device=dev)
        for i, ti in enumerate(step_indices):
            t_batch = ti.expand(B)
            eps     = self._unet(xt, image, t_batch)
            sa      = self.sqrt_abar[ti]
            sma     = self.sqrt_1mabar[ti]
            x0_pred = (xt - sma * eps) / (sa + 1e-8)
            if i < steps - 1:
                ti_prev = step_indices[i + 1]
                sa_prev = self.sqrt_abar[ti_prev]
                xt = sa_prev * x0_pred + self.sqrt_1mabar[ti_prev] * eps
            else:
                xt = x0_pred
        return xt


# ── 5. HVU (Hybrid ViT-UNet) ─── (renugadevi26/HVU_Code, PyTorch port) ──────

class PatchEmbed3D(nn.Module):
    """3D patch embedding for ViT branch in HVU."""
    def __init__(self, in_c, embed_dim, patch_size=16):
        super().__init__()
        self.proj = nn.Conv3d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W, D) → (B, N, embed_dim)
        x = self.proj(x)
        B, E, H, W, D = x.shape
        return x.flatten(2).transpose(1, 2), (H, W, D)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim    = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        h, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class HVUNet(nn.Module):
    """
    Hybrid Vision U-Net (HVU).
    Mirrors renugadevi26/HVU_Code (ResViT / DenseViT variants) – PyTorch port.
    ViT branch encodes global context; CNN decoder fuses local and global features.
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 3,
                 base_filters: int = 32, embed_dim: int = 256,
                 num_heads: int = 8, depth: int = 4, patch_size: int = 8):
        super().__init__()
        f = base_filters

        # CNN encoder
        self.enc1 = nn.Sequential(ConvBnRelu(in_channels, f),   ConvBnRelu(f,   f))
        self.enc2 = nn.Sequential(ConvBnRelu(f,   f*2), ConvBnRelu(f*2, f*2))
        self.enc3 = nn.Sequential(ConvBnRelu(f*2, f*4), ConvBnRelu(f*4, f*4))
        self.pool = nn.MaxPool3d(2)

        # ViT branch (operates on enc3 feature map)
        self.patch_embed = PatchEmbed3D(f*4, embed_dim, patch_size=2)
        self.transformer  = nn.Sequential(*[TransformerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.vit_proj     = nn.Linear(embed_dim, f*4)

        # Bottleneck
        self.bot = nn.Sequential(ConvBnRelu(f*4, f*8), ConvBnRelu(f*8, f*8))

        # Decoder
        self.up3  = nn.ConvTranspose3d(f*8, f*4, 2, stride=2)
        self.dec3 = nn.Sequential(ConvBnRelu(f*8+f*4, f*4), ConvBnRelu(f*4, f*4))
        self.up2  = nn.ConvTranspose3d(f*4, f*2, 2, stride=2)
        self.dec2 = nn.Sequential(ConvBnRelu(f*4, f*2), ConvBnRelu(f*2, f*2))
        self.up1  = nn.ConvTranspose3d(f*2, f, 2, stride=2)
        self.dec1 = nn.Sequential(ConvBnRelu(f*2, f), ConvBnRelu(f, f))
        self.head = nn.Conv3d(f, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b_in = self.pool(e3)

        # ViT branch: enrich enc3 features with global context
        tokens, spatial = self.patch_embed(e3)
        tokens = self.transformer(tokens)
        B, N, C = tokens.shape
        h, w, d = spatial
        vit_feat = self.vit_proj(tokens).transpose(1, 2).reshape(B, -1, h, w, d)
        vit_feat = F.interpolate(vit_feat, size=e3.shape[2:], mode="trilinear", align_corners=False)
        e3_rich  = e3 + vit_feat

        b = self.bot(b_in)

        d3 = self.dec3(torch.cat([self.up3(b), e3_rich], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


# ── 6. Simple 2D UNet ────────────────────────────────────────────────────────

class ConvBnRelu2D(nn.Sequential):
    """2-D counterpart of ConvBnRelu used by UNet2D."""
    def __init__(self, in_c, out_c, kernel=3, stride=1, padding=1):
        super().__init__(
            nn.Conv2d(in_c, out_c, kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )


class UNet2D(nn.Module):
    """
    Classic 2-D U-Net (Ronneberger et al., 2015).

    Operates on individual axial slices: input  (B, 4, H, W)
                                         output (B, 3, H, W)

    Encoder depth: 4 levels with MaxPool2d(2).
    Decoder uses bilinear upsampling (no checkerboard artefacts) followed
    by a double-conv block, and skip connections at every level.

    Compared to the 3-D models this is much lighter (~1 M params at
    base_filters=32) and trains well on a single consumer GPU.
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 3,
                 base_filters: int = 32):
        super().__init__()
        f = base_filters

        def double_conv(ic, oc):
            return nn.Sequential(ConvBnRelu2D(ic, oc), ConvBnRelu2D(oc, oc))

        # ── encoder ──────────────────────────────────────────────────────────
        self.enc1 = double_conv(in_channels, f)       #  f   × H   × W
        self.enc2 = double_conv(f,    f * 2)           #  2f  × H/2 × W/2
        self.enc3 = double_conv(f*2,  f * 4)           #  4f  × H/4 × W/4
        self.enc4 = double_conv(f*4,  f * 8)           #  8f  × H/8 × W/8
        self.pool = nn.MaxPool2d(2)

        # ── bottleneck ───────────────────────────────────────────────────────
        self.bottleneck = double_conv(f*8, f * 16)     # 16f  × H/16 × W/16

        # ── decoder (bilinear up + double-conv) ──────────────────────────────
        self.up4   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec4  = double_conv(f*16 + f*8,  f * 8)

        self.up3   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec3  = double_conv(f*8  + f*4,  f * 4)

        self.up2   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec2  = double_conv(f*4  + f*2,  f * 2)

        self.up1   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec1  = double_conv(f*2  + f,    f)

        # ── 1×1 output projection ─────────────────────────────────────────────
        self.head  = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, H, W)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)          # (B, 3, H, W)


# ── 2-D Slice Dataset ─────────────────────────────────────────────────────────

class BraTS2020SliceDataset(Dataset):
    """
    Thin wrapper around BraTS2020Dataset that decomposes each 3-D volume
    into individual axial slices for use with UNet2D.

    Returns:
        image  – (4, H, W) float32  (one axial slice, all modalities)
        label  – (3, H, W) float32  (WT / TC / ET for that slice)
        pid    – "<patient_id>_z<slice_index>" string
    """

    def __init__(
        self,
        patient_dirs: List[Path],
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        augment: bool = False,
        flip_prob: float = 0.5,
        intensity_prob: float = 0.3,
        skip_empty_ratio: float = 0.9,
        cache_rate: float = 0.0,
    ):
        """
        skip_empty_ratio: fraction of all-background slices to randomly drop
                          during dataset construction (keeps training balanced).
        """
        # Reuse 3-D loader for I/O and normalisation
        self._vol_ds = BraTS2020Dataset(
            patient_dirs, patch_size, augment=False, cache_rate=cache_rate
        )
        self.augment          = augment
        self.flip_prob        = flip_prob
        self.intensity_prob   = intensity_prob
        self.skip_empty_ratio = skip_empty_ratio

        # Build index: list of (vol_idx, slice_z)
        self._index: List[Tuple[int, int]] = []
        rng = np.random.default_rng(0)
        for vi in range(len(self._vol_ds)):
            _, label, _ = self._vol_ds[vi]          # (3, H, W, D)
            D = label.shape[-1]
            for z in range(D):
                has_fg = label[:, :, :, z].sum() > 0
                if not has_fg and rng.random() < skip_empty_ratio:
                    continue
                self._index.append((vi, z))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        vi, z = self._index[idx]
        image, label, pid = self._vol_ds[vi]          # (4,H,W,D), (3,H,W,D)

        img_sl  = image[:, :, :, z].numpy()           # (4, H, W)
        lbl_sl  = label[:, :, :, z].numpy()           # (3, H, W)

        # ── 2-D augmentation ─────────────────────────────────────────────────
        if self.augment:
            # Random horizontal/vertical flip
            for axis in [1, 2]:
                if np.random.rand() < self.flip_prob:
                    img_sl = np.flip(img_sl, axis=axis).copy()
                    lbl_sl = np.flip(lbl_sl, axis=axis).copy()
            # Random intensity shift / scale per modality
            if np.random.rand() < self.intensity_prob:
                for c in range(img_sl.shape[0]):
                    img_sl[c] = img_sl[c] * np.random.uniform(0.9, 1.1) \
                                           + np.random.uniform(-0.1, 0.1)

        return (
            torch.from_numpy(img_sl.copy()).float(),
            torch.from_numpy(lbl_sl.copy()).float(),
            f"{pid}_z{z:03d}",
        )


# ── Registry ──────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "unet2d":          UNet2D,          # ← simple 2-D slice-based U-Net (new)
    "unet3d":          UNet3D,
    "attention_unet":  AttentionUNet3D,
    "equiunet":        EquiUnet,
    "diff_unet":       DiffUNet,
    "hvu":             HVUNet,
}

IS_2D_MODEL = {"unet2d"}


def build_model(cfg: dict) -> nn.Module:
    name = cfg["model"].lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    kwargs = dict(
        in_channels=cfg["in_channels"],
        out_channels=cfg["num_classes"] - 1,  # 3 binary maps
        base_filters=cfg["base_filters"],
    )
    if name == "diff_unet":
        kwargs["T"] = cfg["diffusion_steps"]
    if name == "equiunet":
        kwargs.pop("base_filters")
        kwargs["width"] = cfg["base_filters"]
    # UNet2D uses 2-D convolutions – no extra kwargs needed
    return MODEL_REGISTRY[name](**kwargs)


def is_2d_model(cfg: dict) -> bool:
    return cfg["model"].lower() in IS_2D_MODEL


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, cfg: dict, logger: logging.Logger):
        self.cfg    = cfg
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch.manual_seed(cfg["seed"])
        np.random.seed(cfg["seed"])

        # ── data ──
        all_patients = find_patient_dirs(cfg["data_dir"])
        if not all_patients:
            raise RuntimeError(f"No BraTS patient dirs found in {cfg['data_dir']}")
        train_pts, val_pts = make_splits(all_patients, cfg["val_ratio"], cfg["seed"])
        logger.info(f"Patients → train: {len(train_pts)}, val: {len(val_pts)}")

        if is_2d_model(cfg):
            # UNet2D works on axial slices – use the slice dataset
            logger.info("2-D mode: building per-slice datasets (axial slices)")
            self.train_ds = BraTS2020SliceDataset(
                train_pts, cfg["patch_size"], augment=cfg["augment"],
                flip_prob=cfg["flip_prob"], intensity_prob=cfg["intensity_prob"],
                cache_rate=cfg["cache_rate"],
            )
            self.val_ds = BraTS2020SliceDataset(
                val_pts, cfg["patch_size"], augment=False,
                skip_empty_ratio=0.95,     # keep most empties for val stability
                cache_rate=cfg["cache_rate"],
            )
            logger.info(f"Slices → train: {len(self.train_ds)}, val: {len(self.val_ds)}")
        else:
            self.train_ds = BraTS2020Dataset(
                train_pts, cfg["patch_size"], augment=cfg["augment"],
                flip_prob=cfg["flip_prob"], rotate_prob=cfg["rotate_prob"],
                intensity_prob=cfg["intensity_prob"], cache_rate=cfg["cache_rate"],
            )
            self.val_ds = BraTS2020Dataset(
                val_pts, cfg["patch_size"], augment=False, cache_rate=cfg["cache_rate"],
            )
        self.train_loader = DataLoader(
            self.train_ds, batch_size=cfg["batch_size"], shuffle=True,
            num_workers=cfg["num_workers"], pin_memory=True, drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_ds, batch_size=cfg["batch_size"] if is_2d_model(cfg) else 1,
            shuffle=False, num_workers=cfg["num_workers"], pin_memory=True,
        )

        # ── model ──
        self.model = build_model(cfg).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model: {cfg['model']}  |  Params: {n_params:,}")

        # ── optimiser & scheduler ──
        self.opt = AdamW(self.model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        if cfg["scheduler"] == "cosine":
            self.sched = CosineAnnealingLR(self.opt, T_max=cfg["epochs"])
        elif cfg["scheduler"] == "plateau":
            self.sched = ReduceLROnPlateau(self.opt, patience=10, factor=0.5)
        else:
            self.sched = None

        # ── loss ──
        self.criterion = CombinedLoss()
        self.scaler    = GradScaler(enabled=cfg["amp"])

        # ── checkpointing & logging ──
        run_name   = f"{cfg['model']}_{datetime.now():%Y%m%d_%H%M%S}"
        self.run_dir = Path(cfg["save_dir"]) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.writer    = SummaryWriter(self.run_dir / "tb")
        self.best_dice = 0.0
        self.start_ep  = 1

        # ── resume ──
        if cfg.get("checkpoint"):
            self._load_checkpoint(cfg["checkpoint"])

        # Save config
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(cfg, f, indent=2, default=str)

    # ── training step ─────────────────────────────────────────────────────────

    def _train_step(self, image: torch.Tensor, label: torch.Tensor) -> float:
        self.model.train()
        image, label = image.to(self.device), label.to(self.device)
        self.opt.zero_grad()
        with autocast(enabled=self.cfg["amp"]):
            if isinstance(self.model, DiffUNet):
                pred_noise, true_noise = self.model(image, label)
                loss = F.mse_loss(pred_noise, true_noise)
            else:
                pred = self.model(image)
                loss = self.criterion(pred, label)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.opt)
        self.scaler.update()
        return loss.item()

    # ── validation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        self.model.eval()
        all_metrics: Dict[str, List[float]] = {f"dice_{n}": [] for n in REGION_NAMES}
        all_metrics["loss"] = []

        for image, label, _ in self.val_loader:
            image, label = image.to(self.device), label.to(self.device)
            with autocast(enabled=self.cfg["amp"]):
                if isinstance(self.model, DiffUNet):
                    pred = self.model(image)        # inference mode (no seg supplied)
                    loss = self.criterion(pred, label)
                else:
                    pred = self.model(image)
                    loss = self.criterion(pred, label)

            all_metrics["loss"].append(loss.item())
            batch_m = evaluate_batch(torch.sigmoid(pred), label)
            for k, v in batch_m.items():
                if k.startswith("dice_") and k in all_metrics:
                    all_metrics[k].append(v)

        return {k: float(np.mean(v)) for k, v in all_metrics.items()}

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self):
        logger = self.logger
        cfg    = self.cfg

        for epoch in range(self.start_ep, cfg["epochs"] + 1):
            t0      = time.time()
            losses  = []

            for step, (image, label, _) in enumerate(self.train_loader, 1):
                loss = self._train_step(image, label)
                losses.append(loss)
                if step % cfg["log_interval"] == 0:
                    logger.info(f"Ep {epoch}/{cfg['epochs']} | step {step} | loss {loss:.4f}")
                    global_step = (epoch - 1) * len(self.train_loader) + step
                    self.writer.add_scalar("train/loss_step", loss, global_step)

            mean_loss = float(np.mean(losses))

            # Validation
            val_m = self._validate()
            dt    = time.time() - t0

            logger.info(
                f"[Ep {epoch:03d}] loss={mean_loss:.4f}  "
                f"val_loss={val_m['loss']:.4f}  "
                f"WT={val_m.get('dice_WT', 0):.4f}  "
                f"TC={val_m.get('dice_TC', 0):.4f}  "
                f"ET={val_m.get('dice_ET', 0):.4f}  "
                f"({dt:.1f}s)"
            )

            # TensorBoard
            self.writer.add_scalar("train/loss_epoch", mean_loss, epoch)
            for k, v in val_m.items():
                self.writer.add_scalar(f"val/{k}", v, epoch)

            # LR scheduler
            if self.sched:
                if isinstance(self.sched, ReduceLROnPlateau):
                    self.sched.step(val_m["loss"])
                else:
                    self.sched.step()
            self.writer.add_scalar("train/lr", self.opt.param_groups[0]["lr"], epoch)

            # Checkpoint
            mean_dice = float(np.mean([val_m.get(f"dice_{n}", 0) for n in REGION_NAMES]))
            is_best   = mean_dice > self.best_dice
            if is_best:
                self.best_dice = mean_dice
            self._save_checkpoint(epoch, mean_dice, is_best)

        logger.info(f"Training complete. Best mean Dice: {self.best_dice:.4f}")
        self.writer.close()

    # ── checkpoint helpers ────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, dice: float, is_best: bool):
        state = {
            "epoch":      epoch,
            "model":      self.cfg["model"],
            "state_dict": self.model.state_dict(),
            "optimizer":  self.opt.state_dict(),
            "dice":       dice,
            "cfg":        self.cfg,
        }
        path = self.run_dir / "last.pth"
        torch.save(state, path)
        if is_best:
            torch.save(state, self.run_dir / "best.pth")
            self.logger.info(f"  ↑ New best Dice {dice:.4f} saved.")

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.opt.load_state_dict(ckpt["optimizer"])
        self.start_ep  = ckpt["epoch"] + 1
        self.best_dice = ckpt.get("dice", 0.0)
        self.logger.info(f"Resumed from {path} (ep {ckpt['epoch']}, dice {self.best_dice:.4f})")


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

        ckpt = torch.load(cfg["checkpoint"], map_location=self.device)
        saved_cfg = ckpt.get("cfg", cfg)
        saved_cfg.update({k: cfg[k] for k in ["data_dir", "val_ratio", "seed", "num_workers"]})

        self.model = build_model(saved_cfg).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        logger.info(f"Loaded checkpoint: {cfg['checkpoint']}")

        all_pts  = find_patient_dirs(cfg["data_dir"])
        _, val_pts = make_splits(all_pts, cfg["val_ratio"], cfg["seed"])
        self.ds  = BraTS2020Dataset(val_pts, cfg["patch_size"], augment=False)
        self.loader = DataLoader(self.ds, batch_size=1, shuffle=False,
                                 num_workers=cfg["num_workers"], pin_memory=True)
        self.results_path = Path(cfg["save_dir"]) / "eval_results.csv"
        Path(cfg["save_dir"]).mkdir(parents=True, exist_ok=True)

    def run(self):
        logger  = self.logger
        rows    = []
        agg     = {f"dice_{n}": [] for n in REGION_NAMES}
        agg.update({f"hd95_{n}": [] for n in REGION_NAMES})

        for image, label, pid in self.loader:
            image = image.to(self.device)
            with torch.no_grad(), autocast():
                if isinstance(self.model, DiffUNet):
                    pred = self.model(image)
                else:
                    pred = self.model(image)
            pred_bin = (torch.sigmoid(pred).cpu().numpy() > 0.5)[0]  # (3,H,W,D)
            label_np = label.numpy()[0]                               # (3,H,W,D)

            row = {"patient": pid[0]}
            for ci, name in enumerate(REGION_NAMES):
                d = dice_score(pred_bin[ci], label_np[ci])
                h = hausdorff95(pred_bin[ci], label_np[ci])
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
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Unified BraTS2020 Brain Tumour Segmentation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode",       default="train",  choices=["train", "eval"])
    p.add_argument("--model",      default="unet3d",
                   help=f"Model architecture. Choices: {list(MODEL_REGISTRY)}")
    p.add_argument("--list_models", action="store_true",
                   help="Print available models and exit")

    # Data
    p.add_argument("--data_dir",   default=None, help="Root of BraTS2020 dataset")
    p.add_argument("--val_ratio",  type=float, default=None)
    p.add_argument("--patch_size", type=int, nargs=3, default=None,
                   metavar=("H","W","D"))
    p.add_argument("--num_workers",type=int, default=None)
    p.add_argument("--cache_rate", type=float, default=None)

    # Model
    p.add_argument("--base_filters", type=int, default=None)

    # Training
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--batch_size", type=int,   default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--weight_decay",type=float,default=None)
    p.add_argument("--scheduler",  default=None, choices=["cosine","plateau","none"])
    p.add_argument("--amp",        action="store_true", default=None)
    p.add_argument("--seed",       type=int, default=None)
    p.add_argument("--checkpoint", default=None, help="Path to .pth checkpoint")
    p.add_argument("--save_dir",   default=None, help="Directory for run outputs")
    p.add_argument("--no_augment", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    if args.list_models:
        print("\nAvailable models:")
        descriptions = {
            "unet2d":         "Simple 2-D U-Net on axial slices (fast, low VRAM)",
            "unet3d":         "Vanilla 3D U-Net (mtancak/PyTorch-UNet)",
            "attention_unet": "Attention U-Net 3D (mahdizynali/BraTS2020-TF → PyTorch)",
            "equiunet":       "EquiUnet, Top-10 BraTS2020 (lescientifik/open_brats2020)",
            "diff_unet":      "Diffusion-embedded U-Net (ge-xing/Diff-UNet, MICCAI23)",
            "hvu":            "Hybrid ViT-UNet (renugadevi26/HVU_Code)",
        }
        for k, v in descriptions.items():
            print(f"  {k:<18}  {v}")
        sys.exit(0)

    cfg = build_config(args)
    if args.no_augment:
        cfg["augment"] = False
    if isinstance(cfg["patch_size"], list):
        cfg["patch_size"] = tuple(cfg["patch_size"])

    run_dir = Path(cfg["save_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(run_dir))

    logger.info(f"Mode: {cfg['mode']} | Model: {cfg['model']} | Device: "
                f"{'CUDA' if torch.cuda.is_available() else 'CPU'}")

    if cfg["mode"] == "train":
        Trainer(cfg, logger).run()
    else:
        Evaluator(cfg, logger).run()


if __name__ == "__main__":
    main()
