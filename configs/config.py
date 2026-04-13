"""
Edit the dataset paths below before running. All other hyperparameters
are pre-set to the values used in the thesis experiments.
================================================================================
"""

import os
import random
import torch
import numpy as np
from datetime import datetime
from typing import Any, Dict, List

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Run metadata ───────────────────────────────────────────────────────────────
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
# DATASET PATHS — SET THESE BEFORE RUNNING
# ==============================================================================
# Each entry is a dict with keys:
#   name        : human-readable label used in plots and reports
#   real_path   : directory containing real face images
#   fake_path   : directory containing fake/manipulated face images
#   max_samples : int cap per split, or None for full dataset
#
# Supported image formats: .jpg .jpeg .png .bmp .tiff .webp
# ==============================================================================

# ── Router training: GAN side (140K train split only — no test/valid leakage) ─
GAN_TRAIN_DATASETS: List[Dict[str, Any]] = [
    {
        "name":        "140K_train",
        "real_path":   "",   # e.g. /data/140k/train/real
        "fake_path":   "",   # e.g. /data/140k/train/fake
        "max_samples": None,
    },
    {
        "name":        "140K_valid",
        "real_path":   "",   # e.g. /data/140k/valid/real
        "fake_path":   "",   # e.g. /data/140k/valid/fake
        "max_samples": None,
    },
]

# ── Router training: face-swap side (OpenForensics FaceCrops Train + Val) ──────
FACESWAP_TRAIN_DATASETS: List[Dict[str, Any]] = [
    {
        "name":        "OpenForensics_Train",
        "real_path":   "",   # e.g. /data/openforensics/FaceCrops/Train/Real
        "fake_path":   "",   # e.g. /data/openforensics/FaceCrops/Train/Fake
        "max_samples": None,
    },
    {
        "name":        "OpenForensics_Val",
        "real_path":   "",   # e.g. /data/openforensics/FaceCrops/Val/Real
        "fake_path":   "",   # e.g. /data/openforensics/FaceCrops/Val/Fake
        "max_samples": None,
    },
]

# ── Evaluation: GAN side (never seen by router during training) ────────────────
GAN_EVAL_DATASETS: List[Dict[str, Any]] = [
    {
        "name":        "HumanFaces",
        "real_path":   "",   # e.g. /data/humanfaces/real
        "fake_path":   "",   # e.g. /data/humanfaces/fake
        "max_samples": None,
    },
    {
        "name":        "RVF10K_train",
        "real_path":   "",   # e.g. /data/rvf10k/train/real
        "fake_path":   "",   # e.g. /data/rvf10k/train/fake
        "max_samples": None,
    },
    {
        "name":        "RVF10K_valid",
        "real_path":   "",   # e.g. /data/rvf10k/valid/real
        "fake_path":   "",   # e.g. /data/rvf10k/valid/fake
        "max_samples": None,
    },
]

# ── Evaluation: face-swap side (OpenForensics Test only) ──────────────────────
FACESWAP_EVAL_DATASETS: List[Dict[str, Any]] = [
    {
        "name":        "OpenForensics_Test",
        "real_path":   "",   # e.g. /data/openforensics/FaceCrops/Test/Real
        "fake_path":   "",   # e.g. /data/openforensics/FaceCrops/Test/Fake
        "max_samples": None,
    },
]

# ── External validation dataset (optional — SG3-CW14K) ────────────────────────
# Leave empty strings to skip external validation.
EXTERNAL_EVAL_DATASETS: List[Dict[str, Any]] = [
    {
        "name":        "SG3-CW14K",
        "real_path":   "",   # e.g. /data/sg3cw14k/real
        "fake_path":   "",   # e.g. /data/sg3cw14k/fake
        "max_samples": None,
    },
]

# ==============================================================================
# MODEL ARTIFACT PATHS — SET THESE BEFORE RUNNING
# ==============================================================================

# FRED-Fusion: path to the LOSO fold model directory containing:
#   clip_da_LOSO_<FRED_FOLD>.pt
#   xgb_fusion_clip_1-5.json
#   tab_selector_LOSO_<FRED_FOLD>.pkl
#   tab_scaler_LOSO_<FRED_FOLD>.pkl
FRED_MODEL_DIR = ""    # e.g. /models/fred_fusion/loso_rvf10k/
FRED_FOLD      = "RVF10K"

# SBI: path to the SelfBlendedImages repository root (cloned from GitHub)
# and path to the FFraw.tar checkpoint file
SBI_REPO_ROOT  = ""    # e.g. /models/SelfBlendedImages/
SBI_WEIGHTS_DIR = ""   # e.g. /models/SelfBlendedImages/weights/
SBI_CHECKPOINT_CANDIDATES = [
    os.path.join(SBI_WEIGHTS_DIR, "FFraw.tar"),
    os.path.join(SBI_WEIGHTS_DIR, "FFc23.tar"),
    os.path.join(SBI_WEIGHTS_DIR, "sbi_detector.pth"),
]

# ==============================================================================
# OUTPUT DIRECTORIES
# ==============================================================================
BASE_OUT    = ""       # e.g. /outputs/hdra_fusion/
RUN_DIR     = os.path.join(BASE_OUT, f"Run_{RUN_TIMESTAMP}") if BASE_OUT else f"Run_{RUN_TIMESTAMP}"
ROUTER_DIR  = os.path.join(RUN_DIR, "router")
REPORTS_DIR = os.path.join(RUN_DIR, "reports")
PLOTS_DIR   = os.path.join(RUN_DIR, "plots")
CACHE_DIR   = os.path.join(RUN_DIR, "cache")
SUMMARY_DIR = os.path.join(RUN_DIR, "summary")

# ==============================================================================
# ROUTER HYPERPARAMETERS
# ==============================================================================
# Confidence gate: images with P(face_swap) < theta_low  → FRED-Fusion (GAN)
#                  images with P(face_swap) > theta_high → SBI (face-swap)
#                  theta_low <= P <= theta_high           → ABSTAIN
#
# Symmetric placement around 0.5 with Δ=0.25 requires 75% confidence
# before committing to a routing decision. Validated by abstention-AUC
# trade-off analysis (thesis Section 4.2.5): AUC flat across all gate
# widths up to 2Δ=0.5, confirming current gate is optimal.
ROUTER_THETA_LOW  = 0.25
ROUTER_THETA_HIGH = 0.75
ROUTER_VAL_RATIO  = 0.20   # Fraction of training data held out for validation
ROUTER_C          = 1.0    # Logistic regression regularisation strength (default)
ROUTER_MAX_ITER   = 2000   # Maximum iterations for LBFGS solver

# ==============================================================================
# CLIP SETTINGS
# ==============================================================================
CLIP_MODEL_NAME = "ViT-B/32"
CLIP_FEAT_DIM   = 512
IMG_SIZE        = (224, 224)
CLIP_BATCH_SIZE = 64

# ==============================================================================
# SBI SETTINGS
# ==============================================================================
SBI_IMAGE_SIZE  = 380    # EfficientNet-B4 input resolution (FaceForensics++ protocol)
SBI_BATCH_SIZE  = 32

# ==============================================================================
# PLOTTING
# ==============================================================================
DPI          = 300
FIGSIZE_WIDE = (14, 5)
FIGSIZE_TALL = (7, 6)
FIGSIZE_SQ   = (6, 6)

PALETTE = {
    "fred":    "#2E86AB",   # blue   — FRED-Fusion
    "sbi":     "#A23B72",   # purple — SBI
    "router":  "#F18F01",   # amber  — Router / Hybrid
    "abstain": "#C73E1D",   # red    — Abstain zone
    "chance":  "#AAAAAA",   # grey   — Chance diagonal
    "real":    "#44BBA4",   # teal   — Real images
    "fake":    "#E94F37",   # coral  — Fake images
}
