"""
================================================================================
FRED-Fusion Core — Inference Components
================================================================================
Thesis: HDRA-Fusion: Hybrid Detection with Routed Architecture for
        Manipulation-Aware AI Face Forgery Detection
        Florida Institute of Technology

This file contains the inference-only components of FRED-Fusion (Detector A
in the HDRA-Fusion framework). All training pipeline code, dataset ingestion,
augmentation, LOSO splitting, deduplication, and calibration fitting have been
removed. This file is the only dependency required by models/fred_detector.py.

Exposed components:
    DomainAdversarialCLIP   — Fine-tuned CLIP backbone with domain adversarial
                              heads. Loaded from the .pt checkpoint at inference.
    TabularExtractor        — Extracts SRM co-occurrence, PRNU, frequency, and
                              residual forensic features from a single image.

Required constants (imported by fred_detector.py):
    CLIP_MODEL_NAME, CLIP_FEAT_DIM, IMG_SIZE, PARITY_QUALITY,
    SRM_ORDER, SRM_T, JPEG_JITTER_RANGE

Checkpoint download:
    The FRED-Fusion RVF10K fold checkpoint bundle (~560 MB) is hosted on
    GitHub Releases. Download fred_fusion_RVF10K_checkpoint.zip from:
    https://github.com/omarebeid24/HDRA-Fusion/releases/download/v1.0/fred_fusion_RVF10K_checkpoint.zip
    Unzip into your FRED_MODEL_DIR directory (set in configs/config.py).

References:
    [1] Ganin et al. (2016). Domain-Adversarial Training of Neural Networks.
        Journal of Machine Learning Research.
    [2] Fridrich & Kodovský (2012). Rich Models for Steganalysis of Digital
        Images. IEEE Transactions on Information Forensics and Security.
    [3] Radford et al. (2021). Learning Transferable Visual Models From
        Natural Language Supervision. ICML.
================================================================================
"""

from __future__ import annotations

import os
import random
import warnings
from typing import Dict, Optional, Tuple

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from skimage.filters import laplace
from torch.autograd import Function

try:
    import clip
    _HAS_CLIP = True
except ImportError:
    _HAS_CLIP = False
    print(
        "[WARNING] CLIP not installed. "
        "Run: pip install git+https://github.com/openai/CLIP.git"
    )

cv2.setNumThreads(0)
torch.set_num_threads(1)

# ==============================================================================
# CONSTANTS
# These values must match the training configuration used to produce the
# checkpoint. Do not modify unless retraining from scratch.
# ==============================================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# CLIP backbone
CLIP_MODEL_NAME   = "ViT-B/32"
CLIP_FEAT_DIM     = 512         # ViT-B/32 output dimensionality

# Image preprocessing
IMG_SIZE          = (224, 224)  # CLIP ViT-B/32 input resolution
PARITY_QUALITY    = 85          # JPEG quality for tabular feature standardisation
JPEG_JITTER_RANGE = (35, 95)    # Quality range used during training augmentation

# SRM co-occurrence feature parameters
SRM_ORDER         = 3           # Co-occurrence order
SRM_T             = 3           # Quantisation threshold

# Domain adversarial training strength (set to 0 at inference — no effect)
DOMAIN_ADVERSARIAL_LAMBDA = 0.5


# ==============================================================================
# SECTION 1: GRADIENT REVERSAL LAYER
# ==============================================================================

class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer for Domain Adversarial Training.

    Forward pass  : identity (no-op)
    Backward pass : negate and scale gradients by lambda

    At inference lambda_ is set to 0.0 so this has no effect.
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        return -ctx.lambda_ * grads, None


class GradientReversalLayer(nn.Module):
    """Wrapper module around GradientReversalFunction."""

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float):
        self.lambda_ = lambda_


# ==============================================================================
# SECTION 2: DOMAIN-ADVERSARIAL CLIP MODEL
# ==============================================================================

class DomainAdversarialCLIP(nn.Module):
    """
    CLIP ViT-B/32 backbone with domain adversarial training heads.

    Architecture:
        CLIP ViT-B/32 (frozen or fine-tuned backbone)
        → feature_projector  (512 → 512, LayerNorm + GELU)
        → classifier         (512 → 2, real/fake)
        → domain_classifier  (512 → num_domains, via GRL)

    At inference:
        - lambda_ is set to 0.0 so the gradient reversal has no effect.
        - Only get_features() and the classifier head are used.
        - The domain_classifier output is never consumed by fred_detector.py.

    Parameters
    ----------
    clip_model  : base CLIP model loaded with clip.load()
    num_domains : number of training source domains (from checkpoint metadata)
    feat_dim    : CLIP feature dimensionality (512 for ViT-B/32)
    lambda_     : gradient reversal strength (0.0 at inference)
    freeze_clip : whether to freeze the CLIP backbone parameters
    """

    def __init__(
        self,
        clip_model,
        num_domains:  int,
        feat_dim:     int   = CLIP_FEAT_DIM,
        lambda_:      float = DOMAIN_ADVERSARIAL_LAMBDA,
        freeze_clip:  bool  = False,
    ):
        super().__init__()
        self.clip_model = clip_model
        self.feat_dim   = feat_dim
        self.freeze_clip = freeze_clip

        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        # Feature projection head
        self.feature_projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Real/fake classification head
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 2),
        )

        # Domain discriminator with gradient reversal
        self.grl = GradientReversalLayer(lambda_=lambda_)
        self.domain_classifier = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_domains),
        )

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """Extract raw CLIP image features (FP32 output)."""
        return self.clip_model.encode_image(x).float()

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Tuple:
        """
        Full forward pass (used during training).

        Returns
        -------
        (class_logits, domain_logits) or
        (class_logits, domain_logits, features) if return_features=True
        """
        features      = self.encode_image(x)
        features      = self.feature_projector(features)
        class_logits  = self.classifier(features)
        domain_logits = self.domain_classifier(self.grl(features))

        if return_features:
            return class_logits, domain_logits, features
        return class_logits, domain_logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract projected features only.

        This is the method called by FREDFusionDetector during inference.
        Returns the 512-dim domain-adversarially trained embedding used
        as the CLIP component of the XGBoost fusion input.
        """
        features = self.encode_image(x)
        return self.feature_projector(features)

    def set_lambda(self, lambda_: float):
        """Update gradient reversal strength (no effect at inference)."""
        self.grl.set_lambda(lambda_)


# ==============================================================================
# SECTION 3: TABULAR FEATURE EXTRACTOR
# ==============================================================================

class TabularExtractor:
    """
    Extracts classical forensic features from a single image for use as
    the tabular component of the FRED-Fusion XGBoost ensemble input.

    Feature groups:
        Frequency domain  : energy concentration, peak intensity, entropy
        Residual          : entropy, variance, skewness, kurtosis
        GAN artifacts     : Laplacian kurtosis
        PRNU              : residual power, spectral flatness, block CV,
                            edge coupling
        SRM co-occurrence : co-occurrence statistics from 3 SRM kernels
                            (order=3, T=3 — matches training configuration)

    The combined feature vector is ~700-dimensional before SelectPercentile
    reduction. After the saved tab_selector the effective dimension is ~665.

    Parameters
    ----------
    img_size       : target image size for preprocessing (must match training)
    parity_quality : JPEG quality for feature standardisation preprocessing
    srm_order      : SRM co-occurrence order
    srm_t          : SRM quantisation threshold
    """

    def __init__(
        self,
        img_size:       Tuple[int, int] = IMG_SIZE,
        parity_quality: int             = PARITY_QUALITY,
        srm_order:      int             = SRM_ORDER,
        srm_t:          int             = SRM_T,
    ):
        self.img_size       = img_size
        self.parity_quality = parity_quality
        self.srm_order      = srm_order
        self.srm_t          = srm_t

    # ── Image preprocessing ────────────────────────────────────────────────────

    def _std_img(self, img: np.ndarray, jitter: bool = False) -> np.ndarray:
        """
        Resize to target resolution and apply JPEG standardisation.

        At inference jitter=False so a fixed quality of parity_quality is used,
        matching the deterministic preprocessing applied during evaluation.
        """
        if img.shape[:2] != self.img_size:
            img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_LINEAR)

        q = self.parity_quality
        if jitter:
            q = int(np.clip(
                np.random.randint(JPEG_JITTER_RANGE[0], JPEG_JITTER_RANGE[1]),
                35, 98,
            ))

        _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)

    def _residual(self, gray: np.ndarray) -> np.ndarray:
        """Compute normalised noise residual via dual denoising."""
        den1 = cv2.GaussianBlur(gray, (3, 3), 0)
        den2 = cv2.bilateralFilter(gray, 5, 50, 50)
        den  = (den1.astype(np.float32) + den2.astype(np.float32)) / 2.0
        res  = gray.astype(np.float32) - den
        s    = res.std()
        return (res - res.mean()) / (s + 1e-6) if s > 0 else res

    # ── Feature extraction primitives ─────────────────────────────────────────

    def prnu_stats(
        self,
        gray:  np.ndarray,
        resid: np.ndarray,
    ) -> Dict[str, float]:
        """
        Photo-Response Non-Uniformity (PRNU) feature group.

        Computes residual power, spectral flatness, block coefficient of
        variation, and edge coupling — all sensitive to GAN spectral artifacts.
        """
        rp  = float(np.mean(resid ** 2))
        rk  = float(stats.kurtosis(resid.ravel()))
        F   = np.fft.fftshift(np.fft.fft2(resid))
        mag = np.abs(F) + 1e-8
        geo = float(np.exp(np.mean(np.log(mag))))
        arith = float(np.mean(mag))
        sflat = float(geo / arith)

        H, W = resid.shape
        b    = 8
        Hc   = H - (H % b)
        Wc   = W - (W % b)
        R    = resid[:Hc, :Wc].reshape(Hc // b, b, Wc // b, b).swapaxes(1, 2)
        vmap = R.var(axis=(2, 3))
        cv_blocks = float(vmap.std() / (vmap.mean() + 1e-6))

        edges  = cv2.Canny(gray, 50, 150)
        e_mask = edges > 0
        edge_c = float(
            np.mean(np.abs(resid[e_mask]))
            / (np.mean(np.abs(resid[~e_mask])) + 1e-6)
        ) if e_mask.any() else 1.0

        return dict(
            prnu_residual_power   = rp,
            prnu_kurtosis         = rk,
            prnu_spectral_flatness= sflat,
            prnu_block_cv         = cv_blocks,
            prnu_edge_coupling    = edge_c,
        )

    def freq_feats(self, gray: np.ndarray) -> Dict[str, float]:
        """
        Frequency domain feature group.

        Energy concentration in the central quarter of the spectrum,
        peak intensity ratio, and spectral entropy — capture GAN
        high-frequency grid patterns.
        """
        f   = np.fft.fftshift(np.fft.fft2(gray))
        mag = np.abs(f)
        center = mag[
            mag.shape[0] // 4 : 3 * mag.shape[0] // 4,
            mag.shape[1] // 4 : 3 * mag.shape[1] // 4,
        ]
        energy_conc = float(np.sum(center ** 2) / (np.sum(mag ** 2) + 1e-10))
        peak        = float(np.max(mag) / (np.mean(mag) + 1e-10))
        mag_norm    = mag / (np.sum(mag) + 1e-10)
        ent         = float(-np.sum(mag_norm * np.log(mag_norm + 1e-10)))

        return dict(
            energy_concentration = energy_conc,
            peak_intensity       = peak,
            frequency_entropy    = ent,
        )

    def residual_feats(self, residual: np.ndarray) -> Dict[str, float]:
        """
        Noise residual statistical feature group.

        Entropy, variance, skewness, and kurtosis of the noise residual
        distribution — sensitive to GAN generator regularisation artifacts.
        """
        hist, _ = np.histogram(residual.ravel(), bins=256)
        hist    = hist / (hist.sum() + 1e-10)
        rent    = float(-np.sum(hist * np.log(hist + 1e-10)))

        return dict(
            residual_entropy  = rent,
            residual_variance = float(np.var(residual)),
            residual_skewness = float(stats.skew(residual.ravel())),
            residual_kurtosis = float(stats.kurtosis(residual.ravel())),
        )

    def genagn_feats(self, img: np.ndarray) -> Dict[str, float]:
        """
        GAN-specific artifact feature group.

        Laplacian kurtosis — measures sharpness of high-frequency transitions.
        GAN-generated images exhibit characteristic kurtosis shifts due to
        their convolutional upsampling artifacts.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        lapk = float(stats.kurtosis(laplace(gray).ravel()))
        return dict(lap_kurtosis=lapk)

    # ── SRM co-occurrence features ─────────────────────────────────────────────

    def _srm_bank(self):
        """Three SRM high-pass kernels sensitive to steganalysis residuals."""
        k1 = np.array([[0, 0, 0], [-1, 2, -1], [0, 0, 0]], np.float32)
        k2 = np.array([[0, -1, 0], [0, 2, 0], [0, -1, 0]], np.float32)
        k3 = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], np.float32)
        return [k1, k2, k3]

    def _quant(self, r: np.ndarray, T: int) -> np.ndarray:
        """Normalise and quantise residual to [-T, T]."""
        r = r / (np.std(r) + 1e-6)
        return np.clip(np.round(r).astype(np.int32), -T, T)

    def _cooc(self, q: np.ndarray, order: int) -> np.ndarray:
        """Compute horizontal + vertical co-occurrence histogram."""
        minv, maxv = q.min(), q.max()
        off = -minv
        B   = maxv - minv + 1
        if B <= 1:
            return np.zeros(B ** order * 2, np.float32)

        rows, cols = q.shape
        H, V = [], []

        if cols >= order:
            idx = np.arange(cols - order + 1)
            for r in range(rows):
                seq = q[r, :] + off
                for i in idx:
                    code = 0
                    for t in seq[i : i + order]:
                        code = code * B + int(t)
                    H.append(code)

        if rows >= order:
            idx = np.arange(rows - order + 1)
            for c in range(cols):
                seq = q[:, c] + off
                for i in idx:
                    code = 0
                    for t in seq[i : i + order]:
                        code = code * B + int(t)
                    V.append(code)

        size = B ** order
        h = np.bincount(np.array(H, np.int64), minlength=size).astype(np.float32)
        v = np.bincount(np.array(V, np.int64), minlength=size).astype(np.float32)
        h /= (h.sum() + 1e-6)
        v /= (v.sum() + 1e-6)
        return np.concatenate([h, v], axis=0)

    def srm_feats(
        self,
        gray:  np.ndarray,
        order: int = SRM_ORDER,
        T:     int = SRM_T,
    ) -> Dict[str, float]:
        """
        Spatial Rich Model (SRM) co-occurrence feature group.

        Applies three SRM kernels and computes co-occurrence histograms
        of the quantised residuals. Captures GAN synthesis artifacts at
        the pixel-level statistical structure level.
        """
        feats = {}
        for i, k in enumerate(self._srm_bank()):
            r  = cv2.filter2D(
                gray.astype(np.float32), -1, k,
                borderType=cv2.BORDER_REFLECT,
            )
            q  = self._quant(r, T)
            co = self._cooc(q, order)
            for j, val in enumerate(co):
                feats[f"srm{i}_cooc{order}_{j}"] = float(val)
        return feats

    # ── Public inference interface ─────────────────────────────────────────────

    def extract_all(
        self,
        img_path:    str,
        jitter:      bool = False,
        include_srm: bool = True,
    ) -> Optional[Tuple[Dict[str, float], np.ndarray, np.ndarray]]:
        """
        Extract all forensic features from a single image.

        This is the sole method called by FREDFusionDetector.predict()
        during HDRA-Fusion inference.

        Parameters
        ----------
        img_path    : path to the input image file
        jitter      : apply JPEG quality jitter (False at inference)
        include_srm : include SRM co-occurrence features (always True)

        Returns
        -------
        (feats_dict, residual, gray) on success, None if the image
        cannot be loaded.

        feats_dict contains ~700 named float features. After the saved
        tab_selector is applied in FREDFusionDetector the dimensionality
        is reduced to ~665.
        """
        img = cv2.imread(str(img_path))
        if img is None:
            return None

        img  = self._std_img(img, jitter=jitter)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        res  = self._residual(gray)

        feats = {
            **self.freq_feats(gray),
            **self.residual_feats(res),
            **self.genagn_feats(img),
            **self.prnu_stats(gray, res),
        }
        if include_srm:
            feats.update(self.srm_feats(gray, order=self.srm_order, T=self.srm_t))

        return feats, res, gray
