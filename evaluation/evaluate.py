"""
HDRA-Fusion end-to-end evaluation pipeline.

Orchestrates the full hybrid evaluation:
    1. Extract raw CLIP embeddings and route each image via the router.
    2. Run FRED-Fusion on GAN-routed images.
    3. Run SBI on face-swap-routed images.
    4. Compute per-component and end-to-end metrics.
    5. Return a HybridResult containing all predictions and metrics.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from configs.config import CACHE_DIR, DEVICE
from evaluation.metrics import (
    choose_threshold_balanced,
    full_metrics,
)
from models.clip_extractor import extract_clip_embeddings_raw
from models.fred_detector import FREDFusionDetector
from models.router import ManipulationTypeRouter
from models.sbi_detector import SBIDetector


# ── Data containers ────────────────────────────────────────────────────────────

@dataclass
class RoutedPrediction:
    """Per-image routing outcome and detection result."""
    path:       str
    label:      int     # ground-truth: 0=real, 1=fake
    manip_type: int     # ground-truth manipulation type: 0=GAN, 1=face-swap, -1=unknown
    source:     str     # dataset source name (e.g. "HumanFaces", "RVF10K_train")
    p_faceswap: float   # router confidence P(face_swap | image)
    routing:    str     # "fred_fusion" | "sbi" | "abstain"
    p_fake:     float   # final detection probability from assigned detector
    detector:   str     # which detector produced p_fake


@dataclass
class HybridResult:
    """Aggregated results for a full HDRA-Fusion evaluation run."""
    predictions:    List[RoutedPrediction] = field(default_factory=list)
    router_metrics: Dict[str, Any]         = field(default_factory=dict)
    fred_metrics:   Dict[str, Any]         = field(default_factory=dict)
    sbi_metrics:    Dict[str, Any]         = field(default_factory=dict)
    hybrid_metrics: Dict[str, Any]         = field(default_factory=dict)
    routing_counts: Dict[str, int]         = field(default_factory=dict)
    abstain_rate:   float                  = 0.0


# ── Main evaluator ─────────────────────────────────────────────────────────────

class HybridDetector:
    """
    Unified detection system combining FRED-Fusion, SBI, and the router.

    Decision pipeline per image:
        1. Extract raw CLIP-512 features (frozen backbone).
        2. Router predicts P(face_swap).
        3. Route to FRED-Fusion, SBI, or ABSTAIN based on confidence gate.
        4. Assigned detector returns P(fake).

    Abstained images are excluded from per-detector AUC calculations and
    reported separately via the abstention_rate metric.

    Parameters
    ----------
    router        : trained ManipulationTypeRouter
    fred_detector : loaded FREDFusionDetector
    sbi_detector  : loaded SBIDetector
    device        : torch device string
    """

    def __init__(
        self,
        router:        ManipulationTypeRouter,
        fred_detector: FREDFusionDetector,
        sbi_detector:  SBIDetector,
        device:        str = DEVICE,
    ):
        self.router = router
        self.fred   = fred_detector
        self.sbi    = sbi_detector
        self.device = device

    def evaluate(
        self,
        paths:       List[str],
        labels:      np.ndarray,
        manip_types: np.ndarray,
        sources:     Optional[List[str]] = None,
        logger:      Optional[logging.Logger] = None,
        cache_dir:   str = CACHE_DIR,
    ) -> HybridResult:
        """
        Run the full hybrid evaluation on a labelled image set.

        Parameters
        ----------
        paths       : image file paths
        labels      : ground-truth real/fake labels (0/1)
        manip_types : ground-truth manipulation types (0=GAN, 1=face-swap, -1=unknown)
        sources     : dataset source name per image for per-source heatmap
        logger      : optional logger for progress output
        cache_dir   : directory for CLIP embedding cache

        Returns
        -------
        HybridResult with all per-image predictions and aggregate metrics
        """
        log = logger.info if logger else print
        N   = len(paths)
        log(f"[Hybrid] Evaluating {N:,} images …")

        # ── 1. Extract CLIP embeddings and route ───────────────────────────────
        emb_cache = os.path.join(cache_dir, "router_eval_embeddings.npy")
        os.makedirs(cache_dir, exist_ok=True)
        X = extract_clip_embeddings_raw(paths, emb_cache, device=self.device)
        decisions, p_faceswap = self.router.route(X)

        mask_fred    = decisions == ManipulationTypeRouter.ROUTE_GAN
        mask_sbi     = decisions == ManipulationTypeRouter.ROUTE_FACESWAP
        mask_abstain = decisions == ManipulationTypeRouter.ROUTE_ABSTAIN

        routing_counts = {
            "fred_fusion": int(mask_fred.sum()),
            "sbi":         int(mask_sbi.sum()),
            "abstain":     int(mask_abstain.sum()),
        }
        abstain_rate = float(mask_abstain.mean())
        log(
            f"[Hybrid] Routing → FRED={routing_counts['fred_fusion']:,}  "
            f"SBI={routing_counts['sbi']:,}  "
            f"ABSTAIN={routing_counts['abstain']:,} ({abstain_rate * 100:.1f}%)"
        )

        # ── 2. FRED-Fusion predictions ─────────────────────────────────────────
        p_fake_all   = np.full(N, np.nan, dtype=np.float32)
        detector_tag = np.full(N, "", dtype=object)

        if mask_fred.sum() > 0:
            fred_paths = [paths[i] for i in np.where(mask_fred)[0]]
            p_fred     = self.fred.predict(fred_paths)
            for j, i in enumerate(np.where(mask_fred)[0]):
                p_fake_all[i]   = p_fred[j]
                detector_tag[i] = "fred_fusion"

        # ── 3. SBI predictions ─────────────────────────────────────────────────
        if mask_sbi.sum() > 0:
            sbi_paths = [paths[i] for i in np.where(mask_sbi)[0]]
            p_sbi     = self.sbi.predict(sbi_paths)
            for j, i in enumerate(np.where(mask_sbi)[0]):
                p_fake_all[i]   = p_sbi[j]
                detector_tag[i] = "sbi"

        for i in np.where(mask_abstain)[0]:
            detector_tag[i] = "abstain"

        # ── 4. Assemble per-image predictions ──────────────────────────────────
        src_list = sources if sources is not None else ["unknown"] * N
        preds = [
            RoutedPrediction(
                path       = paths[i],
                label      = int(labels[i]),
                manip_type = int(manip_types[i]) if manip_types is not None else -1,
                source     = str(src_list[i]),
                p_faceswap = float(p_faceswap[i]),
                routing    = str(decisions[i]),
                p_fake     = float(p_fake_all[i]) if not np.isnan(p_fake_all[i]) else -1.0,
                detector   = str(detector_tag[i]),
            )
            for i in range(N)
        ]

        # ── 5. Compute metrics ─────────────────────────────────────────────────
        result = HybridResult(
            predictions   = preds,
            routing_counts= routing_counts,
            abstain_rate  = abstain_rate,
        )

        # Router — manipulation-type classification
        known_manip = manip_types != -1
        if known_manip.sum() > 0:
            mt_true = manip_types[known_manip]
            mt_prob = p_faceswap[known_manip]
            mt_thr  = choose_threshold_balanced(mt_true, mt_prob)
            result.router_metrics = full_metrics(mt_true, mt_prob, threshold=mt_thr, tag="router")
            result.router_metrics["optimal_threshold"] = mt_thr
            log(f"[Router] AUC={result.router_metrics['roc_auc']:.4f}  "
                f"Acc={result.router_metrics['accuracy']:.4f}")

        # FRED-Fusion — GAN-routed subset
        fred_eval = mask_fred & ~np.isnan(p_fake_all)
        if fred_eval.sum() > 10:
            thr = choose_threshold_balanced(labels[fred_eval], p_fake_all[fred_eval])
            result.fred_metrics = full_metrics(
                labels[fred_eval], p_fake_all[fred_eval], threshold=thr, tag="fred_fusion"
            )
            log(f"[FRED]   AUC={result.fred_metrics['roc_auc']:.4f}  n={fred_eval.sum():,}")

        # SBI — face-swap-routed subset
        sbi_eval = mask_sbi & ~np.isnan(p_fake_all)
        if sbi_eval.sum() > 10:
            thr = choose_threshold_balanced(labels[sbi_eval], p_fake_all[sbi_eval])
            result.sbi_metrics = full_metrics(
                labels[sbi_eval], p_fake_all[sbi_eval], threshold=thr, tag="sbi"
            )
            log(f"[SBI]    AUC={result.sbi_metrics['roc_auc']:.4f}  n={sbi_eval.sum():,}")

        # End-to-end hybrid (excluding abstained images)
        routed = (
            ~np.isnan(p_fake_all)
            & (decisions != ManipulationTypeRouter.ROUTE_ABSTAIN)
        )
        if routed.sum() > 10:
            thr = choose_threshold_balanced(labels[routed], p_fake_all[routed])
            result.hybrid_metrics = full_metrics(
                labels[routed], p_fake_all[routed], threshold=thr, tag="hybrid_e2e"
            )
            result.hybrid_metrics["abstain_rate"] = abstain_rate
            log(
                f"[Hybrid] E2E AUC={result.hybrid_metrics['roc_auc']:.4f}  "
                f"n={routed.sum():,}  abstain_rate={abstain_rate:.3f}"
            )

        return result

    def predict_single(self, image_path: str) -> Dict[str, Any]:
        """
        Run inference on a single image and return a structured result dict.

        Parameters
        ----------
        image_path : path to the input face image

        Returns
        -------
        dict with keys: image, p_faceswap, routing, detector, p_fake, verdict
        """
        try:
            import clip as _clip
        except ImportError:
            raise ImportError(
                "CLIP not installed. "
                "Run: pip install git+https://github.com/openai/CLIP.git"
            )

        from PIL import Image as _Image
        from configs.config import CLIP_MODEL_NAME

        model, preproc = _clip.load(CLIP_MODEL_NAME, device=self.device)
        model.eval()

        img  = _Image.open(image_path).convert("RGB")
        x    = preproc(img).unsqueeze(0).to(self.device)
        import torch
        with torch.no_grad():
            feat = model.encode_image(x).float().cpu().numpy()
        feat = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8)

        decisions, p_fs = self.router.route(feat)
        routing    = decisions[0]
        p_faceswap = float(p_fs[0])

        if routing == ManipulationTypeRouter.ROUTE_GAN:
            p_fake   = float(self.fred.predict([image_path])[0])
            detector = "fred_fusion"
        elif routing == ManipulationTypeRouter.ROUTE_FACESWAP:
            p_fake   = float(self.sbi.predict([image_path])[0])
            detector = "sbi"
        else:
            p_fake   = None
            detector = "abstain"

        return {
            "image":      image_path,
            "p_faceswap": p_faceswap,
            "routing":    routing,
            "detector":   detector,
            "p_fake":     p_fake,
            "verdict":    (
                "FAKE"    if (p_fake is not None and p_fake >= 0.5) else
                "REAL"    if p_fake is not None else
                "ABSTAIN"
            ),
        }
