"""
FRED-Fusion Detector — Detector A in HDRA-Fusion.

Wraps the trained FRED-Fusion (CLIP + Domain Adversarial + XGBoost) pipeline.
Loads all artifacts from a single LOSO fold directory and exposes a
unified .predict(paths) interface.

Inference pipeline per image:
    1. Extract tabular forensic features (SRM, PRNU, frequency)
    2. Apply tab_selector → tab_scaler → scaled tabular features
    3. Extract fine-tuned CLIP embedding (512-dim)
    4. Concatenate [scaled_tab | clip_emb]
    5. Average XGBoost predictions across 5 seeds → p(fake)

Required artifacts in FRED_MODEL_DIR:
    clip_da_LOSO_<fold>.pt           — Fine-tuned CLIP backbone + projection head
    xgb_fusion_clip_1-5.json         — XGBoost ensemble (5 seeds)
    tab_selector_LOSO_<fold>.pkl     — SelectPercentile feature mask
    tab_scaler_LOSO_<fold>.pkl       — StandardScaler for tabular features

Dependencies:
    xzy2.py must be present in the same directory as this file or importable
    from sys.path. It provides DomainAdversarialCLIP and TabularExtractor.

References
----------
[1] Ganin et al. (2016). Domain-Adversarial Training of Neural Networks. JMLR.
[2] Fridrich & Kodovský (2012). Rich Models for Steganalysis. IEEE TIFS.
[3] Radford et al. (2021). Learning Transferable Visual Models. ICML.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import xgboost as xgb
import joblib
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import clip
    _HAS_CLIP = True
except ImportError:
    _HAS_CLIP = False

from configs.config import DEVICE, FRED_FOLD, FRED_MODEL_DIR
from models.clip_extractor import SimpleImageDataset


def _import_xzy2():
    """
    Import core classes from xzy2.py.

    xzy2.py must be placed in the project root or be importable from sys.path.
    It provides DomainAdversarialCLIP and TabularExtractor used by FRED-Fusion.
    """
    if "xzy2" in sys.modules:
        return sys.modules["xzy2"]

    import importlib.util

    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "xzy2.py"),
        os.path.join(os.path.dirname(__file__), "..", "..", "xzy2.py"),
        "xzy2.py",
    ]
    for cand in candidates:
        cand = os.path.abspath(cand)
        if os.path.exists(cand):
            spec = importlib.util.spec_from_file_location("xzy2", cand)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            sys.modules["xzy2"] = mod
            return mod

    raise ImportError(
        "Cannot locate xzy2.py. Place it in the project root directory "
        "or add its location to sys.path before importing FREDFusionDetector."
    )


class FREDFusionDetector:
    """
    Wrapper around the trained FRED-Fusion pipeline for use in HDRA-Fusion.

    Parameters
    ----------
    model_dir : path to the LOSO fold directory containing model artifacts
    fold_name : fold name used in artifact filenames (e.g. 'RVF10K')
    """

    def __init__(
        self,
        model_dir: str = FRED_MODEL_DIR,
        fold_name: str = FRED_FOLD,
    ):
        self.model_dir     = model_dir
        self.fold_name     = fold_name
        self.clip_model    = None
        self.clip_preproc  = None
        self.tab_extractor = None
        self.tab_selector  = None
        self.tab_scaler    = None
        self.xgb_models    = []
        self._loaded       = False
        self._device       = DEVICE

    def load(self, device: str = DEVICE) -> None:
        """
        Load all FRED-Fusion artifacts into memory.

        Called automatically on first .predict() call if not already loaded.
        """
        print(f"[FRED] Loading artifacts from {self.model_dir}")
        xzy2 = _import_xzy2()

        # ── CLIP model ─────────────────────────────────────────────────────────
        ckpt_path = os.path.join(
            self.model_dir, f"clip_da_LOSO_{self.fold_name}.pt"
        )
        if not os.path.exists(ckpt_path):
            pts = list(Path(self.model_dir).glob("*.pt"))
            if not pts:
                raise FileNotFoundError(
                    f"No .pt checkpoint found in {self.model_dir}"
                )
            ckpt_path = str(pts[0])
        print(f"[FRED] Checkpoint: {ckpt_path}")

        assert _HAS_CLIP, (
            "CLIP not installed. "
            "Run: pip install git+https://github.com/openai/CLIP.git"
        )

        base_clip, self.clip_preproc = clip.load(
            xzy2.CLIP_MODEL_NAME, device="cpu"
        )
        base_clip = base_clip.float()

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        num_domains = ckpt["num_domains"]
        feat_dim    = ckpt.get("feat_dim", xzy2.CLIP_FEAT_DIM)

        self.clip_model = xzy2.DomainAdversarialCLIP(
            clip_model=base_clip,
            num_domains=num_domains,
            feat_dim=feat_dim,
            lambda_=0.0,
            freeze_clip=True,
        ).to(device).float()

        self.clip_model.clip_model.load_state_dict(ckpt["clip_model"])
        self.clip_model.feature_projector.load_state_dict(ckpt["feature_projector"])
        self.clip_model.classifier.load_state_dict(ckpt["classifier"])
        self.clip_model.domain_classifier.load_state_dict(ckpt["domain_classifier"])
        self.clip_model.eval()
        print(f"[FRED] CLIP model loaded  (num_domains={num_domains})")

        # ── Tabular selector and scaler ────────────────────────────────────────
        sel_path = os.path.join(
            self.model_dir, f"tab_selector_LOSO_{self.fold_name}.pkl"
        )
        scl_path = os.path.join(
            self.model_dir, f"tab_scaler_LOSO_{self.fold_name}.pkl"
        )

        if not os.path.exists(sel_path):
            sels = list(Path(self.model_dir).glob("tab_selector_*.pkl"))
            sel_path = str(sels[0]) if sels else None
        if not os.path.exists(scl_path):
            scls = list(Path(self.model_dir).glob("tab_scaler_*.pkl"))
            scl_path = str(scls[0]) if scls else None

        self.tab_selector = joblib.load(sel_path)
        self.tab_scaler   = joblib.load(scl_path)
        print("[FRED] Selector & scaler loaded")

        # ── XGBoost ensemble (5 seeds) ─────────────────────────────────────────
        self.xgb_models = []
        for i in range(1, 6):
            p = os.path.join(self.model_dir, f"xgb_fusion_clip_{i}.json")
            if os.path.exists(p):
                bst = xgb.Booster()
                bst.load_model(p)
                self.xgb_models.append(bst)
        if not self.xgb_models:
            raise FileNotFoundError(
                f"No XGBoost models (xgb_fusion_clip_*.json) found in {self.model_dir}"
            )
        print(f"[FRED] XGBoost ensemble loaded ({len(self.xgb_models)} models)")

        # ── Tabular extractor ──────────────────────────────────────────────────
        self.tab_extractor = xzy2.TabularExtractor(
            img_size=(xzy2.IMG_SIZE[0], xzy2.IMG_SIZE[1]),
            parity_quality=xzy2.PARITY_QUALITY,
            srm_order=xzy2.SRM_ORDER,
            srm_t=xzy2.SRM_T,
        )

        self._device = device
        self._xzy2   = xzy2
        self._loaded = True

    @torch.no_grad()
    def predict(self, paths: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Run FRED-Fusion inference on a list of image paths.

        Parameters
        ----------
        paths      : list of image file paths
        batch_size : CLIP embedding extraction batch size

        Returns
        -------
        np.ndarray, shape (N,) — p(fake) probabilities.
        Failed images (load error) are returned as np.nan.
        """
        if not self._loaded:
            self.load()

        xzy2 = self._xzy2

        # ── 1. Tabular forensic features ───────────────────────────────────────
        tab_rows  = []
        valid_idx = []
        for i, p in enumerate(tqdm(paths, desc="[FRED] Tabular")):
            result = self.tab_extractor.extract_all(
                p, jitter=False, include_srm=True
            )
            if result is not None:
                tab_rows.append(result[0])
                valid_idx.append(i)

        feat_cols = list(tab_rows[0].keys())
        X_tab     = np.array(
            [[r.get(c, 0.0) for c in feat_cols] for r in tab_rows],
            dtype=np.float32,
        )
        X_tab_sel = self.tab_selector.transform(X_tab)
        X_tab_scl = self.tab_scaler.transform(X_tab_sel)

        # ── 2. Fine-tuned CLIP embeddings ──────────────────────────────────────
        valid_paths = [paths[i] for i in valid_idx]
        ds = SimpleImageDataset(valid_paths, self.clip_preproc)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

        all_embs = []
        for imgs, _ in tqdm(dl, desc="[FRED] CLIP emb"):
            imgs = imgs.to(self._device)
            embs = self.clip_model.get_features(imgs).float().cpu().numpy()
            all_embs.append(embs)
        E = np.concatenate(all_embs, axis=0)

        # ── 3. Feature fusion and XGBoost ensemble ─────────────────────────────
        X_fused   = np.concatenate([X_tab_scl, E], axis=1).astype(np.float32)
        dmat      = xgb.DMatrix(X_fused)
        preds_all = np.stack(
            [bst.predict(dmat) for bst in self.xgb_models], axis=0
        )
        p_fake = preds_all.mean(axis=0)

        # Assign NaN for any images that failed tabular extraction
        out = np.full(len(paths), np.nan, dtype=np.float32)
        for j, orig_idx in enumerate(valid_idx):
            out[orig_idx] = p_fake[j]
        return out
