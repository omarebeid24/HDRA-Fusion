"""
Manipulation-Type Router for HDRA-Fusion.

A logistic regression classifier trained on raw CLIP ViT-B/32 embeddings
that predicts the manipulation domain of an input image:
    0 = GAN-synthetic  → route to FRED-Fusion
    1 = Face-swap      → route to SBI

Hard binary routing with a confidence abstention gate:
    P(face_swap) < theta_low             → GAN      (FRED-Fusion)
    P(face_swap) > theta_high            → face-swap (SBI)
    theta_low <= P(face_swap) <= theta_high → ABSTAIN

References
----------
[1] Radford et al. (2021). Learning Transferable Visual Models
    From Natural Language Supervision. ICML.
[2] Shiohara & Yamasaki (2022). Detecting Face Forgery Videos with
    Self-Blended Images. CVPR.
"""

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from configs.config import (
    ROUTER_C,
    ROUTER_MAX_ITER,
    ROUTER_THETA_HIGH,
    ROUTER_THETA_LOW,
    SEED,
)


class ManipulationTypeRouter:
    """
    Logistic regression router that classifies manipulation domain.

    Trained on L2-normalised CLIP-512 embeddings extracted without
    backbone fine-tuning, so routing relies on general visual domain
    characteristics rather than task-specific features.

    Routing thresholds are set symmetrically around 0.5 (Δ=0.25),
    requiring 75% router confidence before committing to either pathway.
    Validated by abstention-AUC trade-off analysis: AUC flat up to
    2Δ=0.5 before abstention cost rises sharply (thesis Section 4.2.5).
    """

    ROUTE_GAN      = "fred_fusion"
    ROUTE_FACESWAP = "sbi"
    ROUTE_ABSTAIN  = "abstain"

    def __init__(
        self,
        theta_low:  float = ROUTER_THETA_LOW,
        theta_high: float = ROUTER_THETA_HIGH,
        C:          float = ROUTER_C,
        max_iter:   int   = ROUTER_MAX_ITER,
    ):
        self.theta_low  = theta_low
        self.theta_high = theta_high
        self.C          = C
        self.max_iter   = max_iter
        self.model:  Optional[LogisticRegression] = None
        self.scaler: Optional[StandardScaler]     = None
        self._val_metrics: Optional[Dict]         = None

    # ── Training ───────────────────────────────────────────────────────────────

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   Optional[np.ndarray] = None,
        y_val:   Optional[np.ndarray] = None,
        logger:  Optional[logging.Logger] = None,
    ) -> Dict[str, float]:
        """
        Fit the logistic regression router.

        Parameters
        ----------
        X_train, y_train : CLIP-512 embeddings and labels (0=GAN, 1=face-swap)
        X_val,   y_val   : held-out validation set for monitoring
        logger           : optional logger for progress messages

        Returns
        -------
        dict of training and validation metrics
        """
        log = logger.info if logger else print

        log(
            f"[Router] Training  n_train={len(y_train):,}  "
            f"(GAN={int((y_train == 0).sum()):,}  "
            f"FS={int((y_train == 1).sum()):,})"
        )

        self.scaler = StandardScaler()
        X_tr = self.scaler.fit_transform(X_train)

        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver="lbfgs",
            multi_class="auto",
            random_state=SEED,
            n_jobs=-1,
        )
        self.model.fit(X_tr, y_train)

        train_acc = self.model.score(X_tr, y_train)
        log(f"[Router] Train accuracy: {train_acc:.4f}")

        if X_val is not None and y_val is not None:
            X_va  = self.scaler.transform(X_val)
            p_val = self.model.predict_proba(X_va)[:, 1]
            val_auc = roc_auc_score(y_val, p_val)
            val_acc = accuracy_score(y_val, (p_val >= 0.5).astype(int))
            self._val_metrics = {
                "val_auc":   float(val_auc),
                "val_acc":   float(val_acc),
                "train_acc": float(train_acc),
            }
            log(f"[Router] Val AUC={val_auc:.4f}  Val Acc={val_acc:.4f}")
            return self._val_metrics

        return {"train_acc": float(train_acc)}

    # ── Prediction ─────────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return P(face_swap | image) for each sample.

        Parameters
        ----------
        X : CLIP-512 embeddings, shape (N, 512)

        Returns
        -------
        np.ndarray, shape (N,) — probability of face-swap class
        """
        assert self.model is not None, "Router not trained. Call .train() first."
        X_s = self.scaler.transform(X)
        return self.model.predict_proba(X_s)[:, 1]

    def route(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply confidence gate and return routing decisions.

        Parameters
        ----------
        X : CLIP-512 embeddings, shape (N, 512)

        Returns
        -------
        decisions : np.ndarray of str — ROUTE_GAN / ROUTE_FACESWAP / ROUTE_ABSTAIN
        probas    : np.ndarray of float — P(face_swap) for each image
        """
        p = self.predict_proba(X)
        decisions = np.where(
            p < self.theta_low,
            self.ROUTE_GAN,
            np.where(p > self.theta_high, self.ROUTE_FACESWAP, self.ROUTE_ABSTAIN),
        )
        return decisions, p

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        """Save router artifacts (model, scaler, metadata) to directory."""
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.model,  os.path.join(directory, "router_logreg.pkl"))
        joblib.dump(self.scaler, os.path.join(directory, "router_scaler.pkl"))
        meta = {
            "theta_low":   self.theta_low,
            "theta_high":  self.theta_high,
            "C":           self.C,
            "max_iter":    self.max_iter,
            "val_metrics": self._val_metrics,
        }
        with open(os.path.join(directory, "router_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[Router] Saved → {directory}")

    @classmethod
    def load(cls, directory: str) -> "ManipulationTypeRouter":
        """Load a previously saved router from directory."""
        meta_path = os.path.join(directory, "router_meta.json")
        with open(meta_path) as f:
            meta = json.load(f)
        router = cls(
            theta_low=meta["theta_low"],
            theta_high=meta["theta_high"],
            C=meta["C"],
            max_iter=meta["max_iter"],
        )
        router.model  = joblib.load(os.path.join(directory, "router_logreg.pkl"))
        router.scaler = joblib.load(os.path.join(directory, "router_scaler.pkl"))
        router._val_metrics = meta.get("val_metrics")
        print(f"[Router] Loaded from {directory}")
        return router
