"""
Evaluation metrics for HDRA-Fusion.

Provides ECE, FPR@TPR, full metric suite, and optimal threshold selection.
All functions operate on numpy arrays and are framework-agnostic.
"""

from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error with equal-width bins.

    Parameters
    ----------
    y_true : binary ground-truth labels
    y_prob : predicted probabilities for the positive class
    n_bins : number of calibration bins

    Returns
    -------
    float — ECE value (lower is better, 0 = perfect calibration)
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.sum() * abs(acc - conf)
    return float(ece / len(y_true))


def compute_fpr_at_tpr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    tpr_target: float,
) -> float:
    """
    FPR at a fixed TPR target (interpolated from the ROC curve).

    Parameters
    ----------
    y_true     : binary ground-truth labels
    y_prob     : predicted probabilities for the positive class
    tpr_target : target true positive rate (e.g. 0.95 for TPR=95%)

    Returns
    -------
    float — interpolated FPR at the requested TPR
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    idx = np.searchsorted(tpr, tpr_target)
    idx = np.clip(idx, 0, len(fpr) - 1)
    return float(fpr[idx])


def full_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    tag: str = "",
) -> Dict[str, Any]:
    """
    Compute the complete metric suite used in the thesis.

    Parameters
    ----------
    y_true    : binary ground-truth labels (0=real, 1=fake)
    y_prob    : predicted probabilities for the positive (fake) class
    threshold : decision threshold for binary predictions
    tag       : label attached to the metrics dict (e.g. "router", "fred_fusion")

    Returns
    -------
    dict containing all metrics reported in the thesis
    """
    pred = (y_prob >= threshold).astype(int)

    try:
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    except ValueError:
        tn = fp = fn = tp = 0

    return {
        "tag":          tag,
        "n":            int(len(y_true)),
        "threshold":    float(threshold),
        "roc_auc":      float(roc_auc_score(y_true, y_prob)),
        "pr_auc":       float(average_precision_score(y_true, y_prob)),
        "accuracy":     float(accuracy_score(y_true, pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, pred)),
        "precision":    float(precision_score(y_true, pred, zero_division=0)),
        "recall":       float(recall_score(y_true, pred, zero_division=0)),
        "f1":           float(f1_score(y_true, pred, zero_division=0)),
        "brier":        float(brier_score_loss(y_true, y_prob)),
        "nll":          float(log_loss(y_true, np.column_stack([1 - y_prob, y_prob]))),
        "ece":          compute_ece(y_true, y_prob),
        "fpr_tpr90":    compute_fpr_at_tpr(y_true, y_prob, 0.90),
        "fpr_tpr95":    compute_fpr_at_tpr(y_true, y_prob, 0.95),
        "fpr_tpr98":    compute_fpr_at_tpr(y_true, y_prob, 0.98),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def choose_threshold_balanced(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """
    Select the decision threshold that maximises balanced accuracy.

    This is the threshold selection strategy used for both the SBI detector
    and the manipulation-type router in the HDRA-Fusion evaluation.

    Parameters
    ----------
    y_true : binary ground-truth labels
    y_prob : predicted probabilities for the positive class

    Returns
    -------
    float — optimal decision threshold in [0, 1]
    """
    thresholds = np.unique(np.round(y_prob, 6))
    thresholds = np.concatenate(([0.0], thresholds, [1.0]))
    best_bal, best_thr = -1.0, 0.5
    for thr in thresholds:
        pred = (y_prob >= thr).astype(int)
        bal  = balanced_accuracy_score(y_true, pred)
        if bal > best_bal:
            best_bal, best_thr = bal, thr
    return float(best_thr)
