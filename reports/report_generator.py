"""
Report generation for HDRA-Fusion evaluation runs.

Produces three output formats:
    - JSON  : structured metrics for programmatic access
    - TXT   : publication-ready text report matching thesis formatting
    - CSV   : per-image predictions for downstream analysis
"""

import json
import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from configs.config import FRED_FOLD, ROUTER_THETA_HIGH, ROUTER_THETA_LOW, RUN_TIMESTAMP
from evaluation.evaluate import HybridResult


def _fmt(v: Any, pct: bool = False) -> str:
    """Format a metric value for the text report."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if pct:
        return f"{v * 100:.2f}%"
    return f"{v:.4f}"


def generate_json_report(result: HybridResult, run_dir: str) -> str:
    """
    Save a structured JSON report for programmatic access.

    Parameters
    ----------
    result  : HybridResult from the evaluation pipeline
    run_dir : root directory for the current run

    Returns
    -------
    str — path to the saved JSON file
    """
    report = {
        "framework":     "HDRA-Fusion Hybrid Detector",
        "version":       "1.0",
        "run_timestamp": RUN_TIMESTAMP,
        "fred_fold":     FRED_FOLD,
        "routing": {
            "theta_low":    ROUTER_THETA_LOW,
            "theta_high":   ROUTER_THETA_HIGH,
            "counts":       result.routing_counts,
            "abstain_rate": result.abstain_rate,
        },
        "router_metrics": result.router_metrics,
        "fred_metrics":   result.fred_metrics,
        "sbi_metrics":    result.sbi_metrics,
        "hybrid_metrics": result.hybrid_metrics,
    }
    path = os.path.join(run_dir, "summary", "hybrid_framework_results.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[Report] JSON → {path}")
    return path


def generate_txt_report(result: HybridResult, run_dir: str) -> str:
    """
    Generate a publication-ready text report.

    Format matches the thesis Appendix and the LOSO sweep TXT report style.

    Parameters
    ----------
    result  : HybridResult from the evaluation pipeline
    run_dir : root directory for the current run

    Returns
    -------
    str — path to the saved TXT file
    """
    lines = []
    add = lines.append

    add("=" * 80)
    add("FRED-Fusion Hybrid Detector Framework — Evaluation Report")
    add(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"FRED-Fusion fold used: {FRED_FOLD}")
    add("=" * 80)
    add("")

    # ── Framework design ───────────────────────────────────────────────────────
    add("FRAMEWORK DESIGN")
    add("-" * 50)
    add("  Detector A (GAN)       : FRED-Fusion (CLIP ViT-B/32 + DA + XGBoost)")
    add("  Detector B (Face-Swap) : SBI (EfficientNet-B4, pre-trained FF++)")
    add("  Router                 : Logistic Regression on raw CLIP-512 features")
    add("  Routing logic          : Hard binary with confidence abstention zone")
    add(f"    θ_low  = {ROUTER_THETA_LOW}   (p < θ_low  → FRED-Fusion)")
    add(f"    θ_high = {ROUTER_THETA_HIGH}   (p > θ_high → SBI)")
    add(f"    θ_low ≤ p ≤ θ_high   → ABSTAIN")
    add("")

    # ── Routing statistics ─────────────────────────────────────────────────────
    add("ROUTING STATISTICS")
    add("-" * 50)
    counts = result.routing_counts
    total  = sum(counts.values())
    add(f"  Total images evaluated : {total:,}")
    add(
        f"  → FRED-Fusion (GAN)    : {counts.get('fred_fusion', 0):,}  "
        f"({100 * counts.get('fred_fusion', 0) / max(total, 1):.1f}%)"
    )
    add(
        f"  → SBI (Face-Swap)      : {counts.get('sbi', 0):,}  "
        f"({100 * counts.get('sbi', 0) / max(total, 1):.1f}%)"
    )
    add(
        f"  → ABSTAIN              : {counts.get('abstain', 0):,}  "
        f"({result.abstain_rate * 100:.1f}%)"
    )
    add("")

    # ── Router performance ─────────────────────────────────────────────────────
    add("ROUTER PERFORMANCE  (Manipulation-Type Classification)")
    add("-" * 50)
    rm = result.router_metrics
    if rm:
        add(f"  {'Metric':<28} {'Value':>10}")
        add(f"  {'-' * 38}")
        for k, label in [
            ("roc_auc",      "AUC (ROC)"),
            ("pr_auc",       "AUC (PR)"),
            ("accuracy",     "Accuracy"),
            ("balanced_acc", "Balanced Accuracy"),
            ("f1",           "F1 Score"),
            ("precision",    "Precision"),
            ("recall",       "Recall"),
            ("ece",          "ECE"),
            ("fpr_tpr95",    "FPR @ TPR=95%"),
            ("n",            "N images"),
        ]:
            v = rm.get(k)
            if isinstance(v, float):
                add(f"  {label:<28} {v:>10.4f}")
            elif v is not None:
                add(f"  {label:<28} {v:>10}")
    add("")

    # ── Per-detector and hybrid performance ────────────────────────────────────
    sections = [
        ("FRED-FUSION",  result.fred_metrics,   "GAN-Routed Subset"),
        ("SBI",          result.sbi_metrics,     "Face-Swap-Routed Subset"),
        ("HYBRID E2E",   result.hybrid_metrics,  "All Routed (excl. Abstain)"),
    ]
    for det_name, det_metrics, label in sections:
        add(f"{det_name}  ({label})")
        add("-" * 50)
        if not det_metrics:
            add("  [No data — subset too small or detector unavailable]")
            add("")
            continue
        add(f"  {'Metric':<28} {'Value':>10}")
        add(f"  {'-' * 38}")
        for k, label2 in [
            ("roc_auc",      "AUC (ROC)"),
            ("pr_auc",       "AUC (PR)"),
            ("accuracy",     "Accuracy"),
            ("balanced_acc", "Balanced Accuracy"),
            ("f1",           "F1 Score"),
            ("brier",        "Brier Score"),
            ("nll",          "NLL"),
            ("ece",          "ECE"),
            ("fpr_tpr90",    "FPR @ TPR=90%"),
            ("fpr_tpr95",    "FPR @ TPR=95%"),
            ("fpr_tpr98",    "FPR @ TPR=98%"),
            ("n",            "N images"),
        ]:
            v = det_metrics.get(k)
            if isinstance(v, float):
                add(f"  {label2:<28} {v:>10.4f}")
            elif v is not None:
                add(f"  {label2:<28} {v:>10}")
        add("")

    # ── Summary comparison table ───────────────────────────────────────────────
    add("SUMMARY COMPARISON TABLE")
    add("-" * 50)
    header = (
        f"  {'Detector':<22} {'AUC':>8} {'PR-AUC':>8} "
        f"{'Acc':>8} {'F1':>8} {'ECE':>8} {'FPR@95':>8} {'N':>8}"
    )
    add(header)
    add("  " + "-" * (len(header) - 2))
    for det_name, dm in [
        ("FRED-Fusion",  result.fred_metrics),
        ("SBI",          result.sbi_metrics),
        ("Hybrid E2E",   result.hybrid_metrics),
    ]:
        if not dm:
            continue
        add(
            f"  {det_name:<22} "
            f"{_fmt(dm.get('roc_auc')):>8} "
            f"{_fmt(dm.get('pr_auc')):>8} "
            f"{_fmt(dm.get('accuracy')):>8} "
            f"{_fmt(dm.get('f1')):>8} "
            f"{_fmt(dm.get('ece')):>8} "
            f"{_fmt(dm.get('fpr_tpr95')):>8} "
            f"{str(dm.get('n', '—')):>8}"
        )
    add("")

    # ── Key findings ───────────────────────────────────────────────────────────
    add("KEY FINDINGS")
    add("-" * 50)
    if result.router_metrics.get("roc_auc", 0) > 0.95:
        add("  ✓ Router achieves high AUC in manipulation-type classification,")
        add("    confirming that raw CLIP representations carry sufficient")
        add("    discriminative power to separate GAN-synthetic from face-swap domains.")
    if result.fred_metrics.get("roc_auc", 0) > 0.95:
        add("  ✓ FRED-Fusion maintains high performance on GAN-routed images,")
        add("    consistent with LOSO evaluation results.")
    if result.sbi_metrics.get("roc_auc", 0) > 0.70:
        add("  ✓ SBI effectively handles face-swap images in the routed subset,")
        add("    addressing the critical limitation identified in FRED-Fusion.")
    if result.abstain_rate > 0:
        add(
            f"  • Abstention rate: {result.abstain_rate * 100:.1f}% of images flagged as"
        )
        add("    ambiguous manipulation type — a conservative forensic feature.")
    add("")
    add("=" * 80)
    add("End of Hybrid Framework Evaluation Report")
    add("=" * 80)

    txt  = "\n".join(lines)
    path = os.path.join(run_dir, "summary", "hybrid_framework_report.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"[Report] TXT  → {path}")
    return path


def save_predictions_csv(result: HybridResult, run_dir: str) -> str:
    """
    Save per-image predictions to CSV for downstream analysis.

    Columns:
        path, source, y_true, manip_type, p_faceswap,
        routing, detector, p_fake, pred_fake

    Parameters
    ----------
    result  : HybridResult from the evaluation pipeline
    run_dir : root directory for the current run

    Returns
    -------
    str — path to the saved CSV file
    """
    rows = [
        {
            "path":       p.path,
            "source":     p.source,
            "y_true":     p.label,
            "manip_type": p.manip_type,
            "p_faceswap": p.p_faceswap,
            "routing":    p.routing,
            "detector":   p.detector,
            "p_fake":     p.p_fake,
            "pred_fake":  (
                1 if p.p_fake >= 0.5 else
                0 if p.p_fake >= 0 else
                -1
            ),
        }
        for p in result.predictions
    ]

    df   = pd.DataFrame(rows)
    path = os.path.join(run_dir, "reports", "predictions_per_image.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[Report] CSV  → {path}  ({len(df):,} rows)")
    return path
