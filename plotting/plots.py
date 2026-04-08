"""
Thesis-quality plot generation for HDRA-Fusion (300 DPI).

All functions save figures to the provided plots_dir and return
the saved file path. Figures match the style used in the thesis
Chapter 4 experimental results.
"""

import os
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

from configs.config import (
    DPI,
    FIGSIZE_SQ,
    FIGSIZE_TALL,
    FIGSIZE_WIDE,
    PALETTE,
    ROUTER_THETA_HIGH,
    ROUTER_THETA_LOW,
)
from evaluation.metrics import compute_ece


def _savefig(fig, name: str, plots_dir: str) -> str:
    """Save figure to plots_dir at 300 DPI and close it."""
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved → {path}")
    return path


def plot_router_confidence_distribution(
    p_faceswap: np.ndarray,
    true_manip:  np.ndarray,
    theta_low:   float,
    theta_high:  float,
    plots_dir:   str,
) -> str:
    """
    Histogram of router P(face_swap) stratified by true manipulation type.

    Annotates the routing zones: GAN / ABSTAIN / face-swap.
    A strongly bimodal distribution supports H1 (CLIP linear separability).
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_TALL)
    bins = np.linspace(0, 1, 51)

    for label, col, name in [
        (0, PALETTE["fred"],  "GAN-synthetic"),
        (1, PALETTE["sbi"],   "Face-swap"),
    ]:
        mask = true_manip == label
        if mask.sum() > 0:
            ax.hist(
                p_faceswap[mask], bins=bins, alpha=0.65,
                color=col, label=name, density=True,
                edgecolor="white", lw=0.3,
            )

    ax.axvspan(0,          theta_low,  alpha=0.08, color=PALETTE["fred"],
               label=f"GAN zone (p < {theta_low})")
    ax.axvspan(theta_low,  theta_high, alpha=0.08, color=PALETTE["abstain"],
               label="Abstain zone")
    ax.axvspan(theta_high, 1.0,        alpha=0.08, color=PALETTE["sbi"],
               label=f"Face-swap zone (p > {theta_high})")
    ax.axvline(theta_low,  color=PALETTE["fred"],  ls="--", lw=1.5)
    ax.axvline(theta_high, color=PALETTE["sbi"],   ls="--", lw=1.5)

    ax.set_xlabel(
        r"Router Confidence  $P(\mathrm{face\_swap}\ |\ \mathbf{x})$",
        fontsize=12,
    )
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        "Manipulation-Type Router: Confidence Distribution\n"
        "by True Manipulation Type",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper center")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    return _savefig(fig, "router_confidence_distribution.png", plots_dir)


def plot_routing_breakdown(
    routing_counts: Dict[str, int],
    plots_dir: str,
) -> str:
    """Bar + pie chart showing the routing decision breakdown."""
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    labels = ["FRED-Fusion\n(GAN)", "SBI\n(Face-Swap)", "Abstain"]
    values = [
        routing_counts.get("fred_fusion", 0),
        routing_counts.get("sbi",         0),
        routing_counts.get("abstain",     0),
    ]
    colors = [PALETTE["fred"], PALETTE["sbi"], PALETTE["abstain"]]
    total  = sum(values)

    bars = axes[0].bar(labels, values, color=colors, width=0.55,
                       edgecolor="white", lw=1.2)
    for bar, v in zip(bars, values):
        pct = 100 * v / total if total > 0 else 0
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01 * max(values),
            f"{v:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=10,
        )
    axes[0].set_ylabel("Number of Images", fontsize=11)
    axes[0].set_title("Routing Decisions — Absolute Count",
                      fontsize=12, fontweight="bold")
    axes[0].set_ylim(0, max(values) * 1.2)
    axes[0].grid(axis="y", alpha=0.3)

    wedges, texts, autotexts = axes[1].pie(
        values, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=1.5),
    )
    for at in autotexts:
        at.set_fontsize(10)
    axes[1].set_title("Routing Decisions — Proportion",
                      fontsize=12, fontweight="bold")

    fig.suptitle("Hybrid Framework: Routing Decision Breakdown",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _savefig(fig, "routing_breakdown.png", plots_dir)


def plot_router_roc_pr(
    router_metrics: Dict[str, Any],
    y_true:   np.ndarray,
    y_prob:   np.ndarray,
    plots_dir: str,
) -> str:
    """Router ROC and Precision-Recall curves (manipulation-type classification)."""
    fpr, tpr, _  = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    roc_auc_val = router_metrics.get("roc_auc", auc(fpr, tpr))
    pr_auc_val  = router_metrics.get("pr_auc",  auc(rec, prec))

    axes[0].plot(fpr, tpr, color=PALETTE["router"], lw=2,
                 label=f"ROC (AUC = {roc_auc_val:.4f})")
    axes[0].plot([0, 1], [0, 1], color=PALETTE["chance"],
                 ls="--", lw=1.2, label="Chance")
    axes[0].set_xlabel("False Positive Rate", fontsize=11)
    axes[0].set_ylabel("True Positive Rate", fontsize=11)
    axes[0].set_title("Router: ROC Curve\n(GAN vs. Face-Swap Classification)",
                      fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].set_xlim(-0.01, 1.01); axes[0].set_ylim(-0.01, 1.01)
    axes[0].grid(alpha=0.3)

    axes[1].plot(rec, prec, color=PALETTE["router"], lw=2,
                 label=f"PR (AP = {pr_auc_val:.4f})")
    axes[1].axhline(y_true.mean(), color=PALETTE["chance"], ls="--", lw=1.2,
                    label=f"Baseline ({y_true.mean():.3f})")
    axes[1].set_xlabel("Recall", fontsize=11)
    axes[1].set_ylabel("Precision", fontsize=11)
    axes[1].set_title("Router: Precision-Recall Curve\n(GAN vs. Face-Swap Classification)",
                      fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].set_xlim(-0.01, 1.01); axes[1].set_ylim(-0.01, 1.05)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    return _savefig(fig, "router_roc_pr.png", plots_dir)


def plot_router_confusion_matrix(
    y_true:    np.ndarray,
    y_pred:    np.ndarray,
    plots_dir: str,
) -> str:
    """Normalised confusion matrix for the router."""
    cm   = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm_n, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    class_names = ["GAN-synthetic\n(→ FRED-Fusion)", "Face-swap\n(→ SBI)"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names, fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.set_xlabel("Predicted Manipulation Type", fontsize=11)
    ax.set_ylabel("True Manipulation Type", fontsize=11)
    ax.set_title("Router Confusion Matrix (Normalised)",
                 fontsize=13, fontweight="bold")

    for i in range(2):
        for j in range(2):
            col = "white" if cm_n[i, j] > 0.6 else "black"
            ax.text(
                j, i, f"{cm_n[i,j]:.3f}\n(n={cm[i,j]:,})",
                ha="center", va="center",
                fontsize=10.5, color=col, fontweight="bold",
            )
    fig.tight_layout()
    return _savefig(fig, "router_confusion_matrix.png", plots_dir)


def plot_per_detector_roc(
    fred_y_true: np.ndarray, fred_y_prob: np.ndarray,
    sbi_y_true:  np.ndarray, sbi_y_prob:  np.ndarray,
    plots_dir:   str,
) -> str:
    """ROC curves for each detector on its routed subset."""
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    for ax, y_true, y_prob, col, name in [
        (axes[0], fred_y_true, fred_y_prob, PALETTE["fred"], "FRED-Fusion (GAN holdout)"),
        (axes[1], sbi_y_true,  sbi_y_prob,  PALETTE["sbi"],  "SBI (Face-Swap holdout)"),
    ]:
        if len(y_true) < 10:
            ax.text(0.5, 0.5, "Insufficient data", ha="center",
                    transform=ax.transAxes)
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ra = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, color=col, lw=2, label=f"AUC = {ra:.4f}")
        ax.plot([0, 1], [0, 1], color=PALETTE["chance"],
                ls="--", lw=1.2, label="Chance")
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title(f"{name}\nROC Curve", fontsize=12, fontweight="bold")
        ax.legend(fontsize=10); ax.grid(alpha=0.3)
        ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)

    fig.suptitle("Per-Detector Performance on Routed Subsets",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _savefig(fig, "per_detector_roc.png", plots_dir)


def plot_end_to_end_roc(
    routed_y_true: np.ndarray,
    routed_y_prob: np.ndarray,
    fred_y_true:   Optional[np.ndarray],
    fred_y_prob:   Optional[np.ndarray],
    sbi_y_true:    Optional[np.ndarray],
    sbi_y_prob:    Optional[np.ndarray],
    plots_dir:     str,
) -> str:
    """Overlaid ROC curves — hybrid E2E vs individual detectors."""
    fig, ax = plt.subplots(figsize=FIGSIZE_TALL)

    curves = [
        (routed_y_true, routed_y_prob, PALETTE["router"], "Hybrid (end-to-end)", 2.5),
        (fred_y_true,   fred_y_prob,   PALETTE["fred"],   "FRED-Fusion (GAN)",   1.5),
        (sbi_y_true,    sbi_y_prob,    PALETTE["sbi"],    "SBI (Face-Swap)",     1.5),
    ]
    for y_t, y_p, col, lbl, lw in curves:
        if y_t is None or len(y_t) < 10:
            continue
        fpr, tpr, _ = roc_curve(y_t, y_p)
        ra = roc_auc_score(y_t, y_p)
        ax.plot(fpr, tpr, color=col, lw=lw, label=f"{lbl}  (AUC = {ra:.4f})")

    ax.plot([0, 1], [0, 1], color=PALETTE["chance"],
            ls="--", lw=1.2, label="Chance")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Hybrid Framework vs Individual Detectors\nEnd-to-End ROC Comparison",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
    fig.tight_layout()
    return _savefig(fig, "hybrid_e2e_roc.png", plots_dir)


def plot_fpr_tpr_operating_points(
    metrics_dict: Dict[str, Dict],
    plots_dir:    str,
) -> str:
    """Grouped bar chart: FPR at fixed TPR thresholds across detectors."""
    tpr_levels = [0.90, 0.95, 0.98]
    keys_fpr   = ["fpr_tpr90", "fpr_tpr95", "fpr_tpr98"]

    detectors = [k for k in metrics_dict if k in ["fred_fusion", "sbi", "hybrid_e2e"]]
    x = np.arange(len(tpr_levels)); width = 0.25
    colors = [PALETTE["fred"], PALETTE["sbi"], PALETTE["router"]]
    labels = {"fred_fusion": "FRED-Fusion", "sbi": "SBI", "hybrid_e2e": "Hybrid (E2E)"}

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (det, col) in enumerate(zip(detectors, colors)):
        if det not in metrics_dict:
            continue
        vals = [metrics_dict[det].get(k, 1.0) for k in keys_fpr]
        bars = ax.bar(x + i * width, vals, width,
                      label=labels.get(det, det),
                      color=col, edgecolor="white", lw=0.8)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v:.4f}", ha="center", va="bottom",
                fontsize=7.5, rotation=45,
            )

    ax.set_xticks(x + width * (len(detectors) - 1) / 2)
    ax.set_xticklabels([f"TPR = {int(t * 100)}%" for t in tpr_levels], fontsize=11)
    ax.set_ylabel("False Positive Rate", fontsize=11)
    ax.set_title("FPR at Fixed TPR Operating Points\n"
                 "Hybrid Framework vs Individual Detectors",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, min(1.0, ax.get_ylim()[1] * 1.3))
    fig.tight_layout()
    return _savefig(fig, "fpr_tpr_operating_points.png", plots_dir)


def plot_calibration(
    y_trues:   Dict[str, np.ndarray],
    y_probs:   Dict[str, np.ndarray],
    plots_dir: str,
) -> str:
    """Reliability diagrams for each detector and the hybrid system."""
    n_cols = len(y_trues)
    fig, axes = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 5.5), sharey=True)
    if n_cols == 1:
        axes = [axes]

    palette_map = {
        "FRED-Fusion": PALETTE["fred"],
        "SBI":         PALETTE["sbi"],
        "Hybrid":      PALETTE["router"],
    }

    for ax, (name, y_true) in zip(axes, y_trues.items()):
        y_prob = y_probs[name]
        if len(y_true) < 10:
            ax.text(0.5, 0.5, "Insufficient data",
                    ha="center", transform=ax.transAxes)
            continue

        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
        ece = compute_ece(y_true, y_prob)
        col = palette_map.get(name, PALETTE["router"])

        ax.plot([0, 1], [0, 1], color=PALETTE["chance"],
                ls="--", lw=1.5, label="Perfect")
        ax.plot(mean_pred, frac_pos, "o-", color=col, lw=2, ms=6,
                label=f"{name}\n(ECE = {ece:.4f})")
        ax.fill_between(mean_pred, frac_pos, mean_pred, alpha=0.15, color=col)
        ax.set_xlabel("Mean Predicted Probability", fontsize=11)
        ax.set_ylabel("Fraction of Positives", fontsize=11)
        ax.set_title(f"{name}\nCalibration", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim(-0.02, 1.02)

    fig.suptitle("Reliability Diagrams — Hybrid Framework Components",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _savefig(fig, "calibration_reliability_diagrams.png", plots_dir)


def plot_summary_metrics_bar(
    metrics_dict: Dict[str, Dict],
    plots_dir:    str,
) -> str:
    """Summary grouped bar chart: AUC, PR-AUC, Accuracy, ECE per detector."""
    metric_keys   = ["roc_auc", "pr_auc", "accuracy", "ece"]
    metric_labels = ["AUC", "PR-AUC", "Accuracy", "ECE\n(lower is better)"]
    detectors     = [k for k in metrics_dict if k in ["fred_fusion", "sbi", "hybrid_e2e"]]
    det_labels    = {"fred_fusion": "FRED-Fusion", "sbi": "SBI", "hybrid_e2e": "Hybrid (E2E)"}
    colors        = {
        "fred_fusion": PALETTE["fred"],
        "sbi":         PALETTE["sbi"],
        "hybrid_e2e":  PALETTE["router"],
    }

    x     = np.arange(len(metric_keys)); width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5.5))

    for i, det in enumerate(detectors):
        if det not in metrics_dict:
            continue
        vals = []
        for mk in metric_keys:
            v = metrics_dict[det].get(mk, 0.0)
            if mk == "accuracy" and v > 1.0:
                v /= 100.0
            vals.append(float(v))

        bars = ax.bar(x + i * width, vals, width,
                      label=det_labels.get(det, det),
                      color=colors.get(det, "gray"),
                      edgecolor="white", lw=0.8)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8.5,
            )

    ax.set_xticks(x + width * (len(detectors) - 1) / 2)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Hybrid Framework — Performance Summary\n"
                 "FRED-Fusion vs SBI vs End-to-End Hybrid",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.15)
    fig.tight_layout()
    return _savefig(fig, "summary_metrics_bar.png", plots_dir)


def plot_abstention_analysis(
    p_faceswap:  np.ndarray,
    labels:      np.ndarray,
    p_fake_all:  np.ndarray,
    decisions:   np.ndarray,
    theta_low:   float,
    theta_high:  float,
    plots_dir:   str,
) -> str:
    """
    AUC vs abstention threshold sweep.

    Shows how the hybrid end-to-end AUC and abstention rate change as
    the confidence gate width is varied. Used to validate the chosen
    threshold pair (theta_low=0.25, theta_high=0.75).
    """
    from models.router import ManipulationTypeRouter

    thresholds = np.linspace(0.05, 0.45, 30)
    aucs, rates = [], []

    for delta in thresholds:
        tl = 0.5 - delta
        th = 0.5 + delta
        routed = (p_faceswap < tl) | (p_faceswap > th)
        routed &= ~np.isnan(p_fake_all)
        if routed.sum() < 20:
            aucs.append(np.nan); rates.append(float(1 - delta * 2))
            continue
        try:
            a = roc_auc_score(labels[routed], p_fake_all[routed])
        except ValueError:
            a = np.nan
        aucs.append(a)
        rates.append(float(1 - routed.mean()))

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.plot(thresholds * 2, aucs, color=PALETTE["router"], lw=2.5,
             label="End-to-End AUC", marker="o", ms=4)
    ax2.plot(thresholds * 2, [r * 100 for r in rates],
             color=PALETTE["abstain"], lw=2, ls="--",
             label="Abstention Rate (%)", marker="s", ms=4)

    current_delta = (theta_high - theta_low) / 2
    ax1.axvline(current_delta * 2, color="black", ls=":", lw=1.5,
                label=f"Current gate (Δ={current_delta:.2f})")

    ax1.set_xlabel("Confidence Gate Width (2Δ around 0.5)", fontsize=11)
    ax1.set_ylabel("ROC-AUC", fontsize=11, color=PALETTE["router"])
    ax2.set_ylabel("Abstention Rate (%)", fontsize=11, color=PALETTE["abstain"])
    ax1.set_title("Abstention–AUC Trade-off\nEffect of Router Confidence Gate",
                  fontsize=13, fontweight="bold")
    ax1.set_ylim(0.4, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="lower right")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    return _savefig(fig, "abstention_auc_tradeoff.png", plots_dir)


def plot_per_source_auc_heatmap(
    source_metrics: Dict[str, Dict],
    plots_dir:      str,
) -> str:
    """Per-source AUC heatmap: router + FRED + SBI columns, one row per source."""
    sources   = sorted(source_metrics.keys())
    det_keys  = ["router_auc", "fred_auc", "sbi_auc"]
    det_names = [
        "Router\n(manip-type)",
        "FRED-Fusion\n(GAN detector)",
        "SBI\n(face-swap detector)",
    ]

    data = np.array([
        [source_metrics[s].get(k, np.nan) for k in det_keys]
        for s in sources
    ])

    fig, ax = plt.subplots(figsize=(
        max(6, len(det_keys) * 2.5),
        max(4, len(sources) * 0.8 + 1.5),
    ))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(det_names)))
    ax.set_xticklabels(det_names, fontsize=10)
    ax.set_yticks(range(len(sources)))
    ax.set_yticklabels(sources, fontsize=9)
    ax.set_title("Per-Source AUC Heatmap — Hybrid Framework Components",
                 fontsize=13, fontweight="bold", pad=12)

    for i in range(len(sources)):
        for j in range(len(det_keys)):
            v = data[i, j]
            if not np.isnan(v):
                col = "white" if v < 0.65 or v > 0.90 else "black"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=10, color=col, fontweight="bold")
    fig.tight_layout()
    return _savefig(fig, "per_source_auc_heatmap.png", plots_dir)
