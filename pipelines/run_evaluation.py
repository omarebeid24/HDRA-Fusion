"""
Full HDRA-Fusion evaluation pipeline.

Runs end-to-end hybrid evaluation, generates all thesis plots,
and saves JSON, TXT, and CSV reports.
"""

import logging
import os
from typing import Dict

import numpy as np
from sklearn.metrics import roc_auc_score

from configs.config import (
    CACHE_DIR,
    DEVICE,
    FACESWAP_EVAL_DATASETS,
    GAN_EVAL_DATASETS,
    PLOTS_DIR,
    ROUTER_THETA_HIGH,
    ROUTER_THETA_LOW,
)
from evaluation.evaluate import HybridDetector, HybridResult
from models.fred_detector import FREDFusionDetector
from models.router import ManipulationTypeRouter
from models.sbi_detector import SBIDetector
from pipelines.collect_paths import collect_image_paths
from plotting.plots import (
    plot_abstention_analysis,
    plot_calibration,
    plot_end_to_end_roc,
    plot_fpr_tpr_operating_points,
    plot_per_detector_roc,
    plot_per_source_auc_heatmap,
    plot_router_confidence_distribution,
    plot_router_confusion_matrix,
    plot_router_roc_pr,
    plot_routing_breakdown,
    plot_summary_metrics_bar,
)
from evaluation.metrics import choose_threshold_balanced


def run_full_evaluation(
    router:   ManipulationTypeRouter,
    fred_det: FREDFusionDetector,
    sbi_det:  SBIDetector,
    logger:   logging.Logger,
) -> HybridResult:
    """
    End-to-end evaluation of the HDRA-Fusion hybrid framework.

    Evaluates on all GAN and face-swap evaluation datasets defined in
    configs/config.py. All evaluation sources must be completely unseen
    during router training to ensure leak-proof evaluation.

    Parameters
    ----------
    router   : trained ManipulationTypeRouter
    fred_det : loaded FREDFusionDetector
    sbi_det  : loaded SBIDetector
    logger   : logger for progress output

    Returns
    -------
    HybridResult with all metrics, predictions, and routing statistics
    """
    logger.info("=" * 72)
    logger.info("PHASE 2: Full Hybrid Evaluation")
    logger.info("=" * 72)

    # ── Collect evaluation images ──────────────────────────────────────────────
    gan_paths, gan_fakes, gan_manip, gan_src = collect_image_paths(
        GAN_EVAL_DATASETS, manip_label=0
    )
    fs_paths, fs_fakes, fs_manip, fs_src = collect_image_paths(
        FACESWAP_EVAL_DATASETS, manip_label=1
    )

    all_paths = gan_paths + fs_paths
    all_fakes = np.array(gan_fakes + fs_fakes, dtype=np.int64)
    all_manip = np.array(gan_manip + fs_manip, dtype=np.int64)
    all_src   = gan_src + fs_src

    logger.info("Evaluation set (unseen sources only):")
    logger.info(f"  GAN sources      : {', '.join(d['name'] for d in GAN_EVAL_DATASETS)}")
    logger.info(f"  Face-swap source : {', '.join(d['name'] for d in FACESWAP_EVAL_DATASETS)}")
    logger.info(f"  GAN images       : {len(gan_paths):,}")
    logger.info(f"  Face-swap images : {len(fs_paths):,}")
    logger.info(f"  Total            : {len(all_paths):,}")

    if not all_paths:
        raise ValueError(
            "No evaluation images collected. "
            "Check paths in GAN_EVAL_DATASETS and FACESWAP_EVAL_DATASETS "
            "in configs/config.py."
        )

    # ── Run hybrid detector ────────────────────────────────────────────────────
    hybrid = HybridDetector(router, fred_det, sbi_det, device=DEVICE)
    result = hybrid.evaluate(
        all_paths, all_fakes, all_manip,
        sources=all_src,
        logger=logger,
        cache_dir=CACHE_DIR,
    )

    # ── Build arrays for plotting ──────────────────────────────────────────────
    preds = result.predictions

    p_faceswap_all = np.array([p.p_faceswap for p in preds])
    decisions_all  = np.array([p.routing    for p in preds])
    p_fake_all     = np.array(
        [p.p_fake if p.p_fake >= 0 else float("nan") for p in preds],
        dtype=np.float32,
    )
    y_true_all = np.array([p.label for p in preds])

    routed_mask = (
        (decisions_all != ManipulationTypeRouter.ROUTE_ABSTAIN)
        & ~np.isnan(p_fake_all)
    )
    fred_mask = decisions_all == ManipulationTypeRouter.ROUTE_GAN
    sbi_mask  = decisions_all == ManipulationTypeRouter.ROUTE_FACESWAP

    # ── Generate all thesis plots ──────────────────────────────────────────────
    logger.info("[Plots] Generating thesis figures …")

    plot_routing_breakdown(result.routing_counts, PLOTS_DIR)

    if all_manip is not None:
        known = all_manip != -1
        plot_router_confidence_distribution(
            p_faceswap_all[known], all_manip[known],
            ROUTER_THETA_LOW, ROUTER_THETA_HIGH, PLOTS_DIR,
        )
        if result.router_metrics:
            mt_prob = p_faceswap_all[known]
            plot_router_roc_pr(result.router_metrics, all_manip[known], mt_prob, PLOTS_DIR)
            thr = result.router_metrics.get("optimal_threshold", 0.5)
            plot_router_confusion_matrix(
                all_manip[known], (mt_prob >= thr).astype(int), PLOTS_DIR
            )

    if fred_mask.sum() > 10 and sbi_mask.sum() > 10:
        plot_per_detector_roc(
            y_true_all[fred_mask & ~np.isnan(p_fake_all)],
            p_fake_all[fred_mask & ~np.isnan(p_fake_all)],
            y_true_all[sbi_mask  & ~np.isnan(p_fake_all)],
            p_fake_all[sbi_mask  & ~np.isnan(p_fake_all)],
            PLOTS_DIR,
        )

    if routed_mask.sum() > 10:
        plot_end_to_end_roc(
            y_true_all[routed_mask],
            p_fake_all[routed_mask],
            y_true_all[fred_mask & ~np.isnan(p_fake_all)] if fred_mask.sum() > 10 else None,
            p_fake_all[fred_mask & ~np.isnan(p_fake_all)] if fred_mask.sum() > 10 else None,
            y_true_all[sbi_mask  & ~np.isnan(p_fake_all)] if sbi_mask.sum()  > 10 else None,
            p_fake_all[sbi_mask  & ~np.isnan(p_fake_all)] if sbi_mask.sum()  > 10 else None,
            PLOTS_DIR,
        )

    all_metrics: Dict = {}
    if result.fred_metrics:   all_metrics["fred_fusion"] = result.fred_metrics
    if result.sbi_metrics:    all_metrics["sbi"]         = result.sbi_metrics
    if result.hybrid_metrics: all_metrics["hybrid_e2e"]  = result.hybrid_metrics

    if all_metrics:
        plot_fpr_tpr_operating_points(all_metrics, PLOTS_DIR)
        plot_summary_metrics_bar(all_metrics, PLOTS_DIR)

    cal_trues, cal_probs = {}, {}
    if fred_mask.sum() > 10 and result.fred_metrics:
        vm = fred_mask & ~np.isnan(p_fake_all)
        cal_trues["FRED-Fusion"] = y_true_all[vm]
        cal_probs["FRED-Fusion"] = p_fake_all[vm]
    if sbi_mask.sum() > 10 and result.sbi_metrics:
        vm = sbi_mask & ~np.isnan(p_fake_all)
        cal_trues["SBI"]  = y_true_all[vm]
        cal_probs["SBI"]  = p_fake_all[vm]
    if routed_mask.sum() > 10 and result.hybrid_metrics:
        cal_trues["Hybrid"] = y_true_all[routed_mask]
        cal_probs["Hybrid"] = p_fake_all[routed_mask]
    if cal_trues:
        plot_calibration(cal_trues, cal_probs, PLOTS_DIR)

    if routed_mask.sum() > 10:
        plot_abstention_analysis(
            p_faceswap_all, y_true_all, p_fake_all, decisions_all,
            ROUTER_THETA_LOW, ROUTER_THETA_HIGH, PLOTS_DIR,
        )

    # ── Per-source AUC heatmap ─────────────────────────────────────────────────
    src_arr         = np.array([p.source for p in preds])
    manip_arr       = np.array([p.manip_type for p in preds])
    unique_sources  = sorted(set(src_arr))
    source_metrics: Dict = {}

    for src_name in unique_sources:
        mask_src = src_arr == src_name
        sm: Dict = {}

        if mask_src.sum() >= 10 and (manip_arr[mask_src] != -1).all():
            try:
                sm["router_auc"] = float(
                    roc_auc_score(manip_arr[mask_src], p_faceswap_all[mask_src])
                )
            except ValueError:
                pass

        fred_src = mask_src & fred_mask & ~np.isnan(p_fake_all)
        if fred_src.sum() >= 10:
            try:
                sm["fred_auc"] = float(
                    roc_auc_score(y_true_all[fred_src], p_fake_all[fred_src])
                )
            except ValueError:
                pass

        sbi_src = mask_src & sbi_mask & ~np.isnan(p_fake_all)
        if sbi_src.sum() >= 10:
            try:
                sm["sbi_auc"] = float(
                    roc_auc_score(y_true_all[sbi_src], p_fake_all[sbi_src])
                )
            except ValueError:
                pass

        if sm:
            source_metrics[src_name] = sm

    if source_metrics:
        logger.info(f"[Plots] Per-source heatmap ({len(source_metrics)} sources)")
        plot_per_source_auc_heatmap(source_metrics, PLOTS_DIR)

    return result
