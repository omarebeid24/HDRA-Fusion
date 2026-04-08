"""
Router training pipeline for HDRA-Fusion.

Trains the manipulation-type logistic regression router on CLIP-512
embeddings extracted from the GAN and face-swap training datasets.

Steps:
    1. Collect image paths from GAN and face-swap training sources.
    2. Extract raw CLIP ViT-B/32 embeddings (cached to disk).
    3. Stratified train/val split (80/20).
    4. Train logistic regression router.
    5. Save router artifacts and generate diagnostic plots.
"""

import logging
import numpy as np
from sklearn.model_selection import train_test_split

from configs.config import (
    CACHE_DIR,
    DEVICE,
    FACESWAP_TRAIN_DATASETS,
    GAN_EVAL_DATASETS,
    GAN_TRAIN_DATASETS,
    FACESWAP_EVAL_DATASETS,
    PLOTS_DIR,
    ROUTER_C,
    ROUTER_DIR,
    ROUTER_MAX_ITER,
    ROUTER_THETA_HIGH,
    ROUTER_THETA_LOW,
    ROUTER_VAL_RATIO,
    SEED,
)
from evaluation.metrics import choose_threshold_balanced
from models.clip_extractor import extract_clip_embeddings_raw
from models.router import ManipulationTypeRouter
from pipelines.collect_paths import collect_image_paths
from plotting.plots import (
    plot_router_confidence_distribution,
    plot_router_confusion_matrix,
    plot_router_roc_pr,
    plot_routing_breakdown,
)


def train_router_pipeline(
    logger: logging.Logger,
    force_reextract: bool = False,
) -> ManipulationTypeRouter:
    """
    Full router training pipeline.

    Parameters
    ----------
    logger          : logger instance for progress output
    force_reextract : if True, re-extract CLIP embeddings even if cache exists

    Returns
    -------
    Trained ManipulationTypeRouter saved to ROUTER_DIR
    """
    logger.info("=" * 72)
    logger.info("PHASE 1: Router Training")
    logger.info("=" * 72)

    # ── Collect training images ────────────────────────────────────────────────
    gan_paths, gan_fakes, gan_manip, gan_src = collect_image_paths(
        GAN_TRAIN_DATASETS, manip_label=0
    )
    fs_paths, fs_fakes, fs_manip, fs_src = collect_image_paths(
        FACESWAP_TRAIN_DATASETS, manip_label=1
    )

    all_paths = gan_paths + fs_paths
    all_manip = np.array(gan_manip + fs_manip, dtype=np.int64)

    eval_names = ", ".join(
        d["name"] for d in GAN_EVAL_DATASETS + FACESWAP_EVAL_DATASETS
    )
    logger.info("Router training dataset summary:")
    logger.info(f"  GAN source       : {', '.join(d['name'] for d in GAN_TRAIN_DATASETS)}")
    logger.info(f"  Face-swap source : {', '.join(d['name'] for d in FACESWAP_TRAIN_DATASETS)}")
    logger.info(f"  GAN images       : {len(gan_paths):,}")
    logger.info(f"  Face-swap images : {len(fs_paths):,}")
    logger.info(f"  Total            : {len(all_paths):,}")
    logger.info(f"  [Eval sources withheld: {eval_names}]")

    if not all_paths:
        raise ValueError(
            "No training images collected. "
            "Check that all paths in GAN_TRAIN_DATASETS and "
            "FACESWAP_TRAIN_DATASETS are set correctly in configs/config.py."
        )

    # ── Extract CLIP embeddings ────────────────────────────────────────────────
    import os
    emb_cache = os.path.join(CACHE_DIR, "router_train_embeddings.npy")
    X = extract_clip_embeddings_raw(
        all_paths, emb_cache, device=DEVICE, force=force_reextract
    )

    # ── Stratified train/val split ─────────────────────────────────────────────
    idx = np.arange(len(all_paths))
    tr_idx, va_idx = train_test_split(
        idx,
        test_size=ROUTER_VAL_RATIO,
        stratify=all_manip,
        random_state=SEED,
    )
    X_tr, y_tr = X[tr_idx], all_manip[tr_idx]
    X_va, y_va = X[va_idx], all_manip[va_idx]

    logger.info(f"Router split: train={len(tr_idx):,}  val={len(va_idx):,}")

    # ── Train ──────────────────────────────────────────────────────────────────
    router = ManipulationTypeRouter(
        theta_low=ROUTER_THETA_LOW,
        theta_high=ROUTER_THETA_HIGH,
        C=ROUTER_C,
        max_iter=ROUTER_MAX_ITER,
    )
    val_metrics = router.train(X_tr, y_tr, X_va, y_va, logger=logger)
    router.save(ROUTER_DIR)

    # ── Diagnostic plots ───────────────────────────────────────────────────────
    p_va = router.predict_proba(X_va)
    decisions_va, _ = router.route(X_va)

    plot_router_confidence_distribution(
        p_va, y_va, ROUTER_THETA_LOW, ROUTER_THETA_HIGH, PLOTS_DIR
    )
    plot_router_roc_pr(val_metrics, y_va, p_va, PLOTS_DIR)

    thr = choose_threshold_balanced(y_va, p_va)
    plot_router_confusion_matrix(y_va, (p_va >= thr).astype(int), PLOTS_DIR)

    routing_counts_va = {
        "fred_fusion": int((decisions_va == ManipulationTypeRouter.ROUTE_GAN).sum()),
        "sbi":         int((decisions_va == ManipulationTypeRouter.ROUTE_FACESWAP).sum()),
        "abstain":     int((decisions_va == ManipulationTypeRouter.ROUTE_ABSTAIN).sum()),
    }
    plot_routing_breakdown(routing_counts_va, PLOTS_DIR)

    logger.info(
        f"[Router] Training complete. "
        f"Val AUC={val_metrics.get('val_auc', 0):.4f}"
    )
    return router
