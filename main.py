"""
================================================================================
HDRA-Fusion: Hybrid Detection with Routed Architecture
================================================================================
Thesis: HDRA-Fusion: Hybrid Detection with Routed Architecture for
        Manipulation-Aware AI Face Forgery Detection
        Florida Institute of Technology

Main entry point. Supports four modes:

  train_router  Extract CLIP embeddings and train the logistic regression router.
  evaluate      Run full hybrid evaluation on all configured datasets.
  infer         Single-image inference — returns routing decision and p(fake).
  full          Train router then evaluate (default).

Usage examples:
  python main.py --mode full
  python main.py --mode train_router --force_reextract
  python main.py --mode evaluate --router_dir /path/to/router/
  python main.py --mode infer --image /path/to/face.jpg

Before running:
  Set all dataset paths and model artifact paths in configs/config.py.
  See docs/dataset_setup.md for download links and folder structure.
================================================================================
"""

import argparse
import os
import sys
import traceback

from configs import config
from utils.logger import make_dirs, setup_logger


def _override_config(args: argparse.Namespace) -> None:
    """Apply CLI argument overrides to config globals."""
    if args.out_dir:
        config.RUN_DIR     = args.out_dir
        config.ROUTER_DIR  = os.path.join(config.RUN_DIR, "router")
        config.REPORTS_DIR = os.path.join(config.RUN_DIR, "reports")
        config.PLOTS_DIR   = os.path.join(config.RUN_DIR, "plots")
        config.CACHE_DIR   = os.path.join(config.RUN_DIR, "cache")
        config.SUMMARY_DIR = os.path.join(config.RUN_DIR, "summary")

    if args.fred_fold:
        config.FRED_FOLD = args.fred_fold

    if args.fred_dir:
        config.FRED_MODEL_DIR = args.fred_dir


def _validate_paths() -> None:
    """
    Warn if any dataset or model paths are still empty strings.
    The run will continue — empty paths are skipped gracefully by
    collect_image_paths() — but the user should be informed.
    """
    from configs.config import (
        FACESWAP_EVAL_DATASETS,
        FACESWAP_TRAIN_DATASETS,
        FRED_MODEL_DIR,
        GAN_EVAL_DATASETS,
        GAN_TRAIN_DATASETS,
        SBI_REPO_ROOT,
    )

    empty = []
    for ds_list in [
        GAN_TRAIN_DATASETS, FACESWAP_TRAIN_DATASETS,
        GAN_EVAL_DATASETS,  FACESWAP_EVAL_DATASETS,
    ]:
        for ds in ds_list:
            for key in ("real_path", "fake_path"):
                if not ds.get(key):
                    empty.append(f"  {ds['name']} → {key}")

    if not FRED_MODEL_DIR:
        empty.append("  FRED_MODEL_DIR (in configs/config.py)")
    if not SBI_REPO_ROOT:
        empty.append("  SBI_REPO_ROOT  (in configs/config.py)")

    if empty:
        print(
            "\n[WARNING] The following paths are not configured:\n"
            + "\n".join(empty)
            + "\n\nEdit configs/config.py before running. "
            "See docs/dataset_setup.md for details.\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HDRA-Fusion Hybrid Detector Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  train_router   Extract CLIP embeddings + train LogReg router
  evaluate       Full hybrid evaluation with all plots and reports
  infer          Single image inference (requires --image)
  full           Train router then evaluate (default)

Examples:
  python main.py --mode full
  python main.py --mode train_router --force_reextract
  python main.py --mode evaluate --router_dir path/to/router
  python main.py --mode infer --image path/to/face.jpg
        """,
    )

    parser.add_argument(
        "--mode", default="full",
        choices=["full", "train_router", "evaluate", "infer"],
        help="Execution mode (default: full)",
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Image path for --mode infer",
    )
    parser.add_argument(
        "--router_dir", type=str, default=None,
        help="Load existing router from this directory (skips training)",
    )
    parser.add_argument(
        "--fred_dir", type=str, default=None,
        help="Override FRED-Fusion model directory",
    )
    parser.add_argument(
        "--fred_fold", type=str, default=None,
        help="Override FRED fold name (e.g. RVF10K)",
    )
    parser.add_argument(
        "--force_reextract", action="store_true",
        help="Force re-extraction of CLIP embeddings even if cache exists",
    )
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="Override output directory (default: auto-timestamped run dir)",
    )

    args = parser.parse_args()
    _override_config(args)

    make_dirs(
        config.RUN_DIR,
        config.ROUTER_DIR,
        config.REPORTS_DIR,
        config.PLOTS_DIR,
        config.CACHE_DIR,
        config.SUMMARY_DIR,
    )
    logger = setup_logger(config.RUN_DIR)

    logger.info("=" * 72)
    logger.info("HDRA-Fusion Hybrid Detector Framework  v1.0")
    logger.info(f"Mode      : {args.mode}")
    logger.info(f"Run dir   : {config.RUN_DIR}")
    logger.info(f"Device    : {config.DEVICE}")
    logger.info(f"FRED fold : {config.FRED_FOLD}")
    logger.info("=" * 72)

    _validate_paths()

    # ── Mode: infer ────────────────────────────────────────────────────────────
    if args.mode == "infer":
        if not args.image:
            parser.error("--mode infer requires --image <path>")

        from models.fred_detector import FREDFusionDetector
        from models.router import ManipulationTypeRouter
        from models.sbi_detector import SBIDetector
        from evaluation.evaluate import HybridDetector

        router_dir = args.router_dir or config.ROUTER_DIR
        router     = ManipulationTypeRouter.load(router_dir)
        fred_det   = FREDFusionDetector(config.FRED_MODEL_DIR, config.FRED_FOLD)
        fred_det.load(config.DEVICE)
        sbi_det    = SBIDetector(config.SBI_REPO_ROOT)
        sbi_det.load(config.DEVICE)
        hybrid     = HybridDetector(router, fred_det, sbi_det)

        out = hybrid.predict_single(args.image)
        print("\n" + "=" * 50)
        print("INFERENCE RESULT")
        print("=" * 50)
        for k, v in out.items():
            print(f"  {k:<15}: {v}")
        print("=" * 50)
        return

    # ── Mode: train_router ─────────────────────────────────────────────────────
    router = None
    if args.mode in ("full", "train_router"):
        from pipelines.train_router import train_router_pipeline
        router = train_router_pipeline(
            logger, force_reextract=args.force_reextract
        )

    # ── Mode: evaluate ─────────────────────────────────────────────────────────
    if args.mode in ("full", "evaluate"):
        from models.fred_detector import FREDFusionDetector
        from models.router import ManipulationTypeRouter
        from models.sbi_detector import SBIDetector
        from pipelines.run_evaluation import run_full_evaluation
        from reports.report_generator import (
            generate_json_report,
            generate_txt_report,
            save_predictions_csv,
        )

        if args.mode == "evaluate":
            router_dir = args.router_dir or config.ROUTER_DIR
            logger.info(f"[Eval] Loading router from {router_dir}")
            router = ManipulationTypeRouter.load(router_dir)

        fred_det = FREDFusionDetector(config.FRED_MODEL_DIR, config.FRED_FOLD)
        sbi_det  = SBIDetector(config.SBI_REPO_ROOT)

        logger.info("[Eval] Loading FRED-Fusion …")
        fred_det.load(config.DEVICE)
        logger.info("[Eval] Loading SBI …")
        sbi_det.load(config.DEVICE)

        result = run_full_evaluation(router, fred_det, sbi_det, logger)

        generate_json_report(result, config.RUN_DIR)
        generate_txt_report(result,  config.RUN_DIR)
        save_predictions_csv(result, config.RUN_DIR)

        logger.info("")
        logger.info("=" * 72)
        logger.info("HDRA-FUSION EVALUATION COMPLETE")
        logger.info(f"  Run dir  : {config.RUN_DIR}")
        logger.info(f"  Plots    : {config.PLOTS_DIR}")
        logger.info(f"  Reports  : {config.SUMMARY_DIR}")
        if result.hybrid_metrics:
            logger.info(
                f"  AUC E2E  : {result.hybrid_metrics.get('roc_auc', '—'):.4f}"
            )
            logger.info(f"  Abstain  : {result.abstain_rate * 100:.1f}%")
        logger.info("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted]")
        sys.exit(0)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
