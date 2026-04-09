# HDRA-Fusion: Hybrid Detection with Routed Architecture for Manipulation-Aware AI Face Forgery Detection

**Florida Institute of Technology**

HDRA-Fusion is a manipulation-aware forensic detection framework that addresses a fundamental limitation of existing deepfake detectors: the assumption that a single model can handle all types of AI face forgery. GAN-synthetic face generation and face-swap manipulation produce entirely different forensic signatures and require dedicated specialist detectors. HDRA-Fusion explicitly separates the manipulation identification step from the detection step through a principled routing architecture.

---

## Framework Overview

```
Input Image
    │
    ▼
[Raw CLIP ViT-B/32 Encoder]  (frozen, no fine-tuning)
    │
    ▼  512-dim embedding
[Manipulation-Type Router]
 LogReg(CLIP-512) → P(face_swap)
    │
    ├── P < 0.25   →  FRED-Fusion (GAN specialist)
    │                  CLIP-DA + XGBoost ensemble
    │                  → p(fake | GAN hypothesis)
    │
    ├── P > 0.75   →  SBI (Face-swap specialist)
    │                  EfficientNet-B4, pretrained FF++
    │                  → p(fake | face-swap hypothesis)
    │
    └── 0.25 ≤ P ≤ 0.75  →  ABSTAIN (ambiguous manipulation type)
```

**Three components:**

| Component | Role | Architecture |
|---|---|---|
| Manipulation-Type Router | Classifies manipulation domain | Logistic Regression on CLIP-512 |
| FRED-Fusion (Detector A) | GAN-synthetic face detection | CLIP ViT-B/32 + Domain Adversarial Training + XGBoost |
| SBI (Detector B) | Face-swap detection | EfficientNet-B4, pretrained on FaceForensics++ |



## Repository Structure

```
hdra_fusion/
├── main.py                        # Entry point — argparse, mode dispatch
├── fred_fusion_core.py            # FRED-Fusion inference components
│                                  # (DomainAdversarialCLIP, TabularExtractor)
│
├── configs/
│   └── config.py                  # All constants and dataset paths
│                                  # Set your paths here before running
│
├── models/
│   ├── clip_extractor.py          # Raw CLIP embedding extractor
│   ├── router.py                  # ManipulationTypeRouter class
│   ├── fred_detector.py           # FREDFusionDetector wrapper
│   └── sbi_detector.py            # SBIDetector wrapper
│
├── evaluation/
│   ├── metrics.py                 # ECE, FPR@TPR, full metric suite
│   └── evaluate.py                # HybridDetector, end-to-end pipeline
│
├── plotting/
│   └── plots.py                   # All thesis figure generation (300 DPI)
│
├── pipelines/
│   ├── collect_paths.py           # Dataset path collection
│   ├── train_router.py            # Phase 1: router training
│   └── run_evaluation.py          # Phase 2: full evaluation
│
├── reports/
│   └── report_generator.py        # JSON, TXT, CSV report generation
│
├── utils/
│   └── logger.py                  # Logging and directory utilities
│
└── requirements.txt
```

---

## Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/omarebeid24/HDRA-Fusion.git
cd HDRA-Fusion
```

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv hdra_env
# Windows
hdra_env\Scripts\activate
# Linux / macOS
source hdra_env/bin/activate
```

### Step 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Install CLIP (not on PyPI)

```bash
pip install git+https://github.com/openai/CLIP.git
```

### Step 5 — Install SBI (Detector B)

SBI is a separate research repository and is not included here. Follow these steps:

```bash
# Clone SBI into a location of your choice
git clone https://github.com/mapooon/SelfBlendedImages.git

# Install SBI's dependency
pip install efficientnet-pytorch
```

Then download the **FFraw.tar** checkpoint from the [SBI releases page](https://github.com/mapooon/SelfBlendedImages/releases) and place it in your SBI weights directory.

---

## Downloading Model Checkpoints

### FRED-Fusion checkpoint (required)

The FRED-Fusion RVF10K fold checkpoint bundle (~560 MB) is hosted on GitHub Releases.

1. Download **fred_fusion_RVF10K_checkpoint.zip** from:
   [https://github.com/omarebeid24/HDRA-Fusion/releases/download/v1.0/fred_fusion_RVF10K_checkpoint.zip](https://github.com/omarebeid24/HDRA-Fusion/releases/download/1.0/FRED_FUSION_CHECKPOINT.zip)

2. Unzip into a directory of your choice. You should get these files:

```
your_checkpoint_dir/
├── clip_da_LOSO_RVF10K_valid_RVF10K_train.pt   (608 MB — CLIP backbone)
├── xgb_fusion_clip_1.json                       (XGBoost seed 1)
├── xgb_fusion_clip_2.json                       (XGBoost seed 2)
├── xgb_fusion_clip_3.json                       (XGBoost seed 3)
├── xgb_fusion_clip_4.json                       (XGBoost seed 4)
├── xgb_fusion_clip_5.json                       (XGBoost seed 5)
├── tab_selector_LOSO_RVF10K_valid_RVF10K_train.pkl
├── tab_scaler_LOSO_RVF10K_valid_RVF10K_train.pkl
└── domain_encoder_LOSO_RVF10K_valid_RVF10K_train.pkl
```

3. Set `FRED_MODEL_DIR` in `configs/config.py` to point to this directory.

---

## Configuration

Before running, open `configs/config.py` and set all dataset paths and model artifact paths. Every path that needs to be set is marked with a comment. Empty strings `""` are placeholders — the system will warn you at startup if any required path is not configured.

```python
# Example configuration (configs/config.py)

# FRED-Fusion model artifacts
FRED_MODEL_DIR = "/path/to/your/checkpoint/dir/"
FRED_FOLD      = "RVF10K"

# SBI repository and weights
SBI_REPO_ROOT   = "/path/to/SelfBlendedImages/"
SBI_WEIGHTS_DIR = "/path/to/SelfBlendedImages/weights/"

# Router training datasets (GAN side — 140K train split)
GAN_TRAIN_DATASETS = [
    {
        "name":      "140K_train",
        "real_path": "/data/140k/train/real",
        "fake_path": "/data/140k/train/fake",
        "max_samples": None,
    },
    ...
]

# Evaluation datasets (unseen sources — never used in router training)
GAN_EVAL_DATASETS = [
    {
        "name":      "HumanFaces",
        "real_path": "/data/humanfaces/real",
        "fake_path": "/data/humanfaces/fake",
        "max_samples": None,
    },
    ...
]

# Output directory
BASE_OUT = "/path/to/your/output/dir/"
```

---


## Usage

### Mode 1 — Full pipeline (train router then evaluate)

```bash
python main.py --mode full
```

This runs both phases in sequence: extracts CLIP embeddings, trains the router, then evaluates the full hybrid system on all configured evaluation datasets.

### Mode 2 — Train router only

```bash
python main.py --mode train_router

# Force re-extraction of CLIP embeddings (skip cache)
python main.py --mode train_router --force_reextract
```

### Mode 3 — Evaluate only (load existing router)

```bash
python main.py --mode evaluate --router_dir path/to/router/

# Override output directory
python main.py --mode evaluate --router_dir path/to/router/ --out_dir path/to/output/
```

### Mode 4 — Single image inference

```bash
python main.py --mode infer --image path/to/face.jpg
```

Output example:
```
==================================================
INFERENCE RESULT
==================================================
  image          : path/to/face.jpg
  p_faceswap     : 0.0312
  routing        : fred_fusion
  detector       : fred_fusion
  p_fake         : 0.9871
  verdict        : FAKE
==================================================
```

### Command-line options

| Argument | Description | Default |
|---|---|---|
| `--mode` | Execution mode: `full`, `train_router`, `evaluate`, `infer` | `full` |
| `--image` | Image path for `--mode infer` | — |
| `--router_dir` | Load existing router from directory | Auto-detected |
| `--fred_dir` | Override FRED-Fusion model directory | From config |
| `--fred_fold` | Override FRED fold name | `RVF10K` |
| `--force_reextract` | Re-extract CLIP embeddings even if cache exists | False |
| `--out_dir` | Override output directory | Auto-timestamped |

---

## Output Files

Each run creates a timestamped directory under `BASE_OUT/Run_YYYYMMDD_HHMMSS/` containing:

```
Run_20260323_111503/
├── router/
│   ├── router_logreg.pkl          # Trained router model
│   ├── router_scaler.pkl          # Router feature scaler
│   └── router_meta.json           # Router metadata and val metrics
│
├── plots/                         # All thesis-quality figures (300 DPI)
│   ├── router_roc_pr.png
│   ├── router_confusion_matrix.png
│   ├── router_confidence_distribution.png
│   ├── routing_breakdown.png
│   ├── per_detector_roc.png
│   ├── hybrid_e2e_roc.png
│   ├── fpr_tpr_operating_points.png
│   ├── calibration_reliability_diagrams.png
│   ├── abstention_auc_tradeoff.png
│   ├── per_source_auc_heatmap.png
│   └── summary_metrics_bar.png
│
├── reports/
│   └── predictions_per_image.csv  # Per-image routing and detection scores
│
├── summary/
│   ├── hybrid_framework_results.json  # Structured metrics
│   └── hybrid_framework_report.txt   # Publication-ready text report
│
└── run.log                        # Full execution log
```

---

## Data Split Design

The evaluation protocol is leak-proof by design. No evaluation source was seen during router training.

| Phase | GAN sources | Face-swap sources |
|---|---|---|
| Router training | 140K (train + valid splits) | OpenForensics (Train + Val splits) |
| Evaluation | HumanFaces, RVF10K, SG3-CW14K | OpenForensics (Test split only) |

Router confidence thresholds (symmetric around 0.5, Δ=0.25):
- `P(face_swap) < 0.25` → route to FRED-Fusion
- `P(face_swap) > 0.75` → route to SBI
- `0.25 ≤ P ≤ 0.75` → ABSTAIN

---

## Citation

If you use HDRA-Fusion in your research, please cite:

```bibtex
@mastersthesis{ebeid2026hdrafusion,
  title   = {HDRA-Fusion: Hybrid Detection with Routed Architecture for
             Manipulation-Aware AI Face Forgery Detection},
  author  = {Ebeid, Omar},
  school  = {Florida Institute of Technology},
  year    = {2026}
}
```

---

## References

This work builds on the following key prior works:

- **SBI:** Shiohara & Yamasaki, "Detecting Face Forgery Videos with Self-Blended Images," CVPR 2022.
- **CLIP:** Radford et al., "Learning Transferable Visual Models From Natural Language Supervision," ICML 2021.
- **DANN:** Ganin et al., "Domain-Adversarial Training of Neural Networks," JMLR 2016.
- **XGBoost:** Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System," KDD 2016.
- **OpenForensics:** Le et al., "OpenForensics: Large-Scale Challenging Dataset For Multi-Face Forgery Detection And Segmentation In-The-Wild," ICCV 2021.

---

## License

This repository is released for academic research purposes. The FRED-Fusion components and HDRA-Fusion routing architecture are original work. SBI is a separate project under its own licence — see the [SBI repository](https://github.com/mapooon/SelfBlendedImages) for terms of use.
