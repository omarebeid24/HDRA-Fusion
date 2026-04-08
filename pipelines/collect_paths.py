"""
Dataset path collection utilities for HDRA-Fusion pipelines.

Walks dataset config dicts (as defined in configs/config.py) and collects
image file paths with their associated ground-truth labels and metadata.
"""

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from configs.config import SEED

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def collect_image_paths(
    datasets: List[Dict[str, Any]],
    manip_label: int,
) -> Tuple[List[str], List[int], List[int], List[str]]:
    """
    Walk dataset config dicts and collect image paths with labels.

    Parameters
    ----------
    datasets    : list of dataset config dicts from configs/config.py.
                  Each dict must contain 'name', 'real_path', 'fake_path'.
                  Optional 'max_samples' caps images per split.
    manip_label : manipulation type label for all images in this call.
                  0 = GAN-synthetic, 1 = face-swap

    Returns
    -------
    paths        : list of image file paths
    fake_labels  : ground-truth real/fake labels (0=real, 1=fake)
    manip_labels : manipulation type labels (all equal to manip_label)
    sources      : dataset source name per image
    """
    paths, fake_labels, manip_labels, sources = [], [], [], []

    for cfg in datasets:
        name = cfg.get("name", "UNK")
        mx   = cfg.get("max_samples", None)

        for split_path, fake_lbl in [
            (cfg.get("real_path"), 0),
            (cfg.get("fake_path"), 1),
        ]:
            if not split_path:
                continue
            if not os.path.exists(split_path):
                print(
                    f"[WARNING] Path not found, skipping: {split_path}\n"
                    f"  Update '{name}' in configs/config.py."
                )
                continue

            files = [
                str(p) for p in Path(split_path).rglob("*")
                if p.suffix.lower() in SUPPORTED_EXTENSIONS
            ]
            if not files:
                print(f"[WARNING] No images found in: {split_path}")
                continue

            if mx and len(files) > mx:
                random.seed(SEED)
                files = random.sample(files, mx)

            paths.extend(files)
            fake_labels.extend([fake_lbl] * len(files))
            manip_labels.extend([manip_label] * len(files))
            sources.extend([name] * len(files))

    return paths, fake_labels, manip_labels, sources
