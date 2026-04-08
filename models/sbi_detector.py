"""
SBI Detector — Detector B in HDRA-Fusion.

Wrapper around the pre-trained Self-Blended Images (SBI) EfficientNet-B4 model.
Used as the face-swap specialist in the HDRA-Fusion routing framework.

SBI is a pretrained model included without modification to demonstrate that
the HDRA-Fusion routing architecture can effectively leverage existing
specialist detectors. See Shiohara & Yamasaki (CVPR 2022) for the original
training methodology.

The FFraw checkpoint (trained on uncompressed FaceForensics++) is used
as it offers maximum discriminative capacity for the high-quality face
crops in the OpenForensics FaceCrops evaluation set.

Setup:
    1. Clone: https://github.com/mapooon/SelfBlendedImages
    2. Place SBI_REPO_ROOT in config.py to point to the cloned repo
    3. Download FFraw.tar weights and place in SBI_WEIGHTS_DIR

References
----------
[1] Shiohara & Yamasaki (2022). Detecting Face Forgery Videos with
    Self-Blended Images. CVPR.
[2] Tan & Le (2019). EfficientNet: Rethinking Model Scaling for
    Convolutional Neural Networks. ICML.
"""

import os
import sys
from typing import List

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config import (
    DEVICE,
    SBI_BATCH_SIZE,
    SBI_CHECKPOINT_CANDIDATES,
    SBI_IMAGE_SIZE,
    SBI_REPO_ROOT,
)
from models.clip_extractor import SimpleImageDataset


class SBIDetector:
    """
    Wrapper around the pretrained SBI EfficientNet-B4 face-swap detector.

    Exposes a unified .predict(paths) interface compatible with
    the HDRA-Fusion evaluation pipeline.

    The model outputs 2-class logits; softmax[:,1] is taken as p(fake).

    Parameters
    ----------
    repo_root : path to the cloned SelfBlendedImages repository
    """

    def __init__(self, repo_root: str = SBI_REPO_ROOT):
        self.repo_root = repo_root
        self.model     = None
        self._loaded   = False
        self._device   = DEVICE
        self.transform = transforms.Compose([
            transforms.Resize((SBI_IMAGE_SIZE, SBI_IMAGE_SIZE)),
            transforms.ToTensor(),
        ])

    def _find_checkpoint(self) -> str:
        """Return the first available SBI checkpoint from the candidate list."""
        for c in SBI_CHECKPOINT_CANDIDATES:
            if c and os.path.exists(c):
                return c
        raise FileNotFoundError(
            "No SBI checkpoint found. Expected one of:\n"
            + "\n".join(f"  {c}" for c in SBI_CHECKPOINT_CANDIDATES if c)
            + "\n\nDownload FFraw.tar from the SBI repository and set "
            "SBI_WEIGHTS_DIR in configs/config.py."
        )

    def load(self, device: str = DEVICE) -> None:
        """
        Load the SBI model into memory.

        Adds the SelfBlendedImages/src directory to sys.path so that
        the SBI Detector class can be imported.

        Called automatically on first .predict() call if not already loaded.
        """
        print(f"[SBI] Loading from repo: {self.repo_root}")
        src_dir = os.path.join(self.repo_root, "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        try:
            from model import Detector  # type: ignore  # noqa: F401
        except ImportError as e:
            raise ImportError(
                f"Failed to import SBI Detector from {src_dir}/model.py.\n"
                "Ensure the SelfBlendedImages repo is cloned and its "
                "dependencies are installed.\n"
                f"Original error: {e}"
            )

        ckpt_path = self._find_checkpoint()
        print(f"[SBI] Checkpoint: {ckpt_path}")

        self.model = Detector().to(device)
        ckpt  = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model", ckpt)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

        self._device = device
        self._loaded = True
        print(f"[SBI] Model loaded — device={device}")

    @torch.no_grad()
    def predict(
        self,
        paths: List[str],
        batch_size: int = SBI_BATCH_SIZE,
    ) -> np.ndarray:
        """
        Run SBI inference on a list of image paths.

        Parameters
        ----------
        paths      : list of image file paths (face-cropped, 380×380 preferred)
        batch_size : number of images per inference batch

        Returns
        -------
        np.ndarray, shape (N,) — p(fake) probabilities from softmax[:,1]
        """
        if not self._loaded:
            self.load()

        ds = SimpleImageDataset(paths, self.transform)
        dl = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=torch.cuda.is_available(),
        )

        probs = []
        for imgs, _ in tqdm(dl, desc="[SBI] Inference"):
            imgs   = imgs.to(self._device)
            logits = self.model(imgs)
            p      = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())

        return np.array(probs, dtype=np.float32)
