"""
Raw CLIP ViT-B/32 feature extractor for the HDRA-Fusion manipulation-type router.

Embeddings are extracted WITHOUT any fine-tuning of the CLIP backbone,
then L2-normalised following the standard CLIP inference convention.
The router learns manipulation modality from general visual semantics
rather than from detection-biased features.
"""

import os
from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import clip
    _HAS_CLIP = True
except ImportError:
    _HAS_CLIP = False
    print(
        "[WARNING] CLIP not installed. "
        "Run: pip install git+https://github.com/openai/CLIP.git"
    )

from configs.config import CLIP_BATCH_SIZE, CLIP_MODEL_NAME, DEVICE


class SimpleImageDataset(Dataset):
    """
    Lightweight image dataset for CLIP feature extraction and SBI inference.

    Returns (transformed_image, path_string) tuples.
    Falls back to a grey blank image if a file cannot be opened.
    """

    def __init__(self, paths: List[str], transform):
        self.paths     = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img), str(path)
        except Exception:
            blank = Image.new("RGB", (224, 224), (128, 128, 128))
            return self.transform(blank), str(path)


@torch.no_grad()
def extract_clip_embeddings_raw(
    paths: List[str],
    cache_path: str,
    device: str = DEVICE,
    batch_size: int = CLIP_BATCH_SIZE,
    force: bool = False,
) -> np.ndarray:
    """
    Extract raw (untuned) CLIP ViT-B/32 image embeddings.

    Embeddings are L2-normalised and cached to disk as a .npy file.
    On subsequent calls the cache is loaded directly, skipping GPU inference.

    Parameters
    ----------
    paths      : list of image file paths
    cache_path : .npy path where embeddings will be saved / loaded from
    device     : torch device string
    batch_size : number of images per inference batch
    force      : if True, re-extract even if cache exists

    Returns
    -------
    np.ndarray, shape (N, 512), dtype float32 — L2-normalised embeddings
    """
    if os.path.exists(cache_path) and not force:
        print(f"[Router/CLIP] Loading cached embeddings → {cache_path}")
        return np.load(cache_path)

    assert _HAS_CLIP, (
        "CLIP not installed. "
        "Run: pip install git+https://github.com/openai/CLIP.git"
    )

    model, preprocess = clip.load(CLIP_MODEL_NAME, device=device)
    model.eval()

    ds = SimpleImageDataset(paths, preprocess)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=0, pin_memory=False)

    all_embs = []
    for imgs, _ in tqdm(dl, desc="[Router/CLIP] Extracting embeddings"):
        imgs  = imgs.to(device)
        feats = model.encode_image(imgs).float().cpu().numpy()
        norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
        all_embs.append(feats / norms)

    embs = np.concatenate(all_embs, axis=0).astype(np.float32)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, embs)
    print(f"[Router/CLIP] Saved embeddings → {cache_path}  shape={embs.shape}")
    return embs
