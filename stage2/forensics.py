"""
Image-forensics features for bill images — no neural network.

A scanned bill carries a fairly uniform layer of sensor/paper noise. When
someone pastes a patch over a printed value (white-out and retype), the
patch usually lacks that noise, so it shows up as a cluster of unusually
"quiet" background blocks. This noise-inconsistency check is a standard
image-forensics cue, and unlike the CASIA-trained CNN it works at full
resolution, so a one-line edit isn't shrunk away.

Two features, both computed on background (non-text) blocks:
  noise_block_min_z  how far the quietest block falls below the page's
                     typical noise level, in robust z-units
  noise_low_frac     fraction of background blocks with less than half
                     the typical noise

Used by fuse_features.py (training data), predict.py (new claims), and
ela_diagnostic.py, so all three compute exactly the same thing.
"""

import numpy as np
from PIL import Image, ImageFilter

BLOCK = 16             # block size in pixels (full resolution)
INK_THRESHOLD = 160    # grayscale below this counts as text
MIN_INK_FRAC = 0.02    # blocks with less ink than this count as background

FORENSIC_FEATURES = ["noise_block_min_z", "noise_low_frac"]


def forensic_features(image):
    """image: a file path or a PIL image. Returns a dict with the
    FORENSIC_FEATURES (NaN if the page has too little background)."""
    img = image if isinstance(image, Image.Image) else Image.open(image)
    gray_img = img.convert("L")
    gray = np.asarray(gray_img, dtype=np.float32)
    resid = gray - np.asarray(gray_img.filter(ImageFilter.MedianFilter(3)), dtype=np.float32)

    h, w = gray.shape
    hb, wb = h // BLOCK, w // BLOCK
    crop = (slice(0, hb * BLOCK), slice(0, wb * BLOCK))
    block_noise = resid[crop].reshape(hb, BLOCK, wb, BLOCK).std(axis=(1, 3))
    block_ink = (gray[crop] < INK_THRESHOLD).reshape(hb, BLOCK, wb, BLOCK).mean(axis=(1, 3))
    bg = block_noise[block_ink < MIN_INK_FRAC]

    if bg.size < 5:
        return {k: np.nan for k in FORENSIC_FEATURES}
    med = np.median(bg)
    mad = np.median(np.abs(bg - med)) * 1.4826 + 1e-6
    return {
        "noise_block_min_z": float((med - bg.min()) / mad),
        "noise_low_frac": float((bg < 0.5 * med).mean()),
    }
