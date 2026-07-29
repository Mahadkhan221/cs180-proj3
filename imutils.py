"""Small image IO helpers shared across the project.

Images live as float64 RGB in [0, 1] while being processed; clipping to
[0, 1] and conversion to uint8 happens once, on save. Grayscale conversion
(for Harris) uses the standard luma weights.
"""

import numpy as np
from skimage.io import imread, imsave


def load_rgb(path):
    """Load an image as HxWx3 float64 in [0, 1]."""
    im = imread(path)
    if im.dtype == np.uint8:
        im = im.astype(np.float64) / 255.0
    else:
        im = im.astype(np.float64)
        if im.max() > 1.0:
            im = im / 255.0
    if im.ndim == 2:
        im = np.stack([im] * 3, axis=-1)
    if im.shape[2] == 4:  # drop alpha
        im = im[:, :, :3]
    return im


def save_rgb(path, im):
    """Clip to [0, 1] and save as 8-bit."""
    out = np.clip(im, 0.0, 1.0)
    imsave(path, (out * 255.0 + 0.5).astype(np.uint8))


def to_gray(im):
    """RGB [0,1] -> grayscale [0,1] with luma weights."""
    if im.ndim == 2:
        return im
    return im[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
