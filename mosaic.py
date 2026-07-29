"""Warp a set of images into one canvas and blend them into a mosaic.

The heavy lifting (homography + inverse warp) lives in homography.py / warp.py.
Here we:
  * compute the union bounding box of all warped images,
  * warp each image (carrying a feather weight as an extra channel), and
  * composite by weighted averaging so seams disappear.

The same stitch() is reused by Part A (H from hand-clicked points) and Part B
(H from RANSAC), which is the whole point of factoring it out.
"""

import numpy as np

from warp import warpImageBilinear, warped_corners


def feather_weight(h, w):
    """Weight ~1 at the image centre, ramping linearly to 0 at every edge."""
    ys = 1.0 - np.abs(np.linspace(-1, 1, h))
    xs = 1.0 - np.abs(np.linspace(-1, 1, w))
    wgt = np.outer(ys, xs)
    return np.clip(wgt, 1e-3, 1.0)


def _union_bbox(images, Hs):
    corners = []
    for im, H in zip(images, Hs):
        corners.append(warped_corners(im.shape, H))
    corners = np.concatenate(corners, axis=0)
    xmin, ymin = np.floor(corners.min(axis=0)).astype(int)
    xmax, ymax = np.ceil(corners.max(axis=0)).astype(int) + 1
    return int(xmin), int(ymin), int(xmax), int(ymax)


def stitch(images, Hs, feather=True):
    """Composite images into a shared canvas.

    images : list of HxWx3 float [0,1]
    Hs     : list of 3x3 homographies mapping each image -> reference frame
    Returns (mosaic_rgb, coverage_mask, bbox).
    """
    bbox = _union_bbox(images, Hs)
    xmin, ymin, xmax, ymax = bbox
    out_h, out_w = ymax - ymin, xmax - xmin

    num = np.zeros((out_h, out_w, 3), np.float64)
    den = np.zeros((out_h, out_w), np.float64)
    for im, H in zip(images, Hs):
        h, w = im.shape[:2]
        wgt = feather_weight(h, w) if feather else np.ones((h, w))
        rgbw = np.concatenate([im[..., :3], wgt[..., None]], axis=2)
        warped, _ = warpImageBilinear(rgbw, H, bbox=bbox)
        rgb = warped[..., :3]
        blend = warped[..., 3] * warped[..., 4]        # feather * coverage
        num += rgb * blend[..., None]
        den += blend
    mask = den > 1e-6
    mosaic = np.zeros_like(num)
    mosaic[mask] = num[mask] / den[mask, None]
    return mosaic, mask.astype(np.float64), bbox


def stitch_overwrite(images, Hs):
    """Naive last-writer-wins composite -- keeps a visible seam, for contrast."""
    bbox = _union_bbox(images, Hs)
    xmin, ymin, xmax, ymax = bbox
    out_h, out_w = ymax - ymin, xmax - xmin
    canvas = np.zeros((out_h, out_w, 3), np.float64)
    for im, H in zip(images, Hs):
        warped, _ = warpImageBilinear(im[..., :3], H, bbox=bbox)
        cov = warped[..., -1] > 0.5
        canvas[cov] = warped[..., :3][cov]
    return canvas, bbox


def chain_to_reference(Hs_pairwise, ref):
    """Turn adjacent pairwise homographies into per-image maps to a reference.

    Hs_pairwise[k] maps image k -> image k+1. ref is the index whose frame the
    mosaic lives in. Returns a list H_to_ref where H_to_ref[k] maps image k ->
    reference frame.
    """
    n = len(Hs_pairwise) + 1
    to_ref = [None] * n
    to_ref[ref] = np.eye(3)
    for k in range(ref, 0, -1):            # walk left of the reference
        H = Hs_pairwise[k - 1]             # image k-1 -> image k
        to_ref[k - 1] = to_ref[k] @ H
    for k in range(ref, n - 1):            # walk right of the reference
        Hinv = np.linalg.inv(Hs_pairwise[k])  # image k+1 -> image k
        to_ref[k + 1] = to_ref[k] @ Hinv
    return [H / H[2, 2] for H in to_ref]
