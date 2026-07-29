"""Part B end-to-end: automatic homography between two images, and automatic
stitching of a whole set. No hand-clicked points anywhere in this path.

    detect (Harris) -> ANMS -> describe -> Lowe match -> RANSAC -> warp/blend

The warp + feather blend are reused verbatim from Part A (mosaic.py), so the
only thing that changed between the manual and automatic panoramas is *where
the correspondences come from*.
"""

import numpy as np

from corners import anms, harris_corners
from features import extract_descriptors, match_features, matched_points
from mosaic import chain_to_reference, stitch
from ransac import ransac_homography


def features_for(im, n_keep=500):
    pts, strengths = harris_corners(im)
    kept, kstr, _ = anms(pts, strengths, n_keep=n_keep)
    desc, dpts = extract_descriptors(im, kept)
    return desc, dpts


def auto_homography(im1, im2, ratio=0.65, thresh=3.0, n_keep=500):
    """Automatic H mapping im1 -> im2, with pipeline diagnostics."""
    d1, p1 = features_for(im1, n_keep)
    d2, p2 = features_for(im2, n_keep)
    matches = match_features(d1, d2, ratio=ratio)
    mp1, mp2 = matched_points(p1, p2, matches)
    H, inliers = ransac_homography(mp1, mp2, thresh=thresh)
    return H, {"n_desc": (len(d1), len(d2)), "n_matches": len(matches),
               "n_inliers": int(inliers.sum()), "mp1": mp1, "mp2": mp2,
               "inliers": inliers}


def auto_stitch(images, ref=None, ratio=0.65, thresh=3.0):
    """Stitch a sequence of overlapping images automatically.

    Returns (mosaic, mask, bbox, per-pair diagnostics list).
    """
    n = len(images)
    ref = n // 2 if ref is None else ref
    Hs_pair, diags = [], []
    for k in range(n - 1):
        H, diag = auto_homography(images[k], images[k + 1], ratio, thresh)
        if H is None:
            raise RuntimeError(f"RANSAC failed on pair {k}->{k+1}")
        Hs_pair.append(H)
        diags.append(diag)
    Hs = chain_to_reference(Hs_pair, ref)
    mosaic, mask, bbox = stitch(images, Hs)
    return mosaic, mask, bbox, diags
