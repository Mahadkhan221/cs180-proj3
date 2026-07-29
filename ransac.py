"""B.4 Robust homography estimation with 4-point RANSAC.

Lowe matching still leaves some wrong pairs (repetitive windows, foliage). A
single least-squares fit over all of them would be dragged off by those
outliers. RANSAC instead samples minimal 4-point subsets, fits a homography
from each, and keeps the one whose model the most matches agree with (inliers
within a pixel threshold); the final H is refit on all inliers.
"""

import numpy as np

from homography import apply_H, computeH


def ransac_homography(p1, p2, n_iter=2000, thresh=3.0, seed=0):
    """Estimate H mapping p1 -> p2 robustly.

    p1, p2 : (N, 2) matched point arrays.
    thresh : inlier reprojection distance in pixels.
    Returns (H, inlier_mask) or (None, all-False) if it can't fit.
    """
    n = len(p1)
    rng = np.random.default_rng(seed)
    if n < 4:
        return None, np.zeros(n, bool)

    best_inliers = np.zeros(n, bool)
    best_count = 0
    for _ in range(n_iter):
        idx = rng.choice(n, 4, replace=False)
        try:
            H = computeH(p1[idx], p2[idx])
        except (np.linalg.LinAlgError, ValueError):
            continue
        proj = apply_H(H, p1)
        d = np.linalg.norm(proj - p2, axis=1)
        inliers = d < thresh
        count = int(inliers.sum())
        if count > best_count:
            best_count, best_inliers = count, inliers

    if best_count < 4:
        return None, best_inliers
    H = computeH(p1[best_inliers], p2[best_inliers])   # refit on all inliers
    return H / H[2, 2], best_inliers


if __name__ == "__main__":
    import json
    import numpy as np
    from corners import anms, harris_corners
    from features import extract_descriptors, match_features, matched_points
    from imutils import load_rgb

    im1, im2 = load_rgb("data/chateau_1.jpg"), load_rgb("data/chateau_2.jpg")
    feats = []
    for im in (im1, im2):
        p, s = harris_corners(im)
        kp, ks, _ = anms(p, s, 500)
        feats.append(extract_descriptors(im, kp))
    m = match_features(feats[0][0], feats[1][0])
    p1, p2 = matched_points(feats[0][1], feats[1][1], m)
    H, inl = ransac_homography(p1, p2)
    Ht = np.array(json.load(open("points/chateau_12.json"))["H_true"])
    print(f"matches {len(p1)}, inliers {inl.sum()}")
    print("auto H vs ground truth, max abs diff:", np.abs(H - Ht).max())
