"""Projective geometry core for Project 3.

computeH: recover a 3x3 homography from >=4 point correspondences by least
squares. apply_H: push 2-D points through a homography. No cv2/skimage
homography helpers are used here -- only numpy linear algebra, which the
rubric explicitly allows.
"""

import numpy as np


def computeH(src_pts, dst_pts):
    """Recover H (3x3) mapping src_pts -> dst_pts.

    Each correspondence (x, y) -> (x', y') contributes two rows to an
    over-determined linear system in the 8 free parameters of H (h33 fixed
    to 1). With >=4 points the system is solved by least squares, so extra
    correspondences make the estimate more stable rather than break it.

    src_pts, dst_pts : (N, 2) arrays of matching (x, y) coordinates.
    Returns a (3, 3) homography with H[2, 2] == 1.
    """
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError("src_pts and dst_pts must both be (N, 2)")
    n = src.shape[0]
    if n < 4:
        raise ValueError(f"need >=4 correspondences, got {n}")

    A = np.zeros((2 * n, 8), dtype=np.float64)
    b = np.zeros((2 * n,), dtype=np.float64)
    for i in range(n):
        x, y = src[i]
        xp, yp = dst[i]
        # xp = (h11 x + h12 y + h13) / (h31 x + h32 y + 1)
        A[2 * i] = [x, y, 1, 0, 0, 0, -x * xp, -y * xp]
        b[2 * i] = xp
        # yp = (h21 x + h22 y + h23) / (h31 x + h32 y + 1)
        A[2 * i + 1] = [0, 0, 0, x, y, 1, -x * yp, -y * yp]
        b[2 * i + 1] = yp

    h, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.array([
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], 1.0],
    ])


def apply_H(H, pts):
    """Apply homography H to (N, 2) points, returning (N, 2) points.

    Points are lifted to homogeneous coordinates, transformed, and divided
    back through the third coordinate.
    """
    pts = np.asarray(pts, dtype=np.float64)
    single = pts.ndim == 1
    if single:
        pts = pts[None, :]
    homog = np.hstack([pts, np.ones((pts.shape[0], 1))])
    warped = homog @ H.T
    w = warped[:, 2:3]
    # guard against division by ~0 for points sent to infinity
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    out = warped[:, :2] / w
    return out[0] if single else out


if __name__ == "__main__":
    # quick smoke test; the real assertions live in test_homography.py
    H_true = np.array([[1.2, 0.1, 30.0],
                       [0.05, 1.1, -20.0],
                       [0.0003, -0.0002, 1.0]])
    pts = np.array([[0, 0], [100, 0], [100, 80], [0, 80], [40, 55]], float)
    mapped = apply_H(H_true, pts)
    H_est = computeH(pts, mapped)
    print("max |H_true - H_est| =", np.abs(H_true - H_est).max())
