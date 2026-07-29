"""Synthetic validation of the homography core -- no photos required.

Mirrors Project 1's test_align.py idea: prove the math independently of any
image. Make a random H, apply it to random points, recover H from the pairs,
and assert it comes back. Run: python test_homography.py
"""

import numpy as np

from homography import apply_H, computeH


def random_homography(rng):
    """A random but well-conditioned homography (near identity + perspective)."""
    H = np.eye(3)
    H[:2, :2] += rng.uniform(-0.2, 0.2, size=(2, 2))   # rotation/shear/scale
    H[:2, 2] = rng.uniform(-50, 50, size=2)             # translation
    H[2, :2] = rng.uniform(-3e-4, 3e-4, size=2)         # perspective
    return H


def test_recovers_known_H():
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        H_true = random_homography(rng)
        pts = rng.uniform(0, 500, size=(np.random.randint(4, 12), 2))
        mapped = apply_H(H_true, pts)
        H_est = computeH(pts, mapped)
        # normalize both (H is scale-free) and compare
        err = np.abs(H_true / H_true[2, 2] - H_est / H_est[2, 2]).max()
        worst = max(worst, err)
    assert worst < 1e-6, f"H recovery error too large: {worst}"
    return worst


def test_overdetermined_is_stable():
    """More points than the minimum should still recover H cleanly."""
    rng = np.random.default_rng(1)
    H_true = random_homography(rng)
    pts = rng.uniform(0, 500, size=(50, 2))
    mapped = apply_H(H_true, pts)
    H_est = computeH(pts, mapped)
    err = np.abs(H_true / H_true[2, 2] - H_est / H_est[2, 2]).max()
    assert err < 1e-8, f"overdetermined recovery error: {err}"
    return err


def test_roundtrip_inverse():
    """Applying H then H^-1 returns the original points."""
    rng = np.random.default_rng(2)
    H = random_homography(rng)
    pts = rng.uniform(0, 500, size=(30, 2))
    back = apply_H(np.linalg.inv(H), apply_H(H, pts))
    err = np.abs(pts - back).max()
    assert err < 1e-8, f"roundtrip error: {err}"
    return err


if __name__ == "__main__":
    w1 = test_recovers_known_H()
    w2 = test_overdetermined_is_stable()
    w3 = test_roundtrip_inverse()
    print(f"[ok] recover known H (4-11 pts): worst err = {w1:.2e}")
    print(f"[ok] overdetermined (50 pts):    err = {w2:.2e}")
    print(f"[ok] warp/unwarp roundtrip:      err = {w3:.2e}")
    print("all homography tests passed.")
