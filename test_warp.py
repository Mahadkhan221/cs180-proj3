"""Synthetic validation of inverse warping -- no photos required.

Checks the samplers on cases with known answers: identity is a no-op, pure
translation shifts by the right amount, and bilinear resampling reproduces a
linear intensity ramp exactly (bilinear interpolation is exact for affine
coordinate maps over linear signals). Run: python test_warp.py
"""

import numpy as np

from warp import warpImageBilinear, warpImageNearestNeighbor


def _make_ramp(h=40, w=60):
    """Intensity = a*x + b*y + c; bilinear should reproduce this exactly."""
    ys, xs = np.mgrid[0:h, 0:w]
    return (0.3 * xs + 0.2 * ys + 5.0).astype(np.float64)


def test_identity_is_noop():
    im = _make_ramp()
    for warp in (warpImageNearestNeighbor, warpImageBilinear):
        out, (xmin, ymin, xmax, ymax) = warp(im, np.eye(3))
        assert (xmin, ymin) == (0, 0)
        assert out.shape[:2] == im.shape, f"{warp.__name__} canvas {out.shape}"
        rgb, mask = out[..., 0], out[..., -1]
        m = mask > 0
        # where a pixel is covered, identity must return it untouched
        err = np.abs(rgb[m] - im[m]).max()
        assert err < 1e-9, f"{warp.__name__} identity err {err}"
    # nearest covers every pixel; bilinear covers all but the far edge
    nn, _ = warpImageNearestNeighbor(im, np.eye(3))
    assert nn[..., -1].all(), "nearest identity should cover every pixel"
    return err


def test_translation_shifts():
    im = _make_ramp()
    H = np.array([[1, 0, 10], [0, 1, -5], [0, 0, 1]], float)  # +10 x, -5 y
    out, (xmin, ymin, xmax, ymax) = warpImageBilinear(im, H)
    assert (xmin, ymin) == (10, -5), f"bbox shifted wrong: {(xmin, ymin)}"
    # the pixel that was at source (0,0) now sits at canvas (10,-5) == out[0,0]
    err = abs(out[0, 0, 0] - im[0, 0])
    assert err < 1e-9, f"translation sample err {err}"
    return err


def test_bilinear_exact_on_ramp():
    """Under an affine warp, bilinear must reproduce a linear ramp exactly."""
    im = _make_ramp(80, 80)
    H = np.array([[1.1, 0.15, 3.0],
                  [-0.1, 1.05, -2.0],
                  [0.0, 0.0, 1.0]])
    out, bbox = warpImageBilinear(im, H)
    rgb, mask = out[..., 0], out[..., -1]
    # compare warped values against the ideal ramp evaluated at source coords
    xmin, ymin, xmax, ymax = bbox
    ys, xs = np.mgrid[ymin:ymax, xmin:xmax]
    from homography import apply_H
    dst = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(float)
    src = apply_H(np.linalg.inv(H), dst)
    ideal = (0.3 * src[:, 0] + 0.2 * src[:, 1] + 5.0).reshape(rgb.shape)
    m = mask > 0
    err = np.abs(rgb[m] - ideal[m]).max()
    assert err < 1e-7, f"bilinear not exact on ramp: {err}"
    return err


if __name__ == "__main__":
    e1 = test_identity_is_noop()
    e2 = test_translation_shifts()
    e3 = test_bilinear_exact_on_ramp()
    print(f"[ok] identity no-op:           err = {e1:.2e}")
    print(f"[ok] translation shift+sample: err = {e2:.2e}")
    print(f"[ok] bilinear exact on ramp:   err = {e3:.2e}")
    print("all warp tests passed.")
