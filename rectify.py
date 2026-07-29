"""A.3 Rectification.

Take a planar rectangle photographed at an angle, declare its four corners to
be a true axis-aligned rectangle, solve the homography that does that, and warp
the whole image through it. The plane comes out fronto-parallel -- the laptop
screen / building facade looks as if shot head-on.

This is the geometric sanity check for computeH + warp: only 4 correspondences,
and the "answer" is judged by eye (do the receding lines become parallel?).
"""

import os

import numpy as np

from homography import apply_H, computeH
from imutils import load_rgb, save_rgb
from warp import warpImageBilinear

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RESULTS = os.path.join(HERE, "results")

# 4 corners of a real rectangle, in TL, TR, BR, BL order, plus the true aspect
# (width:height) of that rectangle so the rectified output keeps natural shape.
CONFIGS = {
    "desk":   {"img": "rectify_desk.jpg",
               "quad": [(70, 142), (918, 50), (915, 282), (182, 297)],
               "aspect": 16 / 10, "label": "laptop screen"},
    "bruges": {"img": "rectify_bruges.jpg",
               "quad": [(658, 52), (845, 78), (848, 218), (655, 228)],
               "aspect": 190 / 158, "label": "building facade"},
}


def rectify(cfg):
    im = load_rgb(os.path.join(DATA, cfg["img"]))
    quad = np.array(cfg["quad"], float)
    # target rectangle: keep roughly the source's pixel scale, honour true aspect
    side = np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1)
    diag = 0.5 * (side[0] + side[2] + side[1] + side[3]) / 2  # mean of w's and h's
    w_r = np.hypot(*(quad[1] - quad[0]))
    h_r = w_r / cfg["aspect"]
    dst = np.array([[0, 0], [w_r, 0], [w_r, h_r], [0, h_r]], float)

    H = computeH(quad, dst)                        # photo -> rectified plane
    # render a window around the rectified rectangle (with context margin)
    mx, my = 0.55 * w_r, 0.55 * h_r
    bbox = (int(-mx), int(-my), int(w_r + mx), int(h_r + my))
    out, _ = warpImageBilinear(im, H, bbox=bbox)
    rectified = out[..., :3]
    save_rgb(os.path.join(RESULTS, f"rectify_{cfg['img'].split('_')[1]}"), rectified)
    return im, quad, rectified, dst


def _draw_quad(im, quad):
    """Return a copy of im with the selected quad drawn (thin red polyline)."""
    out = im.copy()
    q = np.vstack([quad, quad[0]])
    for a, b in zip(q[:-1], q[1:]):
        for t in np.linspace(0, 1, 400):
            x, y = (a * (1 - t) + b * t).astype(int)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < out.shape[0] and 0 <= xx < out.shape[1]:
                        out[yy, xx] = [1, 0, 0]
    return out


def main():
    os.makedirs(RESULTS, exist_ok=True)
    for name, cfg in CONFIGS.items():
        im, quad, rect, dst = rectify(cfg)
        marked = _draw_quad(im, quad)
        save_rgb(os.path.join(RESULTS, f"rectify_{name}_input.jpg"), marked)
        print(f"[ok] rectified '{name}' ({cfg['label']}) -> "
              f"input+result saved, output {rect.shape[1]}x{rect.shape[0]}")


if __name__ == "__main__":
    main()
