"""Build the input image sets for Parts A and B.

There are no hand-shot photos here, so each "photo set" is synthesised from a
single wide real photograph (downloaded from picsum.photos, real Unsplash
imagery) by re-projecting overlapping sub-windows through *known* homographies
-- exactly the image-formation model of a camera pivoting in place. This gives:

  * real image content (rich texture for Harris corners),
  * guaranteed ~50% overlap between neighbours,
  * no black borders (each photo's pre-image stays inside the source), and
  * an exact ground-truth homography per pair, so the manually- and
    automatically-recovered H can both be checked against truth.

Provenance is stated openly in the writeup. Run: python make_data.py
"""

import json
import os

import numpy as np

from homography import apply_H, computeH
from imutils import load_rgb, save_rgb
from warp import warpImageBilinear

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
POINTS = os.path.join(HERE, "points")

# source photo -> the wide real image each set is carved from
SETS = {
    "chateau":   {"src": "sources/p_142.jpg", "yaws": (11, 0, -11), "pitch": (3, 0, -3)},
    "mountains": {"src": "sources/p_29.jpg",  "yaws": (10, 0, -10), "pitch": (-2, 0, 2)},
    "forest":    {"src": "sources/p_28.jpg",  "yaws": (12, 0, -12), "pitch": (2, 0, -2)},
}
RECTIFY_SRC = {"desk": "sources/p_180.jpg", "bruges": "sources/p_164.jpg"}


def tilt_H(cx, cy, f, yaw_deg, pitch_deg):
    """Homography K R K^-1 of a camera rotated by (yaw, pitch) about its centre."""
    y, p = np.deg2rad(yaw_deg), np.deg2rad(pitch_deg)
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    R = Ry @ Rx
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    return K @ R @ np.linalg.inv(K)


def _fit_quad(base_rect, cx, cy, f, yaw, pitch, W, H):
    """Keystone base_rect by a camera tilt, shrinking the tilt until the quad
    still fits inside the [0,W]x[0,H] source (guarantees full photo coverage)."""
    for scale in np.linspace(1.0, 0.0, 21):
        Ht = tilt_H(cx, cy, f, yaw * scale, pitch * scale)
        quad = apply_H(Ht, base_rect)
        if (quad[:, 0].min() >= 0.5 and quad[:, 0].max() <= W - 1.5 and
                quad[:, 1].min() >= 0.5 and quad[:, 1].max() <= H - 1.5):
            return quad
    return base_rect.copy()  # tilt fully collapsed; fall back to axis-aligned


def make_set(name, cfg, n=3, overlap=0.5, seed=0):
    S = load_rgb(os.path.join(DATA, cfg["src"]))
    H, W = S.shape[:2]
    rng = np.random.default_rng(seed)

    vw = int(round(W / (1 + (n - 1) * (1 - overlap))))   # window width
    step = int(round(vw * (1 - overlap)))
    my = int(round(0.12 * H))                            # vertical margin for tilt room
    vh = H - 2 * my
    photo_rect = np.array([[0, 0], [vw, 0], [vw, vh], [0, vh]], float)

    photos, A = [], []
    for i in range(n):
        x0 = min(i * step, W - vw)
        base = np.array([[x0, my], [x0 + vw, my], [x0 + vw, my + vh], [x0, my + vh]], float)
        cx, cy = x0 + vw / 2, H / 2
        quad = _fit_quad(base, cx, cy, 1.3 * vw, cfg["yaws"][i], cfg["pitch"][i], W, H)
        A_i = computeH(quad, photo_rect)                 # source -> photo i
        out, _ = warpImageBilinear(S, A_i, bbox=(0, 0, vw, vh))
        cover = out[..., -1].mean()
        assert cover > 0.999, f"{name} photo {i} only {cover:.3f} covered"
        photo = out[..., :3]
        save_rgb(os.path.join(DATA, f"{name}_{i + 1}.jpg"), photo)
        photos.append(photo)
        A.append(A_i)

    # correspondences + ground-truth H for each adjacent pair
    for i in range(n - 1):
        j = i + 1
        H_true = A[j] @ np.linalg.inv(A[i])              # photo i -> photo j
        H_true = H_true / H_true[2, 2]
        # spread a grid over photo i, keep points that also fall inside photo j
        gx, gy = np.meshgrid(np.linspace(0.1, 0.9, 5) * vw,
                             np.linspace(0.15, 0.85, 4) * vh)
        cand = np.stack([gx.ravel(), gy.ravel()], axis=1)
        mapped = apply_H(H_true, cand)
        inside = ((mapped[:, 0] > 5) & (mapped[:, 0] < vw - 5) &
                  (mapped[:, 1] > 5) & (mapped[:, 1] < vh - 5))
        pts_i = cand[inside]
        pts_j = mapped[inside]
        # keep a well-spread subset of ~12 and add sub-pixel "clicking" noise
        keep = np.linspace(0, len(pts_i) - 1, min(12, len(pts_i))).astype(int)
        pts_i = pts_i[keep] + rng.normal(0, 0.4, (len(keep), 2))
        pts_j = pts_j[keep] + rng.normal(0, 0.4, (len(keep), 2))
        with open(os.path.join(POINTS, f"{name}_{i + 1}{j + 1}.json"), "w") as fh:
            json.dump({"pts_i": pts_i.tolist(), "pts_j": pts_j.tolist(),
                       "H_true": H_true.tolist(),
                       "img_i": f"{name}_{i + 1}.jpg", "img_j": f"{name}_{j + 1}.jpg"},
                      fh, indent=2)
    return [f"{name}_{i + 1}.jpg" for i in range(n)]


def copy_rectify_sources():
    for name, src in RECTIFY_SRC.items():
        im = load_rgb(os.path.join(DATA, src))
        save_rgb(os.path.join(DATA, f"rectify_{name}.jpg"), im)


def main():
    os.makedirs(POINTS, exist_ok=True)
    for name, cfg in SETS.items():
        files = make_set(name, cfg)
        print(f"[ok] set '{name}': {', '.join(files)}")
    copy_rectify_sources()
    print("[ok] rectification sources copied")


if __name__ == "__main__":
    main()
