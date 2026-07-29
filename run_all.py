"""Run the whole project end to end and write every figure into results/.

    python run_all.py

Produces, for each image set: hand-clicked correspondences, the manual mosaic
(feather vs naive seam), ANMS corners, Lowe matches, RANSAC inliers, the
automatic mosaic, and a manual-vs-auto comparison. Also the rectifications,
the nearest-neighbour vs bilinear warp comparison, and sample descriptor
patches. Every headline number is written to results/summary.json so the
writeup can never drift from the code.
"""

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import rectify
from autostitch import auto_homography, features_for
from corners import anms, draw_points, harris_corners
from features import (draw_matches, extract_descriptors, match_features,
                      matched_points)
from homography import apply_H, computeH
from imutils import load_rgb, save_rgb
from mosaic import chain_to_reference, stitch, stitch_overwrite
from ransac import ransac_homography
from warp import warpImageBilinear, warpImageNearestNeighbor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, POINTS, RESULTS = (os.path.join(HERE, d) for d in ("data", "points", "results"))
SETS = ["chateau", "mountains", "forest"]


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, name), dpi=95, bbox_inches="tight")
    plt.close(fig)


def load_pair(name):
    with open(os.path.join(POINTS, f"{name}.json")) as fh:
        return json.load(fh)


def herr(H, Ht):
    """Comparable homography error: max abs diff after normalising H33=1."""
    return float(np.abs(H / H[2, 2] - Ht / Ht[2, 2]).max())


# --------------------------------------------------------------------------- #
def part_a(name, images, summary):
    p12, p23 = load_pair(f"{name}_12"), load_pair(f"{name}_23")
    H12 = computeH(np.array(p12["pts_i"]), np.array(p12["pts_j"]))
    H23 = computeH(np.array(p23["pts_i"]), np.array(p23["pts_j"]))

    # correspondence visualisation for the first pair
    fig, ax = plt.subplots(figsize=(13, 4))
    draw_matches(ax, images[0], images[1],
                 np.array(p12["pts_i"]), np.array(p12["pts_j"]))
    ax.set_title(f"{name}: hand-clicked correspondences (pair 1-2)")
    savefig(fig, f"{name}_corr.png")

    Hs = chain_to_reference([H12, H23], ref=1)
    mosaic, _, _ = stitch(images, Hs)
    naive, _ = stitch_overwrite(images, Hs)
    save_rgb(os.path.join(RESULTS, f"{name}_manual.jpg"), mosaic)
    save_rgb(os.path.join(RESULTS, f"{name}_naive.jpg"), naive)

    summary[name]["manual_H12_err_vs_truth"] = herr(H12, np.array(p12["H_true"]))
    summary[name]["n_manual_points"] = len(p12["pts_i"])
    return Hs


def part_b(name, images, summary):
    feats = [features_for(im) for im in images]        # (desc, pts) per image
    # raw vs ANMS on the middle image
    praw, sraw = harris_corners(images[1])
    kept, _, _ = anms(praw, sraw, 500)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    draw_points(axes[0], images[1], praw, s=2)
    axes[0].set_title(f"raw Harris ({len(praw)})")
    draw_points(axes[1], images[1], kept, s=9)
    axes[1].set_title(f"after ANMS ({len(kept)})")
    savefig(fig, f"{name}_anms.png")

    # pairwise auto homographies + match/inlier viz on pair 1-2
    Hs_pair, diags = [], []
    for k in range(len(images) - 1):
        H, diag = auto_homography(images[k], images[k + 1])
        Hs_pair.append(H)
        diags.append(diag)

    d0 = diags[0]
    mp1, mp2, inl = d0["mp1"], d0["mp2"], d0["inliers"]
    fig, ax = plt.subplots(figsize=(13, 4))
    draw_matches(ax, images[0], images[1], mp1, mp2)
    ax.set_title(f"{name}: {len(mp1)} Lowe matches")
    savefig(fig, f"{name}_matches.png")

    fig, ax = plt.subplots(figsize=(13, 4))
    off = images[0].shape[1]
    h = max(images[0].shape[0], images[1].shape[0])
    canvas = np.ones((h, images[0].shape[1] + images[1].shape[1], 3))
    canvas[:images[0].shape[0], :off] = images[0][..., :3]
    canvas[:images[1].shape[0], off:] = images[1][..., :3]
    ax.imshow(canvas)
    for (a, b), ok in zip(zip(mp1, mp2), inl):
        c = "lime" if ok else "red"
        ax.plot([a[0], b[0] + off], [a[1], b[1]], "-", c=c, lw=0.6,
                alpha=0.9 if ok else 0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{name}: RANSAC inliers (green {int(inl.sum())}) vs "
                 f"outliers (red {int((~inl).sum())})")
    savefig(fig, f"{name}_ransac.png")

    Hs = chain_to_reference(Hs_pair, ref=1)
    mosaic, _, _ = stitch(images, Hs)
    save_rgb(os.path.join(RESULTS, f"{name}_auto.jpg"), mosaic)

    Ht = np.array(load_pair(f"{name}_12")["H_true"])
    summary[name]["auto_H12_err_vs_truth"] = herr(Hs_pair[0], Ht)
    summary[name]["n_matches"] = d0["n_matches"]
    summary[name]["n_inliers"] = d0["n_inliers"]
    summary[name]["n_raw_corners"] = int(len(praw))
    return mosaic


def comparison(name):
    man = load_rgb(os.path.join(RESULTS, f"{name}_manual.jpg"))
    aut = load_rgb(os.path.join(RESULTS, f"{name}_auto.jpg"))
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].imshow(man); axes[0].set_title(f"{name}: manual (hand-clicked)")
    axes[1].imshow(aut); axes[1].set_title(f"{name}: automatic (Harris+RANSAC)")
    for a in axes:
        a.set_xticks([]); a.set_yticks([])
    savefig(fig, f"{name}_compare.png")


def warp_quality_demo(summary):
    """A.3 deliverable: nearest-neighbour vs bilinear (quality + speed)."""
    im = load_rgb(os.path.join(DATA, "chateau_1.jpg"))
    p = load_pair("chateau_12")
    H = computeH(np.array(p["pts_i"]), np.array(p["pts_j"]))
    t0 = time.time(); nn, bb = warpImageNearestNeighbor(im, H); t_nn = time.time() - t0
    t0 = time.time(); bl, _ = warpImageBilinear(im, H, bbox=bb); t_bl = time.time() - t0
    # zoom into a shared textured crop
    y, x = nn.shape[0] // 2, nn.shape[1] // 2
    sl = (slice(max(0, y - 60), y + 60), slice(max(0, x - 90), x + 90))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].imshow(np.clip(nn[sl][..., :3], 0, 1))
    axes[0].set_title(f"nearest neighbour ({t_nn*1000:.0f} ms)")
    axes[1].imshow(np.clip(bl[sl][..., :3], 0, 1))
    axes[1].set_title(f"bilinear ({t_bl*1000:.0f} ms)")
    for a in axes:
        a.set_xticks([]); a.set_yticks([])
    savefig(fig, "warp_nn_vs_bilinear.png")
    summary["warp"] = {"nn_ms": round(t_nn * 1000, 1), "bilinear_ms": round(t_bl * 1000, 1)}


def descriptor_patches_demo():
    im = load_rgb(os.path.join(DATA, "chateau_2.jpg"))
    p, s = harris_corners(im)
    kept, _, _ = anms(p, s, 500)
    desc, dpts = extract_descriptors(im, kept)
    idx = np.linspace(0, len(desc) - 1, 8).astype(int)
    fig, axes = plt.subplots(1, 8, figsize=(12, 1.8))
    for a, i in zip(axes, idx):
        a.imshow(desc[i].reshape(8, 8), cmap="gray")
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("sample 8x8 descriptors (normalised)")
    savefig(fig, "descriptor_patches.png")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    summary = {s: {} for s in SETS}
    t_start = time.time()
    for name in SETS:
        images = [load_rgb(os.path.join(DATA, f"{name}_{i}.jpg")) for i in (1, 2, 3)]
        part_a(name, images, summary)
        part_b(name, images, summary)
        comparison(name)
        print(f"[ok] {name}: manual + auto mosaics, "
              f"{summary[name]['n_inliers']}/{summary[name]['n_matches']} inliers")
    rectify.main()
    warp_quality_demo(summary)
    descriptor_patches_demo()
    summary["_runtime_s"] = round(time.time() - t_start, 1)
    with open(os.path.join(RESULTS, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[done] everything in results/  ({summary['_runtime_s']}s)")


if __name__ == "__main__":
    main()
