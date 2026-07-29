"""B.1 Corner detection + Adaptive Non-Maximal Suppression.

Harris itself comes from the provided starter (harris.py) -- the one piece the
brief allows as given code. ANMS is ours: from the thousands of raw Harris
peaks, keep a fixed number that are both strong and spatially spread out, by
scoring each corner with the radius to the nearest *significantly stronger*
corner and keeping the largest such radii (Brown et al., MOPS).
"""

import numpy as np

from harris import dist2, get_harris_corners
from imutils import to_gray


def harris_corners(im, edge_discard=20):
    """Return (coords Nx2 as (x,y), strengths N) of raw Harris corners."""
    g = to_gray(im)
    h, coords = get_harris_corners(g, edge_discard=edge_discard)
    ys, xs = coords[0], coords[1]
    strengths = h[ys, xs]
    pts = np.stack([xs, ys], axis=1).astype(np.float64)   # (x, y)
    return pts, strengths


def anms(pts, strengths, n_keep=500, c_robust=0.9, prefilter=3000):
    """Adaptive Non-Maximal Suppression.

    For corner i, its suppression radius is the distance to the nearest corner
    j that is meaningfully stronger (strength_j > c_robust * strength_i). Keep
    the n_keep corners with the largest radii -- strong *and* well distributed.
    """
    if len(pts) > prefilter:                       # bound the O(n^2) step
        order = np.argsort(strengths)[::-1][:prefilter]
        pts, strengths = pts[order], strengths[order]

    D = dist2(pts, pts)                            # (n, n) squared distances
    n = len(pts)
    # valid[i, j] = j can suppress i (strictly stronger, not itself)
    valid = strengths[None, :] > c_robust * strengths[:, None]
    np.fill_diagonal(valid, False)
    D = np.where(valid, D, np.inf)
    radii = np.sqrt(D.min(axis=1))                 # inf for the global maximum

    keep = np.argsort(radii)[::-1][:n_keep]
    return pts[keep], strengths[keep], radii[keep]


def draw_points(ax, im, pts, s=6, color="lime"):
    ax.imshow(im)
    ax.scatter(pts[:, 0], pts[:, 1], s=s, c=color, edgecolors="none")
    ax.set_xticks([]); ax.set_yticks([])


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from imutils import load_rgb

    im = load_rgb("data/chateau_2.jpg")
    pts, st = harris_corners(im)
    kept, kst, kr = anms(pts, st, n_keep=500)
    print(f"raw corners: {len(pts)}  ->  ANMS: {len(kept)}")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    draw_points(axes[0], im, pts, s=2); axes[0].set_title(f"raw Harris ({len(pts)})")
    draw_points(axes[1], im, kept, s=8); axes[1].set_title(f"after ANMS ({len(kept)})")
    fig.tight_layout(); fig.savefig("results/_anms_demo.png", dpi=90)
    print("saved results/_anms_demo.png")
