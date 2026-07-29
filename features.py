"""B.2 Feature descriptors + B.3 feature matching.

Descriptor: a 40x40 window around each corner, low-pass filtered and sampled
down to an 8x8 patch, then normalised to zero mean / unit variance so it is
invariant to brightness (bias) and contrast (gain).

Matching: nearest neighbour between descriptor sets, accepted only when it
clearly beats the second nearest (Lowe's ratio test), which removes ambiguous
matches in repetitive texture (windows, foliage) before RANSAC ever runs.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from harris import dist2
from imutils import to_gray

PATCH = 40      # window side in pixels
OUT = 8         # descriptor side after downsampling


def extract_descriptors(im, pts, patch=PATCH, out=OUT):
    """Return (descriptors Mx(out*out), pts_kept Mx2). Corners whose window
    falls off the image are dropped."""
    g = to_gray(im)
    sigma = (patch / out) / 2.0          # anti-alias before decimating
    gb = gaussian_filter(g, sigma)
    H, W = g.shape
    half = patch // 2
    step = patch // out

    desc, kept = [], []
    for x, y in pts:
        xi, yi = int(round(x)), int(round(y))
        if xi - half < 0 or yi - half < 0 or xi + half > W or yi + half > H:
            continue
        win = gb[yi - half:yi + half, xi - half:xi + half]
        patch8 = win[::step, ::step][:out, :out]
        v = patch8.reshape(-1)
        v = v - v.mean()
        s = v.std()
        if s < 1e-6:
            continue
        desc.append(v / s)
        kept.append((x, y))
    return np.array(desc), np.array(kept, dtype=np.float64)


def match_features(desc1, desc2, ratio=0.65):
    """Match by nearest neighbour + Lowe ratio test.

    Returns an array of (i, j) index pairs into desc1 / desc2.
    """
    D = dist2(desc1, desc2)                 # (M1, M2) squared distances
    matches = []
    nn = np.argsort(D, axis=1)[:, :2]
    for i in range(D.shape[0]):
        j1, j2 = nn[i]
        d1, d2 = D[i, j1], D[i, j2]
        if d2 > 0 and (d1 / d2) < ratio * ratio:   # ratio on squared distances
            matches.append((i, j1))
    return np.array(matches, dtype=int)


def matched_points(pts1, pts2, matches):
    """Convenience: split a match list into two aligned Nx2 point arrays."""
    if len(matches) == 0:
        return np.empty((0, 2)), np.empty((0, 2))
    return pts1[matches[:, 0]], pts2[matches[:, 1]]


def draw_matches(ax, im1, im2, p1, p2):
    """Draw two images side by side with lines connecting matched points."""
    h = max(im1.shape[0], im2.shape[0])
    canvas = np.ones((h, im1.shape[1] + im2.shape[1], 3))
    canvas[:im1.shape[0], :im1.shape[1]] = im1[..., :3]
    canvas[:im2.shape[0], im1.shape[1]:] = im2[..., :3]
    off = im1.shape[1]
    ax.imshow(canvas)
    ax.scatter(p1[:, 0], p1[:, 1], s=8, c="yellow", edgecolors="none")
    ax.scatter(p2[:, 0] + off, p2[:, 1], s=8, c="yellow", edgecolors="none")
    for (x1, y1), (x2, y2) in zip(p1, p2):
        ax.plot([x1, x2 + off], [y1, y2], "-", c="deepskyblue", lw=0.5, alpha=0.7)
    ax.set_xticks([]); ax.set_yticks([])


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from imutils import load_rgb
    from corners import anms, harris_corners

    im1, im2 = load_rgb("data/chateau_1.jpg"), load_rgb("data/chateau_2.jpg")
    r = []
    for im in (im1, im2):
        p, s = harris_corners(im)
        kp, ks, _ = anms(p, s, n_keep=500)
        d, pk = extract_descriptors(im, kp)
        r.append((d, pk))
    m = match_features(r[0][0], r[1][0])
    p1, p2 = matched_points(r[0][1], r[1][1], m)
    print(f"descriptors: {len(r[0][0])}, {len(r[1][0])}  ->  {len(m)} Lowe matches")
    fig, ax = plt.subplots(figsize=(14, 4))
    draw_matches(ax, im1, im2, p1, p2)
    ax.set_title(f"{len(m)} matches (Lowe ratio)")
    fig.tight_layout(); fig.savefig("results/_match_demo.png", dpi=90)
    print("saved results/_match_demo.png")
