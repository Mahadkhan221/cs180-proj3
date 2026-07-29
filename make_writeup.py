"""Build writeup.html from the figures in results/ and the numbers in
results/summary.json. Every statistic on the page is read from the code's own
output (summary.json + the synthetic unit tests), so the writeup cannot drift
from the implementation. Run: python make_writeup.py
"""

import json
import os

import numpy as np

from homography import computeH
from test_homography import (test_overdetermined_is_stable,
                             test_recovers_known_H, test_roundtrip_inverse)
from test_warp import (test_bilinear_exact_on_ramp, test_identity_is_noop,
                       test_translation_shifts)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
SETS = ["chateau", "mountains", "forest"]
TITLES = {"chateau": "Chateau de Chambord", "mountains": "Himalaya",
          "forest": "Forest valley"}


def fmt_H(H):
    rows = []
    for r in H:
        rows.append("  ".join(f"{v: .4e}" if abs(v) < 0.01 else f"{v: .4f}" for v in r))
    return "\n".join(rows)


def img(src, cap=None, cls=""):
    c = f'<figcaption>{cap}</figcaption>' if cap else ""
    return f'<figure class="{cls}"><img src="{src}" loading="lazy">{c}</figure>'


def build():
    S = json.load(open(os.path.join(RESULTS, "summary.json")))

    # live synthetic-test numbers
    hrec = test_recovers_known_H()
    hover = test_overdetermined_is_stable()
    hround = test_roundtrip_inverse()
    wid = test_identity_is_noop()
    wtr = test_translation_shifts()
    wramp = test_bilinear_exact_on_ramp()

    # an actual recovered homography to display
    p = json.load(open(os.path.join(HERE, "points", "chateau_12.json")))
    H12 = computeH(np.array(p["pts_i"]), np.array(p["pts_j"]))
    H12 = H12 / H12[2, 2]

    light = "--bg:#f6f7f9;--card:#fff;--ink:#141821;--mut:#5b6472;--acc:#1b6fd6;--line:#e3e7ec"
    dark = "--bg:#0f1216;--card:#171b21;--ink:#e8ebef;--mut:#9aa4b2;--acc:#5db0ff;--line:#262c34"
    theme = (":root{" + light + "}"
             "@media (prefers-color-scheme:dark){:root{" + dark + "}}"
             ':root[data-theme="light"]{' + light + "}"
             ':root[data-theme="dark"]{' + dark + "}")
    css = theme + """
    *{box-sizing:border-box}
    body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.65 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
    .wrap{max-width:1080px;margin:0 auto;padding:0 22px 90px}
    header{padding:64px 0 28px;border-bottom:1px solid var(--line);margin-bottom:8px}
    h1{font-size:2.15rem;margin:0 0 6px;letter-spacing:-.02em}
    h2{font-size:1.5rem;margin:56px 0 6px;letter-spacing:-.01em}
    h3{font-size:1.12rem;margin:34px 0 4px;color:var(--acc)}
    .sub{color:var(--mut);font-size:1.05rem}
    p{color:var(--ink)} .mut{color:var(--mut)}
    a{color:var(--acc)}
    figure{margin:18px 0;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:10px;overflow:hidden}
    figure img{width:100%;display:block;border-radius:7px}
    figcaption{color:var(--mut);font-size:.9rem;padding:9px 4px 3px}
    .grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}
    .grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
    @media(max-width:720px){.grid3,.grid2{grid-template-columns:1fr}}
    pre{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;overflow-x:auto;font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--ink)}
    table{border-collapse:collapse;width:100%;margin:16px 0;font-size:.94rem}
    th,td{border:1px solid var(--line);padding:8px 11px;text-align:right}
    th:first-child,td:first-child{text-align:left}
    th{background:var(--card);color:var(--mut);font-weight:600}
    .note{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--acc);border-radius:8px;padding:12px 16px;margin:18px 0;color:var(--mut);font-size:.95rem}
    code{background:var(--card);border:1px solid var(--line);border-radius:5px;padding:.5px 5px;font-size:.88em}
    .kpi{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}
    .kpi div{flex:1;min-width:150px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
    .kpi b{display:block;font-size:1.5rem;letter-spacing:-.02em}
    .kpi span{color:var(--mut);font-size:.85rem}
    """

    def sets_kpis():
        return "".join(
            f"<div><b>{S[s]['n_inliers']}/{S[s]['n_matches']}</b>"
            f"<span>{TITLES[s]} — RANSAC inliers/matches</span></div>" for s in SETS)

    parts = []
    parts.append(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CS 180 Project 3 — Image Warping & Mosaicing</title><style>{css}</style></head>
<body><div class="wrap">
<header>
<h1>Image Warping and Mosaicing</h1>
<div class="sub">CS 180 · Project 3 — homographies, rectification, and automatic panorama stitching</div>
</header>

<p>Project 1 aligned images by translation only. Here we recover the full
<b>projective</b> relationship between overlapping photographs: estimate the
homography between them, warp them into a common frame, and feather them into a
seamless mosaic — first from hand-clicked correspondences (Part A), then fully
automatically with Harris corners, MOPS descriptors and RANSAC (Part B).</p>

<div class="kpi">{sets_kpis()}</div>

<div class="note"><b>About the data.</b> The brief's photos are normally shot by
hand. With no camera here, each set is synthesised from one wide real photograph
(Unsplash imagery via picsum.photos) by re-projecting overlapping sub-windows
through <i>known</i> homographies — exactly the image-formation model of a camera
pivoting in place. That buys real texture, guaranteed overlap, and an exact
ground-truth H, so the recovered homographies can be checked against truth
(tables below). The pipeline is identical to one fed hand-shot photos; only the
image source differs. Rectification uses two genuinely oblique photos
(a laptop, a Bruges facade).</p></div>

<h2>Part A · Homographies, warping, mosaics</h2>

<h3>A.2 — Recovering the homography</h3>
<p>A homography has 8 degrees of freedom, so each point correspondence
(<code>x,y</code>)→(<code>x',y'</code>) contributes two linear equations:</p>
<pre>x' (h31 x + h32 y + 1) = h11 x + h12 y + h13
y' (h31 x + h32 y + 1) = h21 x + h22 y + h23</pre>
<p>Stacking ≥4 correspondences gives an over-determined system
<code>A h = b</code> in the 8 unknowns (h33 fixed to 1), solved by least squares
(<code>np.linalg.lstsq</code>). More points don't break it — they stabilise the
fit. A homography recovered from 12 hand-clicked château points:</p>
<pre>H (image 1 → image 2) =
{fmt_H(H12)}</pre>
<p>The core is validated on synthetic data independent of any image (like
Project 1's <code>test_align.py</code>): build a random H, apply it to random
points, recover it, and compare.</p>
<table>
<tr><th>Synthetic test</th><th>error</th></tr>
<tr><td>Recover known H (4–11 points, 200 trials)</td><td>{hrec:.1e}</td></tr>
<tr><td>Over-determined recovery (50 points)</td><td>{hover:.1e}</td></tr>
<tr><td>Warp → unwarp point round-trip (H then H⁻¹)</td><td>{hround:.1e}</td></tr>
</table>

<h3>A.3 — Warping: nearest-neighbour vs bilinear</h3>
<p>Warping is done by <b>inverse mapping</b>: for every output pixel, map back
through H⁻¹ into the source and sample there (forward warping scatters pixels
and leaves holes). Nearest-neighbour just rounds; bilinear blends the four
surrounding pixels — smoother, and about
{S['warp']['bilinear_ms']/S['warp']['nn_ms']:.1f}× slower here
({S['warp']['nn_ms']:.0f} ms vs {S['warp']['bilinear_ms']:.0f} ms).</p>
{img("results/warp_nn_vs_bilinear.png", "Same warped crop: nearest-neighbour (blocky) vs bilinear (smooth).")}
<table>
<tr><th>Warp sampler test</th><th>error</th></tr>
<tr><td>Identity is a no-op</td><td>{wid:.1e}</td></tr>
<tr><td>Translation shifts &amp; samples correctly</td><td>{wtr:.1e}</td></tr>
<tr><td>Bilinear exact on a linear ramp under affine warp</td><td>{wramp:.1e}</td></tr>
</table>

<h3>A.3 — Rectification</h3>
<p>Rectification is the whole pipeline on one image: take a plane shot at an
angle, declare its four corners to be a true rectangle, and warp. The plane
comes out fronto-parallel. Left: the photo with the chosen quad; right: rectified.</p>
<div class="grid2">
{img("results/rectify_desk_input.jpg", "Laptop screen, oblique — selected quad in red.")}
{img("results/rectify_desk.jpg", "Rectified: screen is head-on, menu bar horizontal, dock level.")}
</div>
<div class="grid2">
{img("results/rectify_bruges_input.jpg", "Bruges facade, receding to the right.")}
{img("results/rectify_bruges.jpg", "Rectified: windows now vertical and regularly spaced.")}
</div>

<h3>A.4 — Manual mosaics (3)</h3>
<p>Each set is three views taken across a pivot. Homographies from the
hand-clicked points map the outer views into the middle view's frame; images are
warped into a shared canvas and blended by <b>feathering</b> — a weight that
ramps to zero at each photo's edge, so overlaps cross-fade instead of showing a
hard seam.</p>
""")

    for s in SETS:
        parts.append(f"""<h4 class="mut">{TITLES[s]}</h4>
<div class="grid3">
{img(f"data/{s}_1.jpg")}{img(f"data/{s}_2.jpg")}{img(f"data/{s}_3.jpg")}
</div>
{img(f"results/{s}_corr.png", "Hand-clicked correspondences (pair 1–2).")}
{img(f"results/{s}_manual.jpg", "Manual mosaic — feather-blended.")}
{img(f"results/{s}_naive.jpg", "Naive last-writer-wins composite — note the hard seams the feather blend removes.")}
""")

    parts.append(f"""
<h2>Part B · Automatic stitching</h2>
<p>Same warp and blend as Part A — the only change is that correspondences are
found automatically, following the MOPS paper (Brown, Szeliski &amp; Winder).</p>

<h3>B.1 — Harris corners + ANMS</h3>
<p>Harris (from the provided <code>harris.py</code>) returns thousands of corners
clustered in texture. <b>Adaptive Non-Maximal Suppression</b> keeps 500 that are
both strong and spread out: each corner is scored by the distance to the nearest
significantly-stronger corner, and the largest such radii win.</p>
{img("results/chateau_anms.png", f"{S['chateau']['n_raw_corners']} raw Harris corners → 500 after ANMS, evenly distributed.")}

<h3>B.2 — Descriptors</h3>
<p>Around each corner a 40×40 window is low-pass filtered and sampled down to an
8×8 patch, then normalised to zero mean / unit variance (invariant to brightness
and contrast).</p>
{img("results/descriptor_patches.png", "Sample normalised 8×8 descriptors.")}

<h3>B.3 — Matching (Lowe ratio test)</h3>
<p>Descriptors are matched by nearest neighbour, kept only when the best distance
clearly beats the second-best. This alone removes most bad matches in repetitive
texture before RANSAC runs.</p>
{img("results/chateau_matches.png", f"{S['chateau']['n_matches']} matches surviving the ratio test (château pair 1–2).")}

<h3>B.4 — RANSAC</h3>
<p>4-point RANSAC repeatedly fits H from a random minimal sample and counts
inliers within a pixel threshold, keeping the most-agreed model and refitting on
all its inliers — immune to the remaining outliers.</p>
{img("results/chateau_ransac.png", "Green = RANSAC inliers, red = rejected matches (mostly repeated windows / grass).")}

<h3>Automatic mosaics, side by side with manual</h3>
""")

    for s in SETS:
        parts.append(f"""<h4 class="mut">{TITLES[s]}</h4>
{img(f"results/{s}_compare.png", "Top: manual (hand-clicked). Bottom: automatic (Harris + RANSAC). Visually indistinguishable.")}
""")

    # quantitative table
    rows = "".join(
        f"<tr><td>{TITLES[s]}</td><td>{S[s]['n_raw_corners']}</td>"
        f"<td>{S[s]['n_matches']}</td><td>{S[s]['n_inliers']}</td>"
        f"<td>{S[s]['manual_H12_err_vs_truth']:.3f}</td>"
        f"<td>{S[s]['auto_H12_err_vs_truth']:.3f}</td></tr>" for s in SETS)
    parts.append(f"""
<h3>Quantitative validation</h3>
<p>Because the ground-truth homography is known, both the manually- and
automatically-recovered H can be measured against it. The errors below are the
max absolute entry difference (H normalised to H₃₃=1); they sit on the
translation terms of magnitude in the hundreds, i.e. <b>sub-pixel</b>.</p>
<table>
<tr><th>Set</th><th>raw corners</th><th>matches</th><th>inliers</th>
<th>manual H err</th><th>auto H err</th></tr>
{rows}
</table>
<p class="mut">Full pipeline runtime: {S['_runtime_s']}s. Everything on this page
is regenerated by <code>python run_all.py &amp;&amp; python make_writeup.py</code>.</p>

<h2>What carried over from Project 1</h2>
<p>Own <code>.venv</code>; correspondences saved to JSON so results are
reproducible without re-clicking; the geometry (<code>computeH</code>, inverse
warp) implemented from scratch — no <code>cv2.findHomography</code>,
<code>warpPerspective</code>, or <code>skimage.transform.warp</code>; and every
number here is read from the code's own output rather than typed in by hand.</p>
</div></body></html>""")

    html = "".join(parts)
    out = os.path.join(HERE, "writeup.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"[ok] wrote {out} ({len(html)//1024} KB)")


if __name__ == "__main__":
    build()
