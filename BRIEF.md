# CS 180 Project 3 — Image Warping and Mosaicing (Photo Stitching)

Hand this whole file to Claude Code in a new chat to build the project.

Follow-up to Project 1 (Prokudin-Gorskii colorization, `E:\cs180-proj1`).
Project 1 aligned images by *translation only*. This project handles the full
**projective** relationship between photos: take several overlapping shots by
rotating the camera, figure out the geometric transform between them, warp
them into a common frame, and blend them into one wide panorama. Then do it
**automatically** — detect features, match them, and stitch with no clicking.

Official spec (verified links, checked 2026-07-27):
- Overview: https://cal-cs180.github.io/fa25/hw/proj3/
- Part A: https://cal-cs180.github.io/fa25/hw/proj3/partA.html
- Part B: https://cal-cs180.github.io/fa25/hw/proj3/partB.html
- Starter code (Harris detector): https://cal-cs180.github.io/fa25/hw/proj3/harris.py
- Reference paper (MOPS, Brown et al.): https://cal-cs180.github.io/fa25/hw/proj3/Papers/MOPS.pdf

---

## ⚠️ Read this first: the data is PHOTOS THE USER TAKES

There is no dataset to download. The input is **the user's own photographs**,
and they must be shot correctly or nothing downstream works:
- 2+ sets of photos, each set 2–3 images that overlap by ~40–70%.
- **Rotate the camera about a fixed point** (pivot in place) — do NOT walk
  sideways. Same exposure/zoom across a set; lock focus if possible.
- Also shoot 2 photos of **flat rectangular things at an angle** (a poster,
  a book cover, a window, a tiled floor) for the rectification step.

**So the very first thing the new chat should do is ask the user for these
photos, or offer to help plan the shots.** Don't scaffold blindly — you're
blocked without them. (Meanwhile you can build and unit-test the math on
synthetic point sets, like Project 1 validated on synthetic plates.)

---

## PART A — Warping & Mosaicing (manual correspondences)

### A.1 Shoot pictures
Collect the image sets described above into `data/`.

### A.2 Recover homographies
- Implement `computeH(im1_pts, im2_pts)` → a 3×3 **homography** matrix H.
- A homography has 8 degrees of freedom; each point correspondence gives 2
  equations, so you need ≥4 points. With more than 4, set up an
  overdetermined linear system and solve by **least squares** (more points =
  more stable).
- Correspondences are picked by hand (clicking matching points across the two
  images). Provide a small point-picker (matplotlib `ginput`, or load saved
  points from JSON so runs are reproducible).
- Deliverable: show the correspondences on the images, and display the system
  of equations + the resulting H.

### A.3 Warp images (+ rectification)
- Implement **`warpImageNearestNeighbor(im, H)`** and
  **`warpImageBilinear(im, H)`**.
- Use **inverse warping**: for each output pixel, map back through H⁻¹ into
  the source and sample there (forward warping leaves holes). Compute the
  output bounding box from the warped corners.
- **Rectification sanity check:** take one photo of a known rectangle shot at
  an angle, set its 4 corners to a true rectangle, warp — the plane should
  come out fronto-parallel (the poster looks flat-on). Do this for 2 images.
- Deliverable: compare nearest-neighbor vs bilinear (quality vs speed).

### A.4 Blend into a mosaic
- Warp images into a shared canvas and composite. Naive overwrite leaves a
  visible seam; blend with **weighted feathering** (alpha ramps to 0 at
  edges) or a **Laplacian pyramid** blend (you may already have this from
  Project 2's frequency work; if not, feathering is fine for full marks here).
- Deliverable: **3 complete mosaics.**

### A.5 Bells & whistles (optional; required only for the grad section)
- Cylindrical/spherical projection for very wide panoramas.

### Part A hard constraints (rubric)
Implement the geometry yourself. **Prohibited:** `cv2.findHomography`,
`cv2.warpPerspective`, `cv2.getPerspectiveTransform`,
`skimage.transform.ProjectiveTransform`, `skimage.transform.warp`, and
`scipy.interpolate.griddata`. You may use numpy's linear-algebra solvers
(`np.linalg.lstsq`/`svd`) to solve the homography system — that's the math,
not a stitching black box.

---

## PART B — Autostitching (automatic correspondences)

Based on *"Multi-Image Matching using Multi-Scale Oriented Patches,"*
Brown, Szeliski & Winder (the MOPS paper, linked above — read it). Replaces
the hand-clicked points from Part A with an automatic pipeline.

### B.1 Corner detection + ANMS
- Use the provided **`harris.py`** (download it) for single-scale Harris
  corner detection — this one piece is allowed as starter code.
- Implement **Adaptive Non-Maximal Suppression**: from thousands of raw
  corners, keep ~500 that are both strong and **spread out** across the
  image (suppress a corner if a stronger one sits within radius r; keep the
  ones with the largest such r). Show corners before vs after ANMS.

### B.2 Feature descriptors
- For each surviving corner, extract a **40×40** window and downsample to an
  **8×8** patch; normalize to zero mean, unit variance (bias/gain
  invariance). Show a few sample patches.

### B.3 Feature matching
- Match descriptors between two images by nearest neighbor, filtered with
  **Lowe's ratio test** (accept a match only if the best distance is much
  smaller than the 2nd-best — this kills ambiguous matches). Show matches.

### B.4 RANSAC + auto-mosaic
- Implement **4-point RANSAC** from scratch: repeatedly sample 4 matches,
  fit H (reusing A.2's `computeH`), count inliers within a pixel threshold,
  keep the H with the most inliers, refit on all inliers.
- Auto-stitch using Part A's warp/blend. Deliverable: **≥3 automatic
  mosaics**, shown **side-by-side with the manual versions** from Part A.

### B.5 Bells & whistles (choose one)
Multiscale corners/descriptors · rotation-invariant descriptors · panorama
recognition from an unordered pile of images.

---

## Suggested structure (mirror Project 1)

```
cs180-proj3/
├── .venv/                 # own venv: numpy, scipy, pillow, matplotlib  (opencv only for imread/harris helpers if desired)
├── requirements.txt
├── .gitignore             # .venv/, __pycache__/, big raw photos if huge, .env/secrets
├── homography.py          # computeH, apply H to points
├── warp.py                # inverse warp: nearest + bilinear, bounding box
├── rectify.py             # A.3 rectification demo
├── mosaic.py              # A.4 warp-all-into-canvas + feather/pyramid blend
├── corners.py             # B.1 harris (from starter) + ANMS
├── features.py            # B.2 descriptors, B.3 matching (Lowe ratio)
├── ransac.py              # B.4 robust homography
├── autostitch.py          # B end-to-end: two images -> mosaic
├── points/                # saved correspondence JSONs (reproducible runs)
├── make_writeup.py        # builds writeup.html (manual vs auto side by side)
├── data/                  # the user's photos + rectification shots
└── results/               # all mosaics, corner viz, match viz
```

## Validate the math without photos (Project-1 style)
Before the user's photos arrive, unit-test on **synthetic points**: make a
random H, apply it to a set of points, feed the pairs to `computeH`, and
assert it recovers that H. Warp a synthetic grid image and warp it back with
H⁻¹ — you should get the original. This proves the core is correct
independent of any image, exactly like `test_align.py` did in Project 1.

## Lessons carried from Project 1 (do these)
- Own `.venv`; don't lean on another project's environment.
- **No GitHub token in the git remote URL** — use `gh auth login` or
  Credential Manager. `.gitignore` `.venv/` and any `.env`/secrets.
- **No `Co-Authored-By` trailers** in commits (they add extra names to
  GitHub's Contributors list).
- Save clicked correspondences to JSON so results are reproducible and the
  writeup regenerates without re-clicking.
- Verify `writeup.html` actually renders (serve it / open it) and every image
  link resolves and there's no horizontal overflow, before calling it done.
- Course download links can be gated — the ones listed above are verified
  working, but if any fails, say so instead of pretending; don't fabricate.
- Give the user a recommendation, not an options dump; act when the path is
  clear.

## Milestone order
1. Folder + venv + requirements + .gitignore. **Ask the user for photos.**
2. `homography.py` + synthetic unit test (recover a known H). 
3. `warp.py` nearest + bilinear + synthetic round-trip test.
4. A.3 rectification on the user's angled-rectangle shots.
5. A.4 three manual mosaics with feather blending.
6. B.1 Harris + ANMS (download `harris.py`).
7. B.2/B.3 descriptors + Lowe matching.
8. B.4 RANSAC + three auto mosaics, side-by-side with manual.
9. Assemble writeup.html, verify it renders, commit cleanly.
