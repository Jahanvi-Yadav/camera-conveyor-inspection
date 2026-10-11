"""
pyramidal_lk.py -- Module C, step 1: Lucas-Kanade dense optical flow
implemented FROM SCRATCH (the brightness-constancy least-squares system is
solved by hand, not via cv2.calcOpticalFlowPyrLK), using a coarse-to-fine
Gaussian pyramid with per-level warp-and-refine -- a genuinely different
algorithm from a single-scale iterative-Newton refinement: motion that
would be far too large for one fixed window at full resolution is instead
mostly resolved at a coarse level (where it's small in pixel terms), and
each finer level only has to correct the residual.
"""
import numpy as np
import cv2


def gaussian_pyramid(img_gray, levels):
    """Returns [level0=original, level1=half-res, ...], coarsest last... no
    -- coarsest LAST index means index -1 is coarsest. We build fine-to-coarse
    then the caller walks it in reverse (coarse-to-fine)."""
    pyr = [img_gray.astype(np.float32)]
    cur = img_gray.astype(np.float32)
    for _ in range(levels - 1):
        cur = cv2.GaussianBlur(cur, (5, 5), 1.0)
        cur = cv2.resize(cur, (cur.shape[1] // 2, cur.shape[0] // 2), interpolation=cv2.INTER_AREA)
        pyr.append(cur)
    return pyr  # [full_res, half_res, quarter_res, ...]


def warp_by_flow(img, flow_u, flow_v):
    h, w = img.shape
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = grid_x + flow_u
    map_y = grid_y + flow_v
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def lk_increment(img1, img2_warped, window=15):
    """One-shot (non-iterative) dense Lucas-Kanade INCREMENT at a single
    scale: solves, for every pixel's local window, the 2x2 normal-equations
    system built from the brightness-constancy least squares

        [Sxx Sxy] [du]   [-Sxt]
        [Sxy Syy] [dv] = [-Syt]

    using box-filtered sums (vectorized over the whole image, not a python
    loop per pixel). Returns (du, dv, reliability) where reliability is the
    structure tensor's smaller eigenvalue (low near textureless / aperture-
    problem regions).
    """
    Ix = cv2.Sobel(img1, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    Iy = cv2.Sobel(img1, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    It = img2_warped - img1

    Ixx, Iyy, Ixy = Ix * Ix, Iy * Iy, Ix * Iy
    Ixt, Iyt = Ix * It, Iy * It

    ksize = (window, window)
    Sxx = cv2.boxFilter(Ixx, -1, ksize)
    Syy = cv2.boxFilter(Iyy, -1, ksize)
    Sxy = cv2.boxFilter(Ixy, -1, ksize)
    Sxt = cv2.boxFilter(Ixt, -1, ksize)
    Syt = cv2.boxFilter(Iyt, -1, ksize)

    det = Sxx * Syy - Sxy * Sxy
    det_safe = np.where(np.abs(det) < 1e-6, 1e-6, det)

    du = (-Syy * Sxt + Sxy * Syt) / det_safe
    dv = (Sxy * Sxt - Sxx * Syt) / det_safe

    trace = Sxx + Syy
    disc = np.sqrt(np.clip(trace * trace - 4 * det, 0, None))
    min_eig = (trace - disc) / 2.0

    return du, dv, min_eig


def lucas_kanade_pyramidal(img1_gray, img2_gray, levels=3, window=15, reliability_thresh=6.0):
    """Coarse-to-fine dense optical flow: estimate at the coarsest level
    first, then at each finer level warp img2 by the upsampled current
    estimate and solve for the (small) residual on top of it."""
    pyr1 = gaussian_pyramid(img1_gray, levels)
    pyr2 = gaussian_pyramid(img2_gray, levels)

    # start at the coarsest level with zero flow
    coarsest = levels - 1
    h0, w0 = pyr1[coarsest].shape
    flow_u = np.zeros((h0, w0), dtype=np.float32)
    flow_v = np.zeros((h0, w0), dtype=np.float32)
    min_eig = None

    for level in range(coarsest, -1, -1):
        img1, img2 = pyr1[level], pyr2[level]
        if flow_u.shape != img1.shape:
            scale_y = img1.shape[0] / flow_u.shape[0]
            scale_x = img1.shape[1] / flow_u.shape[1]
            flow_u = cv2.resize(flow_u, (img1.shape[1], img1.shape[0])) * scale_x
            flow_v = cv2.resize(flow_v, (img1.shape[1], img1.shape[0])) * scale_y

        img2_warped = warp_by_flow(img2, flow_u, flow_v)
        du, dv, min_eig = lk_increment(img1, img2_warped, window=window)
        flow_u = flow_u + du
        flow_v = flow_v + dv

    reliable = min_eig > reliability_thresh
    return flow_u, flow_v, reliable
