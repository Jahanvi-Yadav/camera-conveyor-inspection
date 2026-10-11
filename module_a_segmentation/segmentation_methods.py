"""
segmentation_methods.py -- Module A: seven classical segmentation methods,
each returning an integer label image the same size as the input frame.

Deliberately a different set from "all 8 methods including a from-scratch
quadtree split-and-merge pass": this version skips the combined
split-and-merge method entirely and instead runs region SPLITTING and
region MERGING as two separate standalone passes (which is explicitly
allowed by the brief's item 4, "if not already covered by #2"), covering
active contours / watershed / graph-based / mean shift from the 6-method
menu, plus the separately-required Normalized Cut -- 7 methods total.
"""
import numpy as np
import cv2
from skimage.segmentation import active_contour, felzenszwalb, slic
from skimage.color import rgb2gray, label2rgb
from skimage.filters import gaussian
from skimage.feature import peak_local_max
from skimage import graph as skgraph
from scipy.ndimage import label as ndi_label
from sklearn.cluster import MeanShift, estimate_bandwidth

from vision_lib.colors import HUE_RANGES


def _any_parcel_mask(frame_bgr):
    """OR of all three parcel hue ranges -- used only to seed markers /
    define the region of interest for methods that need a starting point,
    never fed forward as if it were the answer."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    for ranges in HUE_RANGES.values():
        for lo, hi in ranges:
            mask |= ((h >= lo) & (h <= hi)).astype(np.uint8) * 255
    mask &= ((s > 70) & (v > 35)).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def method_active_contours(frame_bgr, n_points=140):
    """skimage's energy-minimization snake, one contour per detected blob,
    initialized as a circle around each blob's centroid and let converge."""
    gray = rgb2gray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    smoothed = gaussian(gray, sigma=2, preserve_range=True)
    mask = _any_parcel_mask(frame_bgr)
    n_labels, comp_labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    label_img = np.zeros(frame_bgr.shape[:2], dtype=np.int32)
    next_label = 1
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 200:
            continue
        cy, cx = centroids[i][1], centroids[i][0]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        radius = 0.75 * max(w, h) / 2.0 + 8
        s = np.linspace(0, 2 * np.pi, n_points)
        init = np.array([cy + radius * np.sin(s), cx + radius * np.cos(s)]).T
        try:
            snake = active_contour(smoothed, init, alpha=0.015, beta=8, gamma=0.001,
                                    w_edge=1.2, w_line=0, max_num_iter=250)
        except Exception:
            snake = init
        poly = np.stack([snake[:, 1], snake[:, 0]], axis=1).astype(np.int32)
        blob_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
        cv2.fillPoly(blob_mask, [poly], 1)
        label_img[blob_mask > 0] = next_label
        next_label += 1
    return label_img


def method_watershed(frame_bgr):
    """Distance-transform markers inside each blob, flooded apart with
    cv2.watershed. Markers come from LOCAL MAXIMA of the distance transform
    (skimage.feature.peak_local_max, one peak per touching object's own
    'core'), not a single global threshold -- a global threshold on a
    connected blob of two touching parcels stays one component and can't
    split them; per-blob local maxima can, as long as each object still has
    its own distinct core region."""
    mask = _any_parcel_mask(frame_bgr)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    coords = peak_local_max(dist, min_distance=25, labels=mask, exclude_border=False)
    peak_mask = np.zeros(dist.shape, dtype=bool)
    peak_mask[tuple(coords.T)] = True
    marker_seeds, n_markers = ndi_label(peak_mask)

    sure_bg = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=3)
    unknown = cv2.subtract(sure_bg, (marker_seeds > 0).astype(np.uint8) * 255)

    markers = marker_seeds + 1
    markers[unknown > 0] = 0
    frame_for_ws = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
    cv2.watershed(frame_for_ws, markers)
    markers[markers == -1] = 0
    markers[markers == 1] = 0  # background marker
    return markers.astype(np.int32)


def method_felzenszwalb(frame_bgr, scale=90, sigma=0.6, min_size=60):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    labels = felzenszwalb(rgb, scale=scale, sigma=sigma, min_size=min_size)
    return labels.astype(np.int32)


def method_mean_shift(frame_bgr, downsample=4, spatial_weight=0.35):
    """Clusters pixels in color+position feature space. Downsampled for
    tractable runtime, then the label map is upsampled with nearest-
    neighbor back to full resolution."""
    small = cv2.resize(frame_bgr, (frame_bgr.shape[1] // downsample, frame_bgr.shape[0] // downsample))
    h, w = small.shape[:2]
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    ys, xs = np.mgrid[0:h, 0:w]
    features = np.stack([
        lab[..., 0].ravel(), lab[..., 1].ravel(), lab[..., 2].ravel(),
        xs.ravel() * spatial_weight, ys.ravel() * spatial_weight,
    ], axis=1)
    bandwidth = estimate_bandwidth(features, quantile=0.15, n_samples=min(1500, len(features)))
    bandwidth = max(bandwidth, 3.0)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, max_iter=120)
    labels_small = ms.fit_predict(features).reshape(h, w)
    labels_full = cv2.resize(labels_small.astype(np.int32), (frame_bgr.shape[1], frame_bgr.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
    return labels_full


def method_region_splitting(frame_bgr, min_block=8, var_thresh=60.0):
    """Standalone top-down quadtree SPLIT pass only -- recursively halves
    a block while its intensity variance exceeds the threshold, with no
    merge step afterward, so the raw over-segmentation is visible."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    label_img = np.zeros((h, w), dtype=np.int32)
    counter = [0]

    def split(y0, y1, x0, x1):
        block = gray[y0:y1, x0:x1]
        if block.size == 0:
            return
        if (y1 - y0) <= min_block or (x1 - x0) <= min_block or block.var() <= var_thresh:
            counter[0] += 1
            label_img[y0:y1, x0:x1] = counter[0]
            return
        ym, xm = (y0 + y1) // 2, (x0 + x1) // 2
        split(y0, ym, x0, xm)
        split(y0, ym, xm, x1)
        split(ym, y1, x0, xm)
        split(ym, y1, xm, x1)

    split(0, h, 0, w)
    return label_img


def method_region_merging(frame_bgr, block=16, mean_thresh=10.0):
    """Standalone bottom-up MERGE pass: starts from a regular grid of small
    blocks (a fixed, deliberately naive over-segmentation -- not #2's
    quadtree split output) and repeatedly merges adjacent regions whose
    mean intensity is close, via union-find, until no more merges apply."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    by, bx = h // block, w // block
    parent = list(range(by * bx))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    means = np.zeros(by * bx)
    for r in range(by):
        for c in range(bx):
            idx = r * bx + c
            means[idx] = gray[r * block:(r + 1) * block, c * block:(c + 1) * block].mean()

    changed = True
    while changed:
        changed = False
        for r in range(by):
            for c in range(bx):
                idx = r * bx + c
                ridx = find(idx)
                for dr, dc in [(0, 1), (1, 0)]:
                    r2, c2 = r + dr, c + dc
                    if r2 >= by or c2 >= bx:
                        continue
                    idx2 = r2 * bx + c2
                    r2idx = find(idx2)
                    if r2idx != ridx and abs(means[ridx] - means[r2idx]) < mean_thresh:
                        union(idx, idx2)
                        changed = True

    label_img = np.zeros((h, w), dtype=np.int32)
    root_to_label = {}
    next_label = 1
    for r in range(by):
        for c in range(bx):
            idx = r * bx + c
            root = find(idx)
            if root not in root_to_label:
                root_to_label[root] = next_label
                next_label += 1
            label_img[r * block:(r + 1) * block, c * block:(c + 1) * block] = root_to_label[root]
    return label_img


def method_normalized_cut(frame_bgr, n_segments=140, downsample=2, thresh=0.012):
    """SLIC superpixels -> region adjacency graph (mean color) ->
    skimage.graph.cut_normalized spectral partitioning. Run on a
    downsampled image since N-Cut is the heaviest method here."""
    small = cv2.resize(frame_bgr, (frame_bgr.shape[1] // downsample, frame_bgr.shape[0] // downsample))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    superpixels = slic(rgb, n_segments=n_segments, compactness=12, start_label=1)
    rag = skgraph.rag_mean_color(rgb, superpixels, mode="similarity")
    ncut_labels = skgraph.cut_normalized(superpixels, rag, thresh=thresh)
    full = cv2.resize(ncut_labels.astype(np.int32), (frame_bgr.shape[1], frame_bgr.shape[0]),
                       interpolation=cv2.INTER_NEAREST)
    return full


METHODS = {
    "active_contours": method_active_contours,
    "watershed": method_watershed,
    "felzenszwalb": method_felzenszwalb,
    "mean_shift": method_mean_shift,
    "region_splitting": method_region_splitting,
    "region_merging": method_region_merging,
    "normalized_cut": method_normalized_cut,
}


def label_to_color(label_img):
    return (label2rgb(label_img, bg_label=0) * 255).astype(np.uint8)[..., ::-1]  # RGB->BGR
