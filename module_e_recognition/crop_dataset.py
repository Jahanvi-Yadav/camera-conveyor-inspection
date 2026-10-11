"""
crop_dataset.py -- Module E, step 1: builds the per-type crop dataset by
sampling REAL frames from the line video across its whole duration (each
parcel's own continuous yaw drift already gives natural rotation
diversity, so no separate synthetic augmentation pass is needed for
rotation -- just brightness jitter is added on top), rather than generating
a dedicated augmented-crop dataset offline.

Two crop styles are extracted per sample, matching what each Module E
method actually needs:
  - a SQUARE, perspective-warped canonical patch (alignment + eigenspace --
    both care about appearance, and a fixed square size makes PCA/keypoint
    matching simple)
  - a RAW, aspect-preserving, rotation-corrected silhouette (Hu moments --
    these need the true footprint shape, which a square warp destroys)
"""
import os
import sys
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from vision_lib.colors import get_parcel_quad
from vision_lib.camera_rig import IMG_W, IMG_H
from vision_lib.parcels import CATALOG

VIDEO_PATH = os.path.join(HERE, "..", "data", "line_video.mp4")
PATCH_SIZE = 96
N_PER_TYPE = 24
FRAME_STRIDE = 3  # sample every Nth frame to get spread-out poses/lighting


def order_quad_corners(corners):
    """boxPoints() order isn't guaranteed to start at a fixed physical
    corner across samples; re-order deterministically by angle around the
    centroid so the square warp doesn't randomly flip/rotate 90 degrees
    between samples."""
    c = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 1] - c[1], corners[:, 0] - c[0])
    order = np.argsort(angles)
    return corners[order]


def square_warp(frame, corners, size=PATCH_SIZE):
    corners = order_quad_corners(corners).astype(np.float32)
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)
    Hmat = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(frame, Hmat, (size, size))


def raw_upright_silhouette(mask, corners, size=PATCH_SIZE):
    """Warps the mask's own rotated bounding box (via its 4 box points,
    directly to an axis-aligned rectangle of the SAME aspect ratio) rather
    than rotating the whole mask by minAreaRect's angle and cropping --
    that angle's sign/reference-axis convention is easy to get backwards
    and silently produces a sheared sliver instead of the true footprint.
    A direct 4-point perspective warp has no such ambiguity: box points in,
    an upright rectangle of the same (rw,rh) out, always, preserving the
    true aspect ratio Hu moments need."""
    rect = cv2.minAreaRect(corners.astype(np.float32))
    (cx, cy), (rw, rh), angle = rect
    rw_i, rh_i = max(2, int(round(rw))), max(2, int(round(rh)))
    box_pts = cv2.boxPoints(rect).astype(np.float32)  # 4 corners, consistent cyclic order
    dst = np.array([[0, rh_i - 1], [0, 0], [rw_i - 1, 0], [rw_i - 1, rh_i - 1]], dtype=np.float32)
    Hmat = cv2.getPerspectiveTransform(box_pts, dst)
    upright = cv2.warpPerspective(mask, Hmat, (rw_i, rh_i))

    scale = (size * 0.85) / max(rw_i, rh_i)
    new_w, new_h = max(1, int(rw_i * scale)), max(1, int(rh_i * scale))
    resized = cv2.resize(upright, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((size, size), dtype=np.uint8)
    x_off, y_off = (size - new_w) // 2, (size - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def jitter_brightness(patch, rng, low=0.7, high=1.3):
    factor = rng.uniform(low, high)
    return np.clip(patch.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def build_dataset():
    """Returns {ptype: [{'square': img, 'silhouette': mask, 'frame': fi}, ...]}"""
    rng = np.random.default_rng(42)
    cap = cv2.VideoCapture(VIDEO_PATH)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    dataset = {ptype: [] for ptype in CATALOG}
    for fi in range(0, n_frames, FRAME_STRIDE):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue
        for ptype in CATALOG:
            if len(dataset[ptype]) >= N_PER_TYPE:
                continue
            quad = get_parcel_quad(frame, ptype, top_face_only=True)
            if quad is None:
                continue
            w, h = quad["size_px"]
            if min(w, h) < 15:  # too small to be a reliable crop
                continue
            xs, ys = quad["corners"][:, 0], quad["corners"][:, 1]
            margin = 3
            if xs.min() < margin or ys.min() < margin or xs.max() > IMG_W - margin or ys.max() > IMG_H - margin:
                continue  # parcel is clipped by the frame border -- not its true shape
            square = square_warp(frame, quad["corners"])
            square = jitter_brightness(square, rng)
            silhouette = raw_upright_silhouette(quad["mask"], quad["corners"])
            dataset[ptype].append({"square": square, "silhouette": silhouette, "frame": fi})
    cap.release()
    return dataset


if __name__ == "__main__":
    ds = build_dataset()
    for ptype, samples in ds.items():
        print(f"{ptype}: {len(samples)} samples, frames {[s['frame'] for s in samples]}")
