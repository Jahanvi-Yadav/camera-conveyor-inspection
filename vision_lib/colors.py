"""
colors.py -- hue-based masking/classification for the three parcel colors.
Hue ranges are centered on this project's own palette (amber/teal/magenta),
not the red/blue/green split a plain "colored boxes" scene would use.
"""
import numpy as np
import cv2

from .parcels import CATALOG

HUE_RANGES = {
    "poly_mailer": [(8, 32)],     # amber/orange
    "cube_carton": [(86, 110)],   # teal/blue
    "long_box": [(138, 168)],     # magenta/purple
}

REAL_SIZES_CM = {k: (v.length_cm, v.width_cm, v.height_cm) for k, v in CATALOG.items()}


def hue_mask(frame_bgr, parcel_type, top_face_only=False):
    """Binary mask for one parcel type. top_face_only adds a
    value-channel floor to exclude the darker (0.5x-shaded) side faces,
    isolating just the printed top texture."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    for lo, hi in HUE_RANGES[parcel_type]:
        mask |= ((h >= lo) & (h <= hi)).astype(np.uint8) * 255
    mask &= ((s > 70) & (v > 35)).astype(np.uint8) * 255
    if top_face_only:
        mask &= (v > 120).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def classify_hue(frame_bgr, mask):
    """Given an already-segmented mask (any source), votes on the dominant
    parcel-color hue inside it and returns the matching catalog key."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hue_vals = hsv[..., 0][mask > 0]
    if len(hue_vals) == 0:
        return "unknown"
    median_hue = np.median(hue_vals)
    best, best_dist = "unknown", 1e9
    for key, ranges in HUE_RANGES.items():
        for lo, hi in ranges:
            center = (lo + hi) / 2.0
            dist = abs(median_hue - center)
            if dist < best_dist:
                best_dist = dist
                best = key
    return best


def largest_quad(mask):
    """Largest contour in a binary mask -> (corners(4,2), center, (len_px,wid_px))
    via minAreaRect, or None if the mask is empty/too small."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < 150:
        return None
    rect = cv2.minAreaRect(biggest)
    box = cv2.boxPoints(rect)
    (cx, cy), (rw, rh), _ = rect
    return {"corners": box, "center": (cx, cy), "size_px": (rw, rh), "mask": mask, "contour": biggest}


def get_parcel_quad(frame_bgr, parcel_type, top_face_only=False):
    mask = hue_mask(frame_bgr, parcel_type, top_face_only=top_face_only)
    return largest_quad(mask)
