"""
parcels.py -- the three parcel types on this sorting line, and rendering
helpers to draw them into a frame given the CameraRig.

Deliberately a different catalog from a plain "colored boxes" scene: sizes,
BGR colors, and surface-texture patterns are all distinct so downstream color
masks, Hu-moment aspect ratios and recognition crops are genuinely different
data, not the same fixtures under new names.
"""
from dataclasses import dataclass
import numpy as np
import cv2

from .camera_rig import CameraRig


@dataclass(frozen=True)
class ParcelSpec:
    key: str
    length_cm: float   # along local x (belt-travel-facing edge)
    width_cm: float     # along local y
    height_cm: float    # along z
    color_bgr: tuple    # side-face base color


CATALOG = {
    # a flat poly mailer -- wide and shallow
    "poly_mailer": ParcelSpec("poly_mailer", length_cm=28.0, width_cm=19.0, height_cm=3.5,
                               color_bgr=(30, 170, 235)),   # amber/orange
    # a roughly-cubic small carton
    "cube_carton": ParcelSpec("cube_carton", length_cm=15.0, width_cm=15.0, height_cm=15.0,
                               color_bgr=(180, 140, 20)),   # teal-ish blue
    # a long, narrow box
    "long_box": ParcelSpec("long_box", length_cm=34.0, width_cm=13.0, height_cm=11.0,
                             color_bgr=(150, 40, 150)),      # magenta/purple
}


def make_top_texture(spec: ParcelSpec, px=256):
    """Procedural top-face texture, one distinct pattern family per type
    (diagonal hatching / concentric rings / chevrons) -- chosen to differ
    from a simple stripes-and-corner-dots set."""
    tex = np.full((px, px, 3), spec.color_bgr, dtype=np.uint8)
    dark = tuple(int(c * 0.55) for c in spec.color_bgr)
    light = tuple(min(255, int(c * 1.35 + 20)) for c in spec.color_bgr)

    if spec.key == "poly_mailer":
        # diagonal hatch lines (like packing-tape stripes)
        for offset in range(-px, px, 24):
            cv2.line(tex, (offset, 0), (offset + px, px), dark, 6)
        cv2.rectangle(tex, (10, 10), (px - 10, px - 10), light, 4)
    elif spec.key == "cube_carton":
        # concentric rings
        center = (px // 2, px // 2)
        for r in range(20, px // 2, 22):
            cv2.circle(tex, center, r, dark, 3)
        cv2.circle(tex, center, 14, light, -1)
    else:  # long_box
        # chevron / zigzag band down the middle
        pts = []
        step = 20
        for i, xcoord in enumerate(range(0, px + step, step)):
            ycoord = px // 2 + (30 if i % 2 == 0 else -30)
            pts.append([xcoord, ycoord])
        cv2.polylines(tex, [np.array(pts, dtype=np.int32)], False, dark, 8)
        cv2.rectangle(tex, (0, 0), (px - 1, px - 1), light, 5)

    return tex


def local_corners(spec: ParcelSpec, top_only=False):
    """Corners of the box in its own local frame, centered at (0,0,0) at the
    base, +z up. Returns the 4 top-face corners (or all 8 if top_only=False),
    ordered consistently for texture warping."""
    hl, hw = spec.length_cm / 2.0, spec.width_cm / 2.0
    top = np.array([
        [-hl, -hw, spec.height_cm],
        [hl, -hw, spec.height_cm],
        [hl, hw, spec.height_cm],
        [-hl, hw, spec.height_cm],
    ])
    if top_only:
        return top
    base = top.copy()
    base[:, 2] = 0.0
    return np.vstack([base, top])


def world_corners(spec: ParcelSpec, center_xy, yaw_deg, top_only=False):
    """Places local corners at a world (x,y) position on the belt (z=0 base)
    with a rotation about the vertical (Z) axis."""
    corners = local_corners(spec, top_only=top_only)
    theta = np.radians(yaw_deg)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    rotated = corners @ rot.T
    rotated[:, 0] += center_xy[0]
    rotated[:, 1] += center_xy[1]
    return rotated


def render_belt_background(w, h, n_stripes=14):
    """A grayish belt surface with faint perpendicular guide stripes running
    across the direction of travel (a different pattern from lengthwise
    lines, so Module A's false-positive story plays out differently)."""
    bg = np.full((h, w, 3), (98, 100, 96), dtype=np.uint8)
    noise = (np.random.randn(h, w, 3) * 3).astype(np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for i in range(n_stripes):
        y = int(h * (i + 0.5) / n_stripes)
        cv2.line(bg, (0, y), (w, y), (78, 80, 76), 2)
    return bg


def render_parcel(frame, rig: CameraRig, spec: ParcelSpec, center_xy, yaw_deg, texture_px=256):
    """Paints one parcel into `frame` in-place: shaded side quads (painter's
    algorithm by camera-space depth) then a perspective-warped top texture."""
    all_corners = world_corners(spec, center_xy, yaw_deg, top_only=False)
    base = all_corners[:4]
    top = all_corners[4:]

    faces = []
    for i in range(4):
        j = (i + 1) % 4
        quad = np.array([base[i], base[j], top[j], top[i]])
        faces.append(quad)

    face_depths = []
    for quad in faces:
        d = rig.camera_space_depth(quad).mean()
        face_depths.append(d)
    order = np.argsort(face_depths)[::-1]  # far to near

    dark = tuple(int(c * 0.5) for c in spec.color_bgr)
    for idx in order:
        quad = faces[idx]
        pix = rig.project(quad).astype(np.int32)
        cv2.fillConvexPoly(frame, pix, dark, lineType=cv2.LINE_AA)

    # top face: perspective-correct texture warp
    tex = make_top_texture(spec, px=texture_px)
    src_pts = np.array([[0, texture_px - 1], [texture_px - 1, texture_px - 1],
                         [texture_px - 1, 0], [0, 0]], dtype=np.float32)
    dst_pts = rig.project(top).astype(np.float32)
    Hmat = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(tex, Hmat, (frame.shape[1], frame.shape[0]),
                                  borderMode=cv2.BORDER_TRANSPARENT)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst_pts.astype(np.int32), 255)
    frame[mask > 0] = warped[mask > 0]

    return dst_pts  # top-face pixel corners, for ground truth


def add_sensor_noise(frame, sigma=2.5):
    noise = np.random.randn(*frame.shape) * sigma
    return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
