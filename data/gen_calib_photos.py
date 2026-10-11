"""
gen_calib_photos.py -- synthesizes 12 checkerboard photographs from varying
angles/distances, "captured" with the same CameraRig used for the line video,
for cv2.calibrateCamera to recover intrinsics + distortion from.

Board: 9x6 squares (8x5 internal corners), 4.4 cm squares -- a different
board size/count from a 9x6-corner, 5cm-square, 15-photo setup, and photos
are placed by back-projecting a *desired framing pixel* at a chosen
camera-space depth, so each one reliably lands the board in-frame without
manual trial and error.
"""
import os
import sys
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from vision_lib.camera_rig import CameraRig, IMG_W, IMG_H

OUT_DIR = os.path.join(HERE, "checkerboards")
os.makedirs(OUT_DIR, exist_ok=True)

SQUARES_X, SQUARES_Y = 9, 6
SQUARE_CM = 4.4
BOARD_W_CM = SQUARES_X * SQUARE_CM
BOARD_H_CM = SQUARES_Y * SQUARE_CM

N_PHOTOS = 16


def render_board_texture(px_per_square=60):
    w, h = SQUARES_X * px_per_square, SQUARES_Y * px_per_square
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    for r in range(SQUARES_Y):
        for c in range(SQUARES_X):
            if (r + c) % 2 == 0:
                y0, y1 = r * px_per_square, (r + 1) * px_per_square
                x0, x1 = c * px_per_square, (c + 1) * px_per_square
                img[y0:y1, x0:x1] = (20, 20, 20)
    return img


def place_at_pixel(rig: CameraRig, px, py, depth_cm):
    """Back-projects pixel (px,py) through the (undistorted) ray and returns
    the world-space point lying at camera-space depth `depth_cm` along it."""
    norm = cv2.undistortPoints(np.array([[[px, py]]], dtype=np.float64), rig.K, rig.dist)
    x, y = norm[0, 0]
    point_cam = np.array([x * depth_cm, y * depth_cm, depth_cm])
    R, t = rig.rotation_translation()
    world = R.T @ (point_cam - t)
    return world


def board_plane_corners_world(center_world, yaw_deg, pitch_deg, roll_deg):
    """4 corners of the (planar) board in world space, centered at
    `center_world`, oriented by yaw/pitch/roll (degrees) about that center."""
    hl, hw = BOARD_W_CM / 2.0, BOARD_H_CM / 2.0
    local = np.array([
        [-hl, -hw, 0], [hl, -hw, 0], [hl, hw, 0], [-hl, hw, 0],
    ])
    ry, rp, rr = np.radians([yaw_deg, pitch_deg, roll_deg])
    Rz = np.array([[np.cos(ry), -np.sin(ry), 0], [np.sin(ry), np.cos(ry), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(rp), -np.sin(rp)], [0, np.sin(rp), np.cos(rp)]])
    Ry = np.array([[np.cos(rr), 0, np.sin(rr)], [0, 1, 0], [-np.sin(rr), 0, np.cos(rr)]])
    rot = Rz @ Rx @ Ry
    world = (rot @ local.T).T + center_world
    return world


def render_frame_with_board(rig, center_world, yaw_deg, pitch_deg, roll_deg, tex):
    bg = np.full((IMG_H, IMG_W, 3), (210, 208, 205), dtype=np.uint8)
    bg = np.clip(bg.astype(np.int16) + (np.random.randn(IMG_H, IMG_W, 3) * 3), 0, 255).astype(np.uint8)

    board_world = board_plane_corners_world(center_world, yaw_deg, pitch_deg, roll_deg)
    dst = rig.project(board_world).astype(np.float32)
    th, tw = tex.shape[:2]
    src = np.array([[0, th - 1], [tw - 1, th - 1], [tw - 1, 0], [0, 0]], dtype=np.float32)
    Hmat = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(tex, Hmat, (IMG_W, IMG_H), borderMode=cv2.BORDER_TRANSPARENT)
    mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst.astype(np.int32), 255)
    bg[mask > 0] = warped[mask > 0]
    return bg


def main():
    rig = CameraRig()
    tex = render_board_texture()

    rng = np.random.default_rng(7)
    saved = 0
    attempt = 0
    while saved < N_PHOTOS and attempt < N_PHOTOS * 4:
        attempt += 1
        # frame the board somewhere comfortably inside the image, at a
        # varying depth, with varying tilt so poses are genuinely diverse
        # spans much closer to the full frame (including toward the edges)
        # than a narrow central crop, so the recovered distortion model is
        # actually constrained out where the parcels themselves are imaged
        px = rng.uniform(0.06, 0.94) * IMG_W
        py = rng.uniform(0.15, 0.85) * IMG_H
        depth = rng.uniform(50, 140)
        center = place_at_pixel(rig, px, py, depth)
        yaw = rng.uniform(-35, 35)
        pitch = rng.uniform(-25, 25)
        roll = rng.uniform(-15, 15)

        frame = render_frame_with_board(rig, center, yaw, pitch, roll, tex)
        found, _ = cv2.findChessboardCorners(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (SQUARES_X - 1, SQUARES_Y - 1),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            continue

        frame = rig.apply_lens_distortion(frame)
        out_path = os.path.join(OUT_DIR, f"board_{saved:02d}.png")
        cv2.imwrite(out_path, frame)
        saved += 1

    print(f"Saved {saved}/{N_PHOTOS} checkerboard photos to {OUT_DIR}")
    if saved < N_PHOTOS:
        print("WARNING: fewer boards than requested were in-frame/detectable.")


if __name__ == "__main__":
    main()
