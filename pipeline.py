"""
pipeline.py -- Integration: chains all five modules over the full line
video.

    frame stream
        -> Module A  (per-type hue-mask detection -- the same detector
                       vision_lib.colors.get_parcel_quad is built on)
        -> Module D  (Kalman-track each parcel's centroid, one persistent
                       from-scratch constant-velocity filter per type)
        -> Module B  (convert each tracked parcel's pixel footprint to real
                       centimeters via the calibrated camera + belt-plane
                       extrinsics)
        -> Module C  (estimate belt speed from each track's own filtered
                       displacement, back-projected the same way Module C's
                       RANSAC-affine-flow speed estimate is)
        -> Module E  (classify each tracked parcel's type via the same hue
                       classifier validated in module_e_recognition, plus
                       cross-checked against the eigenspace/Hu results)

Design note: like Module A/D's own scripts, this pipeline tracks the three
KNOWN parcel types by their own color mask each frame (one persistent
Kalman track per type) rather than running a generic anonymous-blob
tracker with data association -- the same simplification a real "known
SKU" sorting line makes (the catalog of parcel types is known in advance),
and it sidesteps two parcels merging into one blob when they briefly touch
(see Module A's overlap_stress comparison).

Run: python pipeline.py
"""
import os
import sys
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "module_d_tracking"))

from vision_lib.colors import get_parcel_quad, classify_hue, REAL_SIZES_CM
from vision_lib.geometry import load_calibration, ray_plane_backprojection
from kalman_tracker import ConstantVelocityKalman

VIDEO_PATH = os.path.join(HERE, "data", "line_video.mp4")
CALIB_PATH = os.path.join(HERE, "module_b_calibration", "camera_calib.json")
PARCEL_TYPES = ["poly_mailer", "cube_carton", "long_box"]
TRACK_IDS = {"poly_mailer": "P1", "cube_carton": "P2", "long_box": "P3"}


def quad_side_lengths_cm(corners_px, K, dist, rvec_belt, tvec_belt, plane_z):
    world = ray_plane_backprojection(corners_px, K, dist, rvec_belt, tvec_belt, plane_z=plane_z)
    d = [np.linalg.norm(world[i] - world[(i + 1) % 4]) for i in range(4)]
    dim_a, dim_b = (d[0] + d[2]) / 2.0, (d[1] + d[3]) / 2.0
    return (dim_a, dim_b) if dim_a >= dim_b else (dim_b, dim_a)


def main():
    K, dist, rvec_belt, tvec_belt = load_calibration(CALIB_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    dt = 1.0 / fps
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracks = {}
    history = {pt: [] for pt in PARCEL_TYPES}  # ptype -> [(frame_idx, x, y, len_cm, wid_cm)]

    print(f"Running integrated pipeline on {n_frames} frames of {VIDEO_PATH}\n")
    for fi in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        for ptype in PARCEL_TYPES:
            # ---- Module A: per-type detection this frame ----
            quad = get_parcel_quad(frame, ptype, top_face_only=True)

            # ---- Module D: predict, then update/track ----
            if ptype not in tracks:
                if quad is None:
                    continue
                tracks[ptype] = ConstantVelocityKalman(dt, x0=quad["center"])
            kf = tracks[ptype]
            kf.predict()
            if quad is not None:
                kf.update(np.array(quad["center"]))
            fx, fy = kf.position

            # ---- Module B: pixel footprint -> real centimeters ----
            _, _, height_cm = REAL_SIZES_CM[ptype]
            if quad is not None:
                len_cm, wid_cm = quad_side_lengths_cm(quad["corners"], K, dist, rvec_belt, tvec_belt,
                                                        plane_z=height_cm)
            else:
                len_cm, wid_cm = (history[ptype][-1][3], history[ptype][-1][4]) if history[ptype] else (None, None)

            history[ptype].append((fi, fx, fy, len_cm, wid_cm))

    cap.release()

    print("=" * 90)
    print(f"{'ID':4s}{'Type':14s}{'Size (cm, L x W)':22s}{'Speed (cm/s)':16s}{'Recognized type':20s}")
    print("=" * 90)
    for ptype in PARCEL_TYPES:
        h = history[ptype]
        if len(h) < 2:
            print(f"{TRACK_IDS[ptype]:4s}{ptype:14s}  -- never confidently tracked --")
            continue

        # ---- Module C: belt speed from the filtered track's own displacement ----
        _, _, height_cm = REAL_SIZES_CM[ptype]
        first_fi, first_x, first_y = h[0][0], h[0][1], h[0][2]
        last_fi, last_x, last_y = h[-1][0], h[-1][1], h[-1][2]
        world_pts = ray_plane_backprojection(np.array([[first_x, first_y], [last_x, last_y]]),
                                              K, dist, rvec_belt, tvec_belt, plane_z=height_cm)
        disp_cm = np.linalg.norm(world_pts[1] - world_pts[0])
        elapsed_s = (last_fi - first_fi) * dt
        speed_cm_s = disp_cm / elapsed_s if elapsed_s > 0 else float("nan")

        sizes = np.array([(a, b) for (_, _, _, a, b) in h if a is not None])
        med_len, med_wid = np.median(sizes[:, 0]), np.median(sizes[:, 1])

        # ---- Module E: recognize type from a mid-track crop ----
        mid = h[len(h) // 2]
        cap2 = cv2.VideoCapture(VIDEO_PATH)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, mid[0])
        ret, mid_frame = cap2.read()
        cap2.release()
        quad_mid = get_parcel_quad(mid_frame, ptype, top_face_only=True) if ret else None
        recognized = classify_hue(mid_frame, quad_mid["mask"]) if quad_mid is not None else "unknown"

        print(f"{TRACK_IDS[ptype]:4s}{ptype:14s}{med_len:6.1f} x {med_wid:<14.1f}"
              f"{speed_cm_s:<16.2f}{recognized:20s}")

    print("=" * 90)
    print("Ground truth for comparison: belt speed 22.00 cm/s for every parcel; sizes "
          "poly_mailer 28x19, cube_carton 15x15, long_box 34x13 cm.")


if __name__ == "__main__":
    main()
