"""
gen_line_video.py -- renders the main test clip: three parcels riding the
sorting line at a constant belt speed, seen through the synthetic
CameraRig (real perspective projection + lens distortion + sensor noise).

Different from a simple "N boxes, one shared schedule" script: each
parcel gets its own lane depth (world Y), its own start delay, and a small
continuous yaw drift (instead of a fixed constant angle) so its silhouette
visibly rotates as it travels -- which matters later for Module E's
rotation-invariance test.

220 frames @ 24 fps (~9.2s), belt speed 22.0 cm/s.
"""
import os
import sys
import json
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from vision_lib.camera_rig import CameraRig, IMG_W, IMG_H
from vision_lib.parcels import CATALOG, render_belt_background, render_parcel, add_sensor_noise

OUT_VIDEO = os.path.join(HERE, "line_video.mp4")
OUT_GT = os.path.join(HERE, "ground_truth_line_video.json")

FPS = 24
N_FRAMES = 220
BELT_SPEED_CM_S = 22.0

# (id, parcel_type, start_x_cm, lane_depth_y_cm, start_delay_s, yaw0_deg, yaw_drift_deg_per_s)
# Each parcel shares the same start_x/speed but a later delay, which nets a
# constant ~44cm running gap between them once all three are moving --
# enough clearance given the largest parcel is 34cm long -- and each rides
# its own lane depth so they don't stack in the projected image either.
SCHEDULE = [
    ("P1", "poly_mailer", -80.0, 20.0, 0.0, -8.0, 3.0),
    ("P2", "cube_carton", -80.0, 32.0, 2.0, 12.0, -4.5),
    ("P3", "long_box", -80.0, 45.0, 4.0, -5.0, 2.0),
]


def main():
    rig = CameraRig()
    ground_truth = {"fps": FPS, "belt_speed_cm_s": BELT_SPEED_CM_S, "frames": []}

    writer = cv2.VideoWriter(OUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (IMG_W, IMG_H))

    for fi in range(N_FRAMES):
        t = fi / FPS
        frame = render_belt_background(IMG_W, IMG_H)

        frame_gt = {"frame": fi, "parcels": {}}
        for pid, ptype, start_x, lane_y, delay, yaw0, yaw_rate in SCHEDULE:
            dt = max(0.0, t - delay)
            x = start_x + BELT_SPEED_CM_S * dt
            yaw = yaw0 + yaw_rate * dt
            spec = CATALOG[ptype]

            top_px = render_parcel(frame, rig, spec, (x, lane_y), yaw)
            centroid_px = top_px.mean(axis=0)

            frame_gt["parcels"][pid] = {
                "type": ptype,
                "world_x": x, "world_y": lane_y, "yaw_deg": yaw,
                "centroid_px": centroid_px.tolist(),
                "size_cm": [spec.length_cm, spec.width_cm, spec.height_cm],
            }

        frame = rig.apply_lens_distortion(frame)
        frame = add_sensor_noise(frame, sigma=2.2)
        writer.write(frame)
        ground_truth["frames"].append(frame_gt)

    writer.release()
    with open(OUT_GT, "w") as f:
        json.dump(ground_truth, f)

    print(f"Wrote {N_FRAMES} frames ({N_FRAMES/FPS:.1f}s @ {FPS}fps) to {OUT_VIDEO}")
    print(f"Ground truth: {OUT_GT}")
    print(f"Belt speed: {BELT_SPEED_CM_S} cm/s; parcels: {[s[0]+':'+s[1] for s in SCHEDULE]}")


if __name__ == "__main__":
    main()
