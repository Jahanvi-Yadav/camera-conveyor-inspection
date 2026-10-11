"""
gen_overlap_frame.py -- a single purpose-built frame with two parcels
pushed together so their silhouettes touch/overlap, used ONLY as an extra
stress-test frame for Module A's segmentation comparison (the main line
video's own schedule keeps a clear running gap between parcels, since a
real sorting line is routed to avoid collisions -- so this frame is
rendered separately, deliberately, to still get an honest touching-objects
comparison out of the segmentation methods).
"""
import os
import sys
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from vision_lib.camera_rig import CameraRig, IMG_W, IMG_H
from vision_lib.parcels import CATALOG, render_belt_background, render_parcel, add_sensor_noise

OUT_PATH = os.path.join(HERE, "overlap_stress_frame.png")


def main():
    rig = CameraRig()
    frame = render_belt_background(IMG_W, IMG_H)
    # cube_carton and long_box pushed together at the same lane depth
    render_parcel(frame, rig, CATALOG["long_box"], (110.0, 34.0), yaw_deg=6.0)
    render_parcel(frame, rig, CATALOG["cube_carton"], (128.0, 30.0), yaw_deg=-10.0)
    frame = rig.apply_lens_distortion(frame)
    frame = add_sensor_noise(frame, sigma=2.2)
    cv2.imwrite(OUT_PATH, frame)
    print(f"Saved overlap stress-test frame to {OUT_PATH}")


if __name__ == "__main__":
    main()
