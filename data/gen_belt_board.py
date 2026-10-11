"""
gen_belt_board.py -- one extra checkerboard photo with the board lying FLAT
on the sorting line's belt plane (world Z=0), at a known world position, so
Module B's second solvePnP call can recover the camera's pose relative to
the belt itself (needed to back-project any pixel to real belt-plane cm).
"""
import os
import sys
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from vision_lib.camera_rig import CameraRig, IMG_W, IMG_H
from data.gen_calib_photos import render_board_texture, render_frame_with_board, BOARD_W_CM, BOARD_H_CM

OUT_PATH = os.path.join(HERE, "belt_board.png")
# Closer to the camera and off to the near side of the line (rather than
# dead-center under it) so the board spans a much wider angular extent and
# picks up real perspective foreshortening -- a small, near-fronto-parallel
# patch near the image center is a classic recipe for the planar pose
# ambiguity (accurate reprojection on the board's own corners, but a
# rotation/depth that doesn't generalize to the rest of the scene).
BELT_BOARD_CENTER_WORLD = np.array([95.0, 0.0, 0.0])


def main():
    rig = CameraRig()
    tex = render_board_texture()
    frame = render_frame_with_board(rig, BELT_BOARD_CENTER_WORLD, yaw_deg=6.0, pitch_deg=0.0, roll_deg=0.0, tex=tex)
    frame = rig.apply_lens_distortion(frame)
    cv2.imwrite(OUT_PATH, frame)
    print(f"Saved belt-plane board photo to {OUT_PATH}")
    print(f"Known board center (world cm): {BELT_BOARD_CENTER_WORLD.tolist()}, "
          f"size {BOARD_W_CM:.1f} x {BOARD_H_CM:.1f} cm")


if __name__ == "__main__":
    main()
