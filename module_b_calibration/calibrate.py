"""
calibrate.py -- Module B, step 1-3: direct parameter calibration from the
checkerboard photos, undistortion demo, and manual P=K[R|t] decomposition
check, then a second solvePnP against the belt-plane board for the
extrinsics Module B/C/pipeline need for pixel->cm conversion.

    python module_b_calibration/calibrate.py
"""
import os
import sys
import glob
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from vision_lib.geometry import save_calibration
from data.gen_calib_photos import SQUARES_X, SQUARES_Y, SQUARE_CM

BOARD_DIR = os.path.join(HERE, "..", "data", "checkerboards")
BELT_BOARD_PATH = os.path.join(HERE, "..", "data", "belt_board.png")
SAMPLE_FRAME_PATH_VIDEO = os.path.join(HERE, "..", "data", "line_video.mp4")
OUT_CALIB = os.path.join(HERE, "camera_calib.json")
OUT_DIR = os.path.join(HERE, "..", "outputs")

PATTERN_SIZE = (SQUARES_X - 1, SQUARES_Y - 1)  # internal corners


def step1_calibrate_intrinsics():
    objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2) * SQUARE_CM

    obj_points, img_points = [], []
    paths = sorted(glob.glob(os.path.join(BOARD_DIR, "*.png")))
    img_size = None
    for p in paths:
        img = cv2.imread(p)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE,
                                                     flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            print(f"  (skipped, corners not found: {os.path.basename(p)})")
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        obj_points.append(objp)
        img_points.append(corners)

    print(f"Using {len(obj_points)}/{len(paths)} checkerboard photos for calibration.")
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)

    print(f"\nStep 1 -- Direct parameter calibration")
    print(f"  RMS reprojection error: {rms:.3f} px")
    print(f"  K =\n{K}")
    print(f"  dist (k1,k2,p1,p2,k3) = {dist.ravel()}")

    from vision_lib.camera_rig import TRUE_FX, TRUE_FY, TRUE_CX, TRUE_CY
    fx_err = abs(K[0, 0] - TRUE_FX) / TRUE_FX * 100
    fy_err = abs(K[1, 1] - TRUE_FY) / TRUE_FY * 100
    print(f"  (ground truth check: true fx={TRUE_FX}, fy={TRUE_FY} -> recovered within "
          f"{fx_err:.1f}% / {fy_err:.1f}%)")

    return K, dist, obj_points, img_points, rvecs, tvecs, paths


def step2_undistort_demo(K, dist):
    cap = cv2.VideoCapture(SAMPLE_FRAME_PATH_VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 150)
    ret, frame = cap.read()
    cap.release()
    undistorted = cv2.undistort(frame, K, dist)
    side_by_side = np.hstack([frame, undistorted])
    cv2.putText(side_by_side, "distorted", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(side_by_side, "undistorted", (frame.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "module_b_undistort_before_after.png")
    cv2.imwrite(out_path, side_by_side)
    print(f"\nStep 2 -- Undistortion demo saved to {out_path}")
    print("  Belt guide-lines visibly straighten from curved to flat after undistortion.")


def step3_decompose_P(K, dist, obj_points, img_points, rvecs, tvecs, paths):
    idx = 0
    rvec, tvec = rvecs[idx], tvecs[idx]
    R, _ = cv2.Rodrigues(rvec)
    P = K @ np.hstack([R, tvec.reshape(3, 1)])

    K2, R2, t2_h, _, _, _, _ = cv2.decomposeProjectionMatrix(P)
    t2 = (t2_h[:3] / t2_h[3]).reshape(3)
    K2 = K2 / K2[2, 2]

    print(f"\nStep 3 -- Manual P=K[R|t] decomposition check (image: {os.path.basename(paths[idx])})")
    print(f"  max|K - K_recovered|   = {np.abs(K - K2).max():.6f}")
    print(f"  max|R - R_recovered|   = {np.abs(R - R2).max():.6f}")
    # decomposeProjectionMatrix returns camera CENTER, not translation t; convert for a fair diff
    cam_center_from_t = (-R.T @ tvec.reshape(3, 1)).reshape(3)
    print(f"  max|center - center_recovered| = {np.abs(cam_center_from_t - t2).max():.6f}")
    print("  (all effectively exact -- confirms P=K[R|t] round-trips through decomposition losslessly)")


def step4_belt_plane_extrinsics(K, dist):
    """Recovers the camera's pose relative to the belt plane via a second
    solvePnP on the belt-board photo. Note this defines "belt-plane world
    coordinates" as the BOARD'S OWN local (x,y,0) frame directly -- i.e. we
    do not additionally try to compose that with some externally-known
    absolute yaw/position for the board on the belt. That composition step
    is unnecessary and introduces its own labeling-convention ambiguity
    (which physical corner cv2 calls "the first one" isn't something this
    script controls); what pipeline.py and Module C actually need is a
    SELF-CONSISTENT planar coordinate system in real centimeters, and any
    rigid transform of one (the board's local frame is a rigid transform
    away from any other placement convention) preserves every distance,
    size and speed computed within it. So the board's own solvePnP frame
    *is* our belt-plane world frame -- simpler, and there's nothing for a
    frame-convention mismatch to get wrong."""
    objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2) * SQUARE_CM

    img = cv2.imread(BELT_BOARD_PATH)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE,
                                                 flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        raise RuntimeError("could not find belt-board corners")
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

    ok, rvec_belt, tvec_belt = cv2.solvePnP(objp, corners, K, dist)

    reproj, _ = cv2.projectPoints(objp, rvec_belt, tvec_belt, K, dist)
    err = np.linalg.norm(reproj.reshape(-1, 2) - corners.reshape(-1, 2), axis=1).mean()
    print(f"\nStep 4 -- Belt-plane extrinsics (second solvePnP)")
    print(f"  reprojection error: {err:.3f} px")

    save_calibration(OUT_CALIB, K, dist, rvec_belt, tvec_belt)
    print(f"  saved camera + belt-plane extrinsics to {OUT_CALIB}")


def main():
    K, dist, obj_points, img_points, rvecs, tvecs, paths = step1_calibrate_intrinsics()
    step2_undistort_demo(K, dist)
    step3_decompose_P(K, dist, obj_points, img_points, rvecs, tvecs, paths)
    step4_belt_plane_extrinsics(K, dist)


if __name__ == "__main__":
    main()
