"""
kalman_tracker.py -- Module D: a from-scratch constant-velocity Kalman
filter tracking one parcel's centroid across the full line video, fed noisy
centroid measurements from Module A-style hue-mask detection.

State is ordered [x, vx, y, vy] (axis-major, not [x,y,vx,vy]) -- a
different convention from a position-then-velocity block layout, built as
two independent constant-velocity sub-blocks (x-axis, y-axis) that happen
to share one state vector.

    python module_d_tracking/kalman_tracker.py
"""
import os
import sys
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from vision_lib.colors import get_parcel_quad

VIDEO_PATH = os.path.join(HERE, "..", "data", "line_video.mp4")
GT_PATH = os.path.join(HERE, "..", "data", "ground_truth_line_video.json")
OUT_PLOT = os.path.join(HERE, "..", "outputs", "module_d_tracking_plot.png")

TRACK_PARCEL_TYPE = "poly_mailer"
MEASUREMENT_NOISE_PX = 3.2
DROPOUT_FRAMES = range(140, 144)   # 4 consecutive frames -- simulated occlusion
PRINT_FRAMES = range(70, 80)       # 10 consecutive frames of predict/measure/update


class ConstantVelocityKalman:
    """State x = [px, vx, py, vy]^T. Two independent constant-velocity
    axes sharing one filter so a single predict()/update() call advances
    both at once."""

    def __init__(self, dt, x0, measurement_noise_px=3.0, process_noise_accel=8.0):
        self.dt = dt
        self.x = np.array([x0[0], 0.0, x0[1], 0.0], dtype=np.float64)

        F1 = np.array([[1, dt], [0, 1]])
        self.F = np.zeros((4, 4))
        self.F[0:2, 0:2] = F1
        self.F[2:4, 2:4] = F1

        self.H = np.array([[1, 0, 0, 0],
                            [0, 0, 1, 0]], dtype=np.float64)

        # discrete white-noise-acceleration process noise, one 2x2 block
        # per axis, driven by process_noise_accel (cm/s^2-ish tuning knob)
        q = process_noise_accel
        Q1 = q * np.array([[dt**4 / 4, dt**3 / 2],
                            [dt**3 / 2, dt**2]])
        self.Q = np.zeros((4, 4))
        self.Q[0:2, 0:2] = Q1
        self.Q[2:4, 2:4] = Q1

        self.R = np.eye(2) * (measurement_noise_px ** 2)
        self.P = np.eye(4) * 50.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x.copy()

    @property
    def position(self):
        return self.x[0], self.x[2]


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    dt = 1.0 / fps
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    rng = np.random.default_rng(11)
    kf = None
    raw_track, filtered_track, gt_track = [], [], []

    import json
    gt = json.load(open(GT_PATH))
    # find the ground-truth parcel id matching TRACK_PARCEL_TYPE
    gt_id = None
    for pid, info in gt["frames"][100]["parcels"].items():
        if info["type"] == TRACK_PARCEL_TYPE:
            gt_id = pid
            break

    print(f"Tracking '{TRACK_PARCEL_TYPE}' (ground truth id {gt_id}) across {n_frames} frames.\n")
    print(f"{'frame':>6s} | {'predicted (x,y)':>18s} | {'measured (x,y)':>18s} | {'corrected (x,y)':>18s}")
    print("-" * 72)

    for fi in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        quad = get_parcel_quad(frame, TRACK_PARCEL_TYPE, top_face_only=True)
        z = None
        if quad is not None and fi not in DROPOUT_FRAMES:
            cx, cy = quad["center"]
            noisy = np.array([cx, cy]) + rng.normal(0, MEASUREMENT_NOISE_PX, size=2)
            z = noisy

        if kf is None:
            if z is None:
                continue
            kf = ConstantVelocityKalman(dt, x0=z, measurement_noise_px=MEASUREMENT_NOISE_PX)
            pred = kf.x[0], kf.x[2]
        else:
            pred = kf.predict()
            pred = (pred[0], pred[2])
            if z is not None:
                kf.update(z)

        cx_f, cy_f = kf.position
        raw_track.append((fi, z[0], z[1]) if z is not None else (fi, np.nan, np.nan))
        filtered_track.append((fi, cx_f, cy_f))
        gt_track.append((fi, gt["frames"][fi]["parcels"][gt_id]["centroid_px"][0],
                          gt["frames"][fi]["parcels"][gt_id]["centroid_px"][1]))

        if fi in PRINT_FRAMES:
            meas_str = f"({z[0]:6.1f},{z[1]:6.1f})" if z is not None else "   -- dropped --  "
            print(f"{fi:6d} | ({pred[0]:6.1f},{pred[1]:6.1f}) | {meas_str:>18s} | "
                  f"({cx_f:6.1f},{cy_f:6.1f})")
        if fi == min(DROPOUT_FRAMES):
            print(f"\n  -- simulating occlusion: frames {DROPOUT_FRAMES.start}-{DROPOUT_FRAMES.stop - 1} "
                  f"have their measurement discarded entirely --")
        if fi == max(DROPOUT_FRAMES):
            pred_only = kf.position
            print(f"  -- occlusion ends; pure-prediction estimate was ({pred_only[0]:.1f},{pred_only[1]:.1f}) --\n")

    cap.release()

    # -- error vs ground truth --
    errs = []
    for (fi, fx, fy), (_, gx, gy) in zip(filtered_track, gt_track):
        errs.append(np.hypot(fx - gx, fy - gy))
    print(f"Mean tracking error vs ground truth: {np.mean(errs):.2f} px over {len(errs)} frames")

    plot_results(raw_track, filtered_track, gt_track)


def plot_results(raw_track, filtered_track, gt_track):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = np.array(raw_track, dtype=np.float64)
    filt = np.array(filtered_track, dtype=np.float64)
    gtt = np.array(gt_track, dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, coord, label in [(axes[0], 1, "x (px)"), (axes[1], 2, "y (px)")]:
        ax.plot(gtt[:, 0], gtt[:, coord], color="gray", lw=1, label="ground truth")
        ax.scatter(raw[:, 0], raw[:, coord], s=8, color="crimson", alpha=0.5, label="raw noisy measurement")
        ax.plot(filt[:, 0], filt[:, coord], color="royalblue", lw=1.6, label="Kalman-filtered")
        dropout_mask = (filt[:, 0] >= DROPOUT_FRAMES.start) & (filt[:, 0] < DROPOUT_FRAMES.stop)
        ax.plot(filt[dropout_mask, 0], filt[dropout_mask, coord], color="orange", lw=2.8,
                label="prediction-only (dropout)")
        ax.set_xlabel("frame")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        ax.set_title(f"{label} vs frame")

    fig.suptitle(f"Module D -- Kalman tracking of '{TRACK_PARCEL_TYPE}' "
                 f"(orange = {DROPOUT_FRAMES.stop - DROPOUT_FRAMES.start}-frame occlusion)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PLOT), exist_ok=True)
    fig.savefig(OUT_PLOT, dpi=130)
    print(f"Saved plot to {OUT_PLOT}")


if __name__ == "__main__":
    main()
