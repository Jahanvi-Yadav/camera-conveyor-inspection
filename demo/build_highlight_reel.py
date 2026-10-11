"""
build_highlight_reel.py -- builds a short SILENT highlight reel from this
project's own output images -- not a substitute for the assignment's
requested 10-minute narrated screen recording (this session has no voice or
the student's own screen to record), but a ready visual aid:
narration_script.md is timed to be read aloud alongside a live run of the
module scripts, and this reel is a quick sanity-check that every required
visual actually exists and looks right.
"""
import os
import subprocess
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "outputs")
FRAMES_DIR = os.path.join(HERE, "frames")
os.makedirs(FRAMES_DIR, exist_ok=True)

CANVAS_W, CANVAS_H = 1280, 720
SECONDS_PER_SLIDE = 5

ACCENT = (235, 170, 30)   # BGR amber -- matches poly_mailer's palette
BG = (24, 24, 24)
HEADER_H = 60

SLIDES = [
    ("TITLE", "Camera-Based Sorting-Line\nInspection & Tracking System", None),
    ("TEXT", "Module A -- Segmentation\n7 classical methods, incl. a dedicated touching-parcels stress frame", None),
    ("IMG", "module_a_comparison_grid.png", "Watershed's local-maxima markers split the touching pair; active contours cannot"),
    ("TEXT", "Module B -- Camera Calibration\ncv2.calibrateCamera + undistortion + P=K[R|t] + belt-plane solvePnP", None),
    ("IMG", "module_b_undistort_before_after.png", "Recovered lens distortion straightens the belt's guide lines"),
    ("IMG", "module_b_measure_frame.png", "Reference (green) vs unknown (red) parcel used for the 4 camera-model size estimate"),
    ("TEXT", "Module C -- Optical Flow\nFrom-scratch PYRAMIDAL Lucas-Kanade + RANSAC affine -> belt speed", None),
    ("IMG", "module_c_dense_flow.png", "Dense flow arrows on the parcels; belt surface correctly flagged unreliable"),
    ("IMG", "module_c_ransac_inliers.png", "RANSAC inlier/outlier split -> belt speed within ~0.5% of ground truth"),
    ("TEXT", "Module D -- Kalman Filter Tracking\nFrom-scratch constant-velocity filter, state [x,vx,y,vy], 220 frames", None),
    ("IMG", "module_d_tracking_plot.png", "Smoothed trajectory tracked straight through a simulated 4-frame occlusion"),
    ("TEXT", "Module E -- Object Recognition\nSIFT alignment vs Eigenspace (PCA) vs Hu moments -- crops sampled from the video itself", None),
    ("IMG", "module_e_templates_and_samples.png", "Reference templates vs held-out samples, per parcel type"),
    ("IMG", "module_e_eigenvectors.png", "Top-3 'eigenboxes' learned by PCA"),
    ("TEXT", "Integration: pipeline.py\nID  Type          Size(cm)      Speed(cm/s)  Recognized\nP1  poly_mailer   30.6x20.4     20.9         poly_mailer\nP2  cube_carton   21.4x18.0     24.0         cube_carton\nP3  long_box      38.6x15.4     19.3         long_box", None),
    ("TEXT", "All code + outputs in this repo.\nSee narration_script.md for the full 10-minute walkthrough.", None),
]


def make_header(title_text):
    bar = np.full((HEADER_H, CANVAS_W, 3), ACCENT, dtype=np.uint8)
    cv2.putText(bar, title_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
    return bar


def make_text_slide(text):
    canvas = np.full((CANVAS_H, CANVAS_W, 3), BG, dtype=np.uint8)
    lines = text.split("\n")
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.15 if len(lines) <= 2 else 0.72
    thickness = 2
    line_h = int(46 * scale) + 22
    total_h = line_h * len(lines)
    y0 = (CANVAS_H - total_h) // 2 + line_h // 2
    for i, line in enumerate(lines):
        (tw, th), _ = cv2.getTextSize(line, font, scale, thickness)
        x = (CANVAS_W - tw) // 2
        y = y0 + i * line_h
        cv2.putText(canvas, line, (x, y), font, scale, (235, 235, 235), thickness, cv2.LINE_AA)
    canvas[0:4, :] = ACCENT
    return canvas


def make_image_slide(img_path, caption, header_text):
    img = cv2.imread(img_path)
    canvas = np.full((CANVAS_H, CANVAS_W, 3), BG, dtype=np.uint8)
    canvas[0:HEADER_H] = make_header(header_text)
    cap_h = 60 if caption else 0
    avail_h = CANVAS_H - HEADER_H - cap_h
    h, w = img.shape[:2]
    scale = min(CANVAS_W / w, avail_h / h) * 0.94
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    x0 = (CANVAS_W - new_w) // 2
    y0 = HEADER_H + (avail_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    if caption:
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(caption, font, 0.62, 1)
        cx = (CANVAS_W - tw) // 2
        cv2.putText(canvas, caption, (cx, CANVAS_H - 22), font, 0.62, (210, 210, 150), 1, cv2.LINE_AA)
    return canvas


def main():
    idx = 0
    current_header = "Sorting-Line Inspection System"
    for kind, a, b in SLIDES:
        if kind in ("TITLE", "TEXT"):
            slide = make_text_slide(a)
        else:
            slide = make_image_slide(os.path.join(OUT_DIR, a), b, current_header)
        for _ in range(SECONDS_PER_SLIDE):
            cv2.imwrite(os.path.join(FRAMES_DIR, f"frame_{idx:03d}.png"), slide)
            idx += 1

    out_path = os.path.join(HERE, "demo_highlights.mp4")
    cmd = [
        "ffmpeg", "-y", "-framerate", "1", "-i", os.path.join(FRAMES_DIR, "frame_%03d.png"),
        "-vf", "fps=25,format=yuv420p", "-c:v", "libx264", out_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"Wrote {out_path} ({idx} seconds)")


if __name__ == "__main__":
    main()
