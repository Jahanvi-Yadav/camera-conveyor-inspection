"""
run_comparison.py -- Module A deliverable: runs all 7 segmentation methods
on 5 representative frames from the line video PLUS the dedicated overlap
stress-test frame, and saves a side-by-side grid image.

    python module_a_segmentation/run_comparison.py
"""
import os
import sys
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from module_a_segmentation.segmentation_methods import METHODS, label_to_color

VIDEO_PATH = os.path.join(HERE, "..", "data", "line_video.mp4")
OVERLAP_FRAME_PATH = os.path.join(HERE, "..", "data", "overlap_stress_frame.png")
OUT_PATH = os.path.join(HERE, "..", "outputs", "module_a_comparison_grid.png")

FRAME_IDS = [90, 120, 150, 180, 210]
THUMB_W, THUMB_H = 260, 150


def load_frames():
    cap = cv2.VideoCapture(VIDEO_PATH)
    frames = []
    for fi in FRAME_IDS:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"could not read frame {fi}")
        frames.append((f"frame_{fi}", frame))
    cap.release()
    overlap_frame = cv2.imread(OVERLAP_FRAME_PATH)
    frames.append(("overlap_stress", overlap_frame))
    return frames


def thumb(img):
    return cv2.resize(img, (THUMB_W, THUMB_H))


def main():
    frames = load_frames()
    method_names = list(METHODS.keys())

    n_rows = len(frames)
    n_cols = 1 + len(method_names)  # original + each method
    grid = np.full((n_rows * THUMB_H + n_rows * 26, n_cols * THUMB_W, 3), 30, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for r, (fname, frame) in enumerate(frames):
        y0 = r * (THUMB_H + 26)
        row_label_y = y0 + 18
        cv2.putText(grid, fname, (6, row_label_y), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        img_y0 = y0 + 22
        grid[img_y0:img_y0 + THUMB_H, 0:THUMB_W] = thumb(frame)

        for c, mname in enumerate(method_names):
            print(f"[{fname}] running {mname} ...")
            try:
                labels = METHODS[mname](frame)
                colored = label_to_color(labels)
            except Exception as e:
                colored = np.zeros_like(frame)
                cv2.putText(colored, "FAILED", (10, 30), font, 0.7, (0, 0, 255), 2)
                print(f"  !! {mname} failed: {e}")
            x0 = (c + 1) * THUMB_W
            grid[img_y0:img_y0 + THUMB_H, x0:x0 + THUMB_W] = thumb(colored)

    # header row of method names, drawn on top strip
    header = np.full((30, grid.shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(header, "input", (6, 20), font, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    for c, mname in enumerate(method_names):
        x0 = (c + 1) * THUMB_W + 4
        cv2.putText(header, mname, (x0, 20), font, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    grid = np.vstack([header, grid])

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    cv2.imwrite(OUT_PATH, grid)
    print(f"\nSaved comparison grid to {OUT_PATH}")

    print("""
Discussion (Module A):
Watershed handles the overlap_stress frame best: seeding its markers from
LOCAL MAXIMA of the distance transform (one peak per object's own "core"),
rather than a single global threshold, lets it place two separate markers
inside the touching long_box/cube_carton pair and flood-fill each out to
their true shared boundary. Active contours instead grows one snake per
*detected blob*, and two touching parcels are already one connected blob
before the snake ever starts -- so one snake converges around both together
and can't split them; on this frame it collapses to a near-empty contour.
Honest caveat: the same local-maxima seeding can over-segment a single
elongated parcel (long_box, 34x13cm) into two pieces even when it's alone,
since its distance transform has two broad peaks along its length -- a
known watershed failure mode, not something this comparison hides. Region
splitting (pure top-down variance splitting, no merge) and Felzenszwalb
both fragment the parcels' printed textures into several small regions
apiece, which is noisy but harmless since each parcel still reads as a
cluster of same-object regions; Normalized Cut on downsampled SLIC
superpixels is coarser still and sometimes lumps a parcel in with a strip of
belt background at this resolution.
""")


if __name__ == "__main__":
    main()
