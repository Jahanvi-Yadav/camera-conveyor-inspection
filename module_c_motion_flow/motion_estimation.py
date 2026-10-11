"""
motion_estimation.py -- Module C, steps 2-4: sparse feature-based flow for
comparison, a from-scratch RANSAC-robust affine fit to one parcel's dense
flow vectors, and the resulting belt-speed estimate in cm/s.

    python module_c_motion_flow/motion_estimation.py
"""
import os
import sys
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from module_c_motion_flow.pyramidal_lk import lucas_kanade_pyramidal
from vision_lib.geometry import load_calibration, ray_plane_backprojection
from vision_lib.colors import hue_mask, REAL_SIZES_CM

VIDEO_PATH = os.path.join(HERE, "..", "data", "line_video.mp4")
CALIB_PATH = os.path.join(HERE, "..", "module_b_calibration", "camera_calib.json")
OUT_DIR = os.path.join(HERE, "..", "outputs")

FRAME_A, FRAME_B = 150, 151   # one-frame gap -- small-motion regime
FLOW_PARCEL_TYPE = "poly_mailer"  # fully in-frame and well-textured at this frame pair


def load_frame_pair():
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_A)
    ret1, frame_a = cap.read()
    ret2, frame_b = cap.read()
    cap.release()
    if not (ret1 and ret2):
        raise RuntimeError("could not read frame pair")
    return frame_a, frame_b


def draw_flow_arrows(base_img, flow_u, flow_v, reliable, step=18, scale=4.0):
    vis = base_img.copy()
    h, w = flow_u.shape
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            du, dv = flow_u[y, x], flow_v[y, x]
            if not reliable[y, x] or (du * du + dv * dv) < 0.05:
                color = (60, 60, 200)  # unreliable / near-zero: dim red dot
                cv2.circle(vis, (x, y), 1, color, -1)
                continue
            end = (int(x + du * scale), int(y + dv * scale))
            cv2.arrowedLine(vis, (x, y), end, (0, 255, 0), 1, tipLength=0.35)
    return vis


def sparse_flow_opencv(gray_a, gray_b):
    corners = cv2.goodFeaturesToTrack(gray_a, maxCorners=250, qualityLevel=0.01, minDistance=8)
    if corners is None:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0,), dtype=bool)
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(gray_a, gray_b, corners, None,
                                                    winSize=(21, 21), maxLevel=3)
    status = status.reshape(-1).astype(bool)
    prev = corners.reshape(-1, 2)[status]
    nxt = next_pts.reshape(-1, 2)[status]
    return prev, nxt, status


def draw_sparse_flow(base_img, prev_pts, next_pts):
    vis = base_img.copy()
    for p, n in zip(prev_pts, next_pts):
        cv2.arrowedLine(vis, tuple(p.astype(int)), tuple(n.astype(int)), (0, 200, 255), 1, tipLength=0.3)
        cv2.circle(vis, tuple(p.astype(int)), 2, (255, 0, 0), -1)
    return vis


def fit_affine_lstsq(xy, uv):
    """u = a0*x + a1*y + a2 ; v = a3*x + a4*y + a5 -- fit by ordinary least
    squares. Coordinates are expected already centered on their own
    centroid (critical for numerical conditioning: fitting on raw ~700px
    image coordinates over a ~100px-wide sample region is ill-conditioned
    and can blow up under a 3-point RANSAC minimal sample)."""
    n = len(xy)
    A = np.zeros((2 * n, 6))
    b = np.zeros(2 * n)
    A[0::2, 0] = xy[:, 0]
    A[0::2, 1] = xy[:, 1]
    A[0::2, 2] = 1.0
    b[0::2] = uv[:, 0]
    A[1::2, 3] = xy[:, 0]
    A[1::2, 4] = xy[:, 1]
    A[1::2, 5] = 1.0
    b[1::2] = uv[:, 1]
    params, *_ = np.linalg.lstsq(A, b, rcond=None)
    return params  # a0..a5


def predict_affine(params, xy):
    a0, a1, a2, a3, a4, a5 = params
    u = a0 * xy[:, 0] + a1 * xy[:, 1] + a2
    v = a3 * xy[:, 0] + a4 * xy[:, 1] + a5
    return np.stack([u, v], axis=1)


def ransac_affine(xy, uv, n_iters=400, sample_size=3, inlier_thresh=0.6, rng=None):
    """From-scratch RANSAC: repeatedly fit an affine flow model to a
    minimal random sample, count inliers (predicted vs actual flow within
    inlier_thresh px), keep the best-supported model, then refit on all its
    inliers. xy is expected pre-centered on its own centroid."""
    rng = rng or np.random.default_rng(3)
    n = len(xy)
    best_inliers = None
    best_count = -1
    for _ in range(n_iters):
        if n < sample_size:
            break
        idx = rng.choice(n, size=sample_size, replace=False)
        try:
            params = fit_affine_lstsq(xy[idx], uv[idx])
        except np.linalg.LinAlgError:
            continue
        pred = predict_affine(params, xy)
        residual = np.linalg.norm(pred - uv, axis=1)
        inliers = residual < inlier_thresh
        count = inliers.sum()
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_inliers.sum() < sample_size:
        final_params = fit_affine_lstsq(xy, uv)
        return final_params, np.ones(n, dtype=bool)

    final_params = fit_affine_lstsq(xy[best_inliers], uv[best_inliers])
    return final_params, best_inliers


def main():
    frame_a, frame_b = load_frame_pair()
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

    print(f"Dense from-scratch pyramidal Lucas-Kanade on frames {FRAME_A}->{FRAME_B} ...")
    flow_u, flow_v, reliable = lucas_kanade_pyramidal(gray_a, gray_b, levels=3, window=15)
    dense_vis = draw_flow_arrows(frame_a, flow_u, flow_v, reliable, step=16, scale=12.0)
    os.makedirs(OUT_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(OUT_DIR, "module_c_dense_flow.png"), dense_vis)

    print("OpenCV sparse feature-based flow (goodFeaturesToTrack + calcOpticalFlowPyrLK) ...")
    prev_pts, next_pts, _ = sparse_flow_opencv(gray_a, gray_b)
    sparse_vis = draw_sparse_flow(frame_a, prev_pts, next_pts)
    cv2.imwrite(os.path.join(OUT_DIR, "module_c_sparse_flow.png"), sparse_vis)

    # -- RANSAC affine fit on one parcel's own dense flow vectors --
    mask = hue_mask(frame_a, FLOW_PARCEL_TYPE, top_face_only=True)
    ys, xs = np.where(mask > 0)
    if len(xs) < 10:
        raise RuntimeError(f"not enough {FLOW_PARCEL_TYPE} pixels in frame {FRAME_A} to fit flow")

    xy_raw = np.stack([xs, ys], axis=1).astype(np.float64)
    uv = np.stack([flow_u[ys, xs], flow_v[ys, xs]], axis=1).astype(np.float64)
    good = reliable[ys, xs]
    xy_raw, uv = xy_raw[good], uv[good]

    centroid = xy_raw.mean(axis=0)
    xy_centered = xy_raw - centroid  # numerical conditioning: see fit_affine_lstsq docstring

    params, inlier_mask = ransac_affine(xy_centered, uv)
    n_in, n_total = inlier_mask.sum(), len(inlier_mask)
    print(f"RANSAC affine fit on {FLOW_PARCEL_TYPE}'s flow: {n_in}/{n_total} inliers")
    print(f"  fitted params (a0..a5): {np.round(params, 4)}")

    # visualize inliers (green) vs outliers (red) over the parcel region
    ransac_vis = frame_a.copy()
    for (x, y), is_in in zip(xy_raw.astype(int), inlier_mask):
        color = (0, 255, 0) if is_in else (0, 0, 255)
        cv2.circle(ransac_vis, (x, y), 2, color, -1)
    cv2.imwrite(os.path.join(OUT_DIR, "module_c_ransac_inliers.png"), ransac_vis)

    # translation component of the fitted affine model = (a2, a5) --
    # the flow at the parcel's own centroid (since xy is centered there)
    du_centroid, dv_centroid = params[2], params[5]

    # -- convert to a real belt-speed estimate using Module B calibration --
    K, dist, rvec_belt, tvec_belt = load_calibration(CALIB_PATH)
    _, _, height_cm = REAL_SIZES_CM[FLOW_PARCEL_TYPE]
    p0_px = centroid
    p1_px = centroid + np.array([du_centroid, dv_centroid])
    world_pts = ray_plane_backprojection(np.stack([p0_px, p1_px]), K, dist, rvec_belt, tvec_belt,
                                          plane_z=height_cm)
    disp_cm = np.linalg.norm(world_pts[1] - world_pts[0])

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.release()
    dt = 1.0 / fps
    speed_cm_s = disp_cm / dt

    print(f"\nPixel displacement (1 frame, {FLOW_PARCEL_TYPE} centroid): "
          f"({du_centroid:.2f}, {dv_centroid:.2f}) px")
    print(f"Back-projected world displacement: {disp_cm:.3f} cm over {dt*1000:.1f} ms")
    print(f"==> Estimated belt speed: {speed_cm_s:.2f} cm/s   (ground truth: 22.00 cm/s)")

    print("""
Discussion: dense LK flags large stretches of the plain belt surface (and
the flat, low-contrast interior of some parcel top faces) as UNRELIABLE --
that's the aperture problem, since a locally uniform patch's structure
tensor has a near-zero smaller eigenvalue and the brightness-constancy
system is under-determined there. Sparse feature-based flow never has this
failure mode as a visible "wrong answer" because goodFeaturesToTrack simply
never PLACES a corner on those same textureless regions to begin with --
the two methods aren't disagreeing on an answer there so much as dense flow
is the only one of the two that even attempts one, and honestly says so.
""")


if __name__ == "__main__":
    main()
