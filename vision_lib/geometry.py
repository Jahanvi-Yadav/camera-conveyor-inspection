"""
geometry.py -- pixel <-> world back-projection shared by calibration,
optical flow's speed conversion, and the integration pipeline.
"""
import json
import numpy as np
import cv2


def ray_plane_backprojection(pixel_pts: np.ndarray, K, dist, rvec, tvec, plane_z=0.0):
    """For each (u,v) pixel, undistorts it to a normalized ray in camera
    space, rotates the ray into world space, and intersects it with the
    horizontal plane Z=plane_z. Returns (N,3) world coordinates.

    This is the exact-geometry equivalent of "where on the belt (or at
    parcel-top height) is this pixel actually looking" -- used instead of a
    single global pixels-per-cm scale.
    """
    pixel_pts = np.asarray(pixel_pts, dtype=np.float64).reshape(-1, 1, 2)
    norm_pts = cv2.undistortPoints(pixel_pts, K, dist)  # (N,1,2), camera-space rays z=1
    rays_cam = np.concatenate([norm_pts.reshape(-1, 2), np.ones((len(norm_pts), 1))], axis=1)

    R, _ = cv2.Rodrigues(rvec)
    R_inv = R.T
    cam_center_world = (-R_inv @ tvec.reshape(3, 1)).reshape(3)

    rays_world = (R_inv @ rays_cam.T).T  # direction vectors in world space

    # solve cam_center_world + s*ray = (x, y, plane_z) for s
    s = (plane_z - cam_center_world[2]) / rays_world[:, 2]
    world_pts = cam_center_world[None, :] + s[:, None] * rays_world
    return world_pts


def save_calibration(path, K, dist, rvec_belt, tvec_belt, extra=None):
    payload = {
        "K": K.tolist(),
        "dist": dist.tolist(),
        "rvec_belt": np.asarray(rvec_belt).reshape(-1).tolist(),
        "tvec_belt": np.asarray(tvec_belt).reshape(-1).tolist(),
    }
    if extra:
        payload.update(extra)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_calibration(path):
    with open(path) as f:
        payload = json.load(f)
    K = np.array(payload["K"], dtype=np.float64)
    dist = np.array(payload["dist"], dtype=np.float64)
    rvec = np.array(payload["rvec_belt"], dtype=np.float64).reshape(3, 1)
    tvec = np.array(payload["tvec_belt"], dtype=np.float64).reshape(3, 1)
    return K, dist, rvec, tvec
