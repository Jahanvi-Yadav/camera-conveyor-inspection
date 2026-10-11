"""
camera_rig.py -- a small, class-based virtual camera used to synthesize the
test footage for this project (see README for why synthetic footage is used
at all). World frame: X runs along the sorting line's direction of travel,
Y is depth away from the camera's nadir point, Z is height above the belt
(belt surface = Z=0), all in centimeters.

This is intentionally a different frame (a CameraRig class with methods)
from a flat functions-only camera module -- it bundles intrinsics,
distortion and pose together as one object that the data generators and
Module B both instantiate.
"""
from dataclasses import dataclass, field
import numpy as np
import cv2

IMG_W, IMG_H = 1024, 576

# Intrinsics: a slightly longer lens / narrower field of view than a phone
# camera, consistent with a ceiling unit mounted a bit further back.
TRUE_FX, TRUE_FY = 760.0, 760.0
TRUE_CX, TRUE_CY = 512.0, 288.0

# Radial (k1,k2,k3) + tangential (p1,p2) distortion of the synthetic lens.
TRUE_DIST = np.array([-0.18, 0.04, 0.0009, -0.0006, -0.006], dtype=np.float64)

# Camera pose: mounted higher and looking down at a steeper angle than a
# typical handheld shot, further back along Y so the whole line is framed.
CAM_POSITION = np.array([135.0, -95.0, 178.0])
CAM_LOOKAT = np.array([135.0, 30.0, 0.0])
WORLD_UP = np.array([0.0, 0.0, 1.0])


@dataclass
class CameraRig:
    fx: float = TRUE_FX
    fy: float = TRUE_FY
    cx: float = TRUE_CX
    cy: float = TRUE_CY
    dist: np.ndarray = field(default_factory=lambda: TRUE_DIST.copy())
    position: np.ndarray = field(default_factory=lambda: CAM_POSITION.copy())
    lookat: np.ndarray = field(default_factory=lambda: CAM_LOOKAT.copy())

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx],
                          [0, self.fy, self.cy],
                          [0, 0, 1]], dtype=np.float64)

    def rotation_translation(self):
        """Look-at construction of R (world->camera) and t, with a
        determinant/orthonormality safety check (a flipped basis would
        silently mirror every downstream projection)."""
        forward = self.lookat - self.position
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, WORLD_UP)
        right = right / np.linalg.norm(right)
        true_up = np.cross(right, forward)
        # camera-space axes: x=right, y=down (=-true_up), z=forward
        R = np.stack([right, -true_up, forward], axis=0)
        if abs(np.linalg.det(R) - 1.0) > 1e-6:
            raise ValueError(f"CameraRig basis is not a proper rotation (det={np.linalg.det(R):.4f})")
        t = -R @ self.position
        return R, t

    def rvec_tvec(self):
        R, t = self.rotation_translation()
        rvec, _ = cv2.Rodrigues(R)
        return rvec, t.reshape(3, 1)

    def project(self, world_pts: np.ndarray) -> np.ndarray:
        """world_pts: (N,3) -> pixel coords (N,2), with lens distortion applied."""
        rvec, tvec = self.rvec_tvec()
        pix, _ = cv2.projectPoints(world_pts.reshape(-1, 1, 3).astype(np.float64),
                                    rvec, tvec, self.K, self.dist)
        return pix.reshape(-1, 2)

    def camera_space_depth(self, world_pts: np.ndarray) -> np.ndarray:
        R, t = self.rotation_translation()
        cam_pts = (R @ world_pts.T).T + t
        return cam_pts[:, 2]

    def build_distortion_remap(self, w=IMG_W, h=IMG_H):
        """Forward-distorts a clean rendered image by building a remap that
        answers, for every DISTORTED pixel, which UNDISTORTED pixel it
        pulls from -- via cv2.undistortPoints run on the full pixel grid."""
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        distorted_pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float64)
        undistorted_norm = cv2.undistortPoints(distorted_pts.reshape(-1, 1, 2), self.K, self.dist)
        undistorted_norm = undistorted_norm.reshape(-1, 2)
        ux = undistorted_norm[:, 0] * self.fx + self.cx
        uy = undistorted_norm[:, 1] * self.fy + self.cy
        map_x = ux.reshape(h, w).astype(np.float32)
        map_y = uy.reshape(h, w).astype(np.float32)
        return map_x, map_y

    def apply_lens_distortion(self, clean_image: np.ndarray) -> np.ndarray:
        map_x, map_y = self.build_distortion_remap(clean_image.shape[1], clean_image.shape[0])
        return cv2.remap(clean_image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
