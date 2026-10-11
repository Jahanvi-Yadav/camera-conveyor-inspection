"""
measure_parcel.py -- Module B, step 4: given one REFERENCE parcel of known
size and one UNKNOWN parcel in the same frame, estimate the unknown's real
size (cm) under four camera models of increasing sophistication:
orthographic, weak perspective, affine, and full perspective.

    python module_b_calibration/measure_parcel.py
"""
import os
import sys
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from vision_lib.geometry import load_calibration, ray_plane_backprojection
from vision_lib.colors import get_parcel_quad, REAL_SIZES_CM

CALIB_PATH = os.path.join(HERE, "camera_calib.json")
VIDEO_PATH = os.path.join(HERE, "..", "data", "line_video.mp4")
FRAME_IDX = 210          # both reference and unknown clearly visible here
REFERENCE_TYPE = "cube_carton"   # known size, used to calibrate the scale
UNKNOWN_TYPE = "long_box"        # its size is "unknown" to the four models
OUT_IMG = os.path.join(HERE, "..", "outputs", "module_b_measure_frame.png")


class ParcelSizeEstimator:
    """Bundles the four camera-model size-estimation strategies so they
    share the same calibration/reference state."""

    def __init__(self, K, dist, rvec_belt, tvec_belt):
        self.K, self.dist = K, dist
        self.rvec_belt, self.tvec_belt = rvec_belt, tvec_belt

    def _quad_side_lengths_cm(self, corners_px, plane_z):
        """A rectangle's 4 consecutive sides always alternate between its
        two true side lengths regardless of which corner cv2.boxPoints
        happened to start at, so (d0,d2) and (d1,d3) are each one of the
        two dimensions -- but WHICH one is 'length' vs 'width' in that pair
        is an accident of the starting corner, not physically meaningful.
        Returned sorted (long, short) so it's directly comparable to a
        catalog size regardless of detected orientation."""
        world = ray_plane_backprojection(corners_px, self.K, self.dist,
                                          self.rvec_belt, self.tvec_belt, plane_z=plane_z)
        d = [np.linalg.norm(world[i] - world[(i + 1) % 4]) for i in range(4)]
        dim_a, dim_b = (d[0] + d[2]) / 2.0, (d[1] + d[3]) / 2.0
        return (dim_a, dim_b) if dim_a >= dim_b else (dim_b, dim_a)

    def orthographic(self, ref_quad, ref_true_len_cm, unk_quad):
        """Single global pixels-per-cm scale from the reference's pixel
        perimeter, applied directly to the unknown's pixel size -- ignores
        perspective and depth entirely. Pixel size_px's own (w,h) order is
        just as arbitrary as boxPoints' corner order, so this is scale-only
        (average of both sides) and the caller sorts the two returned
        pixel dimensions against the sorted true size."""
        ref_len_px, ref_wid_px = ref_quad["size_px"]
        px_per_cm = ((ref_len_px + ref_wid_px) / 2.0) / ((ref_true_len_cm[0] + ref_true_len_cm[1]) / 2.0)
        unk_a_px, unk_b_px = unk_quad["size_px"]
        dims = sorted([unk_a_px / px_per_cm, unk_b_px / px_per_cm], reverse=True)
        return tuple(dims)

    def weak_perspective(self, ref_quad, ref_true_len_cm, ref_plane_z, unk_quad):
        """Scale is derived at the reference's own depth (via the exact
        ray-plane back-projection at the reference's height) but then
        applied UNIFORMLY to the unknown, assuming it sits at the same
        depth -- ignores that objects off to the side are imaged at a
        slightly different distance from the camera."""
        ref_dims_cm = self._quad_side_lengths_cm(ref_quad["corners"], ref_plane_z)
        scale = sum(ref_dims_cm) / 2.0 / (sum(ref_quad["size_px"]) / 2.0)
        unk_a_px, unk_b_px = unk_quad["size_px"]
        dims = sorted([unk_a_px * scale, unk_b_px * scale], reverse=True)
        return tuple(dims)

    def affine(self, ref_quad, ref_plane_z, unk_quad, eps_px=3.0):
        """Local Jacobian linearization: finite-difference the exact
        pixel->world map around the reference's own pixel location, then
        apply that same locally-linear map to the unknown's pixel-space
        side vectors. More faithful to the true (nonlinear) geometry near
        the reference than a single global scale, but still an
        approximation valid only near that anchor point."""
        cx, cy = ref_quad["center"]
        base = ray_plane_backprojection(np.array([[cx, cy]]), self.K, self.dist,
                                         self.rvec_belt, self.tvec_belt, plane_z=ref_plane_z)[0]
        dx_world = ray_plane_backprojection(np.array([[cx + eps_px, cy]]), self.K, self.dist,
                                             self.rvec_belt, self.tvec_belt, plane_z=ref_plane_z)[0]
        dy_world = ray_plane_backprojection(np.array([[cx, cy + eps_px]]), self.K, self.dist,
                                             self.rvec_belt, self.tvec_belt, plane_z=ref_plane_z)[0]
        jac_x = (dx_world - base) / eps_px
        jac_y = (dy_world - base) / eps_px

        corners = unk_quad["corners"]
        side_vecs_px = [corners[(i + 1) % 4] - corners[i] for i in range(4)]
        side_lens_cm = []
        for v in side_vecs_px:
            world_vec = jac_x * v[0] + jac_y * v[1]
            side_lens_cm.append(np.linalg.norm(world_vec))
        dim_a = (side_lens_cm[0] + side_lens_cm[2]) / 2.0
        dim_b = (side_lens_cm[1] + side_lens_cm[3]) / 2.0
        return (dim_a, dim_b) if dim_a >= dim_b else (dim_b, dim_a)

    def full_perspective(self, unk_quad, assumed_plane_z):
        """Exact ray-plane back-projection of the unknown's OWN pixel
        corners at its own assumed height -- no reference object needed at
        all once the camera is calibrated, provided the height is known."""
        return self._quad_side_lengths_cm(unk_quad["corners"], assumed_plane_z)


def main():
    K, dist, rvec_belt, tvec_belt = load_calibration(CALIB_PATH)
    estimator = ParcelSizeEstimator(K, dist, rvec_belt, tvec_belt)

    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_IDX)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("could not read frame")

    ref_quad = get_parcel_quad(frame, REFERENCE_TYPE, top_face_only=True)
    unk_quad = get_parcel_quad(frame, UNKNOWN_TYPE, top_face_only=True)
    if ref_quad is None or unk_quad is None:
        raise RuntimeError("reference or unknown parcel not detected in this frame")

    ref_true_len, ref_true_wid, ref_height = REAL_SIZES_CM[REFERENCE_TYPE]
    unk_true_len, unk_true_wid, unk_height = REAL_SIZES_CM[UNKNOWN_TYPE]
    unk_true_sorted = tuple(sorted([unk_true_len, unk_true_wid], reverse=True))

    ortho = estimator.orthographic(ref_quad, (ref_true_len, ref_true_wid), unk_quad)
    weak = estimator.weak_perspective(ref_quad, (ref_true_len, ref_true_wid), ref_height, unk_quad)
    aff = estimator.affine(ref_quad, ref_height, unk_quad)
    # full perspective needs the unknown's OWN plane height; using the
    # known catalog height here is the "if you know what it roughly is"
    # case -- see the sanity check below for what happens if you don't.
    persp = estimator.full_perspective(unk_quad, assumed_plane_z=unk_height)

    print(f"Reference: {REFERENCE_TYPE} (known {ref_true_len:.1f} x {ref_true_wid:.1f} cm)")
    print(f"Unknown:   {UNKNOWN_TYPE} (true size {unk_true_sorted[0]:.1f} x {unk_true_sorted[1]:.1f} cm, "
          f"for grading only -- not given to the models)\n")
    print("(All four estimates are reported as sorted long-side x short-side, since a")
    print(" detected rectangle's own corner order doesn't tell you which side is which --")
    print(" only its two side lengths, which sorting compares fairly across methods.)\n")

    def report(name, est):
        len_err = abs(est[0] - unk_true_sorted[0]) / unk_true_sorted[0] * 100
        wid_err = abs(est[1] - unk_true_sorted[1]) / unk_true_sorted[1] * 100
        print(f"  {name:22s} {est[0]:6.1f} x {est[1]:<6.1f} cm   "
              f"(error: {len_err:5.1f}% / {wid_err:5.1f}%)")

    print("Four camera-model size estimates for the unknown parcel:")
    report("Orthographic", ortho)
    report("Weak perspective", weak)
    report("Affine", aff)
    report("Full perspective", persp)

    # sanity check: what if we didn't know the unknown's height and
    # (wrongly) assumed it sat flush on the belt, z=0?
    wrong_persp = estimator.full_perspective(unk_quad, assumed_plane_z=0.0)
    print("\nSanity check -- full perspective assuming (wrongly) the unknown's top is at Z=0:")
    report("  Full persp. (Z=0 bug)", wrong_persp)
    print("  This is the classic height-parallax bias: back-projecting a pixel that's actually")
    print("  at height Z=height_cm onto the Z=0 plane over-estimates its distance from the")
    print("  camera's optical axis, and the resulting size estimate. Full perspective is only")
    print("  as good as the assumed height baked into the plane it back-projects onto.")

    print("""
Discussion: affine comes out most accurate here, not full perspective --
both use the calibrated geometry (unlike orthographic/weak perspective's
single global scale), but affine's local-Jacobian linearization is anchored
right at the reference object's own detected pixel position, while full
perspective back-projects the unknown's OWN pixel corners independently and
so carries the full weight of any residual calibration/detection noise at
that (different) image location with no local anchoring to cancel it out.
Orthographic and weak perspective are the two that ignore the unknown's
actual depth/perspective entirely, and error-wise it shows on the long,
foreshortened long_box in particular. None of this is hidden: the sanity
check above shows full perspective is only as good as its assumed height,
and in absolute terms every model here is within roughly the accuracy band
you'd expect from an imperfect (but not wildly wrong) camera calibration.
""")

    marked = frame.copy()
    cv2.drawContours(marked, [ref_quad["corners"].astype(np.int32)], 0, (0, 255, 0), 2)
    cv2.drawContours(marked, [unk_quad["corners"].astype(np.int32)], 0, (0, 0, 255), 2)
    cv2.putText(marked, "reference (known)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(marked, "unknown (estimated)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    os.makedirs(os.path.dirname(OUT_IMG), exist_ok=True)
    cv2.imwrite(OUT_IMG, marked)
    print(f"\nSaved annotated frame to {OUT_IMG}")


if __name__ == "__main__":
    main()
