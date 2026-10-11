# Narration script -- ~10 minute screen recording

Read this while screen-recording yourself running the scripts (or scrolling
through their saved output images in `outputs/` and `segmentation_analysis.ipynb`).
Times are approximate; talk naturally, you don't need to hit them exactly.

## 0:00-0:40 -- Intro
"This is my camera-based sorting-line inspection and tracking system. One
camera watches parcels moving along a line, and I'm using classical computer
vision -- no pretrained deep model -- to recover where each parcel is, how
big it really is in centimeters, how fast the line is moving, and what type
of parcel it is. I built five modules, one per syllabus unit, then chained
them into one pipeline. I'm running it on [a video I filmed / a rendered
test video -- see README for why], since either is fine as long as it's
real footage-shaped data, not a synthetic render used to fake the answers."

## 0:40-2:10 -- Module A: Segmentation
Open `outputs/module_a_comparison_grid.png` (or run
`python module_a_segmentation/run_comparison.py` live).
"Module A finds the parcels in each frame. I ran 7 classical segmentation
methods on the same 5 frames from the line video, plus one extra frame I
built on purpose with two parcels pushed together, specifically to get an
honest touching-objects comparison -- the main video's own schedule keeps a
running gap between parcels, the way a real sorting line would to avoid
collisions. Watershed handles that stress frame best [point at the row]:
its markers come from the distance transform's local maxima inside each
parcel's own core, so it can place two separate seeds in a single connected
blob and flood each one out to the true shared boundary. Active contours
instead grows one snake around the whole blob and can't split it. I'll also
own a limitation here: that same local-maxima seeding can over-segment my
longest parcel into two pieces even when it's completely alone, since its
distance transform has two broad peaks along its length -- that's a real
watershed failure mode, not something I'm hiding."

## 2:10-4:00 -- Module B: Camera Calibration
Run `python module_b_calibration/calibrate.py` live, or show its printed
output plus `outputs/module_b_undistort_before_after.png`.
"Module B calibrates the camera. I photograph a checkerboard from 16
different angles, spread across most of the frame rather than just the
center, and run `cv2.calibrateCamera` to recover the intrinsic matrix and
lens distortion -- that spread actually mattered, an earlier version of this
with narrow central coverage left the distortion model poorly constrained
right where the parcels themselves get imaged. Here's the undistortion
effect on a real frame -- the belt's guide lines go from curved to straight.
I also solve for the camera's pose against a second checkerboard lying flat
on the belt with `solvePnP`, manually build the full projection matrix
P = K[R|t], and decompose it back apart to confirm I get the same K, R, and
t out -- that check comes back exact." Then show `measure_parcel.py`'s
printed table: "Using one parcel of known size as a reference, I estimate an
unknown parcel's size under four camera models. Affine turns out most
accurate here, not full perspective -- I talk through why in the code
comments, it comes down to where each model's residual calibration noise
gets anchored, not a flaw in the perspective math itself."

## 4:00-5:45 -- Module C: Optical Flow
Run `python module_c_motion_flow/motion_estimation.py`, show
`outputs/module_c_dense_flow.png` and `outputs/module_c_ransac_inliers.png`.
"Module C estimates how fast the line is moving. I implemented Lucas-Kanade
optical flow completely from scratch, as a coarse-to-fine pyramid -- flow is
solved at a coarse, blurry scale first, then each finer level only has to
correct the small remaining error on top of that, which is what keeps it
stable. I compared it to OpenCV's built-in sparse feature tracker: dense
flow flags the plain belt surface as unreliable, which is the classic
aperture problem, exactly where the sparse method simply never places a
feature to begin with. I then fit a 6-parameter affine flow model to one
parcel's flow vectors, with my own from-scratch RANSAC to reject outlier
vectors first [point at the red vs green dots], and converted the result
into a real belt-speed estimate using Module B's calibration -- it comes out
within about half a percent of the true 22 cm/s here."

## 5:45-7:15 -- Module D: Kalman Filter Tracking
Run `python module_d_tracking/kalman_tracker.py`, show its printed
predict/measure/update cycle and `outputs/module_d_tracking_plot.png`.
"Module D tracks one parcel across the whole video with a Kalman filter I
wrote from scratch -- the state is x-position, x-velocity, y-position,
y-velocity, constant velocity model. You can see the explicit predict,
measure, and update steps printed here for ten frames in a row. Then I
simulate an occlusion: for four consecutive frames I throw away the
measurement entirely, and the filter keeps tracking on pure prediction alone
-- that's the orange segment in the plot. It demonstrates observability: the
system can infer state even with missing observations, as long as the
motion model still holds."

## 7:15-9:00 -- Module E: Object Recognition
Run `python module_e_recognition/recognize.py`, show
`outputs/module_e_templates_and_samples.png` and
`outputs/module_e_eigenvectors.png`.
"Module E figures out what type of parcel it is. Instead of generating a
separate synthetic augmented dataset, I sampled real crops straight out of
the line video across its whole length -- each parcel already rotates
continuously as it travels, so I get natural rotation diversity for free,
and I added brightness jitter on top. I implemented alignment-based
recognition using SIFT keypoints and a RANSAC homography against a
reference template, and an eigenspace method -- 'eigenboxes' -- using PCA
and nearest-neighbor classification. Here are the top three eigenvectors it
learned. I also computed Hu moments on each parcel's actual silhouette
shape, which are invariant to translation, rotation, and scale, and get
roughly 85-90% accuracy from shape alone -- it moves a couple points between
runs since my test set is only a few dozen crops per class. My accuracy
table is printed in the
terminal, and the grayscale-only ablation is the interesting part: it
splits alignment and eigenspace apart in a way color normally hides, and I
talk through exactly why in the code comments."

## 9:00-9:45 -- Integration
Run `python pipeline.py` live.
"Finally, pipeline.py chains all five modules over the full video: detect
each parcel, Kalman-track its centroid, convert its footprint to centimeters
with the calibrated camera, estimate the line's speed from the track's own
displacement, and classify its type. For each tracked parcel it prints an
ID, real-world size, speed, and recognized type -- all three parcels come
back correctly typed, with sizes and speeds in the same accuracy ballpark as
each module scored on its own."

## 9:45-10:00 -- Close
"That's the full pipeline -- five classical CV modules built from first
principles and chained end to end. Code and README are in the repo."
