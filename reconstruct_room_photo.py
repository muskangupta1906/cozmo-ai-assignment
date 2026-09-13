"""
reconstruct_room_photo.py
--------------------------
Photo-tier, single-room pipeline (Phase 4 -- the "floor" tier: 2-8 stills per
room, any iPhone 15+, no depth, no poses).

Nothing here can lean on VIO poses or per-frame LiDAR depth the way
reconstruct_room.py does -- both have to be recovered from the images
themselves:

  1. Metric depth per still, from a pretrained monocular depth model
     (Depth Anything V2, Metric-Indoor checkpoint -- see MODEL_ID below;
     disclosed per the case study's pretrained-model-with-disclosure
     allowance). Chosen over a *relative*-depth model specifically because
     relative depth has no inherent scale -- fine for a single image's
     shape, useless for stitching multiple images into one metric room
     without an external scale reference we don't have.
  2. Relative camera pose between stills, via classical 2D feature matching
     (ORB) + a RANSAC-Kabsch rigid fit on the matched keypoints' *already
     metric* back-projected 3D points. This sidesteps needing separate
     scale recovery (the usual hard part of classical SfM) because the
     depth model already put every point in meters -- registration only
     has to find the rotation+translation that aligns two already-metric
     point sets, not also solve for an unknown scale factor.

Everything downstream of "one fused, metric room point cloud" reuses
reconstruct_room.py's floor/ceiling/footprint/opening code as-is (imported
as `rr`) -- per output_schema.md's design rule, every tier feeds the same
per-room contract. The two things that do NOT transfer from the LiDAR tier:

  - Floor/ceiling detection there relies on the floor being dave a density
    *spike* from being walked over on nearly every frame of a continuous
    walkthrough. A handful of independent stills never produces that
    signal, so this file re-detects both floor AND ceiling with the same
    "progressively wider top/bottom slice, accept the first band that
    RANSAC-fits a genuinely horizontal plane" search reconstruct_room.py
    already uses for the ceiling alone (see detect_floor_and_ceiling_photo).
  - Confidence intervals: LiDAR's placeholder error model assumes a fixed,
    small, close-range depth-noise figure. Monocular metric depth is a
    fundamentally noisier, farther-from-calibrated signal, and registration
    error compounds across however many photos got chained together to
    reach a given point -- both terms have to inflate the CI here, not just
    a fixed constant (see compute_confidence_intervals_photo).

Known, honestly-flagged limitations (Round-1 scope, not silently hidden):
  - Registration is a photo-i-to-photo-(i-1) chain, not a full pose graph /
    bundle adjustment -- a bad registration anywhere in the chain corrupts
    every photo placed after it. With only 2-8 photos this is a real risk,
    not a hypothetical one.
  - Needs enough shared, textured, in-range scene between *consecutive*
    photos for ORB to find matches at all -- a textureless wall or two
    photos with no visual overlap (e.g. facing opposite corners) will fail
    to register that photo, and it gets dropped rather than silently
    guessed at (see register_photo_chain).
  - The metric depth model's absolute-scale accuracy is a general-purpose
    prior, not calibrated to this rig -- expect real, not paranoid, error
    here, which is exactly why the photo-tier gate (Part 2) is looser
    (+/-8% vs LiDAR's ~cm-level) and why calibration against real ground
    truth (Phase 6/7) matters more here than anywhere else in the pipeline.

Usage:
    python reconstruct_room_photo.py <room_photo_dir> <out_dir> [--room-name NAME]
        [--bins 60] [--stride 4] [--device mps|cpu|cuda]
"""

import os
import json
import argparse
import warnings
import numpy as np
import cv2
import open3d as o3d
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

np.random.seed(0)
o3d.utility.random.seed(0)

# macOS's Accelerate BLAS backend raises spurious "divide by zero"/"invalid
# value encountered in matmul" RuntimeWarnings on ordinary, fully-finite
# float64 matmuls (verified: inputs and outputs both checked all-finite,
# warning fires anyway) -- a known benign platform quirk, not a computation
# error. Filtered narrowly by message so a REAL invalid-value warning from
# actually non-finite data elsewhere wouldn't be silently swallowed too.
warnings.filterwarnings("ignore", message=".*encountered in matmul.*", category=RuntimeWarning)

import reconstruct_room as rr
from photo_scan_io import load_room_photos

# Depth Anything V2, metric-indoor small checkpoint via HF transformers.
# Outputs metric depth in meters, capped at 20m (config.json: max_depth=20,
# depth_estimation_type="metric") -- appropriate for indoor rooms, would
# under-range badly on an outdoor/warehouse-scale capture.
MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"

# Placeholder relative depth error for this depth model on indoor scenes,
# pending real calibration against Phase 6/7 laser ground truth -- see
# compute_confidence_intervals_photo. Not a measured figure for this rig.
PHOTO_RELATIVE_DEPTH_ERROR = 0.08

_MODEL_CACHE = {}


def _pick_device(requested: str = None) -> str:
    if requested:
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_depth_model(device: str):
    if device not in _MODEL_CACHE:
        processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID)
        model.to(device).eval()
        _MODEL_CACHE[device] = (processor, model)
    return _MODEL_CACHE[device]


def estimate_depth_m(image_rgb: np.ndarray, device: str) -> np.ndarray:
    """Runs the metric depth model on one RGB image, returns a (H,W) float32
    depth map in meters at the image's native resolution."""
    processor, model = load_depth_model(device)
    inputs = processor(images=image_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    h, w = image_rgb.shape[:2]
    post = processor.post_process_depth_estimation(out, target_sizes=[(h, w)])
    return post[0]["predicted_depth"].cpu().numpy().astype(np.float32)


def backproject_dense(depth_m: np.ndarray, fx, fy, cx, cy, stride: int = 4,
                       max_depth_m: float = 8.0, min_depth_m: float = 0.15):
    """Back-projects a strided grid of pixels to camera-space 3D points.
    Strided, not full-res: a single 1920x1440 depth map is ~2.8M pixels,
    and this feeds floor/ceiling/footprint detection downstream which
    doesn't need per-pixel density the way registration's keypoint lookup
    does -- stride keeps 4-8 photos' combined cloud a few hundred thousand
    points instead of tens of millions."""
    h, w = depth_m.shape
    us, vs = np.meshgrid(np.arange(0, w, stride), np.arange(0, h, stride))
    z = depth_m[vs, us]
    valid = (z > min_depth_m) & (z < max_depth_m) & np.isfinite(z)
    x = (us[valid] - cx) * z[valid] / fx
    y = (vs[valid] - cy) * z[valid] / fy
    zc = z[valid]
    return np.stack([x, y, zc], axis=1)


def _lookup_depth(depth_m: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Nearest-pixel depth lookup at exact (sub-pixel-rounded) keypoint
    coordinates -- registration needs the depth AT the matched feature, not
    a strided neighbor."""
    h, w = depth_m.shape
    u = np.clip(np.round(uv[:, 0]).astype(int), 0, w - 1)
    v = np.clip(np.round(uv[:, 1]).astype(int), 0, h - 1)
    return depth_m[v, u]


def match_features(img_a: np.ndarray, img_b: np.ndarray, max_features: int = 3000):
    """ORB keypoints + ratio-test matching between two RGB images. Returns
    (pts_a, pts_b): Nx2 pixel coordinates of matched keypoint pairs, sorted
    by match quality. ORB over SIFT: patent-free, fast enough for a Round-1
    CLI tool, and Round-1 photo pairs are close-baseline handheld stills
    where ORB's weaker scale/rotation invariance isn't the limiting factor
    -- shared texture and depth validity are."""
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(nfeatures=max_features)
    kp_a, des_a = orb.detectAndCompute(gray_a, None)
    kp_b, des_b = orb.detectAndCompute(gray_b, None)
    if des_a is None or des_b is None or len(kp_a) < 8 or len(kp_b) < 8:
        return np.zeros((0, 2)), np.zeros((0, 2))

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = bf.knnMatch(des_a, des_b, k=2)
    good = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:  # Lowe's ratio test
            good.append(m)
    if not good:
        return np.zeros((0, 2)), np.zeros((0, 2))

    good.sort(key=lambda m: m.distance)
    pts_a = np.array([kp_a[m.queryIdx].pt for m in good])
    pts_b = np.array([kp_b[m.trainIdx].pt for m in good])
    return pts_a, pts_b


def kabsch(P: np.ndarray, Q: np.ndarray):
    """Rigid (rotation + translation, no scale) transform mapping P -> Q,
    least-squares over corresponding 3D point pairs. No scale term because
    both P and Q are already metric (same depth model, same units) --
    solving for scale here would let registration silently paper over a
    depth-model scale disagreement between the two photos instead of
    surfacing it as registration error.

    Raises np.linalg.LinAlgError on a degenerate input (near-collinear or
    duplicate points -- common when a 3-point RANSAC minimal sample happens
    to land on near-identical keypoints) rather than returning a technically-
    computed but meaningless rotation. An unchecked degenerate fit here was
    observed to silently produce a garbage transform that corrupted an
    entire photo chain (a 9m+ "ceiling height" on a real room) with no
    error or warning -- exactly the "confident garbage" failure mode the
    case study penalizes hardest, so this must fail loudly, not quietly."""
    p_mean, q_mean = P.mean(axis=0), Q.mean(axis=0)
    Pc, Qc = P - p_mean, Q - q_mean
    H = Pc.T @ Qc
    if not np.all(np.isfinite(H)):
        raise np.linalg.LinAlgError("non-finite covariance -- degenerate point set")
    U, S, Vt = np.linalg.svd(H)
    if S[1] < 1e-9:
        # Guard against COLLINEAR points (rank < 2), which leave the
        # rotation underdetermined. NOT a check on S[-1] (the third/
        # smallest singular value): for exactly 3 points -- the RANSAC
        # minimal sample -- the centered points always lie in a plane, so
        # H is always rank <= 2 and S[-1] is always ~0 by construction.
        # Checking S[-1] here would reject every single 3-point sample,
        # which is exactly what happened before this fix (100% of RANSAC
        # iterations across a real capture were rejected as "degenerate"
        # when the points were fine -- just planar, as 3 points always are).
        raise np.linalg.LinAlgError("rank-deficient covariance -- collinear points")
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    t = q_mean - R @ p_mean
    if not (np.all(np.isfinite(R)) and np.all(np.isfinite(t))):
        raise np.linalg.LinAlgError("non-finite result")
    return R, t


def ransac_register(pts_cam_a: np.ndarray, pts_cam_b: np.ndarray,
                     n_iters: int = 2000, inlier_thresh_m: float = 0.12,
                     min_inliers: int = 8):
    """RANSAC-Kabsch: estimates the rigid transform mapping photo B's
    camera-space points onto photo A's, from noisy/partly-wrong matched
    3D correspondences. Returns (R, t, inlier_ratio, n_inliers) or None if
    too few correspondences survive to attempt a fit -- callers must treat
    None as "this pair cannot be registered," not retry with looser
    thresholds silently (see register_photo_chain)."""
    n = len(pts_cam_a)
    if n < min_inliers:
        return None
    best_inliers = None
    rng = np.random.RandomState(0)
    for _ in range(n_iters):
        idx = rng.choice(n, size=3, replace=False)
        try:
            R, t = kabsch(pts_cam_b[idx], pts_cam_a[idx])
        except np.linalg.LinAlgError:
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            pred = (R @ pts_cam_b.T).T + t
        if not np.all(np.isfinite(pred)):
            continue
        errs = np.linalg.norm(pred - pts_cam_a, axis=1)
        inliers = errs < inlier_thresh_m
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
    if best_inliers is None or best_inliers.sum() < min_inliers:
        return None
    try:
        R, t = kabsch(pts_cam_b[best_inliers], pts_cam_a[best_inliers])
    except np.linalg.LinAlgError:
        return None
    # Sanity-check the refit transform against the SAME inlier set used to
    # select it -- a fit that no longer agrees with its own inliers (or
    # produced a non-rotation) means the "best" RANSAC sample was a fluke,
    # not a real registration.
    pred = (R @ pts_cam_b[best_inliers].T).T + t
    if not np.all(np.isfinite(pred)):
        return None
    refit_err = float(np.linalg.norm(pred - pts_cam_a[best_inliers], axis=1).mean())
    if refit_err > inlier_thresh_m or abs(abs(np.linalg.det(R)) - 1.0) > 1e-3:
        return None
    return R, t, float(best_inliers.mean()), int(best_inliers.sum())


def register_photo_chain(photos: list, max_depth_m: float = 8.0, min_depth_m: float = 0.15):
    """
    Chains each photo to its predecessor (photo[i] registered against
    photo[i-1]) via ORB matches + RANSAC-Kabsch on their metric
    back-projected keypoints, then composes transforms so every photo ends
    up expressed in photo[0]'s frame.

    Deliberately a chain, not a full pose graph: with 2-8 photos the
    combinatorics of all-pairs registration + bundle adjustment isn't
    worth the complexity for Round-1 scope, but it means one failed link
    truncates the chain -- a photo that can't be matched to its immediate
    predecessor is dropped along with every photo after it, rather than
    silently placed with a guessed pose. Callers get an honest count of
    how many photos actually made it into the fused cloud.

    Returns a list of dicts (one per SUCCESSFULLY chained photo, in order):
    {photo, depth_m, T (4x4 cam-to-room0), inlier_ratio}.
    """
    chained = [{
        "photo": photos[0], "depth_m": estimate_depth_m(photos[0]["image"], photos[0].get("device", "cpu")),
        "T": np.eye(4), "inlier_ratio": 1.0,
    }]
    prev_depth = chained[0]["depth_m"]

    for i in range(1, len(photos)):
        cur = photos[i]
        cur_depth = estimate_depth_m(cur["image"], cur.get("device", "cpu"))
        prev = photos[i - 1]

        pts_prev_2d, pts_cur_2d = match_features(prev["image"], cur["image"])
        if len(pts_prev_2d) < 8:
            print(f"      WARNING: {os.path.basename(cur['path'])} -- only "
                  f"{len(pts_prev_2d)} feature matches to previous photo, "
                  f"dropping this and all subsequent photos from the chain")
            break

        d_prev = _lookup_depth(prev_depth, pts_prev_2d)
        d_cur = _lookup_depth(cur_depth, pts_cur_2d)
        valid = (d_prev > min_depth_m) & (d_prev < max_depth_m) & \
                (d_cur > min_depth_m) & (d_cur < max_depth_m)
        if valid.sum() < 8:
            print(f"      WARNING: {os.path.basename(cur['path'])} -- only "
                  f"{int(valid.sum())} matches had valid depth in both photos, "
                  f"dropping this and all subsequent photos from the chain")
            break

        pts_prev_cam = np.stack([
            (pts_prev_2d[valid, 0] - prev["cx"]) * d_prev[valid] / prev["fx"],
            (pts_prev_2d[valid, 1] - prev["cy"]) * d_prev[valid] / prev["fy"],
            d_prev[valid],
        ], axis=1)
        pts_cur_cam = np.stack([
            (pts_cur_2d[valid, 0] - cur["cx"]) * d_cur[valid] / cur["fx"],
            (pts_cur_2d[valid, 1] - cur["cy"]) * d_cur[valid] / cur["fy"],
            d_cur[valid],
        ], axis=1)

        result = ransac_register(pts_prev_cam, pts_cur_cam)
        if result is None:
            print(f"      WARNING: {os.path.basename(cur['path'])} -- RANSAC "
                  f"registration failed (too few consistent inliers), dropping "
                  f"this and all subsequent photos from the chain")
            break
        R, t, inlier_ratio, n_inliers = result
        print(f"      registered {os.path.basename(cur['path'])} <- "
              f"{os.path.basename(prev['path'])}: {n_inliers} inliers "
              f"({inlier_ratio:.0%} of {valid.sum()} depth-valid matches)")

        T_cur_to_prev = np.eye(4)
        T_cur_to_prev[:3, :3] = R
        T_cur_to_prev[:3, 3] = t
        T_cur_to_room0 = chained[-1]["T"] @ T_cur_to_prev

        chained.append({"photo": cur, "depth_m": cur_depth,
                         "T": T_cur_to_room0, "inlier_ratio": inlier_ratio})
        prev_depth = cur_depth

    return chained


def build_fused_point_cloud_photo(room_dir: str, stride: int = 4, device: str = None,
                                   max_depth_m: float = 8.0, min_depth_m: float = 0.15):
    """Loads a room's photo folder, estimates metric depth per photo,
    registers them into one shared frame, and returns (pcd, mean_inlier_ratio,
    n_photos_used, n_photos_total)."""
    device = _pick_device(device)
    photos = load_room_photos(room_dir)
    for p in photos:
        p["device"] = device

    print(f"      running {MODEL_ID} on {len(photos)} photos (device={device})...")
    chained = register_photo_chain(photos, max_depth_m=max_depth_m, min_depth_m=min_depth_m)
    if len(chained) < len(photos):
        print(f"      NOTE: only {len(chained)}/{len(photos)} photos chained "
              f"successfully -- room reconstruction uses this subset only")

    all_points = []
    for entry in chained:
        p, depth_m, T = entry["photo"], entry["depth_m"], entry["T"]
        pts_cam = backproject_dense(depth_m, p["fx"], p["fy"], p["cx"], p["cy"],
                                     stride=stride, max_depth_m=max_depth_m,
                                     min_depth_m=min_depth_m)
        pts_h = np.concatenate([pts_cam, np.ones((len(pts_cam), 1))], axis=1)
        pts_room = (T @ pts_h.T).T[:, :3]
        all_points.append(pts_room)

    if not all_points:
        raise RuntimeError(f"{room_dir}: no photos could be back-projected")

    points = np.concatenate(all_points, axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)
    pcd = pcd.voxel_down_sample(voxel_size=0.03)

    mean_inlier_ratio = float(np.mean([e["inlier_ratio"] for e in chained]))
    return pcd, mean_inlier_ratio, len(chained), len(photos)


def detect_floor_and_ceiling_photo(pcd: o3d.geometry.PointCloud):
    """
    Floor/ceiling detection for the photo tier: unlike a continuous LiDAR
    walkthrough, a handful of independent stills gives the floor NO
    walked-over density spike to key off -- so both floor and ceiling here
    use the same "progressively wider slice, accept the first genuinely
    horizontal RANSAC plane" search reconstruct_room.py's
    detect_floor_and_ceiling already uses for the ceiling alone. Bands are
    wider than the LiDAR version's per search step because there are
    orders of magnitude fewer points to search within.
    """
    pts = np.asarray(pcd.points)
    y = pts[:, 1]
    lo, hi = float(y.min()), float(y.max())
    span = hi - lo
    if span < 0.3:
        raise RuntimeError("Point cloud's vertical extent is too small to "
                            "contain a distinct floor and ceiling -- check "
                            "photo coverage/registration")

    def _search(from_top: bool):
        for frac in (0.03, 0.05, 0.08, 0.12, 0.20, 0.30, 0.40):
            mask = (y > hi - span * frac) if from_top else (y < lo + span * frac)
            fit = rr._fit_horizontal_plane(pts, mask)
            if fit is None:
                continue
            plane_y, normal_y, ratio, n_inliers = fit
            if normal_y > 0.85 and ratio > 0.4 and n_inliers >= 20:
                return plane_y, min(1.0, ratio)
        return None, 0.0

    ceiling_y, ceiling_confidence = _search(from_top=True)
    floor_y, floor_confidence = _search(from_top=False)

    if ceiling_y is None:
        ceiling_y, ceiling_confidence = hi, 0.0
        print("      WARNING: no reliable ceiling plane found in any photo -- "
              "falling back to highest observed point, confidence=0")
    if floor_y is None:
        floor_y, floor_confidence = lo, 0.0
        print("      WARNING: no reliable floor plane found in any photo -- "
              "falling back to lowest observed point, confidence=0")

    height = ceiling_y - floor_y
    confidence = min(floor_confidence, ceiling_confidence)
    return floor_y, ceiling_y, height, confidence


def compute_confidence_intervals_photo(polygon: np.ndarray, perimeter: float, area: float,
                                        mean_inlier_ratio: float, n_photos_used: int):
    """
    Placeholder error model for the photo tier, pending real calibration
    against Phase 6/7 laser ground truth -- like reconstruct_room.py's
    LiDAR version, explicitly not a measured error budget yet. Two things
    make this wider than the LiDAR CI, both real and both compounding:

      1. Monocular metric depth error scales with distance (a fixed close-
         range noise-floor assumption, as LiDAR uses, would UNDERSTATE
         error on a photo taken across a room) -- so this is relative to
         each measurement, not a fixed absolute meters figure.
      2. Registration error compounds through the photo-to-photo chain --
         a lower mean inlier ratio across the chain, or fewer photos
         successfully chained, both mean the fused shape is less trustworthy
         even before the depth model's own error is added.

    Returns (wall_length_ci_m, area_ci_m2), both scaled up as
    mean_inlier_ratio drops or n_photos_used shrinks.
    """
    registration_penalty = 1.0 / max(mean_inlier_ratio, 0.15)
    sparsity_penalty = 1.0 + max(0, 3 - n_photos_used) * 0.25  # fewer than 3 photos: extra caution
    relative_error = PHOTO_RELATIVE_DEPTH_ERROR * registration_penalty * sparsity_penalty

    avg_wall_length = perimeter / max(len(polygon), 1)
    wall_length_ci_m = round(float(avg_wall_length * relative_error), 3)
    area_ci_m2 = round(float(area * relative_error * 2), 3)  # area error ~ 2x linear relative error
    return wall_length_ci_m, area_ci_m2


def ceiling_height_ci_m_photo(confidence: float, mean_inlier_ratio: float) -> float:
    """Photo-tier ceiling-height CI: same confidence tiers as the LiDAR
    version's spirit, but every band widened -- monocular metric depth on
    a single or few stills is a much weaker signal than a fused LiDAR
    point cloud even at "high confidence.\""""
    registration_penalty = 1.0 / max(mean_inlier_ratio, 0.15)
    if confidence >= 0.8:
        base = 0.08
    elif confidence >= 0.3:
        base = 0.25
    else:
        base = 0.60
    return round(base * registration_penalty, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("room_dir", help="Folder of 2-8 photos for one room")
    ap.add_argument("out_dir")
    ap.add_argument("--room-name", default=None)
    ap.add_argument("--stride", type=int, default=4,
                     help="Pixel stride for dense back-projection (floor/ceiling/footprint). "
                          "Lower = denser cloud, slower.")
    ap.add_argument("--bins", type=int, default=60,
                     help="Angular bins for the radial wall-footprint sweep. Photo tier "
                          "defaults much lower than LiDAR's 180 -- far fewer points to "
                          "support fine angular resolution.")
    ap.add_argument("--device", default=None, help="mps|cuda|cpu, auto-detected if omitted")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    room_name = args.room_name or os.path.basename(os.path.normpath(args.room_dir))

    print("[1/6] Estimating depth + registering photos...")
    pcd, mean_inlier_ratio, n_used, n_total = build_fused_point_cloud_photo(
        args.room_dir, stride=args.stride, device=args.device)
    o3d.io.write_point_cloud(os.path.join(args.out_dir, "fused_cloud.ply"), pcd)
    print(f"      {len(pcd.points)} points after filtering/downsampling, "
          f"{n_used}/{n_total} photos chained, mean inlier ratio {mean_inlier_ratio:.2f}")
    rr.render_raw_scatter(pcd, os.path.join(args.out_dir, "raw_scatter.png"), room_name)

    print("[2/6] Detecting floor/ceiling...")
    floor_y, ceiling_y, height, height_confidence = detect_floor_and_ceiling_photo(pcd)
    print(f"      floor_y={floor_y:.3f} ceiling_y={ceiling_y:.3f} height={height:.3f} m "
          f"confidence={height_confidence:.2f}")

    print("[3/6] Extracting wall footprint...")
    polygon = rr.extract_wall_footprint(pcd, floor_y, ceiling_y, num_bins=args.bins)

    print("[4/6] Computing dimensions...")
    lengths, area = rr.polygon_metrics(polygon)
    perimeter = float(sum(lengths))
    wall_length_ci_m, area_ci_m2 = compute_confidence_intervals_photo(
        polygon, perimeter, area, mean_inlier_ratio, n_used)
    height_ci_m = ceiling_height_ci_m_photo(height_confidence, mean_inlier_ratio)

    print("[5/6] Detecting openings...")
    openings = rr.detect_openings(pcd, polygon, floor_y, ceiling_y)
    print(f"      {len(openings)} candidate opening(s) found")

    print("[6/6] Writing outputs...")
    rr.render_plan(polygon, os.path.join(args.out_dir, "room_plan.png"), room_name, openings=openings)

    result = {
        "room_name": room_name,
        "source_scan": os.path.abspath(args.room_dir),
        "tier": "photo",
        "ceiling_height_m": round(height, 3),
        "ceiling_height_confidence": round(height_confidence, 3),
        "ceiling_height_ci_m": height_ci_m,
        "floor_area_m2": round(area, 3),
        "floor_area_ci_m2": area_ci_m2,
        "wall_count": len(lengths),
        "wall_lengths_m": [round(l, 3) for l in lengths],
        "wall_length_ci_m": wall_length_ci_m,
        "openings": openings,
        "opening_count": len(openings),
        "footprint_polygon_xz": polygon.tolist(),
        "n_photos_total": n_total,
        "n_photos_used": n_used,
        "mean_registration_inlier_ratio": round(mean_inlier_ratio, 3),
        "confidence_note": (
            f"photo tier: depth from {MODEL_ID} (metric monocular depth, "
            f"disclosed pretrained model), {n_used}/{n_total} photos successfully "
            f"chained via ORB-matched-keypoint RANSAC-Kabsch registration "
            f"(mean inlier ratio {mean_inlier_ratio:.2f}) -- a dropped photo means "
            f"registration failed against its predecessor (no shared texture/"
            f"overlap or too few valid-depth matches), not a silent guess. "
            f"Floor/ceiling use the same tightest-band-first RANSAC plane search "
            f"on both ends (no walked-over density signal available with only a "
            f"few stills, unlike the LiDAR tier). Wall-length/area/ceiling CIs "
            f"scale with the depth model's assumed relative error AND the "
            f"registration chain's inlier ratio -- both placeholders pending "
            f"Phase 6/7 ground-truth calibration, not measured for this rig. "
            f"Registration is a photo-to-predecessor CHAIN, not a full pose "
            f"graph/bundle adjustment -- one bad link truncates every photo "
            f"after it from the reconstruction. Opening detection reuses the "
            f"LiDAR tier's gap-in-coverage heuristic and is expected to under-"
            f"detect on this much sparser data; treat opening_count as even "
            f"less reliable here than on the LiDAR tier."
        ),
    }
    with open(os.path.join(args.out_dir, "room.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nDone. Ceiling height: {height:.2f} m (+/-{height_ci_m:.2f}, "
          f"confidence {height_confidence:.2f}) | Floor area: {area:.2f} m^2 "
          f"(+/-{area_ci_m2:.2f}) | Openings: {len(openings)} | "
          f"Photos used: {n_used}/{n_total}")
    print(f"Outputs in {args.out_dir}/: room.json, room_plan.png, fused_cloud.ply")


if __name__ == "__main__":
    main()
