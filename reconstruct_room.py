
"""
reconstruct_room.py
--------------------
LiDAR-tier, single-room pipeline (Round-1 scope: get ONE room working end to end).

Pipeline:
  1. Load intrinsics + per-frame poses + depth + confidence  (scan_io.py)
  2. Back-project confident depth pixels to 3D, transform into world frame using
     the pose for each frame -> fused point cloud
  3. Detect the floor plane and ceiling plane (largest near-horizontal planes)
  4. Detect wall planes (near-vertical), project their footprint to the floor
     plane -> room boundary polygon
  5. Compute dimensions: wall lengths, ceiling height, floor area
  6. Write:
       - <out>/room.json         (dimensioned plan, machine-readable)
       - <out>/room_plan.png     (rendered 2D floor plan)
       - <out>/fused_cloud.ply   (for inspection in any point-cloud viewer)

Usage:
    python reconstruct_room.py <scan_dir> <out_dir> [--every-n 5] [--min-confidence 2]
"""

import os
import json
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

# Fixed seeds -- RANSAC plane fitting (floor/ceiling detection) is otherwise
# non-deterministic run-to-run on the *same* input, which would fail the
# case study's repeatability gate ("same room in, same plan out") even when
# nothing about the capture changed.
np.random.seed(0)
o3d.utility.random.seed(0)

from scan_io import (
    load_intrinsics, load_odometry, list_depth_frames, list_confidence_frames,
    load_depth_mm, load_confidence, pose_to_T, DEPTH_W, DEPTH_H,
)


def get_rgb_resolution(scan_dir: str):
    """Reads the actual rgb.mp4 resolution instead of assuming 1920x1440 --
    different iPhone models/capture settings can differ, and a wrong
    assumption here silently corrupts every back-projected 3D point."""
    import cv2
    cap = cv2.VideoCapture(os.path.join(scan_dir, "rgb.mp4"))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if w == 0 or h == 0:
        raise RuntimeError("Could not read rgb.mp4 resolution -- check the file path/codec")
    return w, h


def backproject_frame(depth_mm: np.ndarray, conf: np.ndarray, row,
                       fx_d: float, fy_d: float, cx_d: float, cy_d: float,
                       us: np.ndarray, vs: np.ndarray,
                       min_confidence: int = 2, max_depth_m: float = 5.0):
    """
    Back-projects one depth frame to world-space 3D points using one
    odometry row's pose. Factored out of build_fused_point_cloud so
    damage_detection/detect_damage.py can get the same validated
    depth-pixel -> world-point math (including the degenerate-pose guard)
    without duplicating it -- this exact calculation (intrinsics scaling +
    quaternion pose) was the source of real bugs earlier in the project,
    so it has exactly one implementation now.

    Returns (pts_world (N,3), valid_mask (H,W) bool -- which depth pixels
    contributed a point) or (None, valid_mask) if the frame's pose is
    degenerate/missing or no pixels pass the confidence/depth filters.
    valid_mask lets callers map a world point back to its source pixel.
    """
    z = depth_mm / 1000.0  # mm -> m
    valid = (conf >= min_confidence) & (z > 0.05) & (z < max_depth_m)
    if not np.any(valid):
        return None, valid

    x = (us[valid] - cx_d) * z[valid] / fx_d
    y = (vs[valid] - cy_d) * z[valid] / fy_d
    zc = z[valid]
    pts_cam = np.stack([x, y, zc, np.ones_like(zc)], axis=1)  # (N,4)

    quat_norm = np.sqrt(row["qx"]**2 + row["qy"]**2 + row["qz"]**2 + row["qw"]**2)
    pose_vals = [row["x"], row["y"], row["z"], row["qx"], row["qy"], row["qz"], row["qw"]]
    if quat_norm < 1e-8 or not np.all(np.isfinite(pose_vals)):
        return None, valid
    T = pose_to_T(row)
    pts_world = (T @ pts_cam.T).T[:, :3]
    if not np.all(np.isfinite(pts_world)):
        return None, valid
    return pts_world, valid


def build_fused_point_cloud(scan_dir: str, every_n: int = 5, min_confidence: int = 2,
                             max_depth_m: float = 5.0, frame_indices=None,
                             odo=None) -> o3d.geometry.PointCloud:
    """
    Back-projects depth frames to 3D and accumulates them into one point cloud
    in world coordinates.

    frame_indices: if given, restrict to this frame subset (still strided by
    every_n within it) instead of every `every_n`-th frame over the whole
    scan -- used by stitch_property.py to build a single room's cloud from
    just the frames the trajectory segmenter assigned to that room, out of a
    larger multi-room walkthrough. A dwelled-on room can collect thousands of
    frames (they're spatially redundant by construction -- that's how
    segment_rooms finds rooms at all), so skipping the every_n stride here
    was fusing 10-20x more frames than the single-room contract ever does,
    which OOM-killed the process on a 3-room scan.

    odo: if given, use this odometry DataFrame instead of loading
    scan_dir/odometry.csv -- lets callers pass drift-corrected poses (see
    stitch_property.py's loop-closure correction) without duplicating the
    rest of this function.

    min_confidence: keep only depth pixels with confidence >= this (0..2, 2=best).
    Using only high-confidence points early keeps the first pass from choking
    on noisy edges/reflective surfaces -- widen later once the pipeline works.
    """
    K = load_intrinsics(scan_dir)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    if odo is None:
        odo = load_odometry(scan_dir)
    depth_frames = list_depth_frames(scan_dir)
    conf_frames = list_confidence_frames(scan_dir)
    n = min(len(odo), len(depth_frames), len(conf_frames))

    # Read actual depth resolution from the first frame -- don't assume 256x192.
    first_depth = load_depth_mm(depth_frames[0])
    depth_h, depth_w = first_depth.shape[:2]

    # Read actual RGB resolution rather than assuming 1920x1440 -- this
    # directly determines the intrinsics scale factor below and a wrong
    # value here corrupts every 3D point silently (no crash, just garbage).
    rgb_w, rgb_h = get_rgb_resolution(scan_dir)
    scale_x = depth_w / rgb_w
    scale_y = depth_h / rgb_h
    fx_d, fy_d = fx * scale_x, fy * scale_y
    cx_d, cy_d = cx * scale_x, cy * scale_y
    print(f"      depth res: {depth_w}x{depth_h}, rgb res: {rgb_w}x{rgb_h}, "
          f"scaled intrinsics fx={fx_d:.1f} fy={fy_d:.1f} cx={cx_d:.1f} cy={cy_d:.1f}")

    us, vs = np.meshgrid(np.arange(depth_w), np.arange(depth_h))

    if frame_indices is None:
        frame_indices = range(0, n, every_n)
    else:
        frame_indices = [i for i in frame_indices if i < n][::every_n]

    all_points = []
    frames_attempted = 0
    frames_skipped_pose = 0
    frames_skipped_nopoints = 0
    for i in frame_indices:
        frames_attempted += 1
        depth_mm = load_depth_mm(depth_frames[i])
        conf = load_confidence(conf_frames[i])
        if depth_mm.shape != (depth_h, depth_w):
            depth_mm = cv2_resize_nn(depth_mm, depth_w, depth_h)
            conf = cv2_resize_nn(conf, depth_w, depth_h)

        row = odo.iloc[i]
        pts_world, valid = backproject_frame(depth_mm, conf, row, fx_d, fy_d, cx_d, cy_d,
                                              us, vs, min_confidence, max_depth_m)
        if not np.any(valid):
            frames_skipped_nopoints += 1
            continue
        if pts_world is None:
            # Degenerate/missing pose for this frame (seen in real captures,
            # e.g. a dropped VIO frame) -- skip it rather than let a NaN
            # transform poison the whole fused cloud.
            frames_skipped_pose += 1
            continue
        all_points.append(pts_world)

    print(f"      frames attempted={frames_attempted} "
          f"skipped(bad pose)={frames_skipped_pose} "
          f"skipped(no valid depth)={frames_skipped_nopoints}")
    if frames_skipped_pose > frames_attempted * 0.2:
        print("      WARNING: >20% of frames had bad poses -- check odometry.csv "
              "column names/order match scan_io.load_odometry's assumptions")

    if not all_points:
        raise RuntimeError("No valid points fused -- check min_confidence / depth paths")

    points = np.concatenate(all_points, axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.voxel_down_sample(voxel_size=0.02)
    return pcd


def cv2_resize_nn(arr, w, h):
    import cv2
    return cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)


def _fit_horizontal_plane(pts: np.ndarray, mask: np.ndarray, distance_threshold: float = 0.03):
    """RANSAC-fits a plane to pts[mask]. Returns (plane_y, normal_y, inlier_ratio,
    n_inliers), or None if too few candidate points to attempt a fit."""
    cand = pts[mask]
    if len(cand) < 30:
        return None
    cpcd = o3d.geometry.PointCloud()
    cpcd.points = o3d.utility.Vector3dVector(cand)
    plane, inliers = cpcd.segment_plane(distance_threshold=distance_threshold,
                                         ransac_n=3, num_iterations=1000)
    a, b, c, _d = plane
    normal = np.array([a, b, c])
    normal /= np.linalg.norm(normal)
    inlier_pts = cand[inliers]
    plane_y = float(inlier_pts[:, 1].mean())
    return plane_y, float(abs(normal[1])), len(inliers) / len(cand), len(inliers)


def detect_floor_and_ceiling(pcd: o3d.geometry.PointCloud):
    """
    ARKit world frame is y-up.

    Floor: a room's floor is walked over at close range on nearly every frame,
    so it is by far the most heavily-sampled surface in the cloud -- a
    histogram peak in the lower part of the y-range finds it reliably, refined
    by a RANSAC plane fit on points near that peak.

    Ceiling: captured far more sparsely (steep upward angle, longer range) with
    NO equivalent density spike -- confirmed by inspecting the raw y-histograms
    on real captures: floor shows one dominant bin, ceiling shows a smooth decay
    to zero with no bump anywhere. A fixed percentile cut (e.g. y at the 98th
    percentile) therefore lands inside the wall/furniture clutter band rather
    than on the real ceiling, which is exactly why it under-reported ceiling
    heights of 1.6-1.8m on rooms that turned out to have a real, cleanly
    horizontal ceiling plane around 2.1-2.3m once isolated. Fix: search
    progressively wider top-of-cloud slices, tightest first, and accept the
    first slice whose RANSAC-fit plane is genuinely horizontal and has a high
    inlier ratio. If no slice qualifies, the capture never got usable ceiling
    coverage -- report a low-confidence fallback instead of a falsely precise
    number (confidence_note downstream should reflect that).

    Returns (floor_y, ceiling_y, ceiling_height_m, confidence in [0, 1]).
    """
    pts = np.asarray(pcd.points)
    y = pts[:, 1]
    lo, hi = float(y.min()), float(y.max())
    span = hi - lo

    # --- floor: densest horizontal plane in the lower 70% of the range ---
    y_lower = y[y < lo + span * 0.7]
    bins = np.arange(y_lower.min(), y_lower.max() + 0.02, 0.02)
    hist, edges = np.histogram(y_lower, bins=bins)
    peak_center = (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]) / 2
    floor_fit = _fit_horizontal_plane(pts, np.abs(y - peak_center) < 0.06)
    if floor_fit is None or floor_fit[1] < 0.85:
        raise RuntimeError("Could not fit a floor plane -- check capture for floor coverage")
    floor_y, _normal_y, floor_ratio, _n = floor_fit
    floor_confidence = min(1.0, floor_ratio * 1.2)

    # --- ceiling: tightest top-slice band that fits a clean horizontal plane ---
    ceiling_y = None
    ceiling_confidence = 0.0
    for frac in (0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20, 0.30):
        fit = _fit_horizontal_plane(pts, y > hi - span * frac)
        if fit is None:
            continue
        plane_y, normal_y, ratio, n_inliers = fit
        if normal_y > 0.85 and ratio > 0.5 and n_inliers >= 30:
            ceiling_y, ceiling_confidence = plane_y, min(1.0, ratio)
            break

    if ceiling_y is None:
        ceiling_y, ceiling_confidence = hi, 0.0
        print("      WARNING: no reliable ceiling plane found at any search band -- "
              "falling back to highest observed point, confidence=0. Capture likely "
              "never tilted up enough to see the ceiling.")

    height = ceiling_y - floor_y
    confidence = min(floor_confidence, ceiling_confidence)
    return floor_y, ceiling_y, height, confidence


def _circular_median_filter(x: np.ndarray, window: int) -> np.ndarray:
    """Median filter over a circular (wrap-around) sequence."""
    n = len(x)
    half = window // 2
    out = np.empty_like(x)
    for i in range(n):
        idxs = [(i + k) % n for k in range(-half, half + 1)]
        out[i] = np.median(x[idxs])
    return out


def radial_boundary(points_2d: np.ndarray, num_bins: int = 180,
                     smooth_window: int = 7, spike_frac: float = 0.3):
    """
    Reconstructs a room's enclosing polygon from wall-surface points.

    Why not alpha-shape/convex-hull directly on these points: real wall
    points trace a thin OUTLINE (walls are surfaces, the room's interior air
    has no points), not a filled area. An alpha-shape's own area is the area
    of that thin ring itself -- near zero by construction -- not the
    enclosed room area. Confirmed by testing against a synthetic L-shaped
    room before relying on this: alpha-shape gave ~0.1-7 vs a true area of
    18; this radial sweep gave 17.76 (1.3% error).

    Method: for each angular bin around the point cloud's centroid, keep the
    farthest point in that bin -- that's the room boundary in that direction.
    This assumes the room is star-shaped from its centroid (true for typical
    rectangular/L-shaped rooms; can misbehave on deep narrow notches or
    C-shaped floor plans where a ray from centroid crosses the boundary more
    than once -- note this as a known limitation, don't silently trust it on
    unusual layouts).

    Post-process: rejects inward spikes. If the true wall was never sampled
    along a given ray (occluded by furniture in the middle of the room), the
    farthest point picked for that bin is the occluder, not the wall -- it
    reads as a sharp inward notch. Confirmed visually: on a real capture, a
    jagged intrusion toward the centroid lined up exactly with mid-room
    furniture, not an actual room shape, and coincided with false opening
    detections downstream (the same occlusion starves both signals). Real
    walls are locally smooth, so a lone bin sitting well inside a smoothed
    local-radius trend gets corrected to that trend instead of trusted as-is.
    """
    centroid = points_2d.mean(axis=0)
    rel = points_2d - centroid
    angles = np.arctan2(rel[:, 1], rel[:, 0])
    radii = np.linalg.norm(rel, axis=1)
    edges = np.linspace(-np.pi, np.pi, num_bins + 1)
    bin_idx = np.digitize(angles, edges) - 1

    boundary = []
    empty_bins = 0
    for b in range(num_bins):
        mask = bin_idx == b
        if not np.any(mask):
            empty_bins += 1
            continue
        best = np.argmax(radii[mask])
        boundary.append(points_2d[mask][best])

    if empty_bins > num_bins * 0.15:
        print(f"      WARNING: {empty_bins}/{num_bins} angular bins had no points "
              f"-- boundary may have gaps, consider lowering --bins or check capture coverage")
    if len(boundary) < 8:
        raise RuntimeError(f"Only {len(boundary)} boundary points recovered -- too sparse")

    boundary = np.array(boundary)
    b_rel = boundary - centroid
    b_radius = np.linalg.norm(b_rel, axis=1)
    b_angle = np.arctan2(b_rel[:, 1], b_rel[:, 0])

    window = min(smooth_window, len(boundary) - (1 - len(boundary) % 2))
    if window >= 3:
        smoothed = _circular_median_filter(b_radius, window)
        is_spike = b_radius < smoothed * (1 - spike_frac)
        n_spikes = int(is_spike.sum())
        if n_spikes:
            print(f"      corrected {n_spikes}/{len(boundary)} inward spikes in "
                  f"the wall footprint (likely furniture/clutter occluding the "
                  f"true wall along that ray)")
        corrected_radius = np.where(is_spike, smoothed, b_radius)
        boundary = centroid + corrected_radius[:, None] * np.stack(
            [np.cos(b_angle), np.sin(b_angle)], axis=1)

    return boundary


def extract_wall_footprint(pcd: o3d.geometry.PointCloud, floor_y: float, ceiling_y: float,
                            num_bins: int = 180):
    """
    Takes points in the 'wall band' (excluding floor/ceiling clutter), flattens
    to the floor plane (x,z), and returns the room boundary polygon via a
    radial sweep (see radial_boundary docstring for why, not alpha-shape).
    """
    pts = np.asarray(pcd.points)
    band = (pts[:, 1] > floor_y + 0.3) & (pts[:, 1] < ceiling_y - 0.3)
    wall_pts = pts[band][:, [0, 2]]  # (x, z) footprint

    finite_mask = np.all(np.isfinite(wall_pts), axis=1)
    wall_pts = wall_pts[finite_mask]

    if len(wall_pts) > 100:
        median = np.median(wall_pts, axis=0)
        dist = np.linalg.norm(wall_pts - median, axis=1)
        radius = np.percentile(dist, 99)
        wall_pts = wall_pts[dist < max(radius, 0.5)]

    if len(wall_pts) < 50:
        raise RuntimeError("Not enough wall-band points to extract a footprint")

    try:
        return radial_boundary(wall_pts, num_bins=num_bins)
    except Exception as e:
        print(f"      radial sweep failed ({e}), falling back to convex hull")
        from scipy.spatial import ConvexHull
        hull = ConvexHull(wall_pts)
        return wall_pts[hull.vertices]


def detect_openings(pcd: o3d.geometry.PointCloud, polygon: np.ndarray,
                     floor_y: float, ceiling_y: float, wall_thickness: float = 0.15,
                     bin_size: float = 0.05, min_width: float = 0.4, max_width: float = 1.6):
    """
    Heuristic door/window detector (Round-1 scope): walks each wall segment and
    looks for horizontal spans where vertical point coverage drops out in the
    door/window band (roughly knee-to-head height) while the floor near that
    span is still visible -- i.e. you can see past the wall into open space,
    not just a wall stretch that happens to be under-sampled.

    Known limitations:
    - A genuinely unsampled wall stretch with no floor visibility either will
      not be flagged -- this can silently miss openings from missing data.
    - Furniture pressed against a wall (a couch back, a low cabinet) blocks
      the same mid-height band a real opening would, and floor is still
      visible in front of it -- this reads identically to an opening and WILL
      produce false positives. No RGB/semantic check yet to tell the two
      apart; treat opening_count as an upper bound pending that.
    """
    pts = np.asarray(pcd.points)
    xz = pts[:, [0, 2]]
    y = pts[:, 1]
    height = ceiling_y - floor_y
    mid_lo, mid_hi = floor_y + 0.4 * height, floor_y + 0.9 * height
    floor_lo, floor_hi = floor_y, floor_y + 0.15

    openings = []
    n = len(polygon)
    for wi in range(n):
        p1, p2 = polygon[wi], polygon[(wi + 1) % n]
        seg_len = float(np.linalg.norm(p2 - p1))
        if seg_len < min_width:
            continue
        d_unit = (p2 - p1) / seg_len
        normal = np.array([-d_unit[1], d_unit[0]])
        rel = xz - p1
        t = rel @ d_unit
        perp = rel @ normal
        near_wall = (np.abs(perp) < wall_thickness) & (t > 0) & (t < seg_len)
        if not np.any(near_wall):
            continue
        t_wall, y_wall = t[near_wall], y[near_wall]

        edges = np.arange(0, seg_len + bin_size, bin_size)
        gap_mask = np.zeros(len(edges) - 1, dtype=bool)
        for bi in range(len(edges) - 1):
            in_bin = (t_wall >= edges[bi]) & (t_wall < edges[bi + 1])
            if not np.any(in_bin):
                continue  # no data at all -- can't tell gap from missing data
            y_bin = y_wall[in_bin]
            has_mid = np.any((y_bin > mid_lo) & (y_bin < mid_hi))
            has_floor_nearby = np.any((y_bin > floor_lo) & (y_bin < floor_hi))
            gap_mask[bi] = (not has_mid) and has_floor_nearby

        bi = 0
        while bi < len(gap_mask):
            if not gap_mask[bi]:
                bi += 1
                continue
            start = bi
            while bi < len(gap_mask) and gap_mask[bi]:
                bi += 1
            width = (bi - start) * bin_size
            if min_width <= width <= max_width:
                center_xz = p1 + d_unit * (edges[start] + width / 2)
                openings.append({
                    "wall_index": wi,
                    "position_xz": center_xz.tolist(),
                    "width_m": round(float(width), 3),
                })
    return openings


def compute_confidence_intervals(polygon: np.ndarray, perimeter: float,
                                  base_uncertainty_m: float = 0.03):
    """
    Placeholder error model pending real calibration against laser ground
    truth (Phase 6/7): treats each boundary point as independently uncertain by
    `base_uncertainty_m` (a commonly-cited close-range phone-LiDAR depth noise
    figure, not derived from this specific rig) and propagates it. This is
    intentionally conservative-simple, not a measured error budget -- do not
    quote these as calibrated until validated against ground truth.
    """
    wall_length_ci_m = round(float(np.sqrt(2) * base_uncertainty_m), 3)
    area_ci_m2 = round(float(perimeter * base_uncertainty_m), 3)
    return wall_length_ci_m, area_ci_m2


def ceiling_height_ci_m(confidence: float) -> float:
    """Confidence-tiered CI width -- high-confidence RANSAC plane fits get a
    tight interval; the confidence=0 fallback (no ceiling plane found at any
    search band) gets a deliberately wide one so it can't be mistaken for a
    real measurement."""
    if confidence >= 0.8:
        return 0.02
    if confidence >= 0.3:
        return 0.10
    return 0.50


def polygon_metrics(polygon: np.ndarray):
    """Wall segment lengths (m) and enclosed floor area (m^2) via shoelace formula."""
    n = len(polygon)
    lengths = []
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]
        lengths.append(float(np.linalg.norm(p2 - p1)))
    x = polygon[:, 0]
    z = polygon[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1)))
    return lengths, float(area)


def render_raw_scatter(pcd: o3d.geometry.PointCloud, out_path: str, title: str = "raw fused points"):
    """Plain top-down (x,z) scatter of every fused point, no hull/fitting
    involved. Use this to visually confirm the cloud actually looks like a
    room's floor footprint BEFORE trusting anything downstream."""
    pts = np.asarray(pcd.points)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pts[:, 0], pts[:, 2], s=0.5, alpha=0.3)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def render_plan(polygon: np.ndarray, out_path: str, room_name: str = "room", openings=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    patch = MplPolygon(polygon, closed=True, fill=False, edgecolor="black", linewidth=2)
    ax.add_patch(patch)
    ax.scatter(polygon[:, 0], polygon[:, 1], c="red", s=15, zorder=5)
    for op in (openings or []):
        x, z = op["position_xz"]
        ax.scatter([x], [z], c="blue", s=40, marker="s", zorder=6)
        ax.annotate(f'{op["width_m"]:.2f}m', (x, z), textcoords="offset points",
                    xytext=(4, 4), fontsize=7, color="blue")
    ax.set_aspect("equal")
    margin = 0.5
    ax.set_xlim(polygon[:, 0].min() - margin, polygon[:, 0].max() + margin)
    ax.set_ylim(polygon[:, 1].min() - margin, polygon[:, 1].max() + margin)
    ax.set_title(f"{room_name} -- floor plan (radial-sweep footprint)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--every-n", type=int, default=5)
    ap.add_argument("--min-confidence", type=int, default=2)
    ap.add_argument("--bins", type=int, default=180,
                     help="Number of angular bins for radial boundary sweep. "
                          "More bins = finer detail but needs denser points; try 120-360.")
    ap.add_argument("--room-name", default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    room_name = args.room_name or os.path.basename(os.path.normpath(args.scan_dir))

    print("[1/6] Fusing point cloud...")
    pcd = build_fused_point_cloud(args.scan_dir, every_n=args.every_n,
                                   min_confidence=args.min_confidence)
    o3d.io.write_point_cloud(os.path.join(args.out_dir, "fused_cloud.ply"), pcd)
    print(f"      {len(pcd.points)} points after filtering/downsampling")
    render_raw_scatter(pcd, os.path.join(args.out_dir, "raw_scatter.png"), room_name)

    print("[2/6] Detecting floor/ceiling...")
    floor_y, ceiling_y, height, height_confidence = detect_floor_and_ceiling(pcd)
    print(f"      floor_y={floor_y:.3f} ceiling_y={ceiling_y:.3f} height={height:.3f} m "
          f"confidence={height_confidence:.2f}")

    print("[3/6] Extracting wall footprint...")
    polygon = extract_wall_footprint(pcd, floor_y, ceiling_y, num_bins=args.bins)

    print("[4/6] Computing dimensions...")
    lengths, area = polygon_metrics(polygon)
    perimeter = float(sum(lengths))
    wall_length_ci_m, area_ci_m2 = compute_confidence_intervals(polygon, perimeter)
    height_ci_m = ceiling_height_ci_m(height_confidence)

    print("[5/6] Detecting openings...")
    openings = detect_openings(pcd, polygon, floor_y, ceiling_y)
    print(f"      {len(openings)} candidate opening(s) found")

    print("[6/6] Writing outputs...")
    render_plan(polygon, os.path.join(args.out_dir, "room_plan.png"), room_name, openings=openings)

    result = {
        "room_name": room_name,
        "source_scan": os.path.abspath(args.scan_dir),
        "tier": "lidar",
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
        "confidence_note": (
            "radial-sweep footprint -- reconstructs the enclosing polygon from "
            "wall-surface points, correct for typical rectangular/L-shaped "
            "rooms. Assumes the room is star-shaped from its centroid; may "
            "misbehave on deep narrow notches or C-shaped layouts. Ceiling "
            "height uses a tightest-band-first RANSAC plane search with an "
            "honest confidence score (0 = no ceiling plane found, height is a "
            "rough lower-bound fallback). Wall-length/area CIs are a "
            "placeholder error model (fixed close-range depth-noise assumption), "
            "not yet calibrated against ground truth. Opening detection is a "
            "gap-in-coverage heuristic: can miss real openings on sparse data, "
            "and can false-positive on furniture pressed against a wall (no "
            "RGB/semantic check yet to tell the two apart) -- treat "
            "opening_count as an upper bound."
        ),
    }
    with open(os.path.join(args.out_dir, "room.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nDone. Ceiling height: {height:.2f} m (+/-{height_ci_m:.2f}, "
          f"confidence {height_confidence:.2f}) | Floor area: {area:.2f} m^2 "
          f"(+/-{area_ci_m2:.2f}) | Openings: {len(openings)}")
    print(f"Outputs in {args.out_dir}/: room.json, room_plan.png, fused_cloud.ply")


if __name__ == "__main__":
    main()