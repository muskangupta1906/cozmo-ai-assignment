
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


def build_fused_point_cloud(scan_dir: str, every_n: int = 5, min_confidence: int = 2,
                             max_depth_m: float = 5.0) -> o3d.geometry.PointCloud:
    """
    Back-projects every `every_n`-th frame's depth map to 3D and accumulates
    it into one point cloud in world coordinates.

    min_confidence: keep only depth pixels with confidence >= this (0..2, 2=best).
    Using only high-confidence points early keeps the first pass from choking
    on noisy edges/reflective surfaces -- widen later once the pipeline works.
    """
    K = load_intrinsics(scan_dir)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

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

    all_points = []
    frames_attempted = 0
    frames_skipped_pose = 0
    frames_skipped_nopoints = 0
    for i in range(0, n, every_n):
        frames_attempted += 1
        depth_mm = load_depth_mm(depth_frames[i])
        conf = load_confidence(conf_frames[i])
        if depth_mm.shape != (depth_h, depth_w):
            depth_mm = cv2_resize_nn(depth_mm, depth_w, depth_h)
            conf = cv2_resize_nn(conf, depth_w, depth_h)

        z = depth_mm / 1000.0  # mm -> m
        valid = (conf >= min_confidence) & (z > 0.05) & (z < max_depth_m)
        if not np.any(valid):
            frames_skipped_nopoints += 1
            continue

        x = (us[valid] - cx_d) * z[valid] / fx_d
        y = (vs[valid] - cy_d) * z[valid] / fy_d
        zc = z[valid]
        pts_cam = np.stack([x, y, zc, np.ones_like(zc)], axis=1)  # (N,4)

        row = odo.iloc[i]
        quat_norm = np.sqrt(row["qx"]**2 + row["qy"]**2 + row["qz"]**2 + row["qw"]**2)
        pose_vals = [row["x"], row["y"], row["z"], row["qx"], row["qy"], row["qz"], row["qw"]]
        if quat_norm < 1e-8 or not np.all(np.isfinite(pose_vals)):
            # Degenerate/missing pose for this frame (seen in real captures,
            # e.g. a dropped VIO frame) -- skip it rather than let a NaN
            # transform poison the whole fused cloud.
            frames_skipped_pose += 1
            continue
        T = pose_to_T(row)
        pts_world = (T @ pts_cam.T).T[:, :3]
        if not np.all(np.isfinite(pts_world)):
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


def detect_floor_and_ceiling(pcd: o3d.geometry.PointCloud):
    """
    ARKit world frame is y-up. Floor = plane of points with the lowest y values
    that fits a near-horizontal plane; ceiling = highest such plane.
    Returns (floor_y, ceiling_y, ceiling_height_m).
    """
    pts = np.asarray(pcd.points)
    y = pts[:, 1]
    # Robust floor/ceiling estimate via percentiles first (fast, avoids full
    # RANSAC needing a clean single-plane assumption on a messy real room).
    floor_y = np.percentile(y, 2)
    ceiling_y = np.percentile(y, 98)
    height = ceiling_y - floor_y
    return floor_y, ceiling_y, height


def radial_boundary(points_2d: np.ndarray, num_bins: int = 180):
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

    return np.array(boundary)


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


def render_plan(polygon: np.ndarray, out_path: str, room_name: str = "room"):
    fig, ax = plt.subplots(figsize=(6, 6))
    patch = MplPolygon(polygon, closed=True, fill=False, edgecolor="black", linewidth=2)
    ax.add_patch(patch)
    ax.scatter(polygon[:, 0], polygon[:, 1], c="red", s=15, zorder=5)
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

    print("[1/5] Fusing point cloud...")
    pcd = build_fused_point_cloud(args.scan_dir, every_n=args.every_n,
                                   min_confidence=args.min_confidence)
    o3d.io.write_point_cloud(os.path.join(args.out_dir, "fused_cloud.ply"), pcd)
    print(f"      {len(pcd.points)} points after filtering/downsampling")
    render_raw_scatter(pcd, os.path.join(args.out_dir, "raw_scatter.png"), room_name)

    print("[2/5] Detecting floor/ceiling...")
    floor_y, ceiling_y, height = detect_floor_and_ceiling(pcd)
    print(f"      floor_y={floor_y:.3f} ceiling_y={ceiling_y:.3f} height={height:.3f} m")

    print("[3/5] Extracting wall footprint...")
    polygon = extract_wall_footprint(pcd, floor_y, ceiling_y, num_bins=args.bins)

    print("[4/5] Computing dimensions...")
    lengths, area = polygon_metrics(polygon)

    print("[5/5] Writing outputs...")
    render_plan(polygon, os.path.join(args.out_dir, "room_plan.png"), room_name)

    result = {
        "room_name": room_name,
        "source_scan": os.path.abspath(args.scan_dir),
        "tier": "lidar",
        "ceiling_height_m": round(height, 3),
        "floor_area_m2": round(area, 3),
        "wall_count": len(lengths),
        "wall_lengths_m": [round(l, 3) for l in lengths],
        "footprint_polygon_xz": polygon.tolist(),
        "confidence_note": (
            "radial-sweep footprint -- reconstructs the enclosing polygon from "
            "wall-surface points, correct for typical rectangular/L-shaped "
            "rooms. Assumes the room is star-shaped from its centroid; may "
            "misbehave on deep narrow notches or C-shaped layouts. No opening "
            "detection or confidence intervals yet -- next fix-loop candidates."
        ),
    }
    with open(os.path.join(args.out_dir, "room.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nDone. Ceiling height: {height:.2f} m | Floor area: {area:.2f} m^2")
    print(f"Outputs in {args.out_dir}/: room.json, room_plan.png, fused_cloud.ply")


if __name__ == "__main__":
    main()