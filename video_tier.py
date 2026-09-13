"""
video_tier.py
--------------
Video-tier pipeline (Phase 3): reconstructs a room from a plain RGB video --
no depth sensor, no device pose. Approach: classical structure-from-motion
(COLMAP, via pycolmap) recovers camera poses + a 3D point cloud up to an
unknown similarity transform (rotation, translation, AND scale are all
free -- unlike the LiDAR tier's ARKit poses, which come pre-aligned to
gravity and metric scale).

This module owns exactly the two problems that are new at this tier and
don't exist for LiDAR:
  1. Gravity alignment: COLMAP's world frame has no relationship to "up".
     estimate_up_vector / align_up fix this by assuming the operator held
     the phone close to upright (not intentionally rolled) while walking,
     so each camera's local up axis, averaged across all frames and
     rotated into world space, is a usable "up" estimate.
  2. Scale recovery: SfM reconstructs shape, not size. estimate_scale
     anchors metric scale to an assumed average phone-carry height above
     the floor -- an explicit, documented placeholder (same spirit as
     reconstruct_room.py's placeholder CI model), NOT a calibrated value.
     Real calibration needs a known-length reference in frame or Phase 6/7
     ground truth.

Once the point cloud is gravity-aligned and metric-scaled, it's handed to
the exact same floor/ceiling + wall-footprint logic reconstruct_room.py
already validated at the LiDAR tier -- no reason to duplicate or
reinvent that geometry once the point cloud is in the same convention
(y-up, meters).

Usage:
    python video_tier.py <scan_dir_with_rgb.mp4> <out_dir> [--fps 2]

Status: sparse-SfM + alignment/scale pipeline is implemented and smoke-
tested. Wiring the aligned cloud into detect_floor_and_ceiling /
extract_wall_footprint (and deciding whether sparse points are dense
enough or a dense stereo pass is needed first) is the next increment --
see PHASE_PLAN.md Phase 3 for current status before trusting any room.json
this produces.
"""

import os
import sys
import glob
import shutil
import subprocess
import argparse
import numpy as np
import pycolmap

VIDEO_EXTS = (".mp4", ".mov")


def find_video_file(scan_dir: str) -> str:
    """Locates the capture's video file. Checks, in order: `rgb.mp4` (the
    LiDAR-tier export's fixed name, kept so a Stray Scanner folder's own RGB
    track can still be used as a stand-in per PHASE_PLAN's Phase 3 notes),
    then any single video file directly in scan_dir or in a `video/`
    subfolder (the shape docs/capture_protocol.md's Handoff section
    specifies for a real video-tier capture: `Assignment/<room>/video/`)."""
    rgb_mp4 = os.path.join(scan_dir, "rgb.mp4")
    if os.path.exists(rgb_mp4):
        return rgb_mp4
    candidates = []
    for d in (scan_dir, os.path.join(scan_dir, "video")):
        if os.path.isdir(d):
            for ext in VIDEO_EXTS:
                candidates.extend(glob.glob(os.path.join(d, f"*{ext}")))
                candidates.extend(glob.glob(os.path.join(d, f"*{ext.upper()}")))
    if not candidates:
        raise FileNotFoundError(
            f"No video file found in {scan_dir} or {scan_dir}/video/ "
            f"(looked for rgb.mp4 and *.mp4/*.mov)")
    if len(candidates) > 1:
        print(f"      NOTE: {len(candidates)} video files found in {scan_dir}, "
              f"using the first one: {sorted(candidates)[0]}")
    return sorted(candidates)[0]


def extract_frames(video_path: str, out_dir: str, fps: float = 2.0):
    """
    Samples video_path at `fps` frames/sec into out_dir/000000.jpg, ...
    using ffmpeg. A fixed fps (not every_n on frame count) so extraction
    density doesn't silently change if someone hands in a differently-
    encoded video -- SfM needs enough visual overlap between consecutive
    frames to match, not a specific frame count.
    """
    os.makedirs(out_dir, exist_ok=True)
    existing = sorted(f for f in os.listdir(out_dir) if f.endswith(".jpg"))
    if existing:
        return out_dir
    cmd = [
        "ffmpeg", "-loglevel", "error", "-i", video_path,
        "-vf", f"fps={fps}",
        os.path.join(out_dir, "%06d.jpg"),
    ]
    subprocess.run(cmd, check=True)
    n = len([f for f in os.listdir(out_dir) if f.endswith(".jpg")])
    print(f"      extracted {n} frames at {fps} fps")
    return out_dir


def run_sfm(image_dir: str, work_dir: str) -> pycolmap.Reconstruction:
    """
    Runs COLMAP's standard sparse pipeline (SIFT features -> sequential
    matching -> incremental mapping) entirely through pycolmap's in-process
    API -- no external `colmap` binary needed.

    Sequential (not exhaustive) matching: a walkthrough video's frames are
    time-ordered and mostly overlap their near neighbors, so sequential
    matching finds the real matches at a fraction of exhaustive's O(n^2)
    cost. Loop-closure style revisits (walking back through the same room)
    are the one case this can miss -- acceptable for a first pass; the
    LiDAR tier's drift-correction ablation covers the loop-closure story
    for this case study, not this tier.

    Returns the largest reconstruction (most registered images) if COLMAP
    fragments the scene into multiple disconnected models -- a real
    failure mode on textureless walls / fast motion, not filtered away
    silently: caller should check num_reg_images against the input frame
    count.
    """
    database_path = os.path.join(work_dir, "database.db")
    sparse_path = os.path.join(work_dir, "sparse")
    os.makedirs(sparse_path, exist_ok=True)
    if os.path.exists(database_path):
        os.remove(database_path)

    pycolmap.extract_features(database_path, image_dir)
    pycolmap.match_sequential(database_path)
    reconstructions = pycolmap.incremental_mapping(database_path, image_dir, sparse_path)

    if not reconstructions:
        raise RuntimeError("COLMAP failed to register any images -- check frame "
                            "overlap/coverage (too few frames, too much blur/rotation-"
                            "only motion, or textureless scene)")

    best = max(reconstructions.values(), key=lambda r: r.num_reg_images())
    print(f"      SfM registered {best.num_reg_images()} images, "
          f"{best.num_points3D()} 3D points "
          f"({len(reconstructions)} disconnected model(s) found, kept largest)")
    return best


def estimate_up_vector(reconstruction: pycolmap.Reconstruction) -> np.ndarray:
    """
    Averages each registered camera's local up axis (image -Y, since image
    coordinates point down) rotated into world space. Assumes the operator
    held the phone close to upright throughout the walk -- if that's
    false (e.g. the phone was rolled 90 deg for part of the capture), this
    estimate is wrong and everything downstream (floor/ceiling, footprint)
    inherits that error silently. No detection for that failure mode yet;
    flagged as a known gap, not fixed blind.
    """
    ups = []
    for image in reconstruction.images.values():
        if not image.has_pose:
            continue
        R = image.cam_from_world().rotation.matrix()  # world -> camera
        up_world = R.T @ np.array([0.0, -1.0, 0.0])
        ups.append(up_world)
    if not ups:
        raise RuntimeError("No registered camera poses to estimate an up vector from")
    up = np.mean(ups, axis=0)
    norm = np.linalg.norm(up)
    if norm < 1e-6:
        raise RuntimeError("Camera up vectors cancel out (near-zero average) -- "
                            "capture likely rolled/rotated too much for this heuristic")
    return up / norm


def _rotation_aligning(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix R such that R @ a ~= b, both unit vectors."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    if s < 1e-8:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))


def align_up(reconstruction: pycolmap.Reconstruction, up_vector: np.ndarray):
    """Rotates the whole reconstruction (points + camera poses) so up_vector -> world +Y."""
    R = _rotation_aligning(up_vector, np.array([0.0, 1.0, 0.0]))
    sim3d = pycolmap.Sim3d(1.0, pycolmap.Rotation3d(R), np.zeros(3))
    reconstruction.transform(sim3d)


def estimate_scale(reconstruction: pycolmap.Reconstruction,
                    assumed_camera_height_m: float = 1.4) -> float:
    """
    Anchors metric scale to an assumed average phone-carry height above
    the floor. Floor, in the already-up-aligned-but-unscaled reconstruction,
    is approximated as the 5th percentile of camera projection-center
    heights (cameras stay near a constant walking height; using camera
    heights rather than point-cloud points sidesteps needing a floor plane
    fit before scale is even known). Explicitly a placeholder, not a
    calibration -- see module docstring. Returns the scale factor applied
    (colmap_units -> meters) for logging/diagnostics.
    """
    cam_ys = np.array([img.projection_center()[1] for img in reconstruction.images.values()
                        if img.has_pose])
    floor_y_unitless = np.percentile(cam_ys, 5)
    median_height_unitless = np.median(cam_ys) - floor_y_unitless
    if median_height_unitless <= 1e-6:
        raise RuntimeError("Degenerate camera-height spread -- can't anchor scale "
                            "(capture may not have enough vertical/walking motion)")
    scale = assumed_camera_height_m / median_height_unitless
    sim3d = pycolmap.Sim3d(scale, pycolmap.Rotation3d(np.eye(3)), np.zeros(3))
    reconstruction.transform(sim3d)
    return scale


def reconstruct_video(scan_dir: str, out_dir: str, fps: float = 2.0,
                       assumed_camera_height_m: float = 1.4):
    os.makedirs(out_dir, exist_ok=True)
    video_path = find_video_file(scan_dir)

    print("[1/4] Extracting frames...")
    image_dir = extract_frames(video_path, os.path.join(out_dir, "frames"), fps=fps)

    print("[2/4] Running structure-from-motion (this can take a few minutes)...")
    reconstruction = run_sfm(image_dir, out_dir)

    print("[3/4] Estimating gravity direction from camera poses...")
    up = estimate_up_vector(reconstruction)
    align_up(reconstruction, up)

    print("[4/4] Anchoring metric scale to assumed camera-carry height...")
    scale = estimate_scale(reconstruction, assumed_camera_height_m)

    points = np.array([p.xyz for p in reconstruction.points3D.values()])
    cam_heights = np.array([img.projection_center()[1] for img in reconstruction.images.values()
                             if img.has_pose])
    aligned_path = os.path.join(out_dir, "sparse_aligned")
    os.makedirs(aligned_path, exist_ok=True)
    reconstruction.write(aligned_path)

    print(f"\nDone. {reconstruction.num_reg_images()} images registered, "
          f"{len(points)} 3D points.")
    print(f"Scale factor applied (colmap units -> meters): {scale:.4f}")
    print(f"Point cloud y-range (post-align/scale, meters): "
          f"[{points[:,1].min():.2f}, {points[:,1].max():.2f}]" if len(points) else
          "Point cloud empty")
    print(f"Camera height spread (meters): median={np.median(cam_heights):.2f} "
          f"min={cam_heights.min():.2f} max={cam_heights.max():.2f}")
    print(f"Outputs in {out_dir}/: frames/, sparse/ (raw), sparse_aligned/ (gravity+scale)")
    return reconstruction, points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--assumed-camera-height-m", type=float, default=1.4)
    args = ap.parse_args()
    reconstruct_video(args.scan_dir, args.out_dir, fps=args.fps,
                       assumed_camera_height_m=args.assumed_camera_height_m)


if __name__ == "__main__":
    main()
