"""
scan_io.py
----------
Loader for raw capture folders in the Stray Scanner format:

    <scan_folder>/
        camera_matrix.csv   # 3x3 intrinsics, no header
        odometry.csv        # header: timestamp, frame, x, y, z, qx, qy, qz, qw
        imu.csv              # accel/gyro readings (not used in v1)
        depth/000000.png ... # 16-bit mm depth, 256x192
        confidence/000000.png ...  # 0/1/2 confidence per pixel
        rgb.mp4              # color video, same frame count/order as depth/

This module ONLY reads and aligns raw data. It makes no geometry decisions.
"""

import os
import glob
import numpy as np
import pandas as pd
import cv2

DEPTH_W, DEPTH_H = 256, 192


def load_intrinsics(scan_dir: str) -> np.ndarray:
    """Returns the 3x3 camera intrinsics matrix (fx, fy, cx, cy on the diagonal/last col)."""
    path = os.path.join(scan_dir, "camera_matrix.csv")
    K = np.loadtxt(path, delimiter=",")
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    return K


def load_odometry(scan_dir: str) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per frame:
        frame, x, y, z, qx, qy, qz, qw
    Poses are camera-position + orientation-quaternion in the ARKit world frame
    (right-handed, y-up, meters).
    """
    path = os.path.join(scan_dir, "odometry.csv")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    # Some exports use 'frame' as an int index, others just rely on row order.
    if "frame" not in df.columns:
        df["frame"] = np.arange(len(df))
    return df


def quat_to_R(qx, qy, qz, qw) -> np.ndarray:
    """Quaternion (x,y,z,w) -> 3x3 rotation matrix."""
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx*qy - qz*qw),     2 * (qx*qz + qy*qw)],
        [2 * (qx*qy + qz*qw),     1 - 2 * (qx**2 + qz**2), 2 * (qy*qz - qx*qw)],
        [2 * (qx*qz - qy*qw),     2 * (qy*qz + qx*qw),     1 - 2 * (qx**2 + qy**2)],
    ])


def pose_to_T(row) -> np.ndarray:
    """Builds the 4x4 camera-to-world transform for one odometry row."""
    R = quat_to_R(row["qx"], row["qy"], row["qz"], row["qw"])
    t = np.array([row["x"], row["y"], row["z"]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def list_depth_frames(scan_dir: str):
    return sorted(glob.glob(os.path.join(scan_dir, "depth", "*.png")))


def list_confidence_frames(scan_dir: str):
    return sorted(glob.glob(os.path.join(scan_dir, "confidence", "*.png")))


def load_depth_mm(path: str) -> np.ndarray:
    """16-bit depth PNG -> float32 array in millimeters, shape (192, 256)."""
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(path)
    return d.astype(np.float32)


def load_confidence(path: str) -> np.ndarray:
    """Confidence PNG -> uint8 array, values in {0, 1, 2}."""
    c = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if c is None:
        raise FileNotFoundError(path)
    return c.astype(np.uint8)


def extract_rgb_frames(scan_dir: str, out_dir: str, every_n: int = 1):
    """
    Extracts frames from rgb.mp4 into out_dir as 000000.jpg, 000001.jpg, ...
    aligned by index with depth/ and confidence/. Returns the sorted list of paths.
    Requires ffmpeg on PATH.
    """
    os.makedirs(out_dir, exist_ok=True)
    video_path = os.path.join(scan_dir, "rgb.mp4")
    existing = sorted(glob.glob(os.path.join(out_dir, "*.jpg")))
    if existing:
        return existing
    cap = cv2.VideoCapture(video_path)
    idx = 0
    saved = 0
    paths = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % every_n == 0:
            p = os.path.join(out_dir, f"{saved:06d}.jpg")
            cv2.imwrite(p, frame)
            paths.append(p)
            saved += 1
        idx += 1
    cap.release()
    return paths


def summarize_scan(scan_dir: str):
    """Quick sanity check: prints frame counts across modalities so mismatches
    (dropped frames) are caught before doing any geometry."""
    K = load_intrinsics(scan_dir)
    odo = load_odometry(scan_dir)
    depth_frames = list_depth_frames(scan_dir)
    conf_frames = list_confidence_frames(scan_dir)
    print(f"scan: {scan_dir}")
    print(f"  intrinsics:\n{K}")
    print(f"  odometry rows: {len(odo)}")
    print(f"  depth frames:  {len(depth_frames)}")
    print(f"  confidence frames: {len(conf_frames)}")
    n = min(len(odo), len(depth_frames), len(conf_frames))
    if len({len(odo), len(depth_frames), len(conf_frames)}) > 1:
        print(f"  WARNING: frame counts differ, will align on first {n} frames")
    return K, odo, depth_frames, conf_frames


if __name__ == "__main__":
    import sys
    summarize_scan(sys.argv[1])