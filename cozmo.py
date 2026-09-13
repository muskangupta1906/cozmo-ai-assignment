"""
cozmo.py
--------
Single front door for the whole pipeline: `python cozmo.py <input_folder> <out_dir>`.
Inspects the capture folder's shape to decide which tier's pipeline applies
(LiDAR / video / photo) so a fresh capture -- from any of the three tiers,
single- or multi-room -- always runs with the same one command, per the case
study's "one command per capture" requirement. `--tier` forces a choice when
detection is ambiguous or wrong (also needed for the walk-in test, where the
evaluator picks the tier on the day).

Detection rules (see detect_tier docstring for the reasoning):
    lidar        <input_folder>/camera_matrix.csv + odometry.csv present
                 (Stray Scanner export) -- routed through stitch_property.py,
                 which degrades correctly to a single "room_0" property when
                 the walkthrough has no distinct multi-room recurrence signal.
    video        a single video file in <input_folder> or <input_folder>/video/
    photo_room   images directly in <input_folder> (2-8 stills, one room), or
                 in <input_folder>/photos/ -- routed through reconstruct_room_photo.py
    photo_property  subfolders of per-room images -- routed through
                 stitch_property_photo.py, which needs an adjacency.json in
                 <input_folder> to actually place rooms relative to each other
                 (reconstructs each room independently, with a NOTE printed,
                 if it's missing -- see that module's load_adjacency docstring)

Usage:
    python cozmo.py <input_folder> <out_dir> [--tier lidar|video|photo] [--room-name NAME]
"""

import os
import sys
import glob
import argparse

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".heic", ".heif")
VIDEO_EXTS = (".mp4", ".mov")


def _has_images(d: str) -> bool:
    if not os.path.isdir(d):
        return False
    for ext in IMAGE_EXTS:
        if glob.glob(os.path.join(d, f"*{ext}")) or glob.glob(os.path.join(d, f"*{ext.upper()}")):
            return True
    return False


def _has_video(d: str) -> bool:
    if not os.path.isdir(d):
        return False
    for ext in VIDEO_EXTS:
        if glob.glob(os.path.join(d, f"*{ext}")) or glob.glob(os.path.join(d, f"*{ext.upper()}")):
            return True
    return False


def detect_tier(capture_dir: str) -> str:
    """Returns one of "lidar", "video", "photo_room", "photo_property"."""
    if not os.path.isdir(capture_dir):
        raise FileNotFoundError(f"Not a directory: {capture_dir}")

    has_lidar_files = (
        os.path.exists(os.path.join(capture_dir, "camera_matrix.csv"))
        and os.path.exists(os.path.join(capture_dir, "odometry.csv"))
    )
    if has_lidar_files:
        return "lidar"

    if os.path.exists(os.path.join(capture_dir, "rgb.mp4")) or _has_video(capture_dir) \
            or _has_video(os.path.join(capture_dir, "video")):
        return "video"

    if _has_images(capture_dir) or _has_images(os.path.join(capture_dir, "photos")):
        return "photo_room"

    subdirs = [
        os.path.join(capture_dir, e) for e in sorted(os.listdir(capture_dir))
        if not e.startswith(".") and os.path.isdir(os.path.join(capture_dir, e))
    ]
    room_like = [d for d in subdirs if _has_images(d) or _has_images(os.path.join(d, "photos"))]
    if room_like:
        return "photo_property"

    raise RuntimeError(
        f"Could not detect a capture tier for {capture_dir}. Expected one of: "
        f"camera_matrix.csv+odometry.csv (LiDAR), a video file or video/ "
        f"subfolder (video), images directly in the folder or in photos/ "
        f"(single-room photo), or subfolders of per-room images (multi-room "
        f"photo). Use --tier to force a choice if this folder's shape is "
        f"non-standard."
    )


def run_lidar(capture_dir: str, out_dir: str, room_name=None,
               min_confidence=2, bins=180, every_n=5):
    import stitch_property as sp
    import reconstruct_room as rr

    print("[cozmo] LiDAR capture -> multi-room stitch with drift-correction ablation "
          "(degrades to a single room automatically if no multi-room signal is found)")
    try:
        with_corr, _ = sp.stitch_with_ablation(capture_dir, out_dir, min_confidence=min_confidence,
                                                bins=bins, every_n=every_n)
        return with_corr
    except RuntimeError as e:
        if "No room clusters found" not in str(e):
            raise
    print("[cozmo] No distinct room clusters found (short/simple walkthrough) "
          "-> falling back to the plain single-room path")
    return rr.reconstruct_room(capture_dir, out_dir, every_n=every_n,
                                min_confidence=min_confidence, bins=bins, room_name=room_name)


def run_video(capture_dir: str, out_dir: str, fps=2.0, assumed_camera_height_m=1.4):
    import video_tier as vt
    print("[cozmo] Video capture -> classical SfM (pycolmap)")
    print("[cozmo] NOTE: video tier is not yet wired to the room.json output "
          "contract (floor/ceiling/footprint) -- see PHASE_PLAN.md Phase 3. "
          "This produces the aligned/scaled point cloud only.")
    return vt.reconstruct_video(capture_dir, out_dir, fps=fps,
                                 assumed_camera_height_m=assumed_camera_height_m)


def run_photo_room(capture_dir: str, out_dir: str, room_name=None, stride=4, bins=60, device=None):
    import reconstruct_room_photo as rp
    room_dir = capture_dir
    if not _has_images(capture_dir) and _has_images(os.path.join(capture_dir, "photos")):
        room_dir = os.path.join(capture_dir, "photos")
    print("[cozmo] Photo capture, single room -> monocular depth + registration chain")
    return rp.reconstruct_room_photo(room_dir, out_dir, room_name=room_name,
                                      stride=stride, bins=bins, device=device)


def run_photo_property(capture_dir: str, out_dir: str, stride=4, bins=60, device=None):
    import stitch_property_photo as spp
    print("[cozmo] Photo capture, multi-room (per-room folders) -> whole-property stitch")
    if not os.path.exists(os.path.join(capture_dir, "adjacency.json")):
        print("[cozmo] NOTE: no adjacency.json found -- rooms will be reconstructed "
              "independently with no real stitching (see stitch_property_photo.py's "
              "load_adjacency docstring). Add adjacency.json for a real property plan.")
    return spp.stitch(capture_dir, out_dir, stride=stride, bins=bins, device=device)


TIER_HANDLERS = {
    "lidar": run_lidar,
    "video": run_video,
    "photo_room": run_photo_room,
    "photo_property": run_photo_property,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_folder", help="Input folder (any tier)")
    ap.add_argument("out_dir")
    ap.add_argument("--tier", choices=sorted(TIER_HANDLERS.keys()), default=None,
                     help="Force a tier instead of auto-detecting from the capture folder's shape.")
    ap.add_argument("--room-name", default=None, help="LiDAR/photo tiers only")
    ap.add_argument("--every-n", type=int, default=5, help="LiDAR tier frame stride")
    ap.add_argument("--min-confidence", type=int, default=2, help="LiDAR tier")
    ap.add_argument("--bins", type=int, default=None,
                     help="Angular bins for wall-footprint sweep (tier-specific default if omitted)")
    ap.add_argument("--fps", type=float, default=2.0, help="Video tier frame-extraction rate")
    ap.add_argument("--stride", type=int, default=4, help="Photo tier back-projection pixel stride")
    ap.add_argument("--device", default=None, help="Photo tier: mps|cuda|cpu, auto-detected if omitted")
    args = ap.parse_args()

    tier = args.tier or detect_tier(args.input_folder)
    print(f"[cozmo] capture: {args.input_folder}")
    print(f"[cozmo] detected tier: {tier}" if not args.tier else f"[cozmo] tier (forced): {tier}")

    os.makedirs(args.out_dir, exist_ok=True)

    if tier == "lidar":
        run_lidar(args.input_folder, args.out_dir, room_name=args.room_name,
                  min_confidence=args.min_confidence, bins=args.bins or 180, every_n=args.every_n)
    elif tier == "video":
        run_video(args.input_folder, args.out_dir, fps=args.fps)
    elif tier == "photo_room":
        run_photo_room(args.input_folder, args.out_dir, room_name=args.room_name,
                        stride=args.stride, bins=args.bins or 60, device=args.device)
    elif tier == "photo_property":
        run_photo_property(args.input_folder, args.out_dir, stride=args.stride,
                            bins=args.bins or 60, device=args.device)
    else:
        raise ValueError(f"Unknown tier: {tier}")


if __name__ == "__main__":
    main()
