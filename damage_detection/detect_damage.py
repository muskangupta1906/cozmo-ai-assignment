"""
damage_detection/detect_damage.py
----------------------------------
Phase 5, first increment only: per-frame damage candidate detection +
pretrained zero-shot classification. Explicitly NOT yet doing 3D
dedup/clustering across frames, metric extent, concealed-damage rules, or
scope line items -- see PHASE_PLAN.md Phase 5 for the full planned
pipeline; this is step 1+2 of it, wired end to end and smoke-tested.

For every sampled RGB frame in a LiDAR-tier scan:
  1. Back-project its depth map to world points, reusing
     reconstruct_room.backproject_frame -- the same validated math the
     LiDAR tier already relies on for geometry -- then mask down to pixels
     that lie near the room's known floor/ceiling/wall planes (computed
     once from the whole scan via the same functions reconstruct_room.py
     uses). This keeps candidate search on actual room surfaces, not
     furniture/people/clutter.
  2. Within that mask, find color-anomalous regions against a blurred
     local background (classical CV, no model) -- candidate damage blobs.
  3. Classify each candidate crop zero-shot with a pretrained CLIP model
     (open_clip, ViT-B-32-quickgelu/openai weights -- disclosed pretrained-
     model use, no fine-tuning, weights fetched by open_clip on first run
     and cached) against a fixed damage-class prompt set plus a "clean
     surface" negative prompt. Below-threshold or negative-class
     predictions are discarded.

Output (in out_dir):
  - damage_detections.json: every surviving detection (frame index/path,
    surface type, pixel bbox, predicted class, confidence, world-space
    centroid when depth was available there).
  - frames_with_damage.json: just the frame list + classes found, for a
    quick "does this capture have anything worth a closer look" summary.
  - damage_plan.png: the room's footprint polygon (same geometry
    reconstruct_room.py's room_plan.png uses) with each detection's world
    (x, z) centroid marked and labeled by class -- a first cut at "show
    the damage location on the room plan." No dedup yet: the same real
    damage seen across multiple frames shows up as multiple markers here,
    on purpose left for the 3D-clustering increment rather than silently
    (and unvalidated-ly) merged by a guess now.

Known limitations, honestly not fixed here (see also PHASE_PLAN.md):
  - DAMAGE_CLASSES below is a generic placeholder set, not matched to
    whatever ends up staged in the Phase 6 damage room -- swap it in once
    that capture exists.
  - No ground truth exists yet to validate accuracy against. Every sample
    scan currently in Assignment/ is an undamaged room, so a clean run
    finding ~0 detections is the *expected*, correct result, not evidence
    the pipeline works -- this is a mechanics smoke test only, same
    honesty-about-data-gap as Phase 3's video tier.
  - Classical color-anomaly region proposal will flag real but non-damage
    high-contrast surface features (light switches, outlet covers,
    picture-hanging marks, shadows) -- CLIP classification is the only
    filter against that right now; no separate "is this even damage-
    shaped" gate yet.

Usage:
    python damage_detection/detect_damage.py <scan_dir> <out_dir> [--every-n 15] [--conf-threshold 0.5]
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import reconstruct_room as rr
from scan_io import (
    load_intrinsics, load_odometry, list_depth_frames, list_confidence_frames,
    load_depth_mm, load_confidence,
)

# Placeholder damage-class vocabulary -- see module docstring. Each is
# turned into a CLIP text prompt at load time; edit freely once the real
# Phase 6 staged-damage classes are known.
DAMAGE_CLASSES = [
    "water stain",
    "mold",
    "crack",
    "hole or puncture damage",
    "peeling or bubbling paint",
    "scuff or scratch mark",
]
NEGATIVE_CLASS = "clean undamaged surface"
CLIP_MODEL_NAME = "ViT-B-32-quickgelu"
CLIP_PRETRAINED = "openai"


def load_clip():
    import torch
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED)
    model.eval()
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    classes = DAMAGE_CLASSES + [NEGATIVE_CLASS]
    prompts = [f"a close-up photo of {c} on an interior wall or ceiling" for c in classes]
    with torch.no_grad():
        text_features = model.encode_text(tokenizer(prompts))
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return model, preprocess, text_features, classes


def classify_crop(model, preprocess, text_features, classes, crop_rgb: np.ndarray):
    import torch
    image = preprocess(Image.fromarray(crop_rgb)).unsqueeze(0)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)[0]
    idx = int(probs.argmax())
    return classes[idx], float(probs[idx])


def _point_segment_distance(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    ab_len2 = float(np.dot(ab, ab))
    if ab_len2 < 1e-12:
        return np.linalg.norm(pts - a, axis=1)
    t = np.clip(((pts - a) @ ab) / ab_len2, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.linalg.norm(pts - proj, axis=1)


def point_to_polygon_distance(points_xz: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Min distance from each point to the polygon's boundary (closed loop of edges)."""
    m = len(polygon)
    min_dist = np.full(len(points_xz), np.inf)
    for i in range(m):
        a, b = polygon[i], polygon[(i + 1) % m]
        min_dist = np.minimum(min_dist, _point_segment_distance(points_xz, a, b))
    return min_dist


def classify_surface(pts_world: np.ndarray, floor_y: float, ceiling_y: float,
                      polygon: np.ndarray, floor_ceiling_tol: float = 0.05,
                      wall_margin: float = 0.1, wall_tol: float = 0.15):
    """
    Per-point surface label ('floor' | 'ceiling' | 'wall' | '' for none),
    using the room's already-computed floor/ceiling heights and wall
    footprint polygon -- the same geometry reconstruct_room.py validated,
    not re-derived here. '' points are furniture/clutter/mid-air and get
    excluded from damage candidate search entirely.
    """
    y = pts_world[:, 1]
    is_floor = np.abs(y - floor_y) < floor_ceiling_tol
    is_ceiling = np.abs(y - ceiling_y) < floor_ceiling_tol
    is_wall_height = (y > floor_y + wall_margin) & (y < ceiling_y - wall_margin)
    wall_dist = point_to_polygon_distance(pts_world[:, [0, 2]], polygon)
    is_wall = is_wall_height & (wall_dist < wall_tol)

    label = np.full(len(pts_world), "", dtype=object)
    label[is_wall] = "wall"
    label[is_ceiling] = "ceiling"
    label[is_floor] = "floor"  # floor takes priority where bands overlap near skirting
    return label


def propose_candidate_regions(rgb: np.ndarray, surface_mask: np.ndarray,
                               min_area_px: int = 300, max_area_frac: float = 0.25):
    """
    Classical color-anomaly region proposal: pixels that deviate from a
    blurred local-background estimate, restricted to surface_mask. No
    model here -- damage isn't known yet, just "doesn't match its
    surroundings." Returns a list of (x, y, w, h) pixel bboxes.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    background = cv2.GaussianBlur(lab, (0, 0), sigmaX=25)
    anomaly = np.linalg.norm(lab - background, axis=2)
    anomaly[~surface_mask] = 0.0

    masked_vals = anomaly[surface_mask]
    if masked_vals.size == 0:
        return []
    threshold = max(12.0, float(np.percentile(masked_vals, 97)))
    binary = (anomaly > threshold).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    num, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    max_area = max_area_frac * surface_mask.sum()
    boxes = []
    for i in range(1, num):  # 0 is background
        x, y, w, h, area = stats[i]
        if min_area_px <= area <= max_area:
            boxes.append((x, y, w, h))
    return boxes


def process_frame(rgb: np.ndarray, depth_mm: np.ndarray, conf: np.ndarray, row,
                   fx_d, fy_d, cx_d, cy_d, us, vs, min_confidence,
                   floor_y, ceiling_y, polygon,
                   clip_model, clip_preprocess, text_features, classes,
                   conf_threshold: float):
    depth_h, depth_w = depth_mm.shape
    rgb_h, rgb_w = rgb.shape[:2]

    pts_world, valid = rr.backproject_frame(depth_mm, conf, row, fx_d, fy_d, cx_d, cy_d,
                                             us, vs, min_confidence)
    if pts_world is None or not np.any(valid):
        return []

    labels = classify_surface(pts_world, floor_y, ceiling_y, polygon)
    ys, xs = np.where(valid)  # (row, col) per pts_world entry, same order (row-major)

    surface_mask_depth = np.zeros((depth_h, depth_w), dtype=bool)
    on_surface = labels != ""
    surface_mask_depth[ys[on_surface], xs[on_surface]] = True

    world_img = np.full((depth_h, depth_w, 3), np.nan, dtype=np.float32)
    world_img[ys, xs] = pts_world

    surface_mask_rgb = cv2.resize(surface_mask_depth.astype(np.uint8), (rgb_w, rgb_h),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)

    boxes = propose_candidate_regions(rgb, surface_mask_rgb)
    if not boxes:
        return []

    scale_x, scale_y = depth_w / rgb_w, depth_h / rgb_h
    detections = []
    for (x, y, w, h) in boxes:
        pad = int(0.1 * max(w, h))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(rgb_w, x + w + pad), min(rgb_h, y + h + pad)
        crop = rgb[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        pred_class, pred_conf = classify_crop(clip_model, clip_preprocess, text_features,
                                               classes, crop)
        if pred_class == NEGATIVE_CLASS or pred_conf < conf_threshold:
            continue

        cx_rgb, cy_rgb = x + w / 2.0, y + h / 2.0
        dr, dc = int(round(cy_rgb * scale_y)), int(round(cx_rgb * scale_x))
        win = world_img[max(0, dr - 2):dr + 3, max(0, dc - 2):dc + 3].reshape(-1, 3)
        win = win[~np.isnan(win).any(axis=1)]
        world_xyz = win.mean(axis=0).tolist() if len(win) else None
        surface_type = labels[on_surface][
            np.argmin(np.abs(ys[on_surface] - dr) + np.abs(xs[on_surface] - dc))
        ] if on_surface.any() else "unknown"

        detections.append({
            "bbox_px": [int(x), int(y), int(w), int(h)],
            "surface_type": str(surface_type),
            "predicted_class": pred_class,
            "confidence": round(pred_conf, 3),
            "world_xyz": world_xyz,
        })
    return detections


def read_rgb_frames_sequential(video_path: str, wanted_indices: set):
    cap = cv2.VideoCapture(video_path)
    i = 0
    wanted_max = max(wanted_indices)
    while True:
        ok, frame_bgr = cap.read()
        if not ok or i > wanted_max:
            break
        if i in wanted_indices:
            yield i, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()


def run(scan_dir: str, out_dir: str, every_n: int = 15, conf_threshold: float = 0.5,
        min_confidence: int = 2):
    os.makedirs(out_dir, exist_ok=True)

    print("[1/4] Computing room geometry (floor/ceiling/footprint)...")
    pcd = rr.build_fused_point_cloud(scan_dir, min_confidence=min_confidence)
    floor_y, ceiling_y, height, height_conf = rr.detect_floor_and_ceiling(pcd)
    polygon = rr.extract_wall_footprint(pcd, floor_y, ceiling_y)
    print(f"      floor_y={floor_y:.3f} ceiling_y={ceiling_y:.3f} height={height:.3f}m")

    print("[2/4] Loading pretrained CLIP model (open_clip, disclosed pretrained use)...")
    clip_model, clip_preprocess, text_features, classes = load_clip()

    K = load_intrinsics(scan_dir)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    odo = load_odometry(scan_dir)
    depth_frames = list_depth_frames(scan_dir)
    conf_frames = list_confidence_frames(scan_dir)
    n = min(len(odo), len(depth_frames), len(conf_frames))

    first_depth = load_depth_mm(depth_frames[0])
    depth_h, depth_w = first_depth.shape[:2]
    rgb_w, rgb_h = rr.get_rgb_resolution(scan_dir)
    scale_x, scale_y = depth_w / rgb_w, depth_h / rgb_h
    fx_d, fy_d, cx_d, cy_d = fx * scale_x, fy * scale_y, cx * scale_x, cy * scale_y
    us, vs = np.meshgrid(np.arange(depth_w), np.arange(depth_h))

    frame_indices = list(range(0, n, every_n))
    print(f"[3/4] Scanning {len(frame_indices)} frames (every {every_n}) for damage candidates...")

    all_detections = []
    frames_with_damage = []
    video_path = os.path.join(scan_dir, "rgb.mp4")
    for idx, rgb in read_rgb_frames_sequential(video_path, set(frame_indices)):
        if rgb.shape[:2] != (rgb_h, rgb_w):
            rgb = cv2.resize(rgb, (rgb_w, rgb_h))
        depth_mm = load_depth_mm(depth_frames[idx])
        conf = load_confidence(conf_frames[idx])
        if depth_mm.shape != (depth_h, depth_w):
            depth_mm = rr.cv2_resize_nn(depth_mm, depth_w, depth_h)
            conf = rr.cv2_resize_nn(conf, depth_w, depth_h)
        row = odo.iloc[idx]

        dets = process_frame(rgb, depth_mm, conf, row, fx_d, fy_d, cx_d, cy_d, us, vs,
                              min_confidence, floor_y, ceiling_y, polygon,
                              clip_model, clip_preprocess, text_features, classes,
                              conf_threshold)
        if dets:
            for d in dets:
                d["frame_index"] = idx
            all_detections.extend(dets)
            frames_with_damage.append({
                "frame_index": idx,
                "classes_found": sorted(set(d["predicted_class"] for d in dets)),
                "detection_count": len(dets),
            })
            print(f"      frame {idx}: {len(dets)} candidate(s) -> "
                  f"{[d['predicted_class'] for d in dets]}")

    print(f"[4/4] Writing outputs ({len(all_detections)} detections across "
          f"{len(frames_with_damage)}/{len(frame_indices)} frames)...")
    with open(os.path.join(out_dir, "damage_detections.json"), "w") as f:
        json.dump(all_detections, f, indent=2)
    with open(os.path.join(out_dir, "frames_with_damage.json"), "w") as f:
        json.dump({
            "scan_dir": os.path.abspath(scan_dir),
            "frames_scanned": len(frame_indices),
            "frames_with_damage": frames_with_damage,
            "damage_classes_considered": DAMAGE_CLASSES,
            "confidence_note": (
                "Placeholder damage-class vocabulary, not matched to real staged "
                "damage yet. No dedup across frames -- the same real damage seen "
                "in multiple frames appears as multiple entries here. CLIP "
                "classification (open_clip, ViT-B-32-quickgelu, openai weights) "
                "is the only filter on classical color-anomaly region proposals; "
                "not validated against ground truth."
            ),
        }, f, indent=2)

    render_damage_plan(polygon, all_detections,
                        os.path.join(out_dir, "damage_plan.png"),
                        os.path.basename(os.path.normpath(scan_dir)))
    return all_detections, frames_with_damage


def render_damage_plan(polygon: np.ndarray, detections: list, out_path: str, room_name: str):
    fig, ax = plt.subplots(figsize=(8, 8))
    patch = MplPolygon(polygon, closed=True, fill=True, alpha=0.15, edgecolor="black",
                        facecolor="tab:blue", linewidth=2)
    ax.add_patch(patch)

    class_colors = {}
    cmap = plt.get_cmap("tab10")
    located = [d for d in detections if d.get("world_xyz") is not None]
    for d in located:
        cls = d["predicted_class"]
        if cls not in class_colors:
            class_colors[cls] = cmap(len(class_colors) % 10)
        x, _, z = d["world_xyz"]
        ax.scatter([x], [z], c=[class_colors[cls]], s=60, marker="x", zorder=5)

    handles = [plt.Line2D([0], [0], marker="x", color=c, linestyle="", label=cls)
               for cls, c in class_colors.items()]
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=8)

    margin = 0.5
    ax.set_xlim(polygon[:, 0].min() - margin, polygon[:, 0].max() + margin)
    ax.set_ylim(polygon[:, 1].min() - margin, polygon[:, 1].max() + margin)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    n_unlocated = len(detections) - len(located)
    title = f"{room_name} -- damage locations ({len(located)} shown"
    title += f", {n_unlocated} unlocated: no depth at centroid)" if n_unlocated else ")"
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--every-n", type=int, default=15,
                     help="Frame stride for damage scanning (independent of the "
                          "geometry-fusion stride, which stays at reconstruct_room.py's default).")
    ap.add_argument("--conf-threshold", type=float, default=0.5)
    ap.add_argument("--min-confidence", type=int, default=2)
    args = ap.parse_args()
    run(args.scan_dir, args.out_dir, every_n=args.every_n,
        conf_threshold=args.conf_threshold, min_confidence=args.min_confidence)


if __name__ == "__main__":
    main()
