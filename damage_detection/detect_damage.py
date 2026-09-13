"""
damage_detection/detect_damage.py
----------------------------------
Phase 5, second increment: per-frame region proposal + a multi-view
consistency prefilter, THEN pretrained zero-shot classification. Metric
extent, surface (wall-index) assignment, concealed-damage rules, and scope
line items are still not built -- see PHASE_PLAN.md Phase 5 for the full
planned pipeline.

Three passes over the sampled frames:
  1. propose_all_candidates: for every frame, back-project depth to world
     points (reusing reconstruct_room.backproject_frame), mask to pixels
     near the room's known floor/wall/ceiling planes, and find color-
     anomalous regions against a blurred local background (classical CV,
     no model). Each candidate's world-space centroid is computed here --
     classification is deliberately NOT run yet.
  2. cluster_and_filter: DBSCAN-clusters all candidates' world centroids
     across every sampled frame, then keeps only clusters seen from
     >= --min-views DISTINCT frames. This is a real, load-bearing filter
     for one specific failure mode -- a one-off artifact (motion blur,
     a stray reflection, a sensor glitch) that only trips the detector in
     a single frame gets dropped. It is explicitly NOT a fix for the
     other failure mode found during manual crop review: floor-wall
     corner shadows and floor plank/grout seams are real, fixed, repeated
     geometric features, so they get seen from just as many distinct
     frames as genuine damage would and survive this filter too. See
     PHASE_PLAN.md's Phase 5 note on the crop-review finding -- fixing
     that specifically needs geometric-edge exclusion or illumination
     normalization, not multi-view consistency, and is not done here.
  3. Only surviving (multi-view-consistent) candidates get classified with
     CLIP (open_clip, ViT-B-32-quickgelu/openai weights -- disclosed
     pretrained-model use, no fine-tuning). Classifying only survivors
     also cuts CLIP calls roughly in proportion to how much the prefilter
     rejects, which matters since it's the slowest step per candidate.
     Surviving clusters are classified per-member (one CLIP call per view)
     and merged into one detection per cluster via majority vote on class
     + mean confidence over agreeing members -- this is real dedup, a
     piece of the previously-deferred 3D-clustering step pulled forward
     because clustering was already needed for the view-count filter.

Output (in out_dir):
  - damage_detections.json: one entry per surviving, classified cluster
    (world centroid, majority class, mean confidence, view count, member
    frame indices+bboxes, surface type).
  - frames_with_damage.json: frame-level summary derived from the merged
    clusters, for a quick "which frames have something worth a closer
    look" answer.
  - damage_plan.png: room footprint polygon with each surviving cluster's
    world (x, z) centroid marked and labeled by class.

Known limitations, honestly not fixed here (see also PHASE_PLAN.md):
  - DAMAGE_CLASSES below is a generic placeholder set, not matched to
    whatever ends up staged in the Phase 6 damage room.
  - No ground truth exists yet to validate accuracy against.
  - Corner-shadow / grout-seam false positives (found via manual crop
    review on an earlier run) are NOT fixed by this increment -- see the
    module docstring above and PHASE_PLAN.md.
  - Candidates with no recoverable world position (no depth at their
    pixel location) are dropped entirely during clustering, since view-
    count consistency can't be checked without a location -- they never
    reach classification, even if they'd have been real damage.

Usage:
    python damage_detection/detect_damage.py <scan_dir> <out_dir> \\
        [--every-n 15] [--conf-threshold 0.5] [--min-views 2] [--cluster-eps 0.1]
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
from sklearn.cluster import DBSCAN

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


def propose_frame_candidates(rgb: np.ndarray, depth_mm: np.ndarray, conf: np.ndarray, row,
                              fx_d, fy_d, cx_d, cy_d, us, vs, min_confidence,
                              floor_y, ceiling_y, polygon):
    """
    Pass 1 for one frame: region proposal + world-centroid lookup, no
    classification. Returns a list of raw candidate dicts (no
    predicted_class/confidence yet).
    """
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
    candidates = []
    for (x, y, w, h) in boxes:
        cx_rgb, cy_rgb = x + w / 2.0, y + h / 2.0
        dr, dc = int(round(cy_rgb * scale_y)), int(round(cx_rgb * scale_x))
        win = world_img[max(0, dr - 2):dr + 3, max(0, dc - 2):dc + 3].reshape(-1, 3)
        win = win[~np.isnan(win).any(axis=1)]
        world_xyz = win.mean(axis=0).tolist() if len(win) else None
        surface_type = labels[on_surface][
            np.argmin(np.abs(ys[on_surface] - dr) + np.abs(xs[on_surface] - dc))
        ] if on_surface.any() else "unknown"

        candidates.append({
            "bbox_px": [int(x), int(y), int(w), int(h)],
            "surface_type": str(surface_type),
            "world_xyz": world_xyz,
        })
    return candidates


def cluster_and_filter(candidates: list, eps: float = 0.1, min_views: int = 2):
    """
    DBSCAN-clusters candidates' world centroids (min_samples=1, so every
    candidate lands in some cluster) and keeps only clusters backed by
    >= min_views DISTINCT frames -- not just >= min_views candidates,
    since several candidates from the SAME frame shouldn't count as
    "multiple views." Candidates with no world position are dropped here
    (can't check consistency without a location) -- counted and reported,
    not silently lost.

    Returns (surviving_candidates, n_dropped_no_location, n_dropped_single_view).
    Surviving candidates get two new fields: cluster_id, cluster_view_count.
    """
    located = [c for c in candidates if c["world_xyz"] is not None]
    n_dropped_no_location = len(candidates) - len(located)
    if not located:
        return [], n_dropped_no_location, 0

    xyz = np.array([c["world_xyz"] for c in located])
    labels = DBSCAN(eps=eps, min_samples=1).fit_predict(xyz)

    clusters = {}
    for lab, c, frame_idx in zip(labels, located, [c["frame_index"] for c in located]):
        clusters.setdefault(int(lab), []).append(c)

    surviving = []
    n_dropped_single_view = 0
    for lab, members in clusters.items():
        distinct_frames = sorted(set(m["frame_index"] for m in members))
        if len(distinct_frames) >= min_views:
            for m in members:
                m["cluster_id"] = lab
                m["cluster_view_count"] = len(distinct_frames)
            surviving.extend(members)
        else:
            n_dropped_single_view += len(members)
    return surviving, n_dropped_no_location, n_dropped_single_view


def classify_clusters(surviving_candidates: list, video_path: str,
                       clip_model, clip_preprocess, text_features, classes,
                       conf_threshold: float):
    """
    Pass 3: classifies every surviving candidate (one CLIP call per view)
    by re-reading only the frames actually needed, then merges each
    cluster into one detection via majority vote on class + mean
    confidence over members that agree with the majority. A cluster
    survives only if its majority class isn't the negative class and the
    mean agreeing confidence clears conf_threshold.
    """
    by_frame = {}
    for c in surviving_candidates:
        by_frame.setdefault(c["frame_index"], []).append(c)

    for idx, rgb in read_rgb_frames_sequential(video_path, set(by_frame.keys())):
        for c in by_frame[idx]:
            x, y, w, h = c["bbox_px"]
            pad = int(0.1 * max(w, h))
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(rgb.shape[1], x + w + pad), min(rgb.shape[0], y + h + pad)
            crop = rgb[y0:y1, x0:x1]
            if crop.size == 0:
                c["predicted_class"], c["confidence"] = NEGATIVE_CLASS, 0.0
                continue
            c["predicted_class"], c["confidence"] = classify_crop(
                clip_model, clip_preprocess, text_features, classes, crop)

    by_cluster = {}
    for c in surviving_candidates:
        by_cluster.setdefault(c["cluster_id"], []).append(c)

    merged = []
    for cluster_id, members in by_cluster.items():
        class_votes = {}
        for m in members:
            class_votes.setdefault(m["predicted_class"], []).append(m["confidence"])
        majority_class = max(class_votes, key=lambda k: len(class_votes[k]))
        if majority_class == NEGATIVE_CLASS:
            continue
        mean_conf = float(np.mean(class_votes[majority_class]))
        if mean_conf < conf_threshold:
            continue
        xyz = np.mean([m["world_xyz"] for m in members], axis=0).tolist()
        surface_types = [m["surface_type"] for m in members]
        merged.append({
            "cluster_id": cluster_id,
            "predicted_class": majority_class,
            "confidence": round(mean_conf, 3),
            "view_count": members[0]["cluster_view_count"],
            "surface_type": max(set(surface_types), key=surface_types.count),
            "world_xyz": xyz,
            "members": [{"frame_index": m["frame_index"], "bbox_px": m["bbox_px"],
                         "predicted_class": m["predicted_class"],
                         "confidence": round(m["confidence"], 3)} for m in members],
        })
    return merged


def read_rgb_frames_sequential(video_path: str, wanted_indices: set):
    if not wanted_indices:
        return
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
        min_confidence: int = 2, min_views: int = 2, cluster_eps: float = 0.1):
    os.makedirs(out_dir, exist_ok=True)

    print("[1/5] Computing room geometry (floor/ceiling/footprint)...")
    pcd = rr.build_fused_point_cloud(scan_dir, min_confidence=min_confidence)
    floor_y, ceiling_y, height, height_conf = rr.detect_floor_and_ceiling(pcd)
    polygon = rr.extract_wall_footprint(pcd, floor_y, ceiling_y)
    print(f"      floor_y={floor_y:.3f} ceiling_y={ceiling_y:.3f} height={height:.3f}m")

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
    video_path = os.path.join(scan_dir, "rgb.mp4")

    print(f"[2/5] Pass 1/3: proposing candidate regions across {len(frame_indices)} frames "
          f"(no classification yet)...")
    raw_candidates = []
    for idx, rgb in read_rgb_frames_sequential(video_path, set(frame_indices)):
        if rgb.shape[:2] != (rgb_h, rgb_w):
            rgb = cv2.resize(rgb, (rgb_w, rgb_h))
        depth_mm = load_depth_mm(depth_frames[idx])
        conf = load_confidence(conf_frames[idx])
        if depth_mm.shape != (depth_h, depth_w):
            depth_mm = rr.cv2_resize_nn(depth_mm, depth_w, depth_h)
            conf = rr.cv2_resize_nn(conf, depth_w, depth_h)
        row = odo.iloc[idx]

        cands = propose_frame_candidates(rgb, depth_mm, conf, row, fx_d, fy_d, cx_d, cy_d,
                                          us, vs, min_confidence, floor_y, ceiling_y, polygon)
        for c in cands:
            c["frame_index"] = idx
        raw_candidates.extend(cands)
    print(f"      {len(raw_candidates)} raw candidates before any filtering")

    print(f"[3/5] Pass 2/3: clustering in 3D, requiring >= {min_views} distinct-frame "
          f"views (eps={cluster_eps}m)...")
    surviving, n_no_loc, n_single_view = cluster_and_filter(
        raw_candidates, eps=cluster_eps, min_views=min_views)
    print(f"      {len(surviving)} candidates survive the view-count filter "
          f"({n_no_loc} dropped: no depth at centroid, "
          f"{n_single_view} dropped: single-view only)")

    print(f"[4/5] Pass 3/3: classifying {len(surviving)} surviving candidates with CLIP "
          f"(open_clip, disclosed pretrained use)...")
    clip_model, clip_preprocess, text_features, classes = load_clip()
    detections = classify_clusters(surviving, video_path, clip_model, clip_preprocess,
                                    text_features, classes, conf_threshold)
    for d in detections:
        print(f"      cluster {d['cluster_id']} ({d['view_count']} views, "
              f"{d['surface_type']}): {d['predicted_class']} (conf {d['confidence']})")

    frames_with_damage = []
    for d in detections:
        for m in d["members"]:
            frames_with_damage.append({
                "frame_index": m["frame_index"],
                "cluster_id": d["cluster_id"],
                "predicted_class": d["predicted_class"],
            })
    frames_with_damage.sort(key=lambda x: x["frame_index"])

    print(f"[5/5] Writing outputs ({len(detections)} merged detections, "
          f"{len(frames_with_damage)} frame sightings)...")
    with open(os.path.join(out_dir, "damage_detections.json"), "w") as f:
        json.dump(detections, f, indent=2)
    with open(os.path.join(out_dir, "frames_with_damage.json"), "w") as f:
        json.dump({
            "scan_dir": os.path.abspath(scan_dir),
            "frames_scanned": len(frame_indices),
            "raw_candidates_before_filtering": len(raw_candidates),
            "dropped_no_location": n_no_loc,
            "dropped_single_view": n_single_view,
            "surviving_before_classification": len(surviving),
            "merged_detections": len(detections),
            "frame_sightings": frames_with_damage,
            "damage_classes_considered": DAMAGE_CLASSES,
            "confidence_note": (
                "Placeholder damage-class vocabulary, not matched to real staged "
                "damage yet. Multi-view consistency filter (>= min_views distinct "
                "frames at the same 3D location) removes one-off/transient false "
                "positives but does NOT remove fixed geometric false positives "
                "(floor-wall corner shadows, floor plank/grout seams) -- those are "
                "real, repeated features and survive this filter too. See "
                "PHASE_PLAN.md Phase 5 for the crop-review finding this doesn't fix."
            ),
        }, f, indent=2)

    render_damage_plan(polygon, detections, os.path.join(out_dir, "damage_plan.png"),
                        os.path.basename(os.path.normpath(scan_dir)))
    return detections, frames_with_damage


def render_damage_plan(polygon: np.ndarray, detections: list, out_path: str, room_name: str):
    fig, ax = plt.subplots(figsize=(8, 8))
    patch = MplPolygon(polygon, closed=True, fill=True, alpha=0.15, edgecolor="black",
                        facecolor="tab:blue", linewidth=2)
    ax.add_patch(patch)

    class_colors = {}
    cmap = plt.get_cmap("tab10")
    for d in detections:
        cls = d["predicted_class"]
        if cls not in class_colors:
            class_colors[cls] = cmap(len(class_colors) % 10)
        x, _, z = d["world_xyz"]
        ax.scatter([x], [z], c=[class_colors[cls]], s=80, marker="x", zorder=5)
        ax.annotate(f"{d['view_count']}v", (x, z), fontsize=6, xytext=(3, 3),
                    textcoords="offset points")

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
    ax.set_title(f"{room_name} -- damage locations ({len(detections)} multi-view-"
                 f"consistent clusters, view count annotated)")
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
    ap.add_argument("--min-views", type=int, default=2,
                     help="A candidate must be seen at the same 3D location from this many "
                          "distinct frames to survive the prefilter before classification.")
    ap.add_argument("--cluster-eps", type=float, default=0.1,
                     help="DBSCAN radius (meters) for grouping candidates into the same "
                          "real-world location.")
    args = ap.parse_args()
    run(args.scan_dir, args.out_dir, every_n=args.every_n,
        conf_threshold=args.conf_threshold, min_confidence=args.min_confidence,
        min_views=args.min_views, cluster_eps=args.cluster_eps)


if __name__ == "__main__":
    main()
