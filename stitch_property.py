
"""
stitch_property.py
-------------------
LiDAR-tier, multi-room pipeline (Phase 2): segments a single continuous
walkthrough capture into per-room point clouds, runs the same single-room
contract (reconstruct_room.py) on each, and stitches them into one
whole-property plan with adjacency.

Key simplification specific to this tier: because the whole walkthrough was
captured in ONE continuous VIO-tracked session, every room's points already
share one coordinate frame -- there is no independent room-to-room
registration problem to solve here (that problem is real for the photo/video
tiers, where each room folder/clip has no shared pose and Phase 3/4 will need
actual registration). What this phase still owns: figuring out which frames
belong to which room in the first place, and correcting the accumulated VIO
drift across a long walk before trusting the stitched shape.

Room segmentation (see segment_rooms): a room's walls get swept from many
angles over time, so its trajectory points are spatially *revisited* --
close in space to other trajectory points from a much different time.
Corridors/doorways are walked through once, briefly, and are not revisited.
Splitting on that distinction, then clustering what's left spatially, turns
one continuous path into disjoint per-room point groups without needing any
semantic room detector.

Drift: your report/gate requires you to not use poses "as-is" and to show an
ablation. This ships the simplest defensible correction (linear loop-closure
redistribution, translation only) and produces both variants side by side --
see apply_loop_closure_correction's docstring for exactly what it does and
does not fix.

Usage:
    python stitch_property.py <scan_dir> <out_dir> [--min-confidence 2] [--bins 180]
"""

import os
import json
import argparse
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

import reconstruct_room as rr
from scan_io import load_odometry


def segment_rooms(odo, radius: float = 0.5, time_gap: int = 60, min_recurrence: int = 5,
                   dbscan_eps: float = 0.3, dbscan_min_samples: int = 10,
                   min_room_frames: int = 100):
    """
    Splits a multi-room walkthrough's trajectory into per-room frame-index
    groups. See module docstring for the recurrence + spatial-clustering
    rationale. Small leftover clusters (<min_room_frames -- a brief lingering
    spot in a doorway/junction, not a real room) are merged into whichever
    retained room cluster is spatially closest, rather than reported as
    standalone rooms.

    Known limitation: a hub/junction point where 3+ spaces meet at close
    range can under-segment (the shared junction area gets swept from all
    directions too, so it reads as "revisited" like a room would) -- this
    shows up as fewer, larger room clusters than the true room count rather
    than a wrong shape, and is left as a documented gap pending ground truth
    (Phase 6/7) to actually tell room count right from wrong.

    Returns (labels, room_ids): labels is an int array the same length as
    odo (-1 = corridor/unassigned), room_ids is the sorted list of final,
    contiguous room ids appearing in labels.
    """
    xz = odo[["x", "z"]].values
    n = len(xz)
    tree = cKDTree(xz)
    recurrence = np.array([
        sum(1 for j in tree.query_ball_point(xz[i], radius) if abs(j - i) > time_gap)
        for i in range(n)
    ])
    room_mask = recurrence >= min_recurrence
    room_idx = np.where(room_mask)[0]

    labels_full = -np.ones(n, dtype=int)
    if len(room_idx) < dbscan_min_samples:
        return labels_full, []

    db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(xz[room_idx])
    sub_labels = db.labels_

    unique = [l for l in set(sub_labels) if l != -1]
    if not unique:
        return labels_full, []
    sizes = {l: int((sub_labels == l).sum()) for l in unique}
    big = [l for l in unique if sizes[l] >= min_room_frames] or unique
    small = [l for l in unique if l not in big]

    centroids = {l: xz[room_idx][sub_labels == l].mean(axis=0) for l in unique}
    remap = {l: l for l in big}
    for l in small:
        nearest = min(big, key=lambda b: np.linalg.norm(centroids[l] - centroids[b]))
        remap[l] = nearest

    final_ids = sorted(set(remap.values()))
    id_map = {old: new for new, old in enumerate(final_ids)}
    for idx, lab in zip(room_idx, sub_labels):
        if lab != -1:
            labels_full[idx] = id_map[remap[lab]]

    return labels_full, list(range(len(final_ids)))


def compute_adjacency(labels):
    """
    Reads adjacency edges directly off the timeline: a maximal run of
    corridor frames (-1) connects whichever real room precedes it to
    whichever real room follows it in time -- valid because every room here
    shares one coordinate frame already (see module docstring). Returns
    {(room_a, room_b): connector_frame_index}, room_a < room_b, one entry
    per distinct room pair (first occurrence's corridor midpoint wins).
    """
    n = len(labels)
    edges = {}
    i = 0
    prev_room = None
    while i < n:
        if labels[i] != -1:
            prev_room = labels[i]
            i += 1
            continue
        start = i
        while i < n and labels[i] == -1:
            i += 1
        next_room = labels[i] if i < n else None
        if prev_room is not None and next_room is not None and prev_room != next_room:
            key = tuple(sorted((int(prev_room), int(next_room))))
            edges.setdefault(key, (start + i) // 2)
    return edges


def apply_loop_closure_correction(odo):
    """
    Linear ("rubber-band") loop-closure correction: measures the
    translational gap between the trajectory's first and last pose (should
    be ~0 for a walk that starts and ends at the same doorway) and
    redistributes it linearly across the whole sequence so the corrected
    trajectory closes exactly.

    Explicitly NOT full pose-graph SLAM: corrects accumulated TRANSLATION
    drift only (rotation untouched), and assumes drift accumulates roughly
    uniformly over time, which is false if e.g. one long straight corridor
    drifts more than a slow room sweep. Documented as the minimum defensible
    alternative to "poses used as-is" (an automatic fail per the case
    study), not a claim that drift is actually solved -- see PHASE_PLAN.md.
    """
    corrected = odo.copy()
    n = len(odo)
    start = odo.iloc[0][["x", "y", "z"]].values.astype(float)
    end = odo.iloc[-1][["x", "y", "z"]].values.astype(float)
    gap = start - end
    t = np.linspace(0.0, 1.0, n)[:, None]
    correction = t * gap[None, :]
    corrected[["x", "y", "z"]] = odo[["x", "y", "z"]].values + correction
    return corrected


def reconstruct_room_from_frames(scan_dir, room_name, frame_indices, odo,
                                  min_confidence=2, bins=180, every_n=5):
    """Runs the same single-room contract as reconstruct_room.py's main(),
    against an explicit frame subset + externally supplied poses."""
    pcd = rr.build_fused_point_cloud(scan_dir, every_n=every_n, min_confidence=min_confidence,
                                      frame_indices=frame_indices, odo=odo)
    floor_y, ceiling_y, height, height_conf = rr.detect_floor_and_ceiling(pcd)
    polygon = rr.extract_wall_footprint(pcd, floor_y, ceiling_y, num_bins=bins)
    lengths, area = rr.polygon_metrics(polygon)
    perimeter = float(sum(lengths))
    wall_ci, area_ci = rr.compute_confidence_intervals(polygon, perimeter)
    height_ci = rr.ceiling_height_ci_m(height_conf)
    openings = rr.detect_openings(pcd, polygon, floor_y, ceiling_y)
    return {
        "room_name": room_name,
        "tier": "lidar",
        "n_frames": len(frame_indices),
        "ceiling_height_m": round(height, 3),
        "ceiling_height_confidence": round(height_conf, 3),
        "ceiling_height_ci_m": height_ci,
        "floor_area_m2": round(area, 3),
        "floor_area_ci_m2": area_ci,
        "wall_count": len(lengths),
        "wall_lengths_m": [round(l, 3) for l in lengths],
        "wall_length_ci_m": wall_ci,
        "openings": openings,
        "opening_count": len(openings),
        "footprint_polygon_xz": polygon.tolist(),
    }


def render_property_plan(rooms, room_polys, adjacency, out_path, property_name):
    fig, ax = plt.subplots(figsize=(9, 9))
    cmap = plt.get_cmap("tab10")
    for i, r in enumerate(rooms):
        rid = int(r["room_name"].split("_")[1])
        poly = room_polys[rid]
        patch = MplPolygon(poly, closed=True, fill=True, alpha=0.25, edgecolor="black",
                            facecolor=cmap(i % 10), linewidth=2)
        ax.add_patch(patch)
        centroid = poly.mean(axis=0)
        ax.annotate(f'{r["room_name"]}\n{r["floor_area_m2"]:.1f}m2', centroid,
                    ha="center", fontsize=8)
        for op in r["openings"]:
            x, z = op["position_xz"]
            ax.scatter([x], [z], c="blue", s=25, marker="s", zorder=6)
    for edge in adjacency:
        x, z = edge["connector_xz"]
        ax.scatter([x], [z], c="red", s=70, marker="*", zorder=7)
        ax.annotate("-".join(edge["rooms"]), (x, z), fontsize=6, color="red")
    all_pts = np.concatenate(list(room_polys.values()), axis=0)
    margin = 0.5
    ax.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
    ax.set_ylim(all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(f"{property_name} -- stitched property plan")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def stitch(scan_dir, out_dir, apply_drift_correction=True, min_confidence=2, bins=180, every_n=5):
    os.makedirs(out_dir, exist_ok=True)
    odo_raw = load_odometry(scan_dir)
    labels, room_ids = segment_rooms(odo_raw)
    if not room_ids:
        raise RuntimeError("No room clusters found -- check segment_rooms params, or "
                            "this capture may not have distinct dwelled spaces")

    odo = apply_loop_closure_correction(odo_raw) if apply_drift_correction else odo_raw

    rooms = []
    room_polys = {}
    for rid in room_ids:
        frame_indices = np.where(labels == rid)[0].tolist()
        room_name = f"room_{rid}"
        try:
            room_json = reconstruct_room_from_frames(scan_dir, room_name, frame_indices, odo,
                                                      min_confidence=min_confidence, bins=bins,
                                                      every_n=every_n)
        except Exception as e:
            print(f"      WARNING: {room_name} ({len(frame_indices)} frames) "
                  f"failed to reconstruct: {e}")
            continue
        rooms.append(room_json)
        room_polys[rid] = np.array(room_json["footprint_polygon_xz"])
        room_dir = os.path.join(out_dir, room_name)
        os.makedirs(room_dir, exist_ok=True)
        with open(os.path.join(room_dir, "room.json"), "w") as f:
            json.dump(room_json, f, indent=2)

    if not rooms:
        raise RuntimeError("Every segmented room failed to reconstruct")

    edges = compute_adjacency(labels)
    xz = odo[["x", "z"]].values
    adjacency = [
        {"rooms": [f"room_{a}", f"room_{b}"], "connector_xz": xz[mid_idx].tolist()}
        for (a, b), mid_idx in edges.items() if a in room_polys and b in room_polys
    ]

    total_area = float(sum(r["floor_area_m2"] for r in rooms))
    all_pts = np.concatenate(list(room_polys.values()), axis=0)
    bbox = {"min_xz": all_pts.min(axis=0).tolist(), "max_xz": all_pts.max(axis=0).tolist()}

    property_json = {
        "property_name": os.path.basename(os.path.normpath(scan_dir)),
        "source_scan": os.path.abspath(scan_dir),
        "drift_correction": "loop_closure_linear" if apply_drift_correction else "none",
        "rooms": [r["room_name"] for r in rooms],
        "adjacency": adjacency,
        "total_floor_area_m2": round(total_area, 3),
        "bounding_box_xz": bbox,
        "confidence_note": (
            "Room segmentation is trajectory-based (recurrence + spatial "
            "clustering), not semantic -- room IDs are arbitrary, not real "
            "room names, and a hub/junction where 3+ spaces meet can "
            "under-segment (see segment_rooms docstring). All rooms share "
            "one continuous VIO coordinate frame (one walkthrough), so no "
            "independent room-to-room registration was needed here -- that "
            "will NOT be true for independently captured photo/video-tier "
            "rooms (Phase 3/4), which need real registration. Drift "
            "correction, if enabled, is a linear translation-only "
            "loop-closure redistribution, not full pose-graph SLAM."
        ),
    }
    with open(os.path.join(out_dir, "property.json"), "w") as f:
        json.dump(property_json, f, indent=2)

    render_property_plan(rooms, room_polys, adjacency,
                          os.path.join(out_dir, "property_plan.png"),
                          property_json["property_name"])
    return property_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--min-confidence", type=int, default=2)
    ap.add_argument("--bins", type=int, default=180)
    ap.add_argument("--every-n", type=int, default=5,
                     help="Frame stride within each room's assigned frames (matches "
                          "reconstruct_room.py's default) -- a dwelled-on room can collect "
                          "thousands of frames, so this bounds fusion cost/memory the same "
                          "way the single-room contract does.")
    args = ap.parse_args()

    print("[1/2] Stitching WITH drift correction...")
    with_corr = stitch(args.scan_dir, os.path.join(args.out_dir, "with_drift_correction"),
                        apply_drift_correction=True, min_confidence=args.min_confidence,
                        bins=args.bins, every_n=args.every_n)
    print("[2/2] Stitching WITHOUT drift correction (ablation)...")
    without_corr = stitch(args.scan_dir, os.path.join(args.out_dir, "without_drift_correction"),
                           apply_drift_correction=False, min_confidence=args.min_confidence,
                           bins=args.bins, every_n=args.every_n)

    ablation = {
        "with_drift_correction": {
            "total_floor_area_m2": with_corr["total_floor_area_m2"],
            "bounding_box_xz": with_corr["bounding_box_xz"],
            "rooms": with_corr["rooms"],
        },
        "without_drift_correction": {
            "total_floor_area_m2": without_corr["total_floor_area_m2"],
            "bounding_box_xz": without_corr["bounding_box_xz"],
            "rooms": without_corr["rooms"],
        },
    }
    with open(os.path.join(args.out_dir, "drift_ablation.json"), "w") as f:
        json.dump(ablation, f, indent=2)

    print(f"\nDone. Rooms found: {len(with_corr['rooms'])}")
    print(f"With correction:    total area={with_corr['total_floor_area_m2']} m^2, "
          f"bbox={with_corr['bounding_box_xz']}")
    print(f"Without correction: total area={without_corr['total_floor_area_m2']} m^2, "
          f"bbox={without_corr['bounding_box_xz']}")
    print(f"Outputs in {args.out_dir}/: with_drift_correction/, without_drift_correction/, "
          f"drift_ablation.json")


if __name__ == "__main__":
    main()
