"""
stitch_property_photo.py
--------------------------
Photo-tier, multi-room pipeline (Phase 4): stitches independently-captured
per-room photo folders into one whole-property plan.

This is a fundamentally different problem from the LiDAR tier's
stitch_property.py. There, every room's points share one continuous VIO
coordinate frame because the whole property was walked in one session --
placing rooms relative to each other is free. Here, per the case study's
own spec ("at the photo tier it arrives as per-room photo folders"), each
room folder is an INDEPENDENT capture with no shared pose, no shared
timestamp, and often no shared visual content with any other room's photos
at all. There is no information in the photos alone to recover how room A
sits relative to room B -- that is not a hard registration problem to be
solved better, it is information that was never captured.

Design decision: adjacency (which rooms connect) and shared-doorway
identity are taken as EXPLICIT input (adjacency.json in the property
folder), not inferred by vision. This mirrors how the capture protocol
already asks a non-engineer to name/order room folders -- asking them to
also note "the kitchen door leads to the hallway" during capture is a
trivial addition to a one-page protocol and is honest about where the
tier's real information boundary is, instead of quietly pretending a CV
model can recover 6DOF relative pose from zero shared observations.

Given a declared adjacency edge, room-to-room PLACEMENT still needs a rule
-- see align_child_to_parent's docstring: it snaps the child room's
declared opening to coincide with the parent's, oriented so the two rooms
fall on opposite sides of that shared wall (no overlap by construction, for
normal room shapes). This is a heuristic, not a measurement: true relative
ROTATION between independently-captured rooms is exactly the information
gap described above, so only the connecting wall is placed correctly, not
necessarily the child room's true compass orientation. Round-1 scope
addition, honestly documented -- see property.json's confidence_note.

Usage:
    python stitch_property_photo.py <property_dir> <out_dir> [--stride 4] [--bins 60]

<property_dir> layout:
    property_dir/
        adjacency.json     # [{"rooms": ["room_a","room_b"],
                            #   "opening_index": {"room_a":0,"room_b":0}}, ...]
        room_a/*.jpg
        room_b/*.jpg
        ...
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

import reconstruct_room_photo as rp
import reconstruct_room as rr


def load_adjacency(property_dir: str):
    path = os.path.join(property_dir, "adjacency.json")
    if not os.path.exists(path):
        print(f"      NOTE: no adjacency.json in {property_dir} -- rooms will "
              f"be reconstructed independently with no stitching (each room's "
              f"own local frame, placed at the origin, guaranteed to overlap "
              f"in the rendered plan). Add adjacency.json to stitch properly.")
        return []
    with open(path) as f:
        return json.load(f)


def list_room_dirs(property_dir: str):
    return sorted(
        d for d in os.listdir(property_dir)
        if os.path.isdir(os.path.join(property_dir, d)) and not d.startswith(".")
    )


def wall_segment(polygon: np.ndarray, wall_index: int):
    n = len(polygon)
    return polygon[wall_index % n], polygon[(wall_index + 1) % n]


def align_child_to_parent(parent_poly, parent_opening_wall_idx, child_poly, child_opening_wall_idx):
    """
    Returns (R 2x2, t 2, ) mapping child_poly's local (x,z) coordinates into
    the property frame, such that:
      - the child's declared opening edge coincides with the parent's
        declared opening edge (shared doorway placed at one location, not
        two)
      - the child room falls on the OPPOSITE side of that shared wall from
        the parent's own interior (both rooms' outward-facing normals at
        the doorway end up pointing at each other, not overlapping)

    This deliberately does NOT try to recover the child's true absolute
    orientation -- it only has enough information to place the shared wall
    correctly. A child room could in reality be rotated/mirrored along that
    wall in a way this can't distinguish from a single photo folder alone
    (e.g. a room that's a mirror image across the doorway looks identical
    to this heuristic) -- flagged in property.json's confidence_note, not
    silently assumed away.
    """
    p1, p2 = wall_segment(parent_poly, parent_opening_wall_idx)
    q1, q2 = wall_segment(child_poly, child_opening_wall_idx)

    parent_centroid = parent_poly.mean(axis=0)
    child_centroid = child_poly.mean(axis=0)

    p_mid = (p1 + p2) / 2.0
    q_mid = (q1 + q2) / 2.0

    p_dir = (p2 - p1) / max(np.linalg.norm(p2 - p1), 1e-9)
    q_dir = (q2 - q1) / max(np.linalg.norm(q2 - q1), 1e-9)

    p_normal = np.array([-p_dir[1], p_dir[0]])
    if np.dot(p_normal, p_mid - parent_centroid) < 0:
        p_normal = -p_normal  # ensure outward-facing (away from parent interior)

    q_normal = np.array([-q_dir[1], q_dir[0]])
    if np.dot(q_normal, q_mid - child_centroid) < 0:
        q_normal = -q_normal  # ensure outward-facing (away from child interior)

    # Rotate child so its outward normal becomes the exact opposite of the
    # parent's outward normal (the two rooms look at each other across the
    # shared wall).
    target_normal = -p_normal
    angle = np.arctan2(target_normal[1], target_normal[0]) - np.arctan2(q_normal[1], q_normal[0])
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])

    q_mid_rotated = R @ q_mid
    t = p_mid - q_mid_rotated
    return R, t


def reconstruct_room_dir(room_dir: str, room_name: str, stride: int, bins: int, device: str):
    """Runs the same single-room photo-tier contract as
    reconstruct_room_photo.py's main(), returning the room dict (not yet
    placed in property space) plus its local footprint polygon."""
    pcd, mean_inlier_ratio, n_used, n_total = rp.build_fused_point_cloud_photo(
        room_dir, stride=stride, device=device)
    floor_y, ceiling_y, height, height_confidence = rp.detect_floor_and_ceiling_photo(pcd)
    polygon = rr.extract_wall_footprint(pcd, floor_y, ceiling_y, num_bins=bins)
    lengths, area = rr.polygon_metrics(polygon)
    perimeter = float(sum(lengths))
    wall_ci, area_ci = rp.compute_confidence_intervals_photo(
        polygon, perimeter, area, mean_inlier_ratio, n_used)
    height_ci = rp.ceiling_height_ci_m_photo(height_confidence, mean_inlier_ratio)
    openings = rr.detect_openings(pcd, polygon, floor_y, ceiling_y)

    room_json = {
        "room_name": room_name,
        "tier": "photo",
        "ceiling_height_m": round(height, 3),
        "ceiling_height_confidence": round(height_confidence, 3),
        "ceiling_height_ci_m": height_ci,
        "floor_area_m2": round(area, 3),
        "floor_area_ci_m2": area_ci,
        "wall_count": len(lengths),
        "wall_lengths_m": [round(l, 3) for l in lengths],
        "wall_length_ci_m": wall_ci,
        "openings": openings,
        "opening_count": len(openings),
        "footprint_polygon_xz": polygon.tolist(),
        "n_photos_total": n_total,
        "n_photos_used": n_used,
        "mean_registration_inlier_ratio": round(mean_inlier_ratio, 3),
    }
    return room_json, polygon


def render_property_plan_photo(rooms_placed, out_path, property_name):
    fig, ax = plt.subplots(figsize=(9, 9))
    cmap = plt.get_cmap("tab10")
    for i, r in enumerate(rooms_placed):
        poly = r["placed_polygon"]
        patch = MplPolygon(poly, closed=True, fill=True, alpha=0.25, edgecolor="black",
                            facecolor=cmap(i % 10), linewidth=2)
        ax.add_patch(patch)
        centroid = poly.mean(axis=0)
        ax.annotate(f'{r["room_name"]}\n{r["floor_area_m2"]:.1f}m2', centroid,
                    ha="center", fontsize=8)
        for op in r["placed_openings"]:
            ax.scatter([op[0]], [op[1]], c="blue", s=40, marker="s", zorder=6)
    all_pts = np.concatenate([r["placed_polygon"] for r in rooms_placed], axis=0)
    margin = 0.5
    ax.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
    ax.set_ylim(all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(f"{property_name} -- stitched property plan (photo tier)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def stitch(property_dir: str, out_dir: str, stride: int = 4, bins: int = 60, device: str = None):
    os.makedirs(out_dir, exist_ok=True)
    device = rp._pick_device(device)
    room_names = list_room_dirs(property_dir)
    if len(room_names) < 2:
        raise RuntimeError(f"{property_dir}: found {len(room_names)} room folder(s), "
                            f"need at least 2 for a multi-room property")
    adjacency_decl = load_adjacency(property_dir)

    rooms = {}
    polygons_local = {}
    for name in room_names:
        print(f"[room {name}] reconstructing...")
        room_dir = os.path.join(property_dir, name)
        try:
            room_json, polygon = reconstruct_room_dir(room_dir, name, stride, bins, device)
        except Exception as e:
            print(f"      WARNING: {name} failed to reconstruct: {e}")
            continue
        rooms[name] = room_json
        polygons_local[name] = polygon
        room_out = os.path.join(out_dir, name)
        os.makedirs(room_out, exist_ok=True)
        rr.render_plan(polygon, os.path.join(room_out, "room_plan.png"), name,
                        openings=room_json["openings"])
        with open(os.path.join(room_out, "room.json"), "w") as f:
            json.dump(room_json, f, indent=2)

    if not rooms:
        raise RuntimeError("Every room failed to reconstruct")

    # Placement: BFS outward from an arbitrary root room, snapping each
    # child to its parent via a declared adjacency edge + opening. Rooms
    # with no path to the root (bad adjacency.json, or reconstruction
    # failed) are placed at an offset, non-overlapping fallback position
    # and flagged -- not silently dropped, not silently mis-stitched.
    edges = {}
    for e in adjacency_decl:
        a, b = e["rooms"]
        if a not in rooms or b not in rooms:
            print(f"      WARNING: adjacency edge {a}-{b} references a room "
                  f"that failed to reconstruct or doesn't exist -- skipping")
            continue
        oi = e.get("opening_index", {})
        edges.setdefault(a, []).append((b, oi.get(a, 0), oi.get(b, 0)))
        edges.setdefault(b, []).append((a, oi.get(b, 0), oi.get(a, 0)))

    root = next(iter(rooms))
    transforms = {root: (np.eye(2), np.zeros(2))}
    visited = {root}
    queue = [root]
    placement_notes = []
    while queue:
        cur = queue.pop(0)
        R_cur, t_cur = transforms[cur]
        for neighbor, cur_op_idx, nb_op_idx in edges.get(cur, []):
            if neighbor in visited:
                continue
            cur_openings = rooms[cur]["openings"]
            nb_openings = rooms[neighbor]["openings"]
            if cur_op_idx >= len(cur_openings) or nb_op_idx >= len(nb_openings):
                print(f"      WARNING: declared opening_index out of range for "
                      f"{cur}<->{neighbor} (has {len(cur_openings)}/{len(nb_openings)} "
                      f"detected openings) -- placing {neighbor} with a fallback offset")
                placement_notes.append(f"{neighbor}: fallback offset (bad opening_index)")
                offset_poly = polygons_local[cur] @ R_cur.T + t_cur
                fallback_t = np.array([offset_poly[:, 0].max() + 1.0, offset_poly[:, 1].mean()])
                transforms[neighbor] = (np.eye(2), fallback_t)
            else:
                cur_wall_idx = cur_openings[cur_op_idx]["wall_index"]
                nb_wall_idx = nb_openings[nb_op_idx]["wall_index"]
                cur_poly_property = polygons_local[cur] @ R_cur.T + t_cur
                R_rel, t_rel = align_child_to_parent(
                    cur_poly_property, cur_wall_idx, polygons_local[neighbor], nb_wall_idx)
                transforms[neighbor] = (R_rel, t_rel)
            visited.add(neighbor)
            queue.append(neighbor)

    # Anything never reached from the root (disconnected adjacency graph,
    # or no adjacency.json at all) gets laid out in a simple non-overlapping
    # row rather than left stacked at the origin.
    unplaced = [n for n in rooms if n not in transforms]
    if unplaced:
        placed_so_far = [polygons_local[n] @ transforms[n][0].T + transforms[n][1] for n in transforms]
        cursor_x = max(p[:, 0].max() for p in placed_so_far) + 1.0 if placed_so_far else 0.0
        for n in unplaced:
            poly = polygons_local[n]
            width = poly[:, 0].max() - poly[:, 0].min()
            t = np.array([cursor_x - poly[:, 0].min(), -poly[:, 1].mean()])
            transforms[n] = (np.eye(2), t)
            cursor_x += width + 1.0
            placement_notes.append(f"{n}: no adjacency path from root -- placed in fallback row")

    rooms_placed = []
    adjacency_out = []
    for name in rooms:
        R_n, t_n = transforms[name]
        placed_poly = polygons_local[name] @ R_n.T + t_n
        placed_openings = [
            (R_n @ np.array(op["position_xz"]) + t_n).tolist()
            for op in rooms[name]["openings"]
        ]
        rooms_placed.append({
            **rooms[name],
            "placed_polygon": placed_poly,
            "placed_openings": placed_openings,
            "transform_xz": [[R_n[0, 0], R_n[0, 1], t_n[0]], [R_n[1, 0], R_n[1, 1], t_n[1]]],
        })
    for e in adjacency_decl:
        a, b = e["rooms"]
        if a in rooms and b in rooms:
            adjacency_out.append({"rooms": [a, b], "opening_index": e.get("opening_index", {})})

    total_area = float(sum(r["floor_area_m2"] for r in rooms_placed))
    property_name = os.path.basename(os.path.normpath(property_dir))

    property_json = {
        "property_name": property_name,
        "source_scan": os.path.abspath(property_dir),
        "tier": "photo",
        "rooms": [
            {"room_name": r["room_name"], "room_json_path": f"{r['room_name']}/room.json",
             "transform_xz": r["transform_xz"]}
            for r in rooms_placed
        ],
        "adjacency": adjacency_out,
        "total_floor_area_m2": round(total_area, 3),
        "drift_correction": {
            "method": "opening_anchored_placement",
            "ablation_note": (
                "not applicable in the LiDAR-tier sense (no continuous trajectory, "
                "no accumulated VIO drift to correct) -- each room is an independent "
                "capture with its own metric error instead; see each room's own "
                "confidence_note."
            ),
        },
        "placement_notes": placement_notes,
        "confidence_note": (
            "Photo-tier stitching cannot recover true relative room position/"
            "orientation from independent, non-overlapping photo folders -- that "
            "information was never captured (see module docstring). Adjacency "
            "(which rooms connect) and which detected opening is the shared "
            "doorway are taken as EXPLICIT input from adjacency.json, not "
            "inferred by vision. Placement snaps each child room's declared "
            "opening to coincide with its parent's, oriented so the two rooms "
            "fall on opposite sides of that wall -- this places the shared wall "
            "correctly but does NOT verify the child room's absolute orientation "
            "beyond that one wall (e.g. a mirror-imaged room layout across the "
            "doorway is indistinguishable from the correct one on this "
            "evidence). Rooms with no adjacency.json entry, a missing/failed "
            "reconstruction, or a bad declared opening_index are placed in a "
            "non-overlapping fallback row instead of guessed at -- see "
            "placement_notes for which rooms (if any) hit this path."
        ),
    }
    with open(os.path.join(out_dir, "property.json"), "w") as f:
        json.dump(property_json, f, indent=2)

    render_property_plan_photo(rooms_placed, os.path.join(out_dir, "property_plan.png"), property_name)
    return property_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("property_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    result = stitch(args.property_dir, args.out_dir, stride=args.stride,
                     bins=args.bins, device=args.device)
    print(f"\nDone. Rooms: {result['rooms']}")
    print(f"Total floor area: {result['total_floor_area_m2']} m^2")
    if result["placement_notes"]:
        print(f"Placement fallbacks used: {result['placement_notes']}")
    print(f"Outputs in {args.out_dir}/: property.json, property_plan.png, <room>/room.json")


if __name__ == "__main__":
    main()
