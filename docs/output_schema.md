# Output schema

Two levels: **per-room** (implemented, `reconstruct_room.py`) and **per-property**
(stitched multi-room plan, Phase 2 — not implemented yet, documented here as the
target shape so per-room output doesn't have to be redesigned to fit it later).

## Per-room (`room.json`) — current, as emitted by `reconstruct_room.py`

```jsonc
{
  "room_name": "living_room",
  "source_scan": "/abs/path/to/raw/capture",
  "tier": "lidar",                    // "lidar" | "video" | "photo"
  "ceiling_height_m": 2.30,
  "ceiling_height_confidence": 1.0,   // 0-1; 0 = no ceiling plane found (fallback estimate)
  "ceiling_height_ci_m": 0.02,        // +/- width, widens as confidence drops
  "floor_area_m2": 21.26,
  "floor_area_ci_m2": 0.45,           // placeholder error model, see confidence_note
  "wall_count": 180,                  // = --bins; polygon vertex count, not literal wall count
  "wall_lengths_m": [...],
  "wall_length_ci_m": 0.042,
  "openings": [                       // doors/windows, gap-in-coverage heuristic
    {"wall_index": 12, "position_xz": [1.2, 0.4], "width_m": 0.86}
  ],
  "opening_count": 1,
  "footprint_polygon_xz": [[x, z], ...],  // room boundary, floor-plane (x,z) coords
  "confidence_note": "free-text caveats, see source for current wording"
}
```

Deliberately not yet present, tracked for later phases:
- Damage regions (class + metric extent) — Phase 5.
- Concealed-damage flags + firing rule — Phase 5.
- Scope line items keyed to surfaces — Phase 5.
- Per-wall openings currently report a single width/position; door vs. window
  classification is not attempted (Round-1 scope note, not a bug).

## Photo tier (`room.json`, `tier: "photo"`) — Phase 4, `reconstruct_room_photo.py`

Same fields as the LiDAR shape above, plus:

```jsonc
{
  "tier": "photo",
  "n_photos_total": 5,               // photos found in the room folder
  "n_photos_used": 3,                // photos that actually chained into the fused cloud
  "mean_registration_inlier_ratio": 0.62,  // across successfully chained pairs
  ...
}
```

`n_photos_used < n_photos_total` means registration broke somewhere in the
chain (see `reconstruct_room_photo.register_photo_chain`) — the room was
still reconstructed, just from fewer photos than were provided, and this
is reported rather than hidden. CIs on this tier scale with both the depth
model's assumed relative error and `mean_registration_inlier_ratio`, not a
fixed absolute figure like the LiDAR tier — see
`compute_confidence_intervals_photo`'s docstring.

## Property (stitched plan), photo tier — Phase 4, `stitch_property_photo.py`

Matches the per-property target shape above, with two photo-tier-specific
differences documented in `property.json`'s own `confidence_note` (not
duplicated here in full — see the module docstring in
`stitch_property_photo.py` for the reasoning):

- `rooms[].transform_xz` is recovered by snapping each room's declared
  shared opening to its neighbor's (an `adjacency.json` the capturer
  provides, not inferred by vision) — true relative room orientation
  beyond that one shared wall is not recoverable from independent photo
  folders with no shared pose, and isn't claimed to be.
- `drift_correction.method` is `"opening_anchored_placement"`, not one of
  the LiDAR tier's trajectory-based methods — there is no continuous pose
  to drift in the first place at this tier.

## Per-property (stitched plan) — target shape, Phase 2

```jsonc
{
  "property_name": "...",
  "rooms": [
    {"room_name": "living_room", "room_json_path": "out/.../room.json",
     "transform_xz": [[cos, -sin, tx], [sin, cos, tz]]}   // room -> property frame
  ],
  "adjacency": [["living_room", "hallway", {"opening_wall_index": 12}]],
  "footprint_area_m2": 0.0,
  "drift_correction": {"method": "loop_closure|pose_graph|plane_anchored|none",
                        "ablation_note": "path to before/after comparison"},
  "confidence_note": "..."
}
```

`transform_xz` and `adjacency` are the two fields multi-room stitching (Phase 2)
needs to add on top of what per-room output already provides — every room's own
`room.json` stays valid and unchanged when stitched, the property file just
references and places them.

## Design rule going forward

Every tier (photo/video/LiDAR) must emit the *same* per-room schema above, with
wider CIs as signal thins — this is what makes the whole-property stitch step
(Phase 2/4) tier-agnostic instead of three separate stitchers.
