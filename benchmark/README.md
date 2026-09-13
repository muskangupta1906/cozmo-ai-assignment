# Benchmark set (Phase 6 — not populated yet)

Layout required by the case study (Part 2 benchmark composition). This is
scaffolding only; real captures + ground truth get dropped in during Phase 6.

- `multi_room/`   — 1 capture, 3+ rooms + a connector, all 3 tiers (photo tier
                    as per-room folders), used for the stitching + drift gates.
- `damage_room/`  — 1 furnished room, staged damage spanning 2+ damage classes,
                    all 3 tiers.
- `repeat_room/`  — at least 1 room captured twice at the same tier, for the
                    repeatability gate (agree within 1cm / 0.5% per wall).
- `ground_truth/` — laser/tape measurements for every room above, plus raw
                    sensor data and measurements as submitted alongside.

Each room capture should land under the matching subfolder using the same
`Assignment/<id>/` raw format the pipeline already reads (LiDAR tier), or
`<room_name>/photos/` and `<room_name>/video.mov` for the other two tiers —
see `docs/capture_protocol.md` for exact capture steps and `docs/output_schema.md`
for what the pipeline emits per room.
