# Photo-tier smoke-test fixture (dev only -- NOT the Phase 6 benchmark)

`property_test/` is 3 room folders of 5 stills each, extracted as the
sharpest frames from contiguous, single-room segments of
`Assignment/1a8384c3f6/rgb.mp4` (segments identified via
`stitch_property.segment_rooms`, see PHASE_PLAN.md Phase 4). Purpose:
give `reconstruct_room_photo.py` and `stitch_property_photo.py` something
to run against end-to-end before any real photo-tier capture exists.

**This is not benchmark data and must not be used to report photo-tier
accuracy numbers.** Concretely:

- These are video frames, not composed stills -- captured mid-walk during
  a LiDAR sweep, not "stand in a corner and take a deliberate photo of the
  room" the way the real capture protocol asks for. Framing, distance, and
  overlap between shots are whatever the walking path happened to produce,
  not representative of real photo-tier behavior.
- No EXIF (extracted via OpenCV from an mp4, not shot as iPhone stills) --
  every photo in this fixture falls back to the assumed-FOV intrinsics
  path (see `photo_scan_io.FALLBACK_HFOV_DEG`), not the EXIF path real
  captures would mostly use.
- No ground truth measurements exist for the room these frames came from
  in this repo, so there's nothing to score reconstructed dimensions
  against.
- `adjacency.json`'s two edges (room_0-room_1, room_1-room_2) are a
  simplified chain -- `segment_rooms`/`compute_adjacency` on the source
  scan actually found all 3 pairwise edges (a hub/junction where all three
  meet), collapsed here for a simpler test topology.

What running it DID verify (see PHASE_PLAN.md Phase 4 for the full
writeup): the pipeline runs end-to-end without crashing on real depth
model output, photo-to-photo RANSAC registration succeeds when photos have
enough shared texture, and it fails safely (drops photos, warns, falls
back) when they don't or when a room has no matching detected opening. It
also caught a real bug (see reconstruct_room_photo.py's `kabsch`
docstring) that would otherwise have shipped broken.

Regenerate or extend this fixture with a similar frame-extraction script
if needed -- it's disposable dev scaffolding, not a tracked deliverable.
