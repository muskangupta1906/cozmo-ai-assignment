# Capture Protocol (Route 2 — stock apps only)

One page, followed literally. If a step is ambiguous, treat that as a bug in this
page, not license to improvise.

## Apps needed (all free, App Store)

| Tier   | App                                   | Device requirement                  |
|--------|----------------------------------------|--------------------------------------|
| Photos | Native **Camera** app                  | iPhone 15 or newer, any model        |
| Video  | Native **Camera** app (Video mode)     | iPhone 15 or newer, any model        |
| LiDAR  | **Stray Scanner** (Stray Robots, free) | iPhone 12 Pro or newer **Pro/Pro Max** (LiDAR scanner required) |

Install Stray Scanner from the App Store before arriving; first launch needs a
one-time camera + motion permission grant.

## Photos tier — 2 to 8 stills per room

1. Open the native Camera app, Photo mode (not Portrait, not Pano).
2. Stand near the center of the room. Take one photo facing each wall in turn
   (4 photos for a rectangular room). If the room has an alcove, connector, or
   irregular corner, add 1–4 extra shots covering it.
3. Hold the phone roughly vertical, at chest height, level (not tilted up/down).
4. Do not zoom digitally. Do not use flash.
5. One folder per room. Name it after the room (e.g. `living_room/`). Drop the
   photos straight into it — no subfolders, no renaming needed.

## Video tier — one handheld walkthrough clip per room

1. Open the native Camera app, Video mode, standard resolution/fps (no
   slow-mo, no cinematic mode).
2. Start recording standing at the room's entrance.
3. Walk the room's perimeter once, slowly and steadily (~0.3 m/s — slower than
   a normal walking pace), keeping the phone at chest height and roughly level.
4. At each corner, pause for 2 seconds and pan slowly across the corner before
   continuing.
5. Keep walls in frame throughout; avoid fast pans or sudden direction changes
   (motion blur breaks the pipeline's pose estimate, not just its picture
   quality).
6. Stop recording once you're back near the starting point. One clip per room,
   named after the room (e.g. `living_room.mov`).

## LiDAR tier — Stray Scanner

1. Open Stray Scanner, tap **Record**.
2. Walk the same slow perimeter path as the video tier (~0.3 m/s), phone at
   chest height, roughly level. One continuous recording per room — don't
   stop/resume mid-room.
3. Deliberately tilt the phone up toward the ceiling and down toward the floor
   at least once per room (ideally at each corner) — the ceiling in particular
   is only captured if you point at it; a walkthrough held level the whole
   time will under-report ceiling height (this is a known pipeline failure
   mode, see `PHASE_PLAN.md`).
4. Tap **Stop** when the loop is complete.
5. From the scan list, tap the recording → the share icon → **Export** → save
   to Files (or AirDrop to the processing machine). This produces one folder
   per scan containing `camera_matrix.csv`, `odometry.csv`, `imu.csv`,
   `rgb.mp4`, `depth/`, `confidence/` — hand that folder to the pipeline as-is,
   unzipped, under `Assignment/<scan_id>/`.

## What to avoid, all tiers

- Mirrors, large glass panes, and wet/glossy floors directly in the capture
  path where possible — these are known failure modes for depth sensing and
  are addressed separately in the pipeline, not by the capture itself.
- Direct backlighting (a bright window behind the subject wall).
- Capturing with other people walking through frame.
- Skipping the corner tilt-up/tilt-down step on the LiDAR tier — see step 3 above.

## Handoff

Raw folders/files go under `Assignment/<capture_id>/` (LiDAR) or
`Assignment/<room_name>/photos|video/` (photo/video tiers) exactly as exported —
no manual renaming, cropping, or reordering. The pipeline reads Stray Scanner's
native export format and one flat folder/clip per room; anything else needs no
extra handling from the person capturing.
