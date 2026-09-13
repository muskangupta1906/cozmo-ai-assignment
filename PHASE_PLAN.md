# Cozmo AI Case Study — Phase Plan

Source: `Applied AI.pdf` (Cozmo AI Case Study, Aug 2026). This file is the persistent
record of the plan so it survives context resets — update it as phases complete or
the approach changes, don't let it drift out of sync with reality.

**Standing instruction: do not `git commit` anything unless explicitly told to for that
specific change.** Process evidence (Part 5) is scored on the commit history looking
organic, which means the user controls when/how commits happen — not batched by an
agent at the end of a session.

## Where things stand right now

- One commit exists (`bafbb61`): a LiDAR-tier, single-room pipeline
  (`reconstruct_room.py` + `scan_io.py`) that fuses depth+pose into a point cloud,
  detects floor/ceiling by percentile, extracts a wall footprint via radial sweep, and
  writes `room.json` + `room_plan.png` + `fused_cloud.ply`.
- Raw captures for 3 scans live in `Assignment/<hash>/` in **Stray Scanner** format
  (camera_matrix.csv, odometry.csv, imu.csv, depth/, confidence/, rgb.mp4). This is a
  real off-the-shelf iOS app — implies **Route 2 (stock capture protocol)** unless
  there's a reason to build a custom app; needs an explicit decision (see Phase 0).
- All 3 sample scans now have LiDAR-tier output in `out/<hash>/`:
  - `1a8384c3f6`: ceiling 1.83 m, floor area 67.3 m²
  - `c00a170fe1`: ceiling 1.61 m, floor area 21.3 m²
  - `c7d28f72c6`: ceiling 3.08 m, floor area 72.4 m²
  - **Flag**: the first two ceiling heights are implausibly low for real rooms
    (typical 2.4–3 m). Likely a bug in floor/ceiling percentile detection or the wall
    band cutoff — strong candidate for the Part 4 fix-loop's "worst-performing gate."
- Phase 2 (multi-room stitching + drift correction, LiDAR tier) is mostly done —
  `stitch_property.py` runs end-to-end on `1a8384c3f6` (3 rooms) and `c00a170fe1`
  (2 rooms) — both turn out to be real multi-room captures, not the single rooms Phase 1
  treated them as. Produces `property.json`/`property_plan.png`/`drift_ablation.json`.
  Hit and fixed an OOM bug along the way (see Phase 2 below); known remaining issue is
  overlapping/noisy room footprints on stitched output on both scans, not yet fixed.
  `c7d28f72c6` stays single-room; its code path was reviewed but not executed.
- Phase 3 (video tier) started: `video_tier.py` implements SfM (pycolmap) + gravity
  alignment + placeholder scale recovery, runs end-to-end without crashing, but is
  blocked on real video-tier capture data before it can be trusted or wired to the
  output contract — the LiDAR scan's own RGB track was tried as a stand-in and produces
  weak/fragmented registration, most likely because its sweep-heavy capture motion
  (built for LiDAR coverage) isn't representative of a steady photogrammetry
  walkthrough. See Phase 3 below before doing more work here blind.
- Not yet started: photo tier, damage detection, concealed-damage flags, scope line
  items, benchmark construction, ground truth capture, repeatability testing,
  head-to-head comparison, fix loop, and all deliverables/report writing.

## Phase 0 — Decisions and scaffolding — DONE (2026-09-12)

- [x] Confirm capture route: **Route 2**, three stock apps — native Camera (photos +
      video tiers) and **Stray Scanner** (LiDAR tier, confirmed by matching the raw file
      format in `Assignment/`). One-page protocol: `docs/capture_protocol.md`.
- [x] Device matrix: `docs/device_matrix.md`.
- [x] JSON output schema (per-room, current; per-property, target for Phase 2):
      `docs/output_schema.md`.
- [x] Benchmark folder skeleton: `benchmark/{multi_room,damage_room,repeat_room,ground_truth}/`
      + `benchmark/README.md` explaining what goes where. Empty until Phase 6 captures exist.

## Phase 1 — LiDAR tier: finish the single-room contract — mostly DONE (2026-09-12)

- [x] **Fixed the floor/ceiling height bug.** Root cause: `detect_floor_and_ceiling`
      used a fixed `percentile(y, 2/98)` cut. Diagnosed by histogramming the fused
      cloud's y-values on all 3 scans: floor always shows one dominant density spike
      (walked over at close range every frame), ceiling shows a smooth decay to zero
      with **no** density spike in any of the 3 scans (steep upward angle, rarely
      pointed at). The fixed 98th-percentile cut was landing inside the wall/furniture
      clutter tail, not on the real ceiling. Fix: search progressively wider top-of-cloud
      slices (tightest first) and RANSAC-fit a plane in each; accept the first slice
      whose plane is genuinely horizontal (normal within ~30° of vertical) with a high
      inlier ratio. Verified against real data before implementing (see chat) — isolating
      the sparse top slice on the two affected scans revealed a clean, nearly-perfect
      horizontal plane at 2.14m and 2.30m respectively, vs. the old 1.61m/1.83m. Third
      scan has no qualifying plane at any band → correctly falls back to a
      confidence=0 estimate instead of a falsely precise number.
      Before/after (needed for a future fix-loop-style writeup):
      `1a8384c3f6`: 1.833m → ~2.30m | `c00a170fe1`: 1.612m → ~2.14m | `c7d28f72c6`: 3.08m
      → low-confidence fallback (no valid re-run numbers logged yet — confirm against
      the latest run's `room.json` before quoting).
- [x] Confidence interval on every measurement: `ceiling_height_confidence` +
      `ceiling_height_ci_m` (confidence-tiered), `wall_length_ci_m` and
      `floor_area_ci_m2` (placeholder fixed-depth-noise error model, explicitly *not*
      calibrated yet — real calibration needs Phase 6/7 ground truth).
- [x] Opening detection (doors/windows): gap-in-wall-coverage heuristic — flags spans
      along each wall where mid-height coverage drops out but floor visibility
      continues (i.e. you can see past the wall). Reports `wall_index`, `position_xz`,
      `width_m` per opening; marked on `room_plan.png`. Known limitation: heuristic
      distinguishes "real gap" from "missing data" only via floor visibility, so it can
      silently miss openings on sparse captures but shouldn't invent phantom ones.
      Not yet validated against the ≤2cm/85% gate — needs Phase 6 ground truth.
- [x] Wall footprint quality — partially fixed, partially documented as a real limit.
      Diagnosed: a deep inward notch in `c00a170fe1`'s footprint lined up exactly with
      mid-room furniture and coincided with 4 of its 5 false-positive openings — when
      the true wall along a ray is occluded, the radial sweep's farthest-point pick
      grabs the occluder instead, reading as a spike toward the centroid. Fix: a
      circular median-filter outlier pass in `radial_boundary` corrects bins that dive
      sharply inward relative to their local trend (real walls are locally smooth).
      Reduced `c00a170fe1`'s opening count 5→3 as a direct, measurable side effect.
      Remaining wider notches (multi-bin-wide, e.g. a real hallway/connector) are left
      uncorrected on purpose — can't distinguish "real recessed architecture" from
      "occlusion" by geometry alone without ground truth; documented as a known
      limitation rather than force-smoothed away speculatively (see Phase 6/7).
- [x] Also fixed while touching this: RANSAC plane fits (floor/ceiling detection) were
      non-deterministic run-to-run on identical input — a direct repeatability-gate risk
      ("same room in, same plan out"). Seeded `np.random` and `o3d.utility.random`;
      verified two back-to-back runs of the same scan now produce byte-identical
      `room.json`.
- [ ] Opening detection has a known false-positive mode (furniture against a wall reads
      the same as a real opening) — no RGB/semantic disambiguation yet. Documented in
      `room.json`'s `confidence_note`; treat `opening_count` as an upper bound until
      Phase 5/6 adds a way to tell the two apart. Not blocking Phase 1 completion since
      it's honestly flagged rather than silently wrong.

## Phase 2 — Multi-room stitching + drift accountability (LiDAR tier) — mostly DONE (2026-09-13)

- [x] Segment a multi-room capture into rooms. Turns out `1a8384c3f6` (previously treated
      as one 67m² "room" in Phase 1 — already flagged there as suspiciously large) is
      actually a real 3-room walkthrough; didn't need a new capture. `stitch_property.py`'s
      `segment_rooms` splits the trajectory on revisit density (rooms get walked/swept
      repeatedly; corridors are crossed once) then DBSCAN-clusters what's left spatially.
      Found 3 rooms + a shared hub/junction on this scan, matching the case study's
      "3+ rooms with a connector" requirement.
- [x] Stitch per-room plans into one whole-property plan with adjacency —
      `property.json` + `property_plan.png`, adjacency read off the timeline (which
      corridor run connects which two rooms).
- [x] Drift correction implemented: linear translation-only loop-closure redistribution
      (documented as explicitly NOT full pose-graph SLAM — see
      `apply_loop_closure_correction` docstring). Satisfies "don't use poses as-is."
- [x] Drift ablation produced (`drift_ablation.json`, with vs without correction). On
      `1a8384c3f6`: 74.455 m² (with) vs 74.353 m² (without) — measured start/end pose gap
      is only 0.17m over a ~13m-extent walk, so a small delta here is real, not a bug
      (verified by printing the raw gap before trusting the ablation numbers).
- [x] **Found + fixed a blocking bug while first running this end-to-end**: fusing a
      room's assigned frames used ALL of them with no subsampling, unlike the
      single-room contract's `every_n=5` default. A dwelled-on room can collect
      thousands of frames (that's the same revisit-density signal segmentation relies
      on), so room_0/room_1 here each fused ~2000 frames -> ~100M+ raw points before
      confidence filtering -> OOM-killed (exit 137, silently — no traceback, empty
      output dir). Root-caused by rerunning with output redirected to a real file
      instead of piped through `tail` (which was masking python's actual exit code).
      Fix: `build_fused_point_cloud` now strides explicit `frame_indices` by `every_n`
      too (`reconstruct_room.py`), and `stitch_property.py` threads an `--every-n` flag
      through consistent with the single-room CLI. Re-ran clean after the fix: exit 0,
      3 rooms, full output written.
- [ ] **Known quality issue, not yet fixed**: the stitched plan for `1a8384c3f6` shows
      real overlap between room_0 and room_1, and room_1's footprint is a noisy
      star-shaped polygon spanning much of the property rather than a clean room
      outline. Root cause is likely that `extract_wall_footprint`'s radial sweep
      (built/tuned in Phase 1 against single, mostly-enclosed rooms) doesn't hold up on
      a room whose frame subset is spatially messier / has more corridor bleed-through
      at its boundary. Not force-smoothed away speculatively — needs either a
      registration/clipping step between adjacent room polygons or ground truth to know
      how much of this is real vs. algorithmic garbage. Flagging as a Phase 7 (or
      earlier fix-loop) candidate rather than guessing a fix blind.
- [x] Sanity-checked on 2 of the 3 sample scans. `c00a170fe1` (previously Phase 1's
      other suspiciously-small-ceiling room) also turns out to be 2 real rooms —
      stitched clean, exit 0, same overlap/noisy-footprint pattern as `1a8384c3f6`,
      confirming that's a systemic wall-footprint-algorithm issue on multi-room
      segments, not one scan's fluke. `c7d28f72c6` stays a single trajectory cluster
      (9677/9745 frames) — reviewed (not run, to avoid the ~10min fusion cost of its
      9677 frames) that `segment_rooms`/`compute_adjacency`/`stitch()` degrade
      correctly to a one-room "property" with no adjacency edges on this path; not
      execution-verified.
- **Decision (2026-09-13): deferring the overlapping/noisy-footprint issue and moving
  on to Phase 3.** Reasoning: it's honestly flagged (not silently wrong), fixing it
  blind without ground truth risks tuning against noise, and Phase 6/7 (real ground
  truth + gate validation) is the right point to know whether it's actually bad enough
  to need a fix vs. within tolerance. Revisit here, or treat as the Phase 9 fix-loop
  candidate, once real numbers exist. `c7d28f72c6`'s single-room path also remains
  execution-unverified — same reasoning, low risk, revisit if time allows before Phase 6.

## Phase 3 — Video tier — IN PROGRESS (started 2026-09-13)

- [x] **Approach decided**: classical SfM via `pycolmap` (COLMAP's Python API, no
      external `colmap` binary needed) rather than a monocular-depth-model route.
      Reasoning: deterministic, no model checkpoint to bundle/download, explainable
      for the report — accepted the heavier native-dependency footprint (pycolmap +
      ffmpeg, both now installed in the `cozmo` conda env) as the tradeoff.
- [x] Built `video_tier.py`: ffmpeg frame extraction at a fixed fps -> SIFT feature
      extraction -> sequential matching -> incremental SfM mapping -> **gravity
      alignment** (COLMAP's world frame has no relationship to "up", unlike ARKit —
      estimated by averaging each registered camera's local up-axis rotated into
      world space, on the assumption the operator held the phone roughly upright)
      -> **scale recovery** (SfM only recovers shape; scale is anchored to an assumed
      1.4m average camera-carry height above the floor — an explicit placeholder, same
      spirit as the LiDAR tier's placeholder CI model, NOT a calibration). Runs
      end-to-end without crashing.
- [ ] **Not yet wired to the output contract.** `detect_floor_and_ceiling` /
      `extract_wall_footprint` from `reconstruct_room.py` are meant to be reused as-is
      once the SfM cloud is in the same (y-up, meters) convention — deliberately not
      wired yet because the sparse point clouds produced so far are too degenerate to
      make that a meaningful test (see next item). Wiring this + deciding whether
      sparse points are dense enough or a dense stereo pass (`pycolmap.patch_match_stereo`
      + `stereo_fusion`, already available in the same API) is needed first, is the
      next concrete step.
- [ ] **Blocked on real video-tier data, not a code bug.** Smoke-tested against
      `Assignment/1a8384c3f6/rgb.mp4` (the LiDAR scan's own RGB track) as a stand-in,
      since no genuine video-tier capture exists yet (that's a Phase 6 task). Result:
      weak registration — only 11/230 frames at 2fps, 56/201 at 8fps (and even that
      fragmented into 5 disconnected sub-models), with an implausibly wide camera-height
      spread (median 0.21m but min/max -1.38m/+0.49m) suggesting the registered subset
      itself is partly wrong, not just sparse. Root cause is very likely the *source
      motion*, not the SfM code: Stray Scanner drives the camera in a sweeping,
      point-it-at-every-surface pattern for LiDAR coverage (see Phase 1/2 notes on
      repeated wall sweeps) — the opposite of the smooth, steady walkthrough classical
      feature-matching SfM needs. Tried denser sampling (2fps -> 8fps) as the obvious
      first fix; it measurably helped (11 -> 56 registered) but didn't come close to
      resolving it, which points at motion profile over sampling rate. **Do not** keep
      tuning SfM hyperparameters against this proxy data looking for a fix — it isn't
      representative of what a real video-tier capture (native Camera app, walked
      steadily per `docs/capture_protocol.md`) will look like, and doing so risks
      overfitting the pipeline to compensate for a data-quality problem that a real
      capture won't have. Real validation needs an actual Phase 6 video-tier capture.
- [ ] Capture/acquire a video-tier version of the same benchmark rooms (Phase 6 task —
      steady handheld walkthrough via native Camera app, not a LiDAR-sweep motion).
- [ ] Once real video-tier footage exists: re-run `video_tier.py`, confirm registration
      rate/quality on genuinely suitable input, then wire in floor/ceiling + footprint
      + the same JSON output contract (tier="video", wider ±3% calibrated CIs) and
      re-test against `1a8384c3f6`'s footage too (still useful as a stress/worst-case
      test once the happy path is proven, just not the primary validation case).

## Phase 4 — Photo tier (the floor: 2–8 stills/room, no depth/poses)

- [ ] Build a sparse multi-view pipeline (e.g. classical SfM or a monocular
      single/few-view depth model) that still produces a stitched whole-property plan.
- [ ] Must handle per-room photo folders and still stitch — a single-room-only path
      fails this gate outright.
- [ ] Gate: ±8% wall lengths with calibrated intervals; whole-property footprint ±8%,
      correct adjacency, no room overlaps.

## Phase 5 — Damage detection + scope

- [ ] Per-surface damage region detection with class + metric extent (needs the
      furnished, staged-damage room from the benchmark, spanning 2 damage classes).
- [ ] Concealed-damage flags with an explicit fired rule (documented heuristic/model,
      not a black box).
- [ ] Scope line items keyed to specific surfaces.

## Phase 6 — Benchmark set construction (own captures, own ground truth)

Required composition — none of this exists yet:
- [ ] 1 multi-room capture: 3+ rooms + a connector, captured at all 3 tiers (photo
      tier as per-room folders).
- [ ] 1 furnished room with staged damage spanning 2 damage classes.
- [ ] Every benchmark room captured at all 3 tiers.
- [ ] At least 1 room captured twice at the same tier (repeatability gate).
- [ ] Laser/tape ground truth measurements on everything; submit raw sensor data +
      measurements alongside.

## Phase 7 — Gate validation, calibration, repeatability

- [ ] Run all gates end to end per tier: opening widths, ceiling height + repeat
      spread, repeatability (±1cm or 0.5%/wall), drift ablation, photo-tier stitch.
- [ ] Calibrate confidence intervals per tier — confident garbage on thin input caps
      the whole score, so intervals must actually reflect tier-appropriate uncertainty.

## Phase 8 — Head-to-head vs incumbent (Part 3)

- [ ] Pick 1 free consumer scanning app (magicplan or Polycam free tier), name + version.
- [ ] Run it on 2 benchmark rooms at the LiDAR tier, export its output.
- [ ] Build one dimension-by-dimension error table (ours vs theirs). Target: beat or
      tie on ≥70% of shared dimensions.

## Phase 9 — Fix loop (25% of score, Part 4)

- [ ] Identify the single worst-performing gate from Phase 7's real numbers (the
      ceiling-height bug flagged above is a strong early candidate, but confirm against
      actual gate results once benchmarks exist).
- [ ] Write the one-page fix declaration: failing number, root-cause hypothesis +
      evidence, the fix, and a predicted post-fix number.
- [ ] Ship the fix. Keep the before-run and after-run both regenerable, plus a readable
      diff — a fix without regenerable before/after scores zero regardless of the result.

## Phase 10 — Deliverables assembly

- [ ] Compliance matrix: requirement → file path → artifact → status.
- [ ] Capture route doc (protocol or TestFlight build) + device matrix.
- [ ] README: fresh capture → running output in <15 min on a clean machine, one
      command per capture.
- [ ] Reproduction bundle: regenerate every reported number from raw inputs (cached
      outputs OK only if they replay deterministically AND the live path still runs).
- [ ] Benchmark report: gates at all 3 tiers, repeatability table, head-to-head table, timing.
- [ ] Fix loop bundle (declaration + before/after + diff).
- [ ] Technical report, max 6 pages: architecture, tier design, device matrix, drift
      handling, error budget, calibration analysis, fix-loop story, known failure modes
      (mirrors, glass, wet-look surfaces, low light — must be addressed somewhere).
- [ ] Raw benchmark data: sensor logs, ground truth, app exports.

## Phase 11 — Walk-in test readiness

- [ ] All 3 tiers runnable cold, on-demand, on an unseen capture, one command each.
- [ ] Sanity-check timing — the pipeline needs to run live in front of evaluators while
      they take laser measurements, so runtime on the largest expected capture matters.

## Ongoing across every phase

- Commit incrementally as real work lands (user-directed, not batched) — Part 5 reads
  the commit history for authenticity, worth 5% directly and colors how the whole
  narrative report is read.
- Keep this file updated when a phase starts/completes or the plan changes, so a future
  session (or context compaction) can pick up state without re-deriving it.
