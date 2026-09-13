# Device Matrix

Which tier runs on which hardware, and what accuracy each tier is expected to
honestly deliver. Targets below are the case study's own gates (Part 2); "measured"
columns get filled in once Phase 6/7 benchmarking with laser ground truth exists —
until then these are targets, not verified numbers, and should not be quoted as such.

| Tier   | Min hardware                                   | Sensors used                  | Wall-length target | Ceiling-height target | Measured (pending Phase 7) |
|--------|--------------------------------------------------|--------------------------------|---------------------|-------------------------|------------------------------|
| Photos | iPhone 15 or newer (any model, no LiDAR needed)  | Camera only (2–8 stills/room)  | ±8%, calibrated CI  | Calibrated CI, no hard gate | — |
| Video  | iPhone 15 or newer (any model, no LiDAR needed)  | Camera only (handheld clip)    | ±3%, calibrated CI  | Calibrated CI, no hard gate | — |
| LiDAR  | iPhone 12 Pro or newer **Pro/Pro Max**, or LiDAR iPad Pro | LiDAR depth + confidence + VIO pose + intrinsics | ≤2cm on ≥85% of openings; other gates per Round-1 spec | ≤1.5cm/room, ≤1cm repeat spread | — |

## Why the split

- Photos and video tiers deliberately exclude any Pro-only sensor so the floor
  case ("any picture in, results out") holds on the widest hardware set the
  case study allows (iPhone 15 and newer, non-Pro included).
- The LiDAR tier requires Pro-class hardware because Stray Scanner (the chosen
  capture app, see `docs/capture_protocol.md`) needs the onboard LiDAR
  scanner, which only Pro/Pro Max models and LiDAR-equipped iPads expose.
- Accuracy naturally degrades photos → video → LiDAR as pose/depth signal
  thins, which is why the case study's own gates loosen in that order (±8% →
  ±3% → gated in cm). The device matrix should widen honestly in the same
  direction once real numbers exist, not just repeat the gate thresholds
  as if they were guaranteed.

## Photo tier implementation notes (Phase 4)

- Depth per still comes from a pretrained monocular metric depth model,
  `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` (disclosed
  pretrained model per the case study's allowance) — runs locally via
  `transformers`, no network calls at inference time, weights fetched by
  script on first run (~100MB, cached by HF's usual cache dir). CPU or MPS
  (Apple Silicon); ~1s/photo on MPS in testing.
- Intrinsics come from EXIF `FocalLengthIn35mmFilm` when present (true
  iPhone stills normally have this); falls back to an assumed 69.4°
  horizontal FOV (iPhone main/wide camera spec) when EXIF is missing or a
  non-iPhone/edited image strips it — see `photo_scan_io.get_intrinsics`.
  The fallback path directly biases every downstream metric distance, which
  is one reason the photo tier's CI is wide.
- Cross-photo registration (needed to fuse 2-8 independent stills into one
  metric room) has a real, observed failure mode on non-ideal source
  material: weak/no shared texture between photos, or too little parallax,
  produces too few RANSAC inliers and that photo (and everything after it
  in the chain) gets dropped rather than guessed at. See
  `dev_fixtures/photo_tier_smoke/` and PHASE_PLAN.md Phase 4 for the
  smoke-test evidence this was actually exercised, not just theorized.

## Known capture-time risk (see PHASE_PLAN.md)

Even on LiDAR-capable hardware, ceiling height is only reliable if the
capture explicitly tilts the phone toward the ceiling per
`docs/capture_protocol.md` step 3 — a walkthrough held level the whole time
starves the pipeline of ceiling signal regardless of hardware tier, which is
exactly the failure mode the floor/ceiling detection fix addressed (see
`PHASE_PLAN.md` bug note and the `confidence` field now reported per capture).
