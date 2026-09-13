#!/usr/bin/env bash
# reproduce.sh
# ------------
# Regenerates every reported number from raw input, from scratch, with one
# command. Re-runs cozmo.py over every raw capture currently in Assignment/
# into a fresh repro_out/ -- nothing here is read from the committed out/,
# so a diff between out/ and repro_out/ is an honest before/after, not a
# copy of stale numbers.
#
# Scope note: this covers what raw captures exist today (the 3 LiDAR scans
# under Assignment/). Once benchmark/ has real captures (Phase 6) and a
# ground-truth-scored gate report exists (Phase 7), this script is the right
# place to extend with per-gate scoring -- not built ahead of that data,
# since a scoring harness with nothing to score against risks getting the
# interface wrong.
#
# Usage: ./reproduce.sh [out_dir]   (defaults to repro_out/)

set -euo pipefail

OUT_DIR="${1:-repro_out}"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

echo "=== Reproducing every LiDAR-tier capture under Assignment/ into $OUT_DIR/ ==="
for scan_dir in Assignment/*/; do
    scan_id="$(basename "$scan_dir")"
    echo ""
    echo "--- $scan_id ---"
    python cozmo.py "$scan_dir" "$OUT_DIR/$scan_id"
done

echo ""
echo "=== Done. Numbers regenerated from raw input in $OUT_DIR/. ==="
echo "Note: $OUT_DIR/ uses cozmo.py's unified layout (<scan_id>/with_drift_correction/,"
echo "<scan_id>/without_drift_correction/), which differs from the committed out/'s"
echo "layout from when each tier's script was run separately -- compare the numbers"
echo "inside each property.json/room.json against PHASE_PLAN.md's reported figures,"
echo "not the directory structure itself."
