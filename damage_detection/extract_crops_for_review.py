"""
One-off inspection helper: given a damage_detections.json, pulls the actual
RGB crop for a sample of detections (full frame + zoomed bbox crop side by
side) so a human (or the assistant) can eyeball whether the classical
region-proposal + CLIP classification found anything real. Not part of the
main pipeline -- delete or keep as a debugging tool, not wired into
detect_damage.py on purpose (keeps the main run fast).

Usage:
    python damage_detection/extract_crops_for_review.py <scan_dir> <detections_json> <out_dir> [--n 16]
"""
import os
import sys
import json
import argparse
import random
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan_dir")
    ap.add_argument("detections_json")
    ap.add_argument("out_dir")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    dets = json.load(open(args.detections_json))
    random.seed(args.seed)
    sample = random.sample(dets, min(args.n, len(dets)))
    sample.sort(key=lambda d: d["frame_index"])
    wanted_frames = {}
    for d in sample:
        wanted_frames.setdefault(d["frame_index"], []).append(d)

    video_path = os.path.join(args.scan_dir, "rgb.mp4")
    cap = cv2.VideoCapture(video_path)
    i = 0
    max_frame = max(wanted_frames)
    saved = []
    while i <= max_frame:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if i in wanted_frames:
            for j, d in enumerate(wanted_frames[i]):
                x, y, w, h = d["bbox_px"]
                pad = 40
                x0, y0 = max(0, x - pad), max(0, y - pad)
                x1, y1 = min(frame_bgr.shape[1], x + w + pad), min(frame_bgr.shape[0], y + h + pad)
                crop = frame_bgr[y0:y1, x0:x1].copy()
                cv2.rectangle(crop, (x - x0, y - y0), (x - x0 + w, y - y0 + h), (0, 0, 255), 3)
                out_path = os.path.join(
                    args.out_dir,
                    f"frame{i:05d}_{j}_{d['predicted_class'].replace(' ', '_')}_"
                    f"conf{d['confidence']:.2f}_{d['surface_type']}.jpg")
                cv2.imwrite(out_path, crop)
                saved.append(out_path)
        i += 1
    cap.release()
    print(f"Saved {len(saved)} crops to {args.out_dir}")
    for p in saved:
        print(" ", p)


if __name__ == "__main__":
    main()
