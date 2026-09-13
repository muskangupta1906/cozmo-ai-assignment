# Cozmo AI — Floor Plan Pipeline

Turn a phone capture (photos, video, or LiDAR) into a dimensioned floor plan.

## 1. Setup

```bash
conda create -n cozmo python=3.10
conda activate cozmo
pip install -r requirements.txt

brew install ffmpeg        # macOS (apt install ffmpeg on Linux)

./download_weights.sh      # one-time model download
```

## 2. Run it

```bash
python cozmo.py <input_folder> <output_folder>
```

That's it — one command, any tier. It auto-detects photos, video, or LiDAR
from what's in `<input_folder>`.

**Photos** — a folder of 2-8 stills of one room:
```bash
python cozmo.py my_room/ out/my_room
```

**Multiple rooms (photos)** — one subfolder per room, plus `adjacency.json`
declaring which rooms connect:
```bash
python cozmo.py my_property/ out/my_property
```

**Video** — a folder containing one walkthrough clip:
```bash
python cozmo.py my_room/ out/my_room
```

**LiDAR** — a Stray Scanner export folder (single or multi-room, detected automatically):
```bash
python cozmo.py my_scan/ out/my_scan
```

## 3. Get your floor plan

Results land in `<output_folder>/`:
- `room_plan.png` / `property_plan.png` — the rendered floor plan
- `room.json` / `property.json` — dimensions, confidence intervals, openings

## Try it now (no capture needed)

```bash
python cozmo.py Assignment/c00a170fe1 out/demo
```
