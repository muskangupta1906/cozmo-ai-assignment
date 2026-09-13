#!/usr/bin/env bash
# download_weights.sh
# --------------------
# Pre-fetches pretrained model weights so the first real pipeline run isn't
# competing with a network download for the README's <15-minute budget, and
# so the models/large binaries are "fetched by script" rather than committed
# to git, per the case study's constraints.
#
# Both models cache themselves on first use anyway (huggingface_hub /
# open_clip's own cache dirs) -- this script just does that fetch up front,
# explicitly, and is safe to re-run (skips anything already cached).
#
#   Photo tier   : depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf (~95MB)
#   Damage tier  : open_clip ViT-B-32-quickgelu/openai (~350MB)
#
# Usage: ./download_weights.sh

set -euo pipefail

echo "[1/2] Depth Anything V2 (Metric-Indoor-Small) -- photo tier..."
python -c "
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
model_id = 'depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf'
AutoImageProcessor.from_pretrained(model_id)
AutoModelForDepthEstimation.from_pretrained(model_id)
print('      cached.')
"

echo "[2/2] CLIP ViT-B-32-quickgelu (openai) -- damage tier..."
python -c "
import open_clip
open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='openai')
print('      cached.')
"

echo "Done. Weights cached locally (huggingface_hub / open_clip default cache dirs)."
