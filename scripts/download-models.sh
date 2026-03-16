#!/usr/bin/env bash
# Download model files required for examples and tests.
# Usage: bash scripts/download-models.sh
#
# Downloads to models/ at repository root:
#   - model.safetensors  (GPT-2 Small, 124M params, ~500MB)
#   - yolov8n.safetensors (YOLOv8-nano, 3.2M params, ~6MB)
#   - bus.ppm (test image for YOLO, ~1.2MB)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODELS_DIR="$REPO_ROOT/models"

mkdir -p "$MODELS_DIR"

# --- GPT-2 Small (safetensors) ---
GPT2_URL="https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors"
GPT2_FILE="$MODELS_DIR/model.safetensors"

if [ -f "$GPT2_FILE" ]; then
    echo "GPT-2 model already exists: $GPT2_FILE"
else
    echo "Downloading GPT-2 Small safetensors (~500MB)..."
    curl -L --progress-bar -o "$GPT2_FILE" "$GPT2_URL"
    echo "GPT-2 downloaded: $GPT2_FILE ($(du -h "$GPT2_FILE" | cut -f1))"
fi

# --- YOLOv8-nano (safetensors) ---
# This requires exporting from ultralytics. Use the Python helper if available.
YOLO_FILE="$MODELS_DIR/yolov8n.safetensors"

if [ -f "$YOLO_FILE" ]; then
    echo "YOLOv8n model already exists: $YOLO_FILE"
else
    echo "YOLOv8n safetensors requires export from ultralytics."
    echo "Run: pip3 install ultralytics safetensors && python3 scripts/export_yolo.py"
    echo "(Skipping YOLOv8n download — not available as direct URL)"
fi

echo ""
echo "Model directory: $MODELS_DIR"
ls -lh "$MODELS_DIR"
