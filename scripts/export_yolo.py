#!/usr/bin/env python3
"""Export YOLOv8-nano weights to safetensors format.

Usage:
    pip3 install ultralytics safetensors
    python3 scripts/export_yolo.py

Exports to models/yolov8n.safetensors at the repository root.
"""

import os
import sys
from pathlib import Path

def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip3 install ultralytics")
        sys.exit(1)

    try:
        import safetensors.torch
    except ImportError:
        print("ERROR: safetensors not installed. Run: pip3 install safetensors")
        sys.exit(1)

    # Find repo root
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    models_dir = repo_root / "models"
    models_dir.mkdir(exist_ok=True)
    output_path = models_dir / "yolov8n.safetensors"

    if output_path.exists():
        print(f"Already exists: {output_path}")
        return

    print("Loading YOLOv8-nano from ultralytics...")
    model = YOLO("yolov8n.pt")

    # Extract state dict
    state_dict = model.model.state_dict()
    # Convert to float32 and CPU
    state_dict = {k: v.float().cpu() for k, v in state_dict.items()}

    print(f"Exporting {len(state_dict)} tensors to {output_path}...")
    safetensors.torch.save_file(state_dict, str(output_path))

    size_mb = output_path.stat().st_size / 1e6
    print(f"Done: {output_path} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()
