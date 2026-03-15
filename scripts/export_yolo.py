#!/usr/bin/env python3
"""Export YOLOv8-nano weights to safetensors format for GPU inference.

Usage:
    pip install ultralytics safetensors
    python scripts/export_yolo.py

Outputs:
    models/yolov8n.safetensors  (~12.6 MB, f32 weights)

Tensor naming convention (matches Ultralytics state_dict):
    model.{layer}.conv.weight          Conv2d weight [C_out, C_in, kH, kW]
    model.{layer}.bn.weight            BatchNorm gamma [C]
    model.{layer}.bn.bias              BatchNorm beta [C]
    model.{layer}.bn.running_mean      BatchNorm running mean [C]
    model.{layer}.bn.running_var       BatchNorm running var [C]
    model.{layer}.cv1.conv.weight      C2f/SPPF cv1 Conv weight
    model.{layer}.cv1.bn.*             C2f/SPPF cv1 BN params
    model.{layer}.cv2.conv.weight      C2f/SPPF cv2 Conv weight
    model.{layer}.cv2.bn.*             C2f/SPPF cv2 BN params
    model.{layer}.m.{j}.cv1.conv.weight  Bottleneck j, first Conv
    model.{layer}.m.{j}.cv2.conv.weight  Bottleneck j, second Conv
    model.22.cv2.{scale}.{sub}.*       Detect box branch (scale=0,1,2)
    model.22.cv3.{scale}.{sub}.*       Detect class branch (scale=0,1,2)

All weights are stored as float32. No Conv-BN fusion is applied.
"""

import sys
from pathlib import Path

def main():
    try:
        from ultralytics import YOLO
        from safetensors.torch import save_file
        import torch
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install ultralytics safetensors torch")
        sys.exit(1)

    # Load YOLOv8-nano (downloads if not cached)
    model = YOLO("yolov8n.pt")
    state_dict = model.model.state_dict()

    # Convert all tensors to float32 contiguous
    tensors = {}
    for name, tensor in state_dict.items():
        t = tensor.float().contiguous()
        tensors[name] = t

    # Save
    out_path = Path(__file__).parent.parent / "models" / "yolov8n.safetensors"
    out_path.parent.mkdir(exist_ok=True)
    save_file(tensors, str(out_path))

    # Print summary
    total_params = sum(t.numel() for t in tensors.values())
    total_bytes = sum(t.numel() * 4 for t in tensors.values())
    print(f"Saved {len(tensors)} tensors to {out_path}")
    print(f"Total parameters: {total_params:,}")
    print(f"File size: {total_bytes / 1024 / 1024:.1f} MB (f32)")

    # Print tensor names for reference
    print("\nTensor names and shapes:")
    for name, tensor in sorted(tensors.items()):
        print(f"  {name}: {list(tensor.shape)}")


if __name__ == "__main__":
    main()
