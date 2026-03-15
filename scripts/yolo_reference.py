#!/usr/bin/env python3
"""Generate reference intermediate outputs from YOLOv8-nano for validation.

Runs YOLOv8-nano inference on bus.jpg and prints:
- Layer 0 output statistics
- Detection results
- Class score statistics at each scale

Usage:
    uv run --with ultralytics --with pillow --with packaging scripts/yolo_reference.py
"""

import sys
from pathlib import Path

def main():
    try:
        from ultralytics import YOLO
        import torch
        from PIL import Image
        import numpy as np
    except ImportError as e:
        print(f"Missing dependency: {e}")
        sys.exit(1)

    model = YOLO("yolov8n.pt")

    # Run inference on bus.jpg
    img_path = Path(__file__).parent.parent / "models" / "bus.ppm"
    if not img_path.exists():
        # Try jpg
        img_path = Path(__file__).parent.parent / "models" / "bus.jpg"
    if not img_path.exists():
        print("No test image found")
        sys.exit(1)

    # Load image
    img = Image.open(img_path).convert("RGB")
    print(f"Image: {img.size}")

    # Run inference
    results = model.predict(img, conf=0.25, iou=0.45, verbose=False)
    result = results[0]

    print(f"\nDetections: {len(result.boxes)}")
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        name = result.names[cls_id]
        print(f"  {name:<15} conf={conf:.3f}  box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")

    # Get intermediate outputs using hooks
    print("\n--- Layer 0 output stats ---")
    layer0_output = [None]
    def hook_l0(module, input, output):
        layer0_output[0] = output.detach()

    # Access model backbone
    backbone = model.model.model
    backbone[0].register_forward_hook(hook_l0)

    # Run again with hook
    results2 = model.predict(img, conf=0.25, verbose=False)

    if layer0_output[0] is not None:
        t = layer0_output[0]
        print(f"  Shape: {list(t.shape)}")
        print(f"  Min: {t.min().item():.6f}")
        print(f"  Max: {t.max().item():.6f}")
        print(f"  Mean: {t.mean().item():.6f}")
        print(f"  Std: {t.std().item():.6f}")
        # Print first 8 values of channel 0
        ch0 = t[0, 0].flatten()[:8]
        print(f"  First 8 values (ch0): {ch0.tolist()}")

    # Also get detect head raw outputs
    print("\n--- Detect head raw output stats ---")
    detect_outputs = {}
    def hook_detect(name):
        def fn(module, input, output):
            detect_outputs[name] = output.detach()
        return fn

    # The detect head is model[22]
    detect = backbone[22]
    # cv2 and cv3 are the box and class branches
    for scale in range(3):
        for sub in range(3):
            detect.cv2[scale][sub].register_forward_hook(hook_detect(f"cv2.{scale}.{sub}"))
            detect.cv3[scale][sub].register_forward_hook(hook_detect(f"cv3.{scale}.{sub}"))

    results3 = model.predict(img, conf=0.25, verbose=False)

    # Print final outputs of each branch
    for scale in range(3):
        cv2_key = f"cv2.{scale}.2"
        cv3_key = f"cv3.{scale}.2"
        if cv2_key in detect_outputs:
            t = detect_outputs[cv2_key]
            print(f"\n  cv2 scale {scale} (box): shape={list(t.shape)}, min={t.min():.4f}, max={t.max():.4f}, mean={t.mean():.4f}")
        if cv3_key in detect_outputs:
            t = detect_outputs[cv3_key]
            sig_t = torch.sigmoid(t)
            print(f"  cv3 scale {scale} (cls): shape={list(t.shape)}, logit min={t.min():.4f}, max={t.max():.4f}")
            print(f"  cv3 scale {scale} (sig): min={sig_t.min():.4f}, max={sig_t.max():.4f}")

    # Export L0 output for comparison
    if layer0_output[0] is not None:
        out_path = Path(__file__).parent.parent / "models" / "ref_l0_output.bin"
        data = layer0_output[0][0].cpu().numpy().astype(np.float32)
        data.tofile(str(out_path))
        print(f"\nSaved L0 reference output: {out_path} ({data.size} values)")


if __name__ == "__main__":
    main()
