#!/usr/bin/env python3
"""Run YOLOv8-nano with exact 640x640 square letterbox for comparison with Rust.

Usage:
    uv run --with ultralytics --with pillow --with packaging --with numpy scripts/yolo_reference_square.py
"""

import struct
import sys
from pathlib import Path

def main():
    try:
        from ultralytics import YOLO
        import torch
        import numpy as np
        from PIL import Image
    except ImportError as e:
        print(f"Missing: {e}")
        sys.exit(1)

    model = YOLO("yolov8n.pt")
    model.model.eval()

    # Load bus.ppm and do EXACT same letterbox as Rust code
    img_path = Path(__file__).parent.parent / "models" / "bus.ppm"
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size
    print(f"Original: {orig_w}x{orig_h}")

    target = 640
    scale = min(target / orig_w, target / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    pad_x = (target - new_w) // 2
    pad_y = (target - new_h) // 2
    print(f"Scale: {scale:.4f}, new: {new_w}x{new_h}, pad: ({pad_x}, {pad_y})")

    # Resize with nearest (to match Rust)
    resized = img.resize((new_w, new_h), Image.NEAREST)

    # Create 640x640 image filled with 0.5 gray (128 in uint8)
    letterboxed = Image.new("RGB", (target, target), (128, 128, 128))
    letterboxed.paste(resized, (pad_x, pad_y))

    # Convert to CHW tensor [1, 3, 640, 640] normalized to [0, 1]
    arr = np.array(letterboxed).astype(np.float32) / 255.0  # [640, 640, 3]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1, 3, 640, 640]
    print(f"Input tensor: {list(tensor.shape)}, min={tensor.min():.4f}, max={tensor.max():.4f}")

    # Run through the backbone manually, layer by layer
    backbone = model.model.model
    with torch.no_grad():
        x = tensor
        save = {}

        for i in range(22):  # layers 0-21
            layer = backbone[i]
            layer_type = type(layer).__name__

            if layer_type == "Concat":
                # Concat layers need input from multiple sources
                from_layers = layer.f  # e.g., [-1, 6]
                inputs = []
                for f in from_layers:
                    if f == -1:
                        inputs.append(x)
                    else:
                        inputs.append(save[f])
                x = torch.cat(inputs, dim=1)
            else:
                x = layer(x)

            save[i] = x

            if i <= 2 or i in [9, 15, 18, 21]:
                print(f"  L{i}: {list(x.shape)}, min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}")
                if i == 0:
                    ch0 = x[0, 0].flatten()
                    print(f"    First 8 (ch0): {ch0[:8].tolist()}")
                    print(f"    (0, 40): {x[0, 0, 0, 40].item():.6f}")

        # Detect head: takes [L15, L18, L21]
        detect = backbone[22]
        detect_input = [save[15], save[18], save[21]]

        # Hook detect outputs
        detect_outs = {}
        for s in range(3):
            for sub in range(3):
                def mk_hook(name):
                    def fn(mod, inp, out):
                        detect_outs[name] = out.detach()
                    return fn
                detect.cv2[s][sub].register_forward_hook(mk_hook(f"cv2.{s}.{sub}"))
                detect.cv3[s][sub].register_forward_hook(mk_hook(f"cv3.{s}.{sub}"))

        det_out = detect(detect_input)

    # Print detect head stats
    print("\n--- Detect head outputs ---")
    for s in range(3):
        cv2_k = f"cv2.{s}.2"
        cv3_k = f"cv3.{s}.2"
        if cv2_k in detect_outs:
            t = detect_outs[cv2_k]
            print(f"  cv2 scale {s}: {list(t.shape)}, min={t.min():.4f}, max={t.max():.4f}, mean={t.mean():.4f}")
        if cv3_k in detect_outs:
            t = detect_outs[cv3_k]
            sig = torch.sigmoid(t)
            print(f"  cv3 scale {s}: logit min={t.min():.4f}, max={t.max():.4f}, sig max={sig.max():.4f}")

    # Save L0 ref
    l0 = save[0]
    ref_path = Path(__file__).parent.parent / "models" / "ref_l0_square.bin"
    l0_np = l0[0].numpy().astype(np.float32)
    l0_np.tofile(str(ref_path))
    print(f"\nSaved L0 ref: {ref_path} ({l0_np.size} values)")

    # Print first 16 values for detailed comparison
    print("L0 first 16 (ch0, row 0):", l0[0, 0, 0, :16].tolist())


if __name__ == "__main__":
    main()
