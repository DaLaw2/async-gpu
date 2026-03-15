#!/usr/bin/env python3
"""Export a test image for YOLO inference validation.

Generates a 640x640 test image with colored rectangles (simulating objects)
and saves it as raw f32 CHW binary and as PPM.

Usage:
    python scripts/export_test_image.py

Outputs:
    models/test_image.ppm       (PPM P6 binary, 640x640)
    models/test_image_chw.bin   (raw f32 CHW [3, 640, 640])
"""

import struct
from pathlib import Path


def main():
    w, h = 640, 640
    # Create a scene with colored rectangles on gray background
    pixels = bytearray(w * h * 3)

    # Fill with gray background
    for i in range(w * h):
        pixels[i * 3] = 128
        pixels[i * 3 + 1] = 128
        pixels[i * 3 + 2] = 128

    def fill_rect(x1, y1, x2, y2, r, g, b):
        for y in range(max(0, y1), min(h, y2)):
            for x in range(max(0, x1), min(w, x2)):
                idx = (y * w + x) * 3
                pixels[idx] = r
                pixels[idx + 1] = g
                pixels[idx + 2] = b

    # Draw several "objects" as colored rectangles
    # Red rectangle (simulating a car-like object)
    fill_rect(50, 200, 250, 400, 200, 30, 30)
    # Blue rectangle (simulating a person-like object)
    fill_rect(300, 100, 400, 500, 30, 30, 200)
    # Green rectangle (simulating another object)
    fill_rect(450, 300, 600, 500, 30, 200, 30)
    # Yellow rectangle
    fill_rect(100, 50, 200, 150, 200, 200, 30)
    # Purple rectangle
    fill_rect(400, 50, 550, 200, 150, 30, 150)

    out_dir = Path(__file__).parent.parent / "models"
    out_dir.mkdir(exist_ok=True)

    # Save as PPM
    ppm_path = out_dir / "test_image.ppm"
    with open(ppm_path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode())
        f.write(bytes(pixels))
    print(f"Saved PPM: {ppm_path}")

    # Save as raw f32 CHW binary
    chw_path = out_dir / "test_image_chw.bin"
    chw_data = bytearray(3 * h * w * 4)
    for c in range(3):
        for y in range(h):
            for x in range(w):
                val = pixels[(y * w + x) * 3 + c] / 255.0
                offset = (c * h * w + y * w + x) * 4
                struct.pack_into("<f", chw_data, offset, val)
    with open(chw_path, "wb") as f:
        f.write(chw_data)
    print(f"Saved CHW binary: {chw_path} ({len(chw_data)} bytes)")


if __name__ == "__main__":
    main()
