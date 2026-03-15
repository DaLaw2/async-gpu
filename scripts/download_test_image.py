#!/usr/bin/env python3
"""Download a COCO sample image and convert to PPM for YOLO testing.

Downloads the classic 'bus.jpg' image used in YOLO demos and converts
it to PPM format (no external image library needed on the Rust side).

Usage:
    uv run --with pillow --with requests scripts/download_test_image.py
"""

import sys
from pathlib import Path

def main():
    try:
        from PIL import Image
        import requests
        from io import BytesIO
    except ImportError as e:
        print(f"Missing dependency: {e}")
        sys.exit(1)

    url = "https://ultralytics.com/images/bus.jpg"
    print(f"Downloading {url}...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    img = Image.open(BytesIO(resp.content)).convert("RGB")
    print(f"Image size: {img.size} ({img.size[0]}x{img.size[1]})")

    out_dir = Path(__file__).parent.parent / "models"
    out_dir.mkdir(exist_ok=True)

    # Save as PPM (lossless, no dependency needed to read)
    ppm_path = out_dir / "bus.ppm"
    img.save(str(ppm_path), format="PPM")
    print(f"Saved: {ppm_path}")

    # Also save dimensions for reference
    print(f"Original dimensions: {img.size[0]}x{img.size[1]}")
    print("Expected detections: bus, person(s) — at least 3-4 objects")


if __name__ == "__main__":
    main()
