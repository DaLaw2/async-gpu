#!/usr/bin/env python3
"""Export pretrained ResNet-18 weights to SafeTensors format.

Uses torchvision's ResNet-18 pretrained on ImageNet, then exports
all weights to SafeTensors for loading with the generic loader.

Usage:
    uv run --with torch --with torchvision --with safetensors scripts/export_resnet.py
"""
import os
import sys

import torch
import torchvision.models as models
from safetensors.torch import save_file

def main():
    # Load pretrained ResNet-18
    print("Loading pretrained ResNet-18...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()

    # Collect all parameters as float32 tensors
    tensors = {}
    for name, param in model.state_dict().items():
        tensors[name] = param.float().contiguous()
        print(f"  {name}: {list(param.shape)}")

    # Save to SafeTensors
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "resnet18.safetensors")
    save_file(tensors, out_path)

    total_params = sum(p.numel() for p in tensors.values())
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"\nSaved: {out_path}")
    print(f"Parameters: {total_params:,} ({size_mb:.1f} MB)")
    print(f"Keys: {len(tensors)}")

if __name__ == "__main__":
    main()
