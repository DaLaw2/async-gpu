#!/usr/bin/env python3
"""Quantize GPT-2 weights to INT4 with per-group scales.

Reads f32 safetensors, quantizes Linear weight matrices to INT4 (4-bit),
saves quantized weights + scales to a new safetensors file.

INT4 packing: 8 values per u32, unsigned [0,15] with zero_point=8.
Per-group quantization: group_size=128, scale = max(|group|) / 7.

Usage:
    uv run --with torch --with safetensors --with packaging scripts/quantize_gpt2_int4.py
"""
import os
import struct

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


def quantize_int4_per_group(weight, group_size=128):
    """Quantize a 2D weight matrix to INT4 with per-group scales.

    Args:
        weight: [out_features, in_features] float32
        group_size: number of elements per quantization group

    Returns:
        packed: [in_features // 8, out_features] uint32 (8 INT4 per u32)
        scales: [in_features // group_size, out_features] float32
    """
    out_f, in_f = weight.shape
    assert in_f % 8 == 0, f"in_features must be divisible by 8, got {in_f}"

    n_groups = (in_f + group_size - 1) // group_size
    k_packed = in_f // 8

    # Transpose to [in_features, out_features] for column-major access
    w_t = weight.T  # [in_f, out_f]

    scales = np.zeros((n_groups, out_f), dtype=np.float32)
    packed = np.zeros((k_packed, out_f), dtype=np.uint32)

    for g in range(n_groups):
        start = g * group_size
        end = min(start + group_size, in_f)
        group = w_t[start:end, :]  # [group_size, out_f]

        # Per-column max absolute value
        max_abs = np.max(np.abs(group), axis=0)  # [out_f]
        scale = np.where(max_abs < 1e-12, 1.0, max_abs / 7.0)
        scales[g, :] = scale

        # Quantize to [-8, 7], shift to [0, 15]
        for k in range(start, end):
            q = np.clip(np.round(w_t[k, :] / scale), -8, 7).astype(np.int32)
            q_unsigned = (q + 8).astype(np.uint32) & 0xF
            byte_idx = k // 8
            bit_pos = (k % 8) * 4
            packed[byte_idx, :] |= q_unsigned << bit_pos

    return packed, scales


def main():
    # Load GPT-2 safetensors
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    input_path = os.path.join(model_dir, "model.safetensors")
    output_path = os.path.join(model_dir, "model_int4.safetensors")

    if not os.path.exists(input_path):
        print(f"Model not found: {input_path}")
        print("Run: bash scripts/download-models.sh")
        return

    print(f"Loading: {input_path}")
    f = safe_open(input_path, framework="numpy")
    keys = sorted(f.keys())

    tensors = {}
    total_f32 = 0
    total_int4 = 0
    group_size = 128

    for key in keys:
        t = f.get_tensor(key)
        total_f32 += t.nbytes

        # Quantize weight matrices (2D, in_features divisible by 8)
        if t.ndim == 2 and t.shape[1] % 8 == 0 and "weight" in key and t.shape[1] >= 128:
            packed, scales = quantize_int4_per_group(t, group_size)
            tensors[f"{key}.int4_packed"] = packed
            tensors[f"{key}.int4_scales"] = scales
            total_int4 += packed.nbytes + scales.nbytes
            print(f"  INT4: {key} {list(t.shape)} → packed {list(packed.shape)} + scales {list(scales.shape)}")
        else:
            # Keep as-is (bias, embedding, etc.)
            tensors[key] = t.astype(np.float32)
            total_int4 += t.nbytes

    save_file(tensors, output_path)

    f32_mb = total_f32 / 1024 / 1024
    int4_mb = os.path.getsize(output_path) / 1024 / 1024
    ratio = f32_mb / int4_mb

    print(f"\nSaved: {output_path}")
    print(f"Size: {f32_mb:.1f} MB (f32) → {int4_mb:.1f} MB (INT4) — {ratio:.1f}x reduction")
    print(f"Keys: {len(tensors)}")


if __name__ == "__main__":
    main()
