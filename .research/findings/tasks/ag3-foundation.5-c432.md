# ag3-foundation.5: Benchmark GPT-2 nn API after weight pre-padding
**Cycle**: 432 | **Theme**: ag3-foundation | **Kind**: experiment | **Status**: done

## Summary
Linear weight pre-padding eliminates per-forward transpose+pad. GPT-2 drops from 164ms to 79ms/token.

## Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| GPT-2 nn API (non-cached) | 164ms/tok | **79ms/tok** | **2.1x** |
| GPT-2 raw kernels | 68ms/tok | 68ms/tok | baseline |
| nn/raw gap | 2.4x | **1.16x** | 84% gap closed |
| MNIST training | 7.9s (5.6x) | 7.9s (5.6x) | no change (uses direct matmul) |

## autograd-v3 criterion: GPT-2 < 120ms/token → **MET (79ms < 120ms)**
