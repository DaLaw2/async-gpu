## Current Focus
Kernel performance optimization (epic: kernel-perf). Closing the gap to cuBLAS/cuDNN
across GEMM, Attention, Conv2D, and memory-bound ops. Current task: perf-attn-v3.1
(investigating warp-cooperative attention design).

## Recent Decisions
- 2026-06-01: GEMM V4 with NVRTC + sm_86 + fast_math achieves 1760 GFLOPS (63% cuBLAS). Next step: cp.async 3-stage pipeline for 70%.
- 2026-05-31: Flash Attention V3 cooperative 4-thread-per-row gives 5.7x speedup (54% cuDNN). Need tiled GEMM for Q·K^T and P·V to reach 70%.
- 2026-05-30: Fused LayerNorm + residual add kernel implemented. Integration into GPT-2 pending.
- 2026-05-28: GPT-2 end-to-end at 25.1ms (from original 221ms). Bottleneck is now attention + conv.

## Tried & Rejected
- GEMM BK=16 tile: numerical issues with larger K-tiles, reverted to BK=8
- cp.async in inline PTX: needs NVRTC with sm_80 flag, can't use inline asm path
- Scalar dot-product attention: 4% of cuDNN, fundamentally wrong approach — need tiled GEMM

## Active Constraints
- NVRTC required for sm_80+ features (cp.async, async copy). Inline PTX path limited to sm_75 features.
- Float4 loads require 16-byte aligned addresses — weight pre-padding needed for non-aligned shapes.
- Winograd F(4×4,3×3) has numerical stability concerns for deep networks (error accumulation).

## Key Metrics
- SGEMM: 1760 GFLOPS @ 4096³ (63% cuBLAS 2800)
- Flash Attention: 54% cuDNN (V3 cooperative, seq=512)
- Conv2D: 13% cuDNN (im2col path, needs Winograd)
- LayerNorm: 157 GB/s (78% peak, single-pass Welford done)
- GPT-2 e2e: 25.1ms per forward pass (was 221ms)

## Next
1. Complete perf-attn-v3.1 (investigation: warp-cooperative attention design)
2. perf-fusion.1 (fused LN+residual) and perf-layernorm.2 (float4) are also active
3. After attention V3: cp.async for GEMM V4 (perf-gemm-v4.2)
4. Conv Winograd is the biggest remaining gap (13% → 70%)
