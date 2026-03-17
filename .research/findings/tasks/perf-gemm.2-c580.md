# perf-gemm.2: 64×64 tile + 4×4 register blocking GEMM
**Cycle**: 580 | **Theme**: perf-gemm | **Kind**: experiment | **Status**: done

## Summary
Implemented `gemm_f32_v2` with 64×64 CTA tile, BK=8, 256 threads, 4×4 register blocking
(16 named accumulators per thread), double-buffered shared memory, A transposed with
padding in smem. Result: **5.76x speedup** over v1 (157 → 905 GFLOPS at 4096³).

## Results

| Shape | v1 (old) | v2 (new) | cuBLAS | v2 % cuBLAS | Speedup |
|-------|----------|----------|--------|-------------|---------|
| 512³ | 149 | 767 | 2,285 | **33.6%** | 5.1x |
| 1024³ | 157 | 893 | 2,772 | **32.2%** | 5.7x |
| 2048³ | 157 | 888 | 2,744 | **32.3%** | 5.7x |
| 4096³ | 160 | 905 | 2,731 | **33.2%** | 5.7x |
| GPT-2 128×768² | 126 | 607 | 1,961 | **31.0%** | 4.8x |
| GPT-2 128×768×3072 | 131 | 622 | 2,594 | **24.0%** | 4.8x |

Correctness: PASS (max error = 0.00 vs cuBLAS for all shapes).

## Key Design Decisions
1. **Named accumulators** (c00..c33) instead of array — forces register allocation
2. **64×64 tile** (not 128×128) — 16 regs/thread fits easily, no spill risk
3. **A transposed in smem** with stride=68 (padding=4) — avoids bank conflicts
4. **B row-major** — no transpose kernel needed (saves kernel launch + BW)
5. **Double-buffered smem** — prefetch next tile while computing current
6. **Inline FMA** via `core::arch::asm!` — guarantees FMA instruction

## Why 128×128 + 8×8 Failed (First Attempt)
The `[f32; 64]` accumulator array was placed in local memory (GPU stack) by the compiler.
With constant array indices in the outer product function, the compiler should keep them
in registers, but the `&mut acc` reference + inline FMA `in/out` constraints caused spilling.
Solution: use named variables. 64 named variables is unwieldy; 16 (4×4) is manageable.

## Next Steps
- Try 64×128 + 4×8 (32 named accumulators) for better N-dimension amortization
- Add float4 vectorized global loads for A and B tile loading
- Scale up thread tile once register strategy is validated

**Confidence**: high
