# showcase-examples.3 — Benchmark example updated with latest perf numbers

## Findings

### Summary
The benchmark example at `examples/std/benchmark/src/main.rs` was outdated: its summary
line reported V1 numbers (157 GFLOPS, 5.6% cuBLAS) while the actual `matmul()` API now
dispatches to V4.1 (2691 GFLOPS at 4096x4096x4096, ~90% cuBLAS). The "SGEMM V2" section
was redundant — it called `matmul_v2()` which has the same dispatch logic as `matmul()`,
so it measured the same code path as the first SGEMM section.

### Changes Made
1. **Replaced redundant V2 section** with a dedicated V4.1 section that calls
   `matmul_v4_1()` directly, isolating the NVRTC kernel performance with correctness
   checks vs cuBLAS.
2. **Updated summary lines** from stale V1 numbers to current metrics:
   - SGEMM V4.1: ~2691 GFLOPS at 4096^3 (~90% cuBLAS)
   - Flash Attention V3: 47-60% theoretical
   - Conv2D: 81-229% equivalent cuBLAS GEMM
   - LayerNorm: ~100% memory bandwidth
3. **Updated module docs** to list all benchmark sections with kernel version descriptions.
4. **Clarified SGEMM section header** to note it uses auto-dispatch.

### Dispatch Analysis
With `cublas` feature enabled (benchmark's Cargo.toml has it):
- `matmul()` -> `matmul_v2()` for all benchmark sizes (M>=4, N>=4, K>=4)
- `matmul_v2()` dispatch:
  - M<=256: cuBLAS fallback (covers GPT-2 shapes: 1x768, 128x768, etc.)
  - M>=512, N>=512, K>=256: V4.1 NVRTC (covers 512^3, 1024^3, 2048^3, 4096^3)
  - Else: V3/V2 PTX kernels

### Verification
- `cargo build` passes
- `cargo +stable fmt --check` passes
- `cargo +stable clippy -- -D warnings` passes (no new warnings)
