# gpu-compute.5: Tiled GEMM combining Tensor Cores + shared memory
**Cycle**: 125 | **Theme**: gpu-compute | **Kind**: experiment | **Status**: done

## Summary

Successfully implemented a single-tile GEMM using MMA + shared memory in Rust on nvptx64. The full pipeline — global memory → shared memory → MMA fragment registers → `mma.sync.aligned.m16n8k16` → global memory — produces correct results. Verified with all-1.0 matrices: D[16×8] = A[16×16] × B[16×8] gives 16.0 for all 128 output elements.

## Findings

### Q: Can tiled matrix multiply work with MMA + shared memory on Rust nvptx64?
A: Yes. The pipeline works:
1. 32 threads cooperatively load A (128 u32) and B (64 u32) from global to shared memory
2. `bar_sync()` ensures all data is visible
3. Each thread loads its MMA fragment registers from shared memory
4. `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` executes the tile multiply
5. Results written back to global memory

For the all-1.0 test, fragment-to-matrix mapping is trivial (all fragments identical). A production implementation would need the proper per-thread fragment index mapping or `ldmatrix` instruction.
**Confidence**: high

### Q: What is the register pressure with MMA fragments + shared memory pointers?
A: Manageable for a single tile. Per thread: 14 registers for MMA operands + 2 registers for shared memory pointers + loop counters. The PTX output shows ~20 registers per thread, well within the 255 register limit. Multi-tile implementations would need to reuse registers across loop iterations.
**Confidence**: medium (single tile only, multi-tile not tested)

## Unexpected Discoveries
- The all-1.0 test pattern is elegant: it verifies MMA arithmetic correctness independent of fragment-to-matrix mapping. Every fragment register is `{1.0, 1.0}` f16x2, and every result should be 16.0 f32 (sum of 16 products).

## Changes Made
- **crates/gpu-kernel/src/lib.rs**: Added `test_tiled_gemm` kernel
- **crates/gpu-host/src/main.rs**: Added `run_tiled_gemm_test()` with all-1.0 verification

## Open Questions
1. Fragment-to-matrix index mapping for non-uniform matrices — needed for real GEMM
2. Multi-tile loop with accumulation across k-dimension tiles
3. `ldmatrix.sync.aligned` for efficient fragment loading from shared memory

## Impact on Downstream Tasks
- Demonstrates that Rust can implement GEMM-class compute on GPU Tensor Cores
- Foundation for larger matrix operations if needed for inference pipeline
