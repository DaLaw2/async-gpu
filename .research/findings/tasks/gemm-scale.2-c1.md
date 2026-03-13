# gemm-scale.2: Multi-block GEMM
**Cycle**: 1 | **Theme**: gemm-scale | **Kind**: experiment | **Status**: done

## Summary
Implemented multi-block GEMM where each CTA (128 threads, 4 warps) computes a 32×16 output tile. `blockIdx.x` selects the M-dimension row range. B matrix is shared across all blocks (loaded redundantly into each block's shared memory). Verified with 2 and 4 blocks, uniform and non-uniform data.

## Findings

### Q: Can blockIdx be used to partition the output matrix across CTAs?
A: Yes. Each block uses `block_m = blockIdx.x` to offset into A (rows `block_m*32..block_m*32+32`) and D (same row range). B is shared — all blocks read the same global B. The kernel is essentially `multi_warp_gemm` with A pointer offset by `block_m * 32 * k_half` and D output offset by `block_m * 32 * n_cols`.
**Confidence**: high

### Q: Does bar.sync correctly scope to block-local synchronization?
A: Yes. `bar.sync 0` (our `bar_sync()`) synchronizes all threads within a single CTA. Multiple blocks execute independently; each block has its own shared memory and barrier. No cross-block synchronization is needed for this tiling strategy.
**Confidence**: high

### Q: How to handle edge tiles when dimensions are not multiples of tile size?
A: Not addressed in this experiment — all test cases use M as a multiple of 32 and N=16 (exactly one N-tile). For gemm-scale.3 (768×768), M=768 is a multiple of 32 and N=768 requires N-tiling (48 tiles of 16 columns each), which will need additional grid dimension or loop.
**Confidence**: n/a

## Design Notes

- **Completion signaling**: Uses `atom.global.add.u32` so each block atomically increments a counter. Host checks `status >= num_blocks` to confirm all blocks completed.
- **Grid dim**: `(M/32, 1, 1)` — one block per 32-row stripe.
- **Shared memory**: Same as single-block (384 u32 = 1536 bytes per block).
- **No N-tiling yet**: Current kernel only handles N=16 (one MMA-tile width). gemm-scale.3 will need to tile along N as well.

## Unexpected Discoveries
None — this was a straightforward extension of gemm-scale.1.

## Open Questions
1. For 768×768, need to tile along N dimension too. Options: (a) grid_dim = (M/32, N/16, 1) with 2D block indices, or (b) single grid dim with linear block ID decoded to (block_m, block_n).

## Impact on Downstream Tasks
- **gemm-scale.3** (768×768 validation): Unblocked. Needs N-dimension tiling extension.
