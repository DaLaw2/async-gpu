# compute-extract.2: Block-level reductions added to gpu_runtime::block
**Cycle**: 329 | **Theme**: compute-extract | **Kind**: experiment | **Status**: done

## Summary
Added 3 block-level reduction functions to `gpu_runtime::block` module: `reduce_sum_f32`,
`reduce_max_f32`, `reduce_min_f32`. Uses halving-stride tree reduction pattern with
shared memory and `bar.sync` barriers. Build verified on nvptx64.

## Functions Added

### gpu_runtime::block (3 new unsafe functions)
- `reduce_sum_f32(val, tid, block_size, smem_offset)` — parallel sum across block
- `reduce_max_f32(val, tid, block_size, smem_offset)` — parallel max across block
- `reduce_min_f32(val, tid, block_size, smem_offset)` — parallel min across block

All use the proven halving-stride pattern from `compute_gemm.rs::test_softmax`:
1. Write val to smem[tid]
2. bar_sync
3. Loop: stride = block_size/2 down to 1, active threads combine positions
4. bar_sync after each step
5. Broadcast result from smem[0]

### API Design Decisions
- `smem_offset` parameter allows multiple reductions in the same kernel with different
  shared memory regions (e.g., softmax needs both max and sum reductions)
- `block_size` parameter because it may not equal block_dim_x (e.g., 3D blocks)
- Result broadcast to ALL threads (extra bar_sync + read from smem[0])

## Build Verification
- `cargo +nightly-2026-03-11 build --release --target nvptx64-nvidia-cuda` — SUCCESS

## Impact on Downstream Tasks
- **compute-extract.3**: warp_softmax + warp_layer_norm can now be added to nn module
- **demo-pipeline.1**: block-level reductions enable full softmax demo
