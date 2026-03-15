# demo-pipeline.2: Implement multi-stage compute pipeline demo
**Cycle**: 332 | **Theme**: demo-pipeline | **Kind**: experiment | **Status**: done

## Summary
Implemented three compute demo kernels in `crates/kernel/gpu-kernel/src/compute_demo.rs`
plus host test in `tests_scaling.rs`. All compile to valid PTX.

## Kernels Implemented

### 1. compute_pipeline_demo (primary demo)
- 7-stage pipeline: generate data → softmax → GELU → reduce → converge → write → timing
- GPU-autonomous convergence loop (no host roundtrip per iteration)
- Uses: math::{sin,cos,abs}, nn::{warp_softmax,gelu}, warp::reduce_sum, index::{thread_idx_x,clock_nanos}
- Launch: (1,1,1) grid, (32,1,1) block (one warp)
- Status buffer reports: iterations, GPU nanoseconds, done flag

### 2. block_softmax_demo
- Block-level softmax using shared memory reduction
- Uses: block::{reduce_max_f32,reduce_sum_f32}, math::exp_f32
- Launch: (1,1,1) grid, (N,1,1) block, N*4 shared mem bytes

### 3. warp_layer_norm_demo
- Warp-level layer normalization
- Uses: nn::warp_layer_norm_f32
- Launch: (1,1,1) grid, (32,1,1) block

## Host Test
- `run_compute_pipeline_demo_test()` in tests_scaling.rs
- Allocates mapped memory for output (32 f32) and status (4 u32)
- Polls done flag, reads timing, verifies convergence
- Prints estimated CUDA launch overhead savings

## Build Verification
- gpu-kernel compiles with all 3 new kernels
- gpu-host compiles with new test
- kernel.ptx contains compute_pipeline_demo entry point
- ONLY_TEST="compute" dispatch added to main.rs

## Note
GPU execution not verified in this session (requires CUDA hardware).
Host test can be run with: `ONLY_TEST=compute cargo run -p gpu-host`

## Impact on Downstream Tasks
- **demo-pipeline.3**: Can benchmark once GPU test runs successfully
