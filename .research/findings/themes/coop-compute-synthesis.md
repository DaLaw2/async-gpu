# Theme Synthesis: coop-compute — Kernel-side compute library

## Status
Active. Investigation (coop-compute.1) complete. MVP experiment (coop-compute.2) next.

## Key Findings
- All existing GEMM kernels are host-launched entry points — none callable from within a running kernel.
- The cooperative_map_with_params API constrains kernel-side compute: single lane per warp, no shared memory, no bar.sync.
- Naive triple-loop matmul (f32, row-major) fits naturally: A=src, C=dst, B+dims in params[4].
- Performance estimate: ~0.3-0.6 GFLOPS (4 warps, lane-0 only). Sufficient for the litmus test demo.
- Upgrade path to tiled GEMM (all 32 lanes, register blocking) shares the same caller API.

## Critical Path
coop-compute.1 (done) → coop-compute.2 (implement naive MVP) → coop-compute.3 (tiled optimization)

## Design Decision
Row-major f32 for all matrices. Row-striped warp partitioning. B pointer encoded in params[3].
No new API needed — cooperative_map_with_params is sufficient for the MVP and the tiled version.
