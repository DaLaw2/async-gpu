# Theme Synthesis: coop-compute — Kernel-side compute library

## Status
Active. Investigation (coop-compute.1) and naive matmul MVP (coop-compute.2) complete.

## Key Findings
- cooperative_map_with_params is sufficient for kernel-side matmul: A=src, C=dst, B+dims in params[4].
- Naive triple-loop matmul verified on GPU: C[8x6] = A[8x4] x B[4x6], all 48 elements match CPU reference.
- No API changes needed — the existing cooperative infrastructure handles compute workloads naturally.
- Lane-0-only execution is correct but leaves 31/32 lanes idle per warp. Tiled optimization is next.
- Static AtomicU32 arrays work for input data in no_std kernels; f32 output via gpu::launch<f32> is seamless.

## Critical Path
coop-compute.1 (done) → coop-compute.2 (done) → coop-compute.3 (tiled, all 32 lanes)

## Design Decision
Row-major f32, row-striped warp partitioning, B pointer in params[3]. No new API needed.
