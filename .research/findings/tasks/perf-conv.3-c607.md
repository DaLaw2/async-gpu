# perf-conv.3 — Direct Conv Kernel for Non-3×3 Sizes

## Status: done
## Summary: Implemented direct convolution via NVRTC for 1×1 (as GEMM reshape) and arbitrary sizes (5×5, 7×7 via tiled shared memory kernel). Auto-routing in conv2d() dispatches 1×1 to reshape+matmul, 3×3/stride=1 to Winograd, others to direct tiled conv.

## Implementation
- `conv2d_1x1()`: Reshapes to matmul — input [N,H,W,C_in] → [N*H*W, C_in], weight [C_out,C_in] → GEMM
- `direct_conv2d`: Naive per-output-element kernel for small tensors
- `direct_conv2d_tiled`: Shared memory tiled kernel with register accumulation for larger tensors
- NVRTC compile with OnceLock caching (same pattern as Winograd)
- Routing: 1×1 → GEMM, 3×3/stride=1 → Winograd, others → direct tiled

## Files Changed:
- crates/core/gpu-host/src/nn/ops/conv.rs (+611 lines)
