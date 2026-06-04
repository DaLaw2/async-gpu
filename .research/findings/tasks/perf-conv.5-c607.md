# perf-conv.5 — Conv Pipeline Integration & Verification

## Routing Logic

`conv2d()` auto-routes based on kernel size and stride:

| Condition | Single-sample path | Batched path |
|---|---|---|
| 1×1 kernel | `conv2d_1x1` (GEMM reshape) | `conv2d_1x1_batched` (batched GEMM) |
| 3×3, stride=1 | `conv2d_winograd_f2x2` (cublas) | `conv2d_batched_winograd` (cublas) |
| Other (5×5, 7×7, 3×3 stride>1) | `conv2d_direct` (cublas) | `conv2d_direct_impl` with batch (cublas) |
| Fallback (no cublas) | im2col + GEMM | im2col per-sample + batched GEMM |

4D input `[N, C, H, W]` auto-delegates to `conv2d_batched()`.

## Bug Fix: OnceLock Caching

**Found and fixed**: Both Winograd and direct conv NVRTC compilation used
`static OnceLock<bool>` to cache "compiled" state. This broke when multiple
`CudaDevice` handles existed (e.g., different unit tests in the same process).
The PTX was loaded into the first device only, but `OnceLock` prevented
reloading for subsequent devices.

**Fix**: Replaced `OnceLock` with `dev.get_func(...).is_none()` checks.
Now NVRTC compilation happens once per device handle, correctly.

Files changed:
- `crates/core/gpu-host/src/nn/ops/conv.rs` (two OnceLock → get_func patterns)
- `crates/core/gpu-host/tests/conv2d_bench.rs` (removed unused `Arc` import)

## Correctness Tests — All Pass

| Test | Result | Max Error |
|---|---|---|
| `test_conv2d_1x1_identity` | OK | < 1e-3 |
| `test_conv2d_3x3_matches_cpu` | OK | < 1e-2 |
| `test_conv2d_multichannel` | OK | < 0.1 |
| `test_conv2d_cifar10_dims` | OK | 2.4e-7 |
| `test_conv2d_batched_matches_cpu` | OK | 3.0e-7 |
| `test_conv2d_gradient_check` | OK | — |
| `winograd_f2x2_correctness` (6 sub-cases) | OK | 0.0 |
| `test_resnet18_forward_cifar10` | OK | finite |
| `bench_conv2d_shapes` | OK | — |

## Model Integration

- **GPT-2**: No conv2d usage (pure transformer). Confirmed clean.
- **ResNet-18**: Uses `Conv2d` layer throughout.
  - conv1: 3×3 stride=1 → Winograd path
  - BasicBlocks: 3×3 stride=1 → Winograd; 3×3 stride=2 → direct tiled
  - Shortcut: 1×1 → GEMM reshape path
- **YOLOv8-nano**: Uses `Conv2d` via `ConvBnSilu` and `ConvBias`.
  - Stem/downsampling: 3×3 stride=2 → direct tiled
  - C2f bottlenecks: 3×3 stride=1 → Winograd
  - 1×1 pointwise convs → GEMM reshape
  - Detect head final: 1×1 → GEMM reshape

All model paths correctly route through optimized kernels when cublas feature
is enabled.

## Benchmark Results (GTX 1660, release mode)

| Shape (Cin→Cout, HxW, K, S) | Time (ms) | GFLOPS |
|---|---|---|
| 3→64, 224×224, 3×3, s1 | 0.63 | 276 |
| 64→64, 56×56, 3×3, s1 | 3.24 | 71 |
| 128→128, 28×28, 3×3, s1 | 3.26 | 71 |
| 256→256, 14×14, 3×3, s1 | 2.90 | 80 |
| 512→512, 7×7, 3×3, s1 | 24.2 | 10 |
| 32→32, 32×32, 3×3, s1 | 0.24 | 78 |
| 16→32, 32×32, 3×3, s2 | 0.04 | 56 |
| 3→16, 640×640, 3×3, s2 | 0.88 | 100 |
| 64→64, 80×80, 3×3, s1 | 5.32 | 89 |

GTX 1660 theoretical peak: 5000 GFLOPS FP32. cuDNN typically 60-80% of peak.

Current performance: **1.5-5.5% of theoretical peak** for most shapes. This
is expected for the current Winograd kernel architecture which:
- Processes C_in channels serially per thread (no shared-memory reduction)
- Does not use tensor cores
- Has no register tiling for the GEMM stage

The **50% of cuDNN target is not yet met**. Key bottlenecks for future work:
1. Winograd element-wise multiply should be batched GEMM (cuBLAS) not per-tile
2. Direct conv needs input channel parallelism (reduction across warps)
3. Large shapes (512×512, 7×7) need better occupancy tuning

## im2col Fallback

The im2col + GEMM path remains as fallback for when the `cublas` feature is
disabled. It is not dead code — it provides baseline functionality without
NVRTC. Added clarifying comment marking it as fallback-only.

## CI Lint

`scripts/ci-lint.sh` passes cleanly.
