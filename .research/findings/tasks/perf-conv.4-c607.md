# perf-conv.4: GPU-side batched conv (eliminate D2H/H2D round-trips)

## Status: DONE

## What changed

Replaced per-sample loops with single-kernel-launch batched convolution for all
three conv paths: Winograd (3x3), direct conv (5x5/7x7), and 1x1 GEMM.

### Files modified

- `crates/core/gpu-host/src/nn/ops/conv.rs` -- host-side batched conv routing
- `crates/core/gpu-host/src/nn/ops/winograd_f2x2.cu` -- CUDA kernel batch support

### 1. Winograd path (3x3 stride=1)

**Before**: `conv2d_batched_winograd` looped over N samples, calling
`conv2d_winograd_f2x2` per sample with D2H/H2D round-trips for each.

**After**: `winograd_conv2d_f2x2` kernel gains batch support via `grid.z = N`.
Each thread uses `blockIdx.z` as the batch index and computes per-sample
input/output offsets (`batch_idx * C_in * H * W`, `batch_idx * C_out * H_out * W_out`).
Single kernel launch processes all samples. Filter transform is still done once
(shared across batch).

New Rust function `conv2d_winograd_f2x2_impl` accepts `batch_size` parameter.
`conv2d_winograd_f2x2` is a thin wrapper with `batch_size=1`.

### 2. Direct conv path (5x5, 7x7, arbitrary)

**Before**: `conv2d_batched_direct` looped over N samples, calling
`conv2d_direct` per sample with D2H/H2D round-trips.

**After**: Both CUDA kernels (`direct_conv2d` and `direct_conv2d_tiled`) support
batch via `blockIdx.z = batch_idx * C_out + c_out`. The kernel decodes batch_idx
and c_out from `blockIdx.z` and applies per-sample offsets.

New Rust function `conv2d_direct_impl` accepts `batch_size` parameter.
`conv2d_direct` wraps with `batch_size=1`. Uses raw `Vec<*mut c_void>` launch
interface because 15 kernel parameters exceed cudarc's 12-tuple LaunchAsync
limit (this was actually a pre-existing compile bug with nn+cublas features).

### 3. 1x1 path

**Before**: `conv2d_batched_direct(is_1x1=true)` looped over N samples.

**After**: New `conv2d_1x1_batched` function transposes input from
`[N, C_in, H, W]` to `[C_in, N, H, W]` (= `[C_in, N*H*W]`), performs a
single matmul `W[C_out, C_in] x input[C_in, N*H*W]`, then transposes back
to `[N, C_out, H_out, W_out]`. For stride=1 padding=0 (common case), this is
just transpose + matmul + transpose.

### 4. Fallback im2col path

Cleaned up to use GPU transpose per sample (instead of CPU transpose), still
assembles the big column matrix on host before one GEMM. This path only runs
when cublas feature is disabled and kernel size is not 1x1.

## Pre-existing bug fixed

The `conv2d_direct` function had 15 kernel parameters passed as a tuple, but
cudarc 0.12 only implements `LaunchAsync` for tuples up to 12 elements. This
meant `conv2d_direct` never compiled with `nn+cublas` features. Fixed by
switching to the raw `&mut Vec<*mut c_void>` launch interface.

## Verification

- `cargo check -p gpu-host --features "nn,cublas"` -- compiles clean
- `cargo check -p gpu-host --features "nn"` -- compiles clean
- `bash scripts/ci-lint.sh` -- all checks pass
- All conv tests pass individually:
  - `test_conv2d_batched_matches_cpu` -- PASS (key test for batched changes)
  - `test_conv2d_3x3_matches_cpu` -- PASS (Winograd single-sample still works)
  - `test_conv2d_1x1_identity` -- PASS
  - `test_conv2d_multichannel` -- PASS
  - `test_conv2d_cifar10_dims` -- PASS

Note: parallel test run shows OnceLock race failures (pre-existing issue with
NVRTC compile caching in `OnceLock<bool>` -- unrelated to this change).

## Expected performance impact

For batch=32:
- **Winograd**: 32 kernel launches -> 1 launch. Saves ~160-320us launch overhead
  plus eliminates 32x D2H + H2D round-trips (~100us each = ~6.4ms saved).
- **Direct conv**: Same savings as Winograd.
- **1x1**: N kernel launches + N D2H/H2D -> 2 transposes + 1 matmul. Net savings
  from eliminating per-sample overhead.
- GPU scheduler can now see all batch work at once, improving SM utilization.
