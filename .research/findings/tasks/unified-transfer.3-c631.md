# unified-transfer.3: GpuVec integration with gpu::launch + par_iter kernels

**Status**: DONE
**Cycle**: 631

## What was implemented

### 1. `gpu::launch_with_gpuvec()` — zero-copy kernel launch with GpuVec I/O

New function in `crates/core/gpu-host/src/gpu.rs` that launches a kernel using
raw CUDA driver API (`cuLaunchKernel`) with `GpuVec` device pointers directly.
No `cudaMemcpy` needed — both input and output are pinned device-mapped memory.

Kernel signature expected: `fn(input: *const T, output: *mut T, n: u32)`

Also added `launch_with_gpuvec_cubin()` variant that accepts pre-compiled cubin
for fast loading (sub-second vs 10+ min PTX JIT).

### 2. `GpuVec::map_gpu()` — one-liner convenience method

New method on `GpuVec<T>` in `crates/core/gpu-host/src/memory.rs`:
- Creates output GpuVec of same size
- Launches kernel via `launch_with_gpuvec`
- Returns output (results immediately readable, zero-copy)

Also added `map_gpu_cubin()` variant for cubin fast-load path.

### 3. CUDA context auto-init in `MappedBuffer`

Added `ensure_cuda_context()` in `memory.rs` that lazily initializes CUDA via
`cuInit(0)` + `cuDevicePrimaryCtxRetain` + `cuCtxSetCurrent`. This allows
standalone `GpuVec` creation without requiring the caller to init CUDA first.
Uses `std::sync::Once` for thread-safe, idempotent initialization.

### 4. Integration tests (3 tests, all passing)

In `crates/core/gpu-host/tests/gpu_integration.rs`:
- `test_gpuvec_launch_zero_copy` — `launch_with_gpuvec` with 1K elements
- `test_gpuvec_map_gpu` — `GpuVec::map_gpu()` convenience with 2K elements
- `test_gpuvec_large_data` — 1M elements, multi-block grid-stride loop

Tests use inline PTX (~20 instructions) that JIT compiles in milliseconds,
avoiding the 10+ minute JIT of the full 8MB kernel_test.ptx.

### 5. Bug fixes (pre-existing)

- Fixed `scheduler.rs` accessing private `MODULE_SEQ` — now uses `fresh_module_name()`
- Made `fresh_module_name()` `pub(crate)` in `gpu.rs`
- Fixed clippy `manual_div_ceil` warnings in `gpu.rs` and `scheduler.rs`

## Key design decisions

1. **Raw CUDA driver API for launch**: cudarc's `LaunchAsync` trait requires
   `CudaSlice`, not raw device pointers. Since `GpuVec` wraps `MappedBuffer`
   (pinned host memory with a device pointer), we bypass cudarc and use
   `cuLaunchKernel` directly — same pattern as `GpuStdModule::launch_raw`.

2. **Cubin fast-path**: Added `_cubin` variants for both `launch_with_gpuvec`
   and `map_gpu` because the kernel_test.ptx is 8.3MB and JIT takes 10+ min.
   Existing cubin at `kernel_std.cubin` is stale (doesn't include multiblock
   kernel), so tests use inline PTX instead.

3. **Inline test PTX**: Rather than depending on the massive kernel_test.ptx or
   stale cubin, the tests embed a 20-instruction PTX kernel that computes the
   same `f(x) = x * 2.0 + 1.0` operation. JIT compiles in ~1ms.

## Verification

```
cargo check -p gpu-host          # OK
cargo clippy -p gpu-host -D warnings  # OK
3/3 new tests pass on GPU (GTX 1660, sm_75)
```

## Files changed

- `crates/core/gpu-host/src/gpu.rs` — `launch_with_gpuvec`, `launch_with_gpuvec_cubin`, `fresh_module_name` pub(crate)
- `crates/core/gpu-host/src/memory.rs` — `ensure_cuda_context`, `GpuVec::map_gpu`, `GpuVec::map_gpu_cubin`
- `crates/core/gpu-host/src/scheduler.rs` — use `fresh_module_name()`, fix div_ceil
- `crates/core/gpu-host/tests/gpu_integration.rs` — 3 new GpuVec integration tests
