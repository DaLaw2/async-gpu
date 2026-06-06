# td-runtime.2: Experiment — Integrate GpuArray with gpu::custom() kernel launch

Status: COMPLETE
Task kind: experiment

## Summary

Integrated `GpuArray<T>` with the `gpu::custom()` builder API by adding
`bind_gpu_array()` and `device()` methods to `GpuContext`. Wrote 6 GPU
integration tests proving the full lifecycle: `GpuArray::from_vec()` ->
auto H2D via `bind_gpu_array()` -> kernel launch -> `mark_device_dirty()`
-> auto D2H via `Deref`. Zero explicit cudaMemcpy in user code.

## Findings

### 1. GpuContext::bind_gpu_array() is the natural integration point

Rather than modifying `CustomLaunchBuilder` to accept `GpuArray` args
directly (which would require changing the launch arg tuple type system),
the cleaner approach is a method on `GpuContext` that:
- Calls `AsDevicePtr::ensure_device(&self.dev)` to lazily upload
- Returns `u64` device pointer for passing to `launch()` as a scalar arg
- Works through `&dyn AsDevicePtr` for dynamic dispatch

This keeps the existing `launch()` signature intact and composes with
cudarc's `LaunchAsync` trait.

### 2. Both backends verified end-to-end with real GPU kernels

- **Small arrays** (< 64 KiB): MappedBuffer backend, zero-copy. Kernel
  reads/writes the same physical memory. D2H sync copies from mapped
  buffer into host Vec.
- **Large arrays** (>= 64 KiB): CudaSlice backend, explicit copies.
  H2D via `htod_sync_copy`, D2H via `dtoh_sync_copy`. Both work
  correctly through `bind_gpu_array()`.

### 3. Modify-reupload cycle works correctly

The HostDirty -> Synced transition via `bind_gpu_array()` correctly
re-uploads modified host data. Tested with:
1. First pass: input = 1.0 -> kernel -> output = 3.0
2. Modify input[0] = 10.0 (triggers HostDirty)
3. Second pass: bind_gpu_array re-uploads -> output[0] = 21.0

### 4. Inline PTX avoids 10-minute JIT compilation

Used a minimal ~20-instruction PTX kernel (`gpu_array_map_f32`) that
JIT-compiles in milliseconds. Same pattern as `gpu_integration.rs` tests.

### 5. All 27 tests pass (21 existing + 6 new)

New tests:
- `gpu_array_custom_launch_small` — 1024 elements, MappedBuffer backend
- `gpu_array_custom_launch_large` — 32768 elements, CudaSlice backend
- `gpu_array_modify_reupload_cycle` — modify -> re-upload -> verify
- `gpu_array_transparent_demo` — zero-cudaMemcpy demo
- `gpu_array_bind_via_trait` — dynamic dispatch via `&dyn AsDevicePtr`
- `gpu_array_try_sync_after_kernel` — fallible sync after real kernel

## Open Questions

1. **mark_device_dirty() is manual**: The user must call it after launch
   for output arrays. A future `launch_with_gpu_arrays()` method could
   automate this by accepting input/output array lists and calling
   `mark_device_dirty()` on outputs automatically.

2. **Device affinity**: `bind_gpu_array()` uses `GpuContext`'s device,
   but `GpuArray` caches the device handle from the first `ensure_device()`.
   If two `GpuContext`s use different devices, the second bind would use
   the cached device pointer from the first. This is correct for
   single-GPU use but needs attention for multi-GPU.
