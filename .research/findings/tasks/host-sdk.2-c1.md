# host-sdk.2: Extract GpuRuntime + MappedBuffer from Test Code
**Cycle**: 1 | **Theme**: host-sdk | **Kind**: experiment | **Status**: done

## Summary
Created two new public modules in gpu-host: `runtime.rs` (GpuRuntime wrapper around CudaDevice) and `memory.rs` (MappedBuffer<T> RAII wrapper for pinned mapped memory). Both compile and pass lint. Existing test binary continues to build unchanged.

## Findings

### Q: Can the test infrastructure be cleanly separated into library API + test binary?
A: Yes. The existing `lib.rs` already separated `error` and `hostcall` modules. Adding `runtime` and `memory` modules was clean. The test modules (`tests_*.rs`) remain `pub(crate)` in `main.rs` and use internal helpers — they are unaffected by the new public API.
**Confidence**: high

### Q: What refactoring is needed to make hostcall buffer management reusable?
A: Minimal. `HostcallBuffer` was already public. The new `MappedBuffer<T>` replaces the raw `alloc_mapped_*` / `free_mapped_*` functions with a typed RAII wrapper. The old functions remain in `mapped_mem.rs` as `pub(crate)` for backward compatibility with existing test code.
**Confidence**: high

## New Public API Surface

### `runtime::GpuRuntime`
- `new(ordinal)` — init CUDA device
- `device()` / `device_arc()` — access underlying CudaDevice
- `load_ptx(src, module, fn_names)` — load PTX module
- `get_func(module, func)` — get kernel function handle
- `alloc_zeros<T>(len)` — allocate zeroed device memory
- `htod_sync_copy(data)` / `dtoh_sync_copy(buf)` — data transfer
- `launch_config(grid, block, smem)` — create LaunchConfig
- `synchronize()` — wait for kernels

### `memory::MappedBuffer<T>`
- `new_zeroed(len)` — allocate pinned mapped buffer
- `dev_ptr()` — device pointer for kernel args
- `host_ptr()` — raw host pointer
- `read(idx)` / `write(idx, val)` — volatile access
- `as_slice()` / `as_mut_slice()` — slice views
- `Drop` — automatic `cuMemFreeHost`

## Design Notes
- `GpuRuntime` does NOT wrap `launch()` with a generic param bound — cudarc's `LaunchAsync` trait is the right level for that. Users call `get_func()` + `f.launch(cfg, params)` directly for maximum flexibility.
- `MappedBuffer<T>` uses `assert!` for bounds checking (panics on OOB). This is deliberate — out-of-bounds access to GPU-mapped memory would cause UB, so panic is the correct behavior.
- `ValidAsZeroBits` bound on `alloc_zeros` comes from cudarc — only types that are valid when zero-filled can be allocated this way.

## Impact on Downstream Tasks
- **host-sdk.3**: Ready to create standalone example using the SDK
- **model-loading**: Can use `GpuRuntime` for weight upload
- **full-inference**: Can use `GpuRuntime` for the inference pipeline
