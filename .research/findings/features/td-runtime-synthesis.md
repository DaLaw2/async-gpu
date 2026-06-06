# td-runtime — Runtime residency tracking and automatic transfers

## What shipped
`GpuArray<T>` in `crates/core/gpu-host/src/gpu_array.rs`:
- 4-state residency machine: HostOnly / Synced / DeviceOnly / HostDirty
- `Deref<Target=[T]>` with auto D2H sync on DeviceOnly reads
- `DerefMut` marking HostDirty for lazy re-upload
- `ensure_device(&Arc<CudaDevice>) -> Result<u64>` for lazy H2D
- 64 KiB threshold: MappedBuffer (zero-copy) vs CudaSlice (VRAM)
- `AsDevicePtr` trait for generic kernel binding
- Re-exported via `gpu_host::{GpuArray, AsDevicePtr, Residency}` and `async_gpu`

## Key decisions
- Struct bounds `T: Copy + Send + DeviceRepr + Unpin + 'static` (all GPU-useful primitives)
- `UnsafeCell` + `Cell<Residency>` for `Deref` compatibility, `!Sync` by design
- Deref panics on D2H failure; `try_sync_to_host()` for graceful handling

## What remains
- `GpuContext::bind()` / `mark_output()` convenience methods (Phase 3)
- `GpuArray` + `gpu::custom()` end-to-end demo with real kernel
- AutoScheduler integration with `par_map(&GpuArray<f32>)` (separate story)
