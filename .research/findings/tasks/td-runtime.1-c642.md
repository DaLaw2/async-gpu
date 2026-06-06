# td-runtime.1: Experiment — Lazy host-device transfer on kernel access

Status: COMPLETE
Task kind: experiment

## Summary

Implemented `GpuArray<T>` with a 4-state residency machine (`HostOnly`, `Synced`,
`DeviceOnly`, `HostDirty`) and automatic lazy host-device transfers. The type
provides `Deref<Target=[T]>` for transparent host reads and `DerefMut` for writes
that mark the host as dirty. Device uploads happen lazily on `ensure_device()`,
and device-to-host syncs happen automatically on `Deref` when the device has been
marked dirty.

## Findings

### 1. Size-threshold backend selection works cleanly

The 64 KiB threshold splits storage into two backends:
- **Below threshold**: `MappedBuffer` (pinned device-mapped memory, zero-copy).
  Host-to-device "copy" is a `ptr::copy_nonoverlapping` into the mapped buffer.
  Device-to-host "sync" reads from the same mapped memory.
- **At/above threshold**: `CudaSlice` (separate VRAM with explicit copies via
  `htod_sync_copy` / `dtoh_sync_copy`). Better for repeated GPU reads.

Both backends work correctly through the same state machine.

### 2. Trait bounds needed tightening

The design spec used `T: Copy + Send + 'static`, but cudarc's transfer APIs
(`htod_sync_copy`, `dtoh_sync_copy`) require `T: DeviceRepr + Unpin`. Since
`Deref` triggers D2H sync, these bounds must be on the struct itself. All
primitive numeric types satisfy `DeviceRepr + Unpin`, so this is not restrictive
in practice.

### 3. UnsafeCell + Cell pattern works for Deref

The `UnsafeCell<Vec<T>>` + `Cell<Residency>` combination is sound because:
- `sync_to_host()` only mutates `host` when `residency == DeviceOnly`, at which
  point no `&[T]` from a prior `deref()` can exist.
- `DerefMut` takes `&mut self`, providing exclusive access.
- The type is `!Sync` (via `Cell`), preventing concurrent access.

### 4. AsDevicePtr trait enables generic kernel binding

The `AsDevicePtr` trait (`ensure_device`, `device_len`, `mark_device_dirty`)
allows `GpuContext::bind()` integration in the future without modifying the
builder API.

### 5. All 21 tests pass on real GPU hardware

Tests cover: host-only lifecycle, size threshold selection, mapped backend,
device-memory backend, full lifecycle (create -> upload -> modify -> re-upload ->
kernel write -> sync), `try_sync_to_host`, zero-length arrays, and trait-object
dispatch.

## Open Questions

1. **CudaDevice caching**: `ensure_device(&dev)` requires the caller to pass an
   `Arc<CudaDevice>`. A future improvement could cache the device handle in
   `GpuArray` at construction time (from `GpuRuntime`).

2. **GpuContext integration**: `ctx.bind(&gpu_array)` and `ctx.mark_output()`
   convenience methods are not yet added to `GpuContext`. This is Phase 3 work
   from the design spec.

3. **Async kernel writes**: With async/await on GPU, `mark_device_dirty()` must
   happen after the kernel future resolves, not at launch time. This requires
   integration with the async executor (future work).
