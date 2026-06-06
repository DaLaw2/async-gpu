# td-design: Transparent Vec<T> Design Synthesis

## Design Complete
`GpuArray<T: Copy+Send+'static>` with 4-state residency machine
(HostOnly, Synced, DeviceOnly, HostDirty). Uses `UnsafeCell<Vec<T>>`
+ `Cell<Residency>` for interior mutability, enabling `Deref<Target=[T]>`
that auto-syncs device-to-host. `!Sync`, `Send`.

## Backend Selection
Size threshold at 64 KiB: below uses MappedBuffer (zero-copy PCIe),
above uses CudaSlice (explicit H2D/D2H copies, VRAM bandwidth).
`DeviceStorage` enum encapsulates the choice.

## Kernel Integration
`AsDevicePtr` trait with `ensure_device() -> Result<u64>` and
`mark_device_dirty()`. GpuContext gains `bind()` / `mark_output()`.
Deref panics on D2H failure; `try_sync_to_host()` for fallible path.

## Implementation Phases
1. Core type + Deref (host-only tests)
2. DeviceStorage + ensure_device (GPU integration test)
3. GpuContext bind/mark_output + example migration
4. AutoScheduler par_map overload (future story)

## Blocked By
- Scheduler par_map overload for GpuArray (Phase 4, separate story)
- Async executor integration for mark_device_dirty timing
