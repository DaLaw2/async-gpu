# Transparent Data

GpuArray<T> with automatic host-device synchronization — zero explicit cudaMemcpy.

## What It Demonstrates

- `GpuArray::from_vec()` wraps a normal `Vec<T>` as GPU-aware data
- `GpuArray::zeroed()` allocates an output buffer
- `gpu::custom()` builder with inline PTX for kernel launch
- `bind_gpu_array()` triggers automatic host-to-device transfer
- `mark_device_dirty()` signals that the kernel wrote to a buffer
- `Deref` on `GpuArray` triggers automatic device-to-host sync
- Residency state machine: HostOnly -> Synced -> DeviceOnly -> Synced

## Running

```bash
cargo run -p transparent-data --release
```

## Key Results

Host-side code never calls cudaMemcpy, htod, or dtoh explicitly. Data flows automatically between host and device based on access patterns.
