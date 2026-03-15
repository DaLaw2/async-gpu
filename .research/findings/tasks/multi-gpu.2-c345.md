# multi-gpu.2 — Device Enumeration + Validation

**Status**: complete
**Cycle**: 345

## What was done

Added multi-GPU device enumeration and validation API to `GpuRuntime`:

### API additions (`crates/core/gpu-host/src/runtime.rs`)

1. **`GpuRuntime::device_count() -> Result<usize>`** — static method returning available CUDA device count via `CudaDevice::count()`
2. **`GpuRuntime::device_ordinal(&self) -> usize`** — returns which device ordinal this runtime is bound to
3. **`GpuRuntime::device_name(&self) -> Result<String>`** — returns human-readable device name via `cudarc::driver::result::device::get_name()`
4. **Stored `ordinal: usize`** in `GpuRuntime` struct (set during `new()`)

### Integration test (`crates/core/gpu-host/tests/gpu_integration.rs`)

Added `test_multi_gpu_enumeration()`:
- Verifies `device_count() >= 1`
- Creates `GpuRuntime` on device 0, checks ordinal and name
- If `count >= 2`, creates a second runtime on device 1

### cudarc API used

- `CudaDevice::count()` → `cuDeviceGetCount` (safe wrapper, returns `i32`)
- `cudarc::driver::result::device::get(ordinal)` → `cuDeviceGet` (gets CUdevice handle)
- `cudarc::driver::result::device::get_name(cu_dev)` → `cuDeviceGetName` (128-byte buffer)

### Verification

- `cargo +stable check` passes (lib + tests)
- `cargo +stable fmt` applied
- No `anyhow` used — all errors go through `GpuHostError::Cudarc`
