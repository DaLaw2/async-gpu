# gpu-error-propagation.2: Implement GPU-side error checking in hostcall helpers
**Cycle**: 184 | **Theme**: gpu-error-propagation | **Kind**: experiment | **Status**: done

## Summary
Changed all hostcall functions from returning `bool`/`(ptr, bool)` to returning `Result<T, GpuError>`.
Defined `GpuError` and `GpuKernelResult` types in gpu-protocol. Updated all callers across
gpu-runtime, gpu-kernel, and gpu-kernel-std. Both host and kernel code compiles cleanly.

## Findings

### Q: Can hostcall helpers return Result instead of bool?
A: YES. Changed `gpu_hostcall_request` from `(*mut u8, bool)` to `Result<*mut u8, GpuError>` and
`gpu_hostcall_print` from `bool` to `Result<(), GpuError>`. The Result-based API is cleaner:
- Pool exhaustion → `Err(GpuError::pool_exhausted())`
- Timeout → `Err(GpuError::timeout())`
- Host-side error → `Err(GpuError::from_encoded(slot0))` (decoded from CONTROL_ERROR + payload)

Key improvement: `gpu_hostcall_request` now auto-detects CONTROL_ERROR, decodes the error category
and errno from payload slot 0, releases the packet, and returns `Err(GpuError)`. Callers no longer
need to manually check success flag and decode errors.

Updated callers:
- `gpu-runtime::sideband` (gpu_bulk_write, gpu_bulk_read)
- `gpu-kernel::helpers` (open, write, close, read, stdin_read, time, grep_buffer)
- `gpu-kernel::hostcall_kernels` (3 print calls)
- `gpu-kernel-std::lib` (gpu_stdout_write)

**Confidence**: high (compiles for both nvptx64 and host targets)

### Q: Does checking CONTROL_ERROR add measurable overhead?
A: The check is a single `if ctrl & CONTROL_ERROR != 0` branch in the existing spin-wait loop,
plus a 64-bit read of payload slot 0 on the error path only. This adds no overhead on the happy
path — the branch is perfectly predicted (errors are rare). The error-path cost (one read + packet
release + struct construction) is negligible compared to the hostcall round-trip latency (~µs).

**Confidence**: high (analysis-based, not measured — error path is cold)

## Types Added

### GpuError (gpu-protocol, 4 bytes)
```rust
pub struct GpuError { pub category: u16, pub raw_errno: u16 }
```
Constructors: `new()`, `from_encoded()`, `pool_exhausted()`, `timeout()`

### GpuKernelResult (gpu-protocol, 64 bytes, cache-line aligned)
```rust
pub struct GpuKernelResult {
    tag: u32, category: u16, raw_errno: u16,
    thread_idx: u16, block_idx: u16, msg_len: u32,
    msg_bytes: [u8; 48],
}
```
TAG_OK=0, TAG_ERR=1, TAG_UNINIT=0xDEAD_BEEF

## Impact on Downstream Tasks
- gpu-error-propagation.3 unblocked: host-side can now read GpuKernelResult after launch
- gpu-error-propagation.4 unblocked: panic handler can write to GpuKernelResult buffer
- std-migration.3 unblocked: async pipeline kernels can use `?` with hostcall Results
