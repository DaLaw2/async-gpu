# large-payload.3: Implement bulk read/write
**Cycle**: 71 | **Theme**: large-payload | **Kind**: experiment | **Status**: done

## Summary

Implemented the sideband buffer bulk data transfer system as designed in large-payload.2
(ADR-7). Added SERVICE_BULK_WRITE/READ to gpu-protocol, sideband allocator + GPU-side
API to gpu-runtime, sideband buffer allocation + I/O thread handlers to gpu-host, and a
4KB round-trip test kernel. All tests pass including the new bulk_io_test.

## Findings

### Q: Does 4KB+ file read work end-to-end?
A: **Yes.** The bulk_io_test kernel successfully:
1. Opens a file for writing via hostcall
2. Generates a 4096-byte test pattern (byte value = index & 0xFF)
3. Writes 4096 bytes via sideband bulk transfer (gpu_bulk_write)
4. Closes and reopens the file for reading
5. Reads 4096 bytes via sideband bulk transfer (gpu_bulk_read)
6. Verifies all 4096 bytes match the original pattern

Round-trip time: ~18ms (includes file I/O latency on host).

**Confidence**: high (verified end-to-end with content matching)

### Q: What is the throughput vs multi-packet approach?
A: With sideband, 4096 bytes transfer in a single hostcall round-trip (~18ms including
host file I/O). The multi-packet approach would require 4096/48 = 86 separate hostcall
round-trips for writing (48 bytes per FILE_WRITE packet), which at ~20µs per round-trip
minimum would take ~1.7ms just for protocol overhead, plus risk of pool exhaustion with
only 64 packets.

**Confidence**: medium (estimate, not measured for multi-packet)

### Q: Does it work under multi-thread contention?
A: Not yet tested with multiple concurrent threads. The bump allocator uses system-scope
atomic fetch_add, which should handle concurrent allocation correctly. Testing with
multiple threads would require a more complex kernel setup.

**Confidence**: medium (single-thread verified, multi-thread untested)

## Changes Made

### gpu-protocol/src/lib.rs
- Added `SERVICE_BULK_WRITE = 11` and `SERVICE_BULK_READ = 12`
- Added sideband buffer constants: `SIDEBAND_HEADER_SIZE`, `SIDEBAND_OFF_ALLOC`,
  `SIDEBAND_OFF_CAPACITY`, `SIDEBAND_DATA_OFFSET`, `DEFAULT_SIDEBAND_SIZE`
- Added payload layout documentation for bulk services

### gpu-runtime/src/lib.rs
- Added `sideband` module with:
  - `sideband_alloc()` — bump allocator using atomic fetch_add
  - `sideband_reset()` — reset allocator to zero
  - `gpu_bulk_write()` — write arbitrary data to file via sideband
  - `gpu_bulk_read()` — read arbitrary data from file via sideband
- Added sideband functions to prelude

### gpu-host/src/hostcall.rs
- Added `sideband_host_ptr`, `sideband_dev_ptr`, `sideband_size` fields to HostcallBuffer
- Split `new()` into `new()` (default 1MB sideband) + `new_with_sideband()`
- Sideband allocated as separate cuMemHostAlloc with DEVICEMAP|PORTABLE
- Added SERVICE_BULK_WRITE/READ to listener dispatch (slow path)
- Added `handle_bulk_write()` — reads data from sideband, writes to file
- Added `handle_bulk_read()` — reads from file, writes to sideband
- Both handlers include bounds checking against sideband capacity
- Updated Drop to free sideband buffer

### gpu-kernel/src/lib.rs
- Added `bulk_io_test` kernel: 4KB write→read→verify round-trip

### gpu-host/src/main.rs
- Added `run_bulk_io_test()` test function
- Registered before panic_test (which calls process::exit)

### gpu-host/kernel.ptx
- Regenerated with bulk_io_test kernel

## Impact on Downstream Tasks

- **large-payload theme**: All 3 tasks complete. Theme can be marked completed.
- **product.8**: Dependency satisfied — can now implement parallel file grep using
  gpu_bulk_read for reading file chunks.
- **gpu-libc**: Can route libc read/write through bulk path when length > 48 bytes.
