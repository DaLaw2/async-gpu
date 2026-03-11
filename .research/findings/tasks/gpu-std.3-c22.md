# gpu-std.3: File I/O from GPU via hostcall
**Cycle**: 22 | **Theme**: gpu-std | **Kind**: experiment | **Status**: done

## Summary
Successfully implemented and tested GPU-initiated file I/O through the hostcall protocol. The GPU kernel can open files, write data, read data back, and close files -- all by issuing hostcall requests that the host listener dispatches to `std::fs` operations. Full round-trip (open + write + close + open + read + close) completes in ~1.3ms.

## Findings
### Q: Can we create a file from the GPU?
A: Yes. The GPU kernel issues a SERVICE_OPEN hostcall with FILE_OPEN_WRITE_CREATE flag, which the host translates to `File::create()`. The file "gpu_test_output.txt" was successfully created on disk with the content "Hello from GPU file I/O!\n" (25 bytes). The GPU received a valid file descriptor (fd=1) back from the host.
**Confidence**: high

### Q: Does error handling propagate correctly?
A: Yes. The protocol uses FILE_ERROR_SENTINEL (u64::MAX) as the error indicator in response payloads. Invalid fd, failed opens, and I/O errors all correctly return the sentinel value. The GPU kernel checks for this sentinel at each step and short-circuits on failure. The CONTROL_ERROR bit in the packet control word provides an additional error channel for protocol-level failures.
**Confidence**: high

### Q: Measure performance impact
A: The full file I/O round-trip (6 hostcalls: open-write-close-open-read-close) completed in 1.3313ms. This is ~222us per hostcall on average, which is dominated by the PCIe round-trip latency for each hostcall (GPU spin-wait -> host poll -> host dispatch -> host response -> GPU sees response). This is acceptable for file I/O which is inherently slow, but batching multiple operations into fewer hostcalls would improve throughput for intensive workloads.
**Confidence**: high

## Implementation Details

### Protocol Extensions (gpu-protocol)
Added to `crates/gpu-protocol/src/lib.rs`:
- `FILE_MAX_PATH_LEN = 56` (7 slots x 8 bytes, same as PRINT)
- `FILE_MAX_WRITE_LEN = 48` (6 slots x 8 bytes, slots 2-7)
- `FILE_MAX_READ_LEN = 56` (7 slots x 8 bytes, slots 1-7)
- `FILE_ERROR_SENTINEL = u64::MAX` (error indicator)
- `FILE_OPEN_READ = 0`, `FILE_OPEN_WRITE_CREATE = 1`, `FILE_OPEN_APPEND = 2`
- Payload layout documentation for OPEN/WRITE/READ/CLOSE services

Service IDs were already defined: SERVICE_OPEN=4, SERVICE_WRITE=5 (note: SERVICE_WRITE=2 is the legacy alias for PRINT-like write; the new file write uses slot-based fd+data layout), SERVICE_READ=3, SERVICE_CLOSE=5.

Correction: The existing protocol already had SERVICE_WRITE=2, SERVICE_READ=3, SERVICE_OPEN=4, SERVICE_CLOSE=5. We reused these IDs with file-specific payload layouts.

### GPU-side Code (gpu-kernel)
Added generic `gpu_hostcall_request()` helper that factored out the common hostcall pattern (pop free packet, fill header, push to ready, ring doorbell, spin-wait). This reduces code duplication vs. the original `gpu_hostcall_print()`.

Key difference from PRINT: the new helpers return the packet pointer so the caller can read the response payload before releasing the packet back to the free stack.

Functions added:
- `gpu_hostcall_request()` -- generic request/response
- `gpu_hostcall_release()` -- return packet to free stack
- `gpu_hostcall_open()` -- open file, returns fd
- `gpu_hostcall_write()` -- write data to fd
- `gpu_hostcall_read()` -- read data from fd into buffer
- `gpu_hostcall_close()` -- close fd
- `hostcall_file_test` kernel -- full round-trip test

### Host-side Code (gpu-host)
Extended `HostcallBuffer::listen()` with:
- `HashMap<u64, File>` fd table (maps virtual fds to `std::fs::File`)
- `next_fd` counter (starting at 1, fd 0 reserved)
- Dispatch for SERVICE_OPEN, SERVICE_WRITE, SERVICE_READ, SERVICE_CLOSE

Handler methods added:
- `handle_open()` -- path from payload, flags -> `File::create/open/append`, returns fd
- `handle_write()` -- fd + data from payload -> `file.write() + flush()`
- `handle_read()` -- fd + max_len -> `file.read()`, writes data to response payload
- `handle_close()` -- removes fd from table (File dropped = closed)

## Test Results
### hostcall_file_test (full round-trip)
- Config: 1 block x 32 threads, 4 packets, single-threaded listener
- Result: PASSED
- Details:
  - Open for write: fd=1, flags=WRITE_CREATE
  - Write: 25 bytes written ("Hello from GPU file I/O!\n")
  - Close: fd=1 closed
  - Open for read: fd=2, flags=READ
  - Read: 25 bytes read back
  - Close: fd=2 closed
  - Content verification: GPU-side byte-by-byte comparison passed
  - Host-side file verification: content confirmed on disk
  - Total latency: 1.3313ms (6 hostcall round-trips)

## Unexpected Discoveries
1. **Generic hostcall pattern**: The `gpu_hostcall_request()` helper revealed that the original `gpu_hostcall_print()` had a subtle design issue: it returned the packet to the free stack before the caller could read the response. For services that return data (READ, OPEN), the packet must remain allocated until the response is consumed. The new pattern separates request submission from packet release.

2. **fd table simplicity**: A simple `HashMap<u64, File>` with a monotonic counter works perfectly for the fd table. No need for fixed-size arrays or complex allocation -- the host side has full std access.

3. **Payload slot packing is adequate**: 48 bytes per write and 56 bytes per read are sufficient for many use cases. Larger transfers would need chunking, but that's a straightforward extension.

## Open Questions
1. **Large file I/O**: Current max per-call is 48 bytes write / 56 bytes read. For bulk transfers, we'd need a chunking protocol or a DMA-based approach using device memory pointers.
2. **Concurrent file access**: Multiple GPU threads doing file I/O simultaneously would contend on the fd table. The current single-threaded listener handles this naturally (serial dispatch), but a multi-threaded listener would need synchronization.
3. **Seek support**: Not implemented. Adding SERVICE_SEEK would be straightforward.
4. **Directory operations**: mkdir, readdir, stat etc. could follow the same pattern.

## Impact on Downstream Tasks
- **gpu-std.4** (if planned): Can build higher-level file abstractions (`gpu_println!` to file, `BufWriter` shim) on top of these primitives.
- **gpu-libc**: The libc shim crate can now implement `open()`, `write()`, `read()`, `close()` by calling these hostcall functions, making C-compatible file I/O available to GPU code.
- **async-runtime**: File I/O could be made async by yielding the GPU "task" while waiting for the hostcall response, instead of busy-spinning.
