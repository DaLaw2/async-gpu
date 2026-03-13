# std-fs.2: Implement gpu-libc open/read/write/close via hostcall
**Cycle**: 188 | **Theme**: std-fs | **Kind**: experiment | **Status**: done

## Summary
Replaced gpu-libc's ENOSYS stubs for open/read/write/close with real hostcall implementations.
Added gpu-runtime + gpu-protocol as dependencies. New `hostcall_io` module routes I/O through
the hostcall protocol. Added `gpu_libc_io_init(buf)` to register the hostcall buffer.

## Findings

### Q: Can gpu-libc stubs call hostcall protocol without depending on gpu-runtime?
A: No — gpu-libc needs gpu-runtime for the hostcall request/release functions. Added as a
dependency (no circular dependency: gpu-kernel → gpu-runtime + gpu-libc, both are leaves).

**Confidence**: high

### Q: How does gpu-libc get access to the hostcall buffer pointer?
A: Via `gpu_libc_io_init(buf)` called at kernel entry. Stores in a `static mut HC_BUF`.
If not initialized, I/O functions return ENOSYS (graceful degradation).

**Confidence**: high

## Changes

### New: `gpu-libc/src/hostcall_io.rs`
- `gpu_libc_io_init(buf)` — register hostcall buffer
- `open()` — maps libc flags to FILE_OPEN_* constants, uses SERVICE_OPEN
- `write()` — uses SERVICE_WRITE, max FILE_MAX_WRITE_LEN per call
- `read()` — uses SERVICE_READ, copies response payload to caller buffer
- `close()` — uses SERVICE_CLOSE

### Error mapping
- GpuError categories mapped to libc errno (ERR_NOT_FOUND→ENOENT, etc.)
- If host provides raw_errno, uses it directly

### Flag mapping
- `O_RDONLY` → FILE_OPEN_READ
- `O_WRONLY|O_CREAT|O_TRUNC` → FILE_OPEN_WRITE_CREATE
- `O_APPEND` → FILE_OPEN_APPEND

## Impact on Downstream Tasks
- std-fs.4 (end-to-end File::create + write_all) — nearly unblocked (needs std-fs.3 for stdin)
