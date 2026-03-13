# std-fs.4: End-to-end File::create + write_all + read with error handling
**Cycle**: 190 | **Theme**: std-fs | **Kind**: experiment | **Status**: done

## Summary
Implemented full std::fs support on GPU via a new CUDA fs PAL module that routes
File::open/read/write/close through gpu-libc's hostcall I/O. Also added a CUDA-specific
io/error module for proper errno retrieval. End-to-end test passes: GPU kernel writes a
file using std::fs::File::create() + write_all(), reads it back with read_to_end(),
and verifies content — all via hostcall protocol.

## Findings

### Q: Does std::fs::File::create() work on GPU via hostcall?
A: YES. Created `patched-std/library/std/src/sys/fs/cuda.rs` which wraps
gpu-libc's `open/read/write/close` extern functions. File::create maps to
`open(path, O_WRONLY|O_CREAT|O_TRUNC, 0o666)` → hostcall SERVICE_OPEN.
write_all loops through the write() method → hostcall SERVICE_WRITE.
File::open maps to `open(path, O_RDONLY, 0o666)` → hostcall SERVICE_OPEN.
read_to_end reads chunks → hostcall SERVICE_READ until EOF.
Drop closes the fd → hostcall SERVICE_CLOSE.

**Confidence**: high (tested end-to-end, 27-byte file round-trip verified)

### Q: Do file I/O errors produce proper std::io::Error?
A: YES. Created `patched-std/library/std/src/sys/io/error/cuda.rs` which reads
errno from gpu-libc's `__errno_location()` (exported as `#[no_mangle] extern "C"`).
Maps errno values to ErrorKind (ENOENT→NotFound, EACCES→PermissionDenied, etc.).
Previously cuda used the generic module (errno always 0, all errors Uncategorized).

**Confidence**: high

## Changes

### New: `patched-std/library/std/src/sys/fs/cuda.rs`
- File struct wrapping libc fd (i32)
- OpenOptions with proper flag mapping (O_RDONLY, O_WRONLY, O_CREAT, O_TRUNC, O_APPEND, O_EXCL)
- read/write via libc extern functions → hostcall protocol
- Drop impl calls close()
- Unsupported ops (stat, readdir, symlink, etc.) return proper unsupported() errors
- Placeholder types for FileAttr, FileType, FilePermissions, etc.

### New: `patched-std/library/std/src/sys/io/error/cuda.rs`
- errno() reads from `__errno_location()` (gpu-libc)
- decode_error_kind maps Linux errno → io::ErrorKind
- error_string provides human-readable error messages

### Modified: `patched-std/library/std/src/sys/fs/mod.rs`
- Added `target_os = "cuda" => { mod cuda; use cuda as imp; }` before catch-all

### Modified: `patched-std/library/std/src/sys/io/error/mod.rs`
- Moved cuda from generic group to its own arm

### Modified: `crates/gpu-libc/src/types.rs`
- Added O_EXCL constant (0o200) for create_new support

### Modified: `crates/gpu-kernel-std/src/lib.rs`
- Added `std_file_io_test` kernel demonstrating std::fs on GPU

### Modified: `crates/gpu-host/src/tests_std.rs`
- Added `run_std_file_io_test()` host-side test function

### Modified: `crates/gpu-host/src/main.rs`
- Added KERNEL_STD_PTX constant and std_fs test entry

## Impact on Downstream Tasks
- std-fs theme is now COMPLETE (all 3 success criteria met)
- std-migration.3 (async pipeline with std) is unblocked
