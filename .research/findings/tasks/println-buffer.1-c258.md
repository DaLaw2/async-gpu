# println-buffer.1: Investigate gpu-libc write() path
**Cycle**: 258 | **Theme**: println-buffer | **Kind**: investigation | **Status**: done

## Summary

Traced the full `println!()` call chain on GPU. The key finding is that **`println!()` does NOT go through gpu-libc `write()` at all**. There are two completely separate I/O paths:

1. **println! path** (patched std): `println!()` -> `_print()` -> `Stdout::write()` -> `gpu_stdout_write()` -> `gpu_hostcall_print()` (SERVICE_PRINT)
2. **File I/O path** (gpu-libc): `std::fs::File::write()` -> libc `write()` -> `gpu_hostcall_request(SERVICE_WRITE)` (SERVICE_WRITE)

Therefore, print_buffer integration must happen in `gpu-kernel-std`'s `gpu_stdout_write()`, NOT in gpu-libc.

## Call Chain

```
println!("hello")
  |
  v
std::io::_print(args)                          [std-patches/io_stdio.patch]
  |  #[cfg(target_arch = "nvptx64")] bypass OnceLock/ReentrantLock
  v
crate::sys::stdio::Stdout::new().write_fmt(args)  [patched-std/.../sys/stdio/cuda.rs]
  |  impl io::Write for Stdout
  v
gpu_stdout_write(buf.as_ptr(), buf.len())       [extern "C" declared in cuda.rs]
  |  linked from gpu-kernel-std
  v
gpu_stdout_write() in gpu-kernel-std/src/lib.rs  [line 20]
  |  reads STDIO_HOSTCALL_BUF: AtomicU64
  |  chunks into 56-byte pieces
  v
gpu_runtime::hostcall::gpu_hostcall_print(hc_buf, ptr, len)  [gpu-runtime/src/lib.rs:392]
  |  pops free packet, fills SERVICE_PRINT, pushes to ready stack
  |  rings doorbell, spin-waits for CONTROL_READY
  v
Host listener receives SERVICE_PRINT, prints to host stdout
```

## Global State

### Current globals (3 separate ones for the hostcall buffer):

| Location | Variable | Type | Purpose |
|---|---|---|---|
| `gpu-kernel-std/src/lib.rs:15` | `STDIO_HOSTCALL_BUF` | `AtomicU64` | hostcall buf ptr for println! path |
| `gpu-libc/src/hostcall_io.rs:12` | `HC_BUF` | `static mut *mut u8` | hostcall buf ptr for file I/O path |
| `gpu-runtime/src/lib.rs:989` | `PANIC_BUF` | `static mut *mut u8` | hostcall buf ptr for panic handler |
| `gpu-runtime/src/lib.rs:993` | `RESULT_BUF` | `static mut *GpuKernelResult` | kernel result for panic handler |

### Sideband pointer: NOT stored globally anywhere

The sideband pointer is only passed as a function parameter to `print_buffer::init()`, `print_buffer::print()`, and `print_buffer::flush()`. There is **no global storage** for it. The existing `print_buffer` API requires both `buf` and `sideband` to be passed explicitly by the kernel.

### Initialization pattern

Each kernel entry point manually calls:
```rust
stdio_init(buf);                    // sets STDIO_HOSTCALL_BUF
gpu_libc::gpu_libc_io_init(buf);    // sets HC_BUF (only if file I/O needed)
// print_buffer::init(sideband, thread_count);  // only in no_std kernels that opt in
```

## Integration Points

To make `println!()` use `print_buffer` instead of direct `gpu_hostcall_print()`:

### Option A: Add globals for sideband in gpu-kernel-std

Add two new globals to `gpu-kernel-std/src/lib.rs`:
```rust
static STDIO_SIDEBAND_PTR: AtomicU64 = AtomicU64::new(0);
static STDIO_PRINT_BUF_READY: AtomicU32 = AtomicU32::new(0);
```

Modify `gpu_stdout_write()` to check `STDIO_PRINT_BUF_READY`:
- If ready: call `print_buffer::print(hc_buf, sideband, ...)` (buffered, fast)
- If not ready: fall back to `gpu_hostcall_print()` (current behavior, works without sideband)

New init function:
```rust
fn stdio_print_buffer_init(buf: *mut u8, sideband: *mut u8, thread_count: u32) {
    STDIO_HOSTCALL_BUF.store(buf as u64, ...);
    STDIO_SIDEBAND_PTR.store(sideband as u64, ...);
    print_buffer::init(sideband, thread_count);
    STDIO_PRINT_BUF_READY.store(1, ...);
}
```

### Option B: Store sideband globally in gpu-runtime

Add a `sideband` module to `gpu-runtime` with global storage, so any code path can access it.

### Key concern: flush semantics

The `print_buffer` REQUIRES a manual `flush()` before kernel exit. With `println!()`, the user never calls flush. Solutions:
1. Kernel wrapper macro auto-inserts flush at end
2. Each `println!()` auto-flushes (defeats the batching purpose)
3. Flush on some heuristic (e.g., every N messages or every M bytes)
4. The host-side kernel launcher inserts a flush call

## Open Questions

1. **Who calls flush?** The current `print_buffer` API requires explicit `flush()` at kernel end. The `println!()` user never does this. A kernel wrapper / proc macro could insert it, but gpu-kernel-std kernels currently don't use one.

2. **Thread safety of globals**: `STDIO_HOSTCALL_BUF` uses `AtomicU64` (safe), but `print_buffer` slot access is per-thread (indexed by tid) with no locking. This is fine for the single-writer-per-thread model, but `Stdout::write()` calls could theoretically come from any context within a thread (e.g., nested formatting).

3. **Interaction with file I/O write()**: The gpu-libc `write()` sends ALL fds (including fd 1/2) through SERVICE_WRITE. If a user opens fd 1 via `std::io::stdout()` vs `File::create("/dev/stdout")`, they'd go through different paths. This is likely fine since no one does the latter on GPU.

4. **Multi-block sideband layout**: `print_buffer` uses `sideband + SIDEBAND_DATA_OFFSET + tid * 512` for per-thread slots. In multi-block launches, different blocks share the same sideband pointer. The `tid()` function only returns intra-block thread ID, so multiple blocks would collide on the same slots. This needs investigation for multi-block std kernels.

5. **Should gpu-libc write() also be buffered?** Currently gpu-libc write() for file I/O goes through SERVICE_WRITE (different from SERVICE_PRINT). If someone calls `write(STDOUT_FILENO, ...)` via libc, it won't be buffered. This is a separate path from println! and probably fine.
