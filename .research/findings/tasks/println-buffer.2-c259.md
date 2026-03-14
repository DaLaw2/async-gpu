# println-buffer.2: Implement buffered println — modify gpu_stdout_write() to use print_buffer
**Cycle**: 259 | **Theme**: println-buffer | **Kind**: experiment | **Status**: done

## Summary

Integrated `print_buffer` with `gpu_stdout_write()` in `gpu-kernel-std` so that `println!()` on GPU automatically uses buffered print when initialized. Added `stdio_print_buffer_init()` and `gpu_print_buffer_flush()` functions, plus a test kernel `std_buffered_println_test`.

## Changes

### crates/gpu-kernel-std/src/lib.rs
- Added `STDIO_SIDEBAND_PTR: AtomicU64` and `STDIO_PRINT_BUF_READY: AtomicU32` globals
- Modified `gpu_stdout_write()` to check `STDIO_PRINT_BUF_READY`:
  - **Fast path**: If ready, routes through `gpu_runtime::print_buffer::print()` (buffered, auto-flush on full)
  - **Slow path**: Falls back to direct `gpu_hostcall_print()` (56-byte chunks, one hostcall per chunk)
- Added `stdio_print_buffer_init(buf, sideband, thread_count)`: sets globals, calls `print_buffer::init()`, sets ready flag
- Added `gpu_print_buffer_flush()`: flushes print buffer for calling thread, no-op if not initialized
- Added `std_buffered_println_test` kernel: 6 println!() calls routed through print_buffer

### crates/gpu-host/src/tests_std.rs
- Added `run_std_buffered_println_test()`: launches std_buffered_println_test kernel, verifies 6+ messages received via SERVICE_BULK_PRINT

### crates/gpu-host/src/main.rs
- Registered `run_std_buffered_println_test` in test suite

### crates/gpu-host/kernel_std.ptx
- Rebuilt with new functions (stdio_print_buffer_init, gpu_print_buffer_flush, std_buffered_println_test all present)

## PTX Verification

All new symbols confirmed in PTX output:
- `STDIO_SIDEBAND_PTR` — `.global .align 8 .u64`
- `STDIO_PRINT_BUF_READY` — `.global .align 4 .u32`
- `gpu_print_buffer_flush()` — `.visible .func`
- `std_buffered_println_test` — `.visible .entry` (3 params: buf, sideband, result)

## Architecture

```
println!("hello")
  │
  ▼
gpu_stdout_write(buf, len)
  │
  ├─ STDIO_PRINT_BUF_READY != 0?
  │   ├─ Yes → print_buffer::print() → accumulate in per-thread sideband slot
  │   │         (auto-flush via SERVICE_BULK_PRINT if slot full)
  │   └─ No  → gpu_hostcall_print() → 1 hostcall per 56-byte chunk
  │
  ▼
kernel exit: gpu_print_buffer_flush() → print_buffer::flush() → SERVICE_BULK_PRINT
```

## Kernel Usage Pattern

```rust
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn my_kernel(buf: *mut u8, sideband: *mut u8) {
    stdio_print_buffer_init(buf, sideband, 1);  // enable buffered path

    println!("these all go through print_buffer");
    println!("batched into a single hostcall");

    gpu_print_buffer_flush();  // flush remaining messages before exit
}
```

## Open Questions

1. **Multi-block support**: `print_buffer` uses `tid.x` for per-thread slot indexing. Multi-block launches would have tid collisions across blocks. Not a concern for single-block kernels.
2. **Automatic flush**: Currently requires explicit `gpu_print_buffer_flush()` before kernel exit. A proc-macro or kernel wrapper could automate this.

## Impact on Downstream Tasks

- println-buffer theme success criteria met: `gpu_stdout_write()` routes through print_buffer when initialized
- Pattern established for any std kernel to opt into buffered printing
