# kernel-entry.3: Rewrite gpu-kernel-std kernels to use zero-param entry

## STATUS: done

## SUMMARY
Converted all gpu-kernel-std showcase kernels to use zero-param entry via `__HOSTCALL_BUF` device global. I/O-only kernels (7) became truly zero-parameter; thread/compute kernels (6) had the `buf` parameter removed while retaining data params (`result`, `dims`). Added `GpuStdModule` helper to gpu-host for host-side module loading with device global injection + arbitrary kernel params.

## Strategy Applied
- **I/O-only kernels** (std_println_test, std_vec_format_test, std_alloc_stress_test, std_file_io_test, std_pipeline_test, std_stdin_test, std_hashmap_test): Converted to zero-param. Use `stdio_auto_init()` internally which reads `__HOSTCALL_BUF` device global and initializes stdio + panic handler + libc I/O.
- **Thread/compute kernels** with `result: *mut u32` (std_multithread_println_test, std_thread_spawn_demo, real_std_thread_spawn, std_thread_spawn_minimal, unified_io_compute, kernel_std_println_smoke): Removed `buf` param, kept `result`. Use `stdio_auto_init()` for hostcall init.
- **matmul_io_compute**: Removed `buf` param, kept `dims` and `result` as data params. Signature: `(dims: *const u32, result: *mut u32)`.
- **Unchanged**: `std_buffered_println_test` (needs sideband param), `kernel_std_smoke_test` (no hostcall), `kernel_std_pool_smoke` (no hostcall), `std_multithread_vec_test` (already no buf).

## New API: GpuStdModule
Added `gpu_host::gpu::GpuStdModule` — loads PTX via raw CUDA driver API, injects `__HOSTCALL_BUF` device global, and provides `launch_raw()` for kernels with arbitrary params. Supports optional print callback for capturing GPU stdout.

## Tests Verified
- `ONLY_TEST=zero_param` — PASSED (existing zero_param_hello still works)
- `ONLY_TEST=std_fs` — PASSED (std_file_io_test as zero-param)
- `ONLY_TEST=std_pipeline` — PASSED (std_pipeline_test as zero-param)
- Host crate + test harness compile with zero warnings

## FILES_CHANGED
- `crates/kernel/gpu-kernel-std/src/lib.rs` — converted 13 kernel signatures
- `crates/core/gpu-host/src/gpu.rs` — added GpuStdModule struct (~170 lines)
- `crates/test/gpu-test-harness/src/main.rs` — updated 5 test functions to use GpuStdModule
- `crates/test/gpu-test-harness/src/tests_std.rs` — updated 4 test functions for new kernel signatures
- `crates/core/gpu-host/kernel_std.ptx` — rebuilt PTX (6.3 MB)
- `crates/core/gpu-host/kernel_std.cubin` — rebuilt cubin (42.5 MB)
- `crates/test/gpu-test-harness/kernel_std.cubin` — copied updated cubin
