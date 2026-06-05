# split-execute.1: Extract stdio infrastructure from lib.rs to gpu-runtime

**Status**: done
**Kind**: experiment
**Cycle**: 614

## Baseline

PTX build of gpu-kernel-std succeeded before any changes (21 warnings, all pre-existing).

## What was done

Extracted 3 static atomics + 5 functions from `crates/kernel/gpu-kernel-std/src/lib.rs`
into new `crates/core/gpu-runtime/src/stdio.rs`:

**Moved to gpu-runtime::stdio:**
- `STDIO_HOSTCALL_BUF: AtomicU64` (module-private)
- `STDIO_SIDEBAND_PTR: AtomicU64` (module-private)
- `STDIO_PRINT_BUF_READY: AtomicU32` (module-private)
- `pub unsafe fn gpu_stdout_write(buf, len) -> usize` — #[no_mangle], called by std PAL
- `pub unsafe fn gpu_stdin_read(out_buf, max_len) -> usize` — #[no_mangle], called by std PAL
- `pub fn stdio_init(buf)` — sets hostcall buffer pointer
- `pub unsafe fn stdio_print_buffer_init(buf, sideband, thread_count)` — sets sideband+ready
- `pub fn gpu_print_buffer_flush()` — flushes print buffer

**Stays in gpu-kernel-std lib.rs:**
- `stdio_auto_init()` — 10-line wrapper that reads `__HOSTCALL_BUF` device global and
  calls `gpu_runtime::stdio::stdio_init()`, `gpu_runtime::panic::gpu_panic_init()`,
  and `gpu_libc::gpu_libc_io_init()`.
- `#[used]` force-link statics for `gpu_stdout_write` and `gpu_stdin_read`.

## Key findings

1. **LTO symbol stripping**: Confirmed that `#[used]` static function pointers
   successfully force-link `gpu_stdout_write` and `gpu_stdin_read` through LTO.
   Both appear as `.visible .func` in the PTX output.

2. **Clippy safety**: Functions that take raw pointers and dereference them must be
   marked `unsafe` in gpu-runtime (which is clippy-checked), unlike in gpu-kernel-std
   (which was not). Changed `gpu_stdout_write`, `gpu_stdin_read`, and
   `stdio_print_buffer_init` to `unsafe fn`.

3. **Inlining**: `stdio_init` (a single atomic store) is inlined away by LTO —
   it doesn't appear as a standalone PTX function. This is expected and correct.

4. **Zero new warnings**: The 21 PTX build warnings are identical to baseline
   (all in compute_gemm.rs, pipeline.rs, etc.).

## Verification

- `cargo +nightly-2026-06-03 build --release` (PTX): PASS (19.7s)
- `cargo +stable fmt --check` (gpu-runtime): PASS
- `cargo +stable clippy -- -D warnings` (gpu-runtime): PASS
- PTX symbols verified: `gpu_stdout_write`, `gpu_stdin_read`, `gpu_print_buffer_flush`, `stdio_print_buffer_init` all present

## Files changed

- `crates/core/gpu-runtime/src/stdio.rs` — NEW: stdio module with extracted code
- `crates/core/gpu-runtime/src/lib.rs` — registered `pub mod stdio`
- `crates/core/gpu-runtime/src/prelude.rs` — added stdio re-exports
- `crates/kernel/gpu-kernel-std/src/lib.rs` — removed 160 lines of stdio code, added force-link + wrapper
