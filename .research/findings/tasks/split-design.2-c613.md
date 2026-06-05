# split-design.2: Stdio Infrastructure Extraction Design

**Task**: Design where stdio infra goes, module structure, and migration plan.
**Date**: 2026-06-05 | **Cycle**: 613

## Summary

After reading every function implementation and tracing all dependency edges, the stdio infrastructure **must go into gpu-runtime**, not a new gpu-kernel-core crate. The code is pure runtime plumbing (atomics + calls to gpu-runtime APIs) with zero kernel-specific logic. Placing it in gpu-runtime eliminates the need for a new crate and leverages the existing rlib that every kernel crate already depends on.

## Design Decision: gpu-runtime (not gpu-kernel-core)

### Why gpu-runtime wins

1. **All callees are already in gpu-runtime.** The stdio functions call:
   - `gpu_runtime::hostcall::gpu_hostcall_print()`
   - `gpu_runtime::hostcall::gpu_hostcall_request_with_timeout()`
   - `gpu_runtime::hostcall::gpu_hostcall_release()`
   - `gpu_runtime::print_buffer::init()`, `print()`, `flush()`
   - `gpu_runtime::entry::hostcall_buf_ptr()`
   - `gpu_runtime::panic::gpu_panic_init()`
   - `gpu_runtime::prelude::{PKT_OFF_PAYLOAD, SERVICE_STDIN}`

   Moving stdio into gpu-runtime creates zero new dependencies. It only calls down into modules that already exist in the same crate.

2. **No circular dependency risk.** The stdio module would be a new leaf module in gpu-runtime, calling into existing modules (hostcall, print_buffer, entry, panic). No module in gpu-runtime needs to call stdio. The dependency arrow is one-way: `stdio → {hostcall, print_buffer, entry, panic}`.

3. **gpu-libc is the only external call**, and it goes *outward*: `stdio_auto_init()` calls `gpu_libc::gpu_libc_io_init(buf)`. This is fine — gpu-libc is NOT a dependency of gpu-runtime (gpu-runtime depends on gpu-protocol + gpu-atomics only). Two options:
   - **Option A (recommended)**: `stdio_auto_init()` stays in each kernel crate as a thin 3-line wrapper that calls `gpu_runtime::stdio::stdio_init()` + `gpu_runtime::panic::gpu_panic_init()` + `gpu_libc::gpu_libc_io_init()`. This keeps gpu-runtime free of the gpu-libc dependency.
   - **Option B**: Add gpu-libc as a dependency of gpu-runtime. Rejected — gpu-libc is a `no_std` libc shim with its own init flow; coupling it to gpu-runtime inverts the expected dependency direction.

4. **A new gpu-kernel-core crate adds complexity for no gain.** It would be another rlib that all kernel crates depend on, but it would itself depend on gpu-runtime for 100% of its functionality. That's a passthrough crate with no independent value.

5. **std PAL contract.** The patched std (`std-patches/sys_stdio_cuda.rs`) declares `extern "C" { fn gpu_stdout_write(...); fn gpu_stdin_read(...); }`. These symbols must be defined as `#[no_mangle]` in each cdylib. Since gpu-runtime is an rlib, its `#[no_mangle]` functions get linked into every cdylib that depends on it — exactly the right behavior.

### split-design.1 recommended gpu-kernel-core — why override?

split-design.1 said "kernel-specific, not runtime-generic." But reading the actual code, these functions contain zero kernel-specific logic. They are pure hostcall protocol wrappers: load an atomic pointer, branch on null, call gpu_runtime hostcall/print_buffer APIs. The "kernel-specific" appearance is only because they currently live in a kernel crate. The code itself is generic runtime infrastructure.

## Module Structure

### New module: `gpu_runtime::stdio`

```
crates/core/gpu-runtime/src/stdio.rs   (NEW — ~130 lines)
```

**Public API:**

```rust
// Global state (must be in the rlib so each cdylib gets its own copy via LTO)
static STDIO_HOSTCALL_BUF: AtomicU64 = AtomicU64::new(0);
static STDIO_SIDEBAND_PTR: AtomicU64 = AtomicU64::new(0);
static STDIO_PRINT_BUF_READY: AtomicU32 = AtomicU32::new(0);

/// Initialize the stdio hostcall buffer pointer.
/// Called at kernel entry — must be called before any println!().
pub fn stdio_init(buf: *mut u8)

/// Initialize buffered printing. After this, gpu_stdout_write routes
/// through print_buffer instead of one-hostcall-per-chunk.
/// Caller MUST call gpu_print_buffer_flush() before kernel exit.
#[unsafe(no_mangle)]
pub fn stdio_print_buffer_init(buf: *mut u8, sideband: *mut u8, thread_count: u32)

/// Flush buffered print messages via SERVICE_BULK_PRINT hostcall.
/// Must be called before kernel exit when buffered printing is active.
#[unsafe(no_mangle)]
pub fn gpu_print_buffer_flush()

/// PAL callback: std Stdout::write() → this function.
#[unsafe(no_mangle)]
pub fn gpu_stdout_write(buf: *const u8, len: usize) -> usize

/// PAL callback: std Stdin::read() → this function.
#[unsafe(no_mangle)]
pub fn gpu_stdin_read(out_buf: *mut u8, max_len: usize) -> usize
```

**Visibility notes:**
- `stdio_init()`: `pub` — kernel crates call this from their `stdio_auto_init()` wrapper.
- `gpu_stdout_write`, `gpu_stdin_read`: `pub` + `#[no_mangle]` — required by the std PAL extern declarations.
- `stdio_print_buffer_init`, `gpu_print_buffer_flush`: `pub` + `#[no_mangle]` — called by kernel entry points that use buffered printing.
- The three `static` atomics: module-private (no `pub`). Only accessed by functions within the `stdio` module. Each cdylib gets its own copy because gpu-runtime is an rlib linked into each cdylib.

### Remaining in each kernel crate: `stdio_auto_init()`

```rust
/// Auto-initialize stdio from __HOSTCALL_BUF device global.
/// Returns the hostcall buffer pointer, or null if not injected.
fn stdio_auto_init() -> *mut u8 {
    let buf = gpu_runtime::entry::hostcall_buf_ptr();
    if !buf.is_null() {
        gpu_runtime::stdio::stdio_init(buf);
        unsafe {
            gpu_runtime::panic::gpu_panic_init(buf);
            gpu_libc::gpu_libc_io_init(buf);
        }
    }
    buf
}
```

This 8-line function stays in each kernel crate because:
1. It calls `gpu_libc::gpu_libc_io_init()` — gpu-libc is a kernel dependency, not a gpu-runtime dependency.
2. It's the kernel's entry-point initialization sequence — kernel-specific in the sense that different kernel crates may initialize different subsystems.

### Static atomics: no duplication concern

Since gpu-runtime is an **rlib** (not cdylib), its statics are compiled into each cdylib that depends on it. Each cdylib (each kernel crate) gets its own independent copy of the three atomics. This is exactly correct: different kernels in different cubins should have independent stdio state.

### gpu-runtime lib.rs changes

Add to `crates/core/gpu-runtime/src/lib.rs`:

```rust
/// GPU-side stdio routing — bridges patched std's Stdout/Stdin to hostcall.
///
/// The patched std PAL (`sys_stdio_cuda.rs`) calls `gpu_stdout_write()` and
/// `gpu_stdin_read()` via extern declarations. This module provides the
/// implementations that route through the hostcall protocol.
///
/// # Usage
///
/// ```rust,ignore
/// // At kernel entry:
/// gpu_runtime::stdio::stdio_init(hostcall_buf);
///
/// // Now println!() works via hostcall:
/// println!("Hello from GPU!");
///
/// // For buffered printing (optional, reduces hostcall overhead):
/// gpu_runtime::stdio::stdio_print_buffer_init(buf, sideband, 1);
/// println!("Buffered message");
/// gpu_runtime::stdio::gpu_print_buffer_flush();
/// ```
pub mod stdio;
```

### Prelude addition

Add to `gpu_runtime::prelude`:
```rust
pub use crate::stdio::{stdio_init, stdio_print_buffer_init, gpu_print_buffer_flush};
```

Note: `gpu_stdout_write` and `gpu_stdin_read` are NOT added to the prelude — they're PAL callbacks, never called by user code directly.

## Migration Plan

### Step 1: Create `gpu_runtime::stdio` module

Create `crates/core/gpu-runtime/src/stdio.rs` with the five functions and three statics, moved verbatim from `gpu-kernel-std/src/lib.rs` lines 49-208.

Adjustment needed: `stdio_auto_init()` does NOT move — only `stdio_init()` (the inner helper) moves. The `stdio_auto_init()` body becomes a thin kernel-side wrapper.

Register the module in `gpu-runtime/src/lib.rs` and add prelude re-exports.

### Step 2: Update gpu-kernel-std to use the new module

In `gpu-kernel-std/src/lib.rs`:
1. Delete the 6 functions and 3 statics (lines 49-208).
2. Rewrite `stdio_auto_init()` as an 8-line wrapper calling `gpu_runtime::stdio::stdio_init()`.
3. Replace all direct calls to `stdio_print_buffer_init()` and `gpu_print_buffer_flush()` with `gpu_runtime::stdio::stdio_print_buffer_init()` and `gpu_runtime::stdio::gpu_print_buffer_flush()`.

### Step 3: Update std-build-test (if in scope)

`crates/test/std-build-test/src/lib.rs` has its own copy of `gpu_stdout_write`/`gpu_stdin_read` (simpler version without print_buffer support). It can be migrated to use `gpu_runtime::stdio` too, gaining print_buffer support for free. However, this crate predates the unified architecture and may be intentionally standalone — defer to kernel-split execution.

### Step 4: Verify correctness

For each step, verify:
- `cargo build --release -p gpu-runtime` succeeds (rlib builds)
- `cargo build --release -p gpu-kernel-std --target nvptx64-nvidia-cuda` succeeds (cdylib links)
- The `#[no_mangle]` symbols `gpu_stdout_write` and `gpu_stdin_read` appear in the generated PTX
- Run `std_println_test` and `std_buffered_println_test` kernels to confirm runtime correctness

## Open Questions

1. **Force-link concern**: When gpu-runtime is an rlib and the `#[no_mangle]` functions are only called via `extern "C"` from the std PAL (not via Rust `use`), will LTO strip them? The kernel crate may need a `#[used]` array (similar to how std-build-test forces gpu-libc symbols) to keep `gpu_stdout_write`/`gpu_stdin_read` alive through LTO. Investigate during implementation.

2. **std-build-test migration**: Should it switch to `gpu_runtime::stdio` or stay standalone? Low priority — it's a test crate, not a production kernel crate.

3. **Future: stdio_auto_init() macro?** If many kernel crates will have the same 8-line `stdio_auto_init()` boilerplate, consider a `gpu_runtime::stdio_auto_init!()` macro that expands to the init sequence (calling into gpu_runtime + gpu_libc). This would require gpu-libc as an optional dependency of gpu-runtime, or the macro would reference `gpu_libc` as an extern crate path.
