# audit-build.2: Fix std-build-test linker symbol collision

## Summary

Fixed the `std-build-test` build failure caused by duplicate `#[no_mangle]` definitions
of `gpu_stdin_read` (and `gpu_stdout_write`). The test crate had standalone implementations
of these symbols that collided with the authoritative versions in `gpu-runtime::stdio`.
Removed ~300 lines of duplicated hostcall code and delegated to `gpu-runtime` instead.

## Findings

### Q: What caused the std-build-test build failure?

`std-build-test/src/lib.rs` defined its own `#[no_mangle] gpu_stdout_write` and
`#[no_mangle] gpu_stdin_read` functions with self-contained inline PTX hostcall
implementations. Meanwhile, `gpu-runtime/src/stdio.rs` provides the same symbols.
Since `std-build-test` depends on `gpu-runtime` (transitively via `gpu-libc`),
the `llvm-bitcode-linker` with fat LTO rejected the duplicate symbols.

### Q: How was it fixed?

1. **Removed local `gpu_stdout_write` and `gpu_stdin_read`** — these were ~300 lines of
   duplicated hostcall protocol code with inline PTX assembly (including helpers
   `gpu_hostcall_print_raw` and `gpu_hostcall_stdin_raw`).
2. **Added `gpu-runtime` as a direct dependency** — was already a transitive dep via
   `gpu-libc`, but now explicitly declared for clarity.
3. **Rewrote `stdio_init`** — now delegates to `gpu_runtime::stdio::stdio_init(buf)` for
   the hostcall buffer setup, plus `gpu_libc::gpu_libc_io_init(buf)` for file I/O.
4. **Updated 3 `gpu_stdin_read` call sites** — changed from local function calls to
   `unsafe { gpu_runtime::stdio::gpu_stdin_read(...) }`.
5. **Removed unused `STDIO_HOSTCALL_BUF` static** — no longer needed since gpu-runtime
   manages its own state.
6. **Removed `use core::sync::atomic::{AtomicU64, Ordering}`** — no longer needed.
7. **Removed 2 "function `membar` is never used" warnings** — deleted along with the
   hostcall helper functions that contained them.

### Q: Does it build successfully?

Yes. After the fix, `cargo build --release` in `crates/test/std-build-test/` completes
successfully. 43/43 crates now compile.

### Unexpected Discoveries

- The `std-build-test` crate had ~300 lines of copy-pasted hostcall protocol code that
  duplicated `gpu-runtime::stdio`. This suggests other test crates may have similar
  duplication worth auditing.

### Impact on Downstream Tasks

- Unblocks any tasks that depend on a clean workspace-wide build.
- The removed code was purely duplicated — no behavioral change to the test crate.

## Open Questions

- Are there other test crates with similar duplicated hostcall implementations that
  should be consolidated?
- The `std-build-test` crate may still produce compiler warnings from other sources —
  a full `cargo build --release 2>&1 | grep warning` pass would confirm.
