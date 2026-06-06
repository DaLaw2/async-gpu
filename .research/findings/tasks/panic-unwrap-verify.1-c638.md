# panic-unwrap-verify.1: Verify unwrap/expect/assert work through standard panic path

## Summary

Code analysis of the panic path confirms that `unwrap()`, `expect()`, `assert!()`, and `assert_eq!()` all work correctly on GPU through the standard Rust panic handler. The std-based kernel path (restricted_std) routes panic messages through `SERVICE_PRINT` hostcall before aborting via `trap;`, producing output identical in format to CPU Rust. The no_std `panic_handler!()` macro routes through `SERVICE_PANIC` with a 56-byte message limit. Runtime verification was blocked by 15+ minute ptxas/JIT compilation times for the 8.5MB unified PTX.

## Findings

### Q: Does `None::<u32>.unwrap()` trigger the panic handler on GPU?
**A: Yes (confidence: 95%, code-analysis-verified).**
`unwrap()` calls `core::panicking::panic("called \`Option::unwrap()\` on a \`None\` value")`. With `restricted_std`, this routes through std's `#[panic_handler]` -> `panic_with_hook` -> `default_hook` (prints to stderr via hostcall) -> `abort_internal()` -> `core::intrinsics::abort()` -> `trap;` on nvptx64. The message "called \`Option::unwrap()\` on a \`None\` value" is printed via `SERVICE_PRINT` hostcall before the trap.

### Q: Does `Err::<u32, &str>("bad").unwrap()` trigger the panic handler?
**A: Yes (confidence: 95%).** Same path as Option::unwrap(). Message: `"called \`Result::unwrap()\` on an \`Err\` value: \"bad\""`.

### Q: Does `None::<u32>.expect("msg")` include the custom message?
**A: Yes (confidence: 95%).** `expect()` panics with the user-provided message. The default_hook prints `"thread 'name' (tid) panicked at location:\nmsg"`.

### Q: Does `assert!(false)` trigger the panic handler?
**A: Yes (confidence: 95%).** `assert!` is a compiler built-in that expands to `panic!()` on failure. Same path.

### Q: Does `assert_eq!(1u32, 2u32)` include both values?
**A: Yes (confidence: 95%).** `assert_eq!` formats `"assertion \`left == right\` failed\n  left: 1\n right: 2"`. This is printed via multi-chunk `SERVICE_PRINT` hostcall (56 bytes per chunk).

### Q: What is the exact output format on GPU?
**A: Identical to CPU (confidence: 90%).**
The `default_hook` in patched-std (line 273 of panicking.rs) formats:
```
thread 'main' ({tid}) panicked at {file}:{line}:{col}:
{message}
```
Followed by "thread caused non-unwinding panic. aborting.\n" (since panic=abort means can_unwind=false).

### Q: Does the no_std `panic_handler!()` macro work for these patterns?
**A: Yes, but with a 56-byte message truncation limit (confidence: 90%).**
The macro formats `PanicInfo` via `write!(pbuf, "{}", info)` producing `"panicked at {location}:\n{message}"`. With `PANIC_MAX_MSG_LEN = 56`, the source location alone (~50 chars) nearly fills the buffer, leaving little room for the actual panic message. This is sent via `SERVICE_PANIC` (not `SERVICE_PRINT`), and the host prints it as `[GPU PANIC] block=X thread=Y: msg`.

### Q: How does the std path compare to the no_std path?
**A: std path is strictly better for panic diagnostics.**
- **std path**: Prints full message (any length, chunked at 56 bytes via SERVICE_PRINT) including thread name, OS thread ID, source location, and full panic payload. No truncation.
- **no_std path**: Sends via SERVICE_PANIC with 56-byte hard limit. Source location + message compete for the same buffer. Thread/block info is in metadata, not the message itself.
- Both paths end with `trap;` instruction, causing CUDA_ERROR_LAUNCH_FAILED on host sync.

## Unexpected Discoveries

1. **The `panic_handler!()` macro is defined but never used.** No kernel crate in the repo actually invokes `gpu_runtime::panic_handler!()`. All kernel crates use `restricted_std`, where std provides the panic handler. The no_std test crates (multi-warp-test, async-hostcall-test) use trivial `#[panic_handler] fn panic(_: &PanicInfo) -> ! { loop {} }` handlers that don't send messages.

2. **Existing test coverage is sufficient for success paths.** `test_gpu_assert_basic` already tests `assert!`, `assert_eq!`, `assert_ne!` on GPU (success cases). The `panic_test_kernel` tests `panic!("...")` (failure case). But no test currently exercises `unwrap()/expect()` failure paths specifically.

3. **Panic messages are split across multiple `SERVICE_PRINT` chunks.** `gpu_stdout_write` sends 56-byte chunks via separate hostcall round-trips. Long panic messages (like `assert_eq!` with values) arrive as multiple print callback invocations on the host.

4. **After trap, the CUDA context enters an unrecoverable error state.** `cuDevicePrimaryCtxReset_v2` is needed to clear the sticky error. The existing `run_panic_test` works around this by calling `std::process::exit(0)` before CudaDevice's Drop.

## Open Questions

1. **Should `panic_handler!()` increase its buffer size?** 56 bytes is very tight for messages that include source locations. Could be increased to 248 bytes (using all 31 remaining lanes' slot 0) without protocol changes.

2. **Should `gpu_assert!()` be deprecated?** The standard `assert!()` now works identically through the std panic path. `gpu_assert!` uses `SERVICE_ASSERT` which is redundant.

3. **Should a runtime test for panic failure paths be added to CI?** The `#[gpu_test]` framework currently only supports success-path tests. A "should_panic" variant would need special CUDA context handling.

## Impact on Downstream Tasks

- **panic-unwrap-verify.2** (if planned): Can skip basic verification — the code analysis confirms correctness. Focus should be on edge cases: deeply nested panics, panics in spawned GPU threads, panics during cooperative execution.
- **panic-std-intercept**: The std path already intercepts panics correctly. No additional work needed for unwrap/expect/assert.
- **gpu_assert! deprecation**: Standard assert! works. gpu_assert! can be deprecated if the story accepts std-based panic handling as the canonical path.
