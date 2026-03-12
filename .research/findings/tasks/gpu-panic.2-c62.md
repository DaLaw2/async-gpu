# gpu-panic.2: Implement GPU Panic Handler
**Cycle**: 62 | **Theme**: gpu-panic | **Kind**: experiment | **Status**: done

## Summary

Implemented the GPU panic handler per ADR-5 design. The `#[panic_handler]` now formats the panic message into a 56-byte buffer, sends it via `SERVICE_PANIC` hostcall with thread/block metadata, then executes `trap; exit;`. Host receives and displays the message with `[GPU PANIC]` prefix in red. Test verified: panic message, thread ID, block ID all correctly transmitted.

## Changes Made

### gpu-protocol/src/lib.rs
- Added `SERVICE_PANIC = 10` opcode
- Added `PANIC_MAX_MSG_LEN = 56` constant
- Added `encode_panic_metadata()`, `panic_thread_idx()`, `panic_block_idx()`, `panic_msg_len()` helpers

### gpu-runtime/src/lib.rs
- Added `#![feature(stdarch_nvptx, asm_experimental_arch)]`
- Added `panic` module with:
  - `PANIC_BUF` global static for hostcall buffer pointer
  - `gpu_panic_init()` to set the buffer pointer at kernel entry
  - `PanicBuf` struct implementing `core::fmt::Write` for 56-byte fixed formatting
  - `send_panic_hostcall()` — best-effort panic message delivery
- Added `panic_handler!()` macro that installs the `#[panic_handler]`
- Added `gpu_panic_init` to prelude

### gpu-kernel/src/lib.rs
- Replaced `loop {}` panic handler with `gpu_runtime::panic_handler!()`
- Added `gpu-runtime` dependency
- Added `panic_test_kernel` — deliberately panics to verify message delivery

### gpu-host/src/hostcall.rs
- Added `handle_panic()` method — decodes metadata, prints `[GPU PANIC]` in red to stderr
- Added `SERVICE_PANIC` arm to both dispatch match blocks

### gpu-host/src/main.rs
- Added `run_panic_test()` — launches panic kernel, uses raw `cuCtxSynchronize` to handle CUDA error, exits cleanly to avoid CudaDevice Drop panic

## Test Results

```
--- GPU Panic Handler Test (gpu-panic.2) ---
  Launching panic_test_kernel (expects GPU panic + trap)...
  [GPU PANIC] block=0 thread=0: panicked at src\lib.rs:1077:5:
  test panic from GPU threa
  Result marker: 1 (expected 1 = reached panic point)
  CUDA sync result: CUDA_ERROR_LAUNCH_FAILED (LAUNCH_FAILED expected from trap)
  panic_test: PASSED (panic message sent via hostcall before trap)
```

## Findings

### Q: Does #[panic_handler] correctly route formatted message through hostcall?
A: Yes. The `PanicBuf` `Write` impl formats `PanicInfo` (including source location) into 56 bytes. The message includes file path and line number (`panicked at src\lib.rs:1077:5:`). Full message is truncated at 56 bytes — the test message `"test panic from GPU thread 0"` was truncated to `"test panic from GPU threa"`.
**Confidence**: high

### Q: Does trap instruction cleanly terminate the thread after sending panic?
A: Yes. `trap` terminates the thread and sets the CUDA context into a "sticky error" state. `cuCtxSynchronize` returns `CUDA_ERROR_LAUNCH_FAILED`. The CudaDevice Drop impl will also panic on this error, so `std::process::exit(0)` is used to exit cleanly.
**Confidence**: high

### Q: Does host-side print panic message with thread/block metadata?
A: Yes. Host decodes `threadIdx.x` and `blockIdx.x` from the metadata slot and prints: `[GPU PANIC] block=0 thread=0: <message>`. Output is in red using ANSI escape codes.
**Confidence**: high

### Q: Does panic work in both sync and async kernel code paths?
A: Verified for sync kernels. The panic handler uses synchronous hostcall (spin-wait), which works regardless of async context. Async kernel testing deferred — the handler doesn't depend on executor state.
**Confidence**: high (sync), medium (async — untested but architecturally sound)

## Unexpected Discoveries

1. **CudaDevice Drop panics after trap**: cudarc's `Drop` impl for `CudaDevice` unwraps CUDA operations that fail with the sticky error from `trap`. Workaround: `std::process::exit(0)` before Drop. This means the panic test must be the last test in the suite.

2. **56-byte truncation is practical**: The formatted `PanicInfo` includes the source location (`src\lib.rs:1077:5`) which consumes ~30 bytes, leaving ~26 bytes for the actual panic message. For simple messages like `"index out of bounds"` this is sufficient. For complex formatted messages, only the beginning is visible.

3. **Message does NOT include the panic message argument itself first** — `PanicInfo`'s `Display` format outputs `"panicked at <location>:\n<message>"`, so the location comes before the message. With 56-byte truncation, short messages are fully visible after the location, but long messages get cut.

## Open Questions

- Should we flip the format order (message first, then location) to prioritize the panic message over the source location?
- Could we use a larger buffer (e.g., multiple packets) for longer messages?

## Impact on Downstream Tasks

- **All kernel crates**: Can now adopt `gpu_runtime::panic_handler!()` instead of `loop {}` — one-line change per crate.
- **host-scaling**: Panic reporting works with the existing hostcall protocol — no changes needed for multi-threaded listener (each SERVICE_PANIC is an independent packet).
- **Developer experience**: Debugging GPU hangs is now possible — panics produce visible output instead of silent infinite loops.
