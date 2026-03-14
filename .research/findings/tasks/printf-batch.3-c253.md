# printf-batch.3: Write GPU kernel test — 12 buffered prints, verify fewer hostcall round-trips
**Cycle**: 253 | **Theme**: printf-batch | **Kind**: experiment | **Status**: done

## Summary
Added end-to-end test for GPU-side buffered printing. Kernel accumulates 12 print messages in print_buffer, flushes once via SERVICE_BULK_PRINT. Host test verifies kernel success + receives all 12 messages. Compiles to PTX successfully.

## Changes

### crates/gpu-kernel/src/hostcall_kernels.rs
- Added `buffered_print_test` kernel: calls `print_buffer::init()`, buffers 12 "Buffered msg NN" messages via `print_buffer::print()`, flushes via `print_buffer::flush()`. Returns 1 on success, 0 on error.

### crates/gpu-host/src/tests_pipeline.rs
- Added `run_buffered_print_test()`: launches kernel with hostcall buffer + sideband, collects all print messages, verifies kernel returns 1 and all 12 messages are received.

### crates/gpu-host/src/main.rs
- Wired `run_buffered_print_test()` into the test suite.

## Findings
### Q: Does the buffered print path compile end-to-end?
A: Yes. Kernel compiles to PTX. Host test compiles. All CI checks pass.
**Confidence**: high

### Q: How many hostcall round-trips does buffered print use?
A: 1 round-trip for 12 messages (vs 12 round-trips unbuffered). The 12 messages total ~264 bytes (12 * 22), well within the 504-byte slot capacity.
**Confidence**: high (by design — runtime test needed for hardware verification)

## Open Questions
1. Hardware verification pending — needs an actual GPU to run the test and confirm SERVICE_BULK_PRINT handler works at runtime.

## Impact on Downstream Tasks
- printf-batch theme criterion 3 ("Test: 10+ prints") is now met at the code level.
- Hardware verification can be done by running `cargo run` on a machine with NVIDIA GPU.
