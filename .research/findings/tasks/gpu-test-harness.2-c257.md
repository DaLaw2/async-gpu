# gpu-test-harness.2: Implement cargo test integration for GPU kernels
**Cycle**: 257 | **Theme**: gpu-test-harness | **Kind**: experiment | **Status**: done

## Summary
Created a `cargo test` integration test harness for GPU kernels. Three proof-of-concept tests demonstrate the pattern: shared `OnceLock<Arc<CudaDevice>>`, PTX constants from library, standard `#[test]` functions.

## Changes

### crates/gpu-host/src/lib.rs
- Added `pub mod ptx` with 7 embedded PTX constants (KERNEL, EMBASSY_TEST, etc.)
- Added `pub mod mapped_mem` to expose allocation helpers

### crates/gpu-host/src/main.rs
- PTX constants now reference `gpu_host::ptx::*` instead of duplicating `include_str!()` calls

### crates/gpu-host/src/mapped_mem.rs
- Changed `pub(crate)` → `pub` for all functions
- Added `# Safety` documentation sections for clippy compliance

### crates/gpu-host/tests/gpu_integration.rs (NEW)
Three integration tests:
1. `test_write_thread_idx` — basic kernel launch + mapped memory readback
2. `test_hostcall_print_hello` — hostcall with listener thread + message verification
3. `test_buffered_print` — SERVICE_BULK_PRINT end-to-end (12 buffered messages)

Usage: `cargo test --test gpu_integration -- --test-threads=1`

## Architecture
```
tests/gpu_integration.rs
    ├── shared_device() → OnceLock<Arc<CudaDevice>>
    ├── test_write_thread_idx    (uses gpu_host::ptx::KERNEL)
    ├── test_hostcall_print_hello (uses HostcallBuffer + listener)
    └── test_buffered_print      (uses SERVICE_BULK_PRINT)
```

## Impact on Downstream Tasks
- Pattern established for migrating remaining 107 tests from main.rs
- PTX constants now accessible from both binary and integration tests
- mapped_mem functions now public API
