# audit-api.2: Fix API Inconsistencies (Quick Wins)

## Summary

Fixed 2 of 6 API issues identified in audit-api.1. Both are non-breaking quick wins:
encapsulating HostcallBuffer raw pointer fields behind accessors, and adding missing
GpuVec + scheduler re-exports to the async-gpu facade crate.

## Changes Made

### Issue 2: HostcallBuffer pub fields → pub(crate) + accessors

**Problem**: `HostcallBuffer` exposed 9 raw pointer/config fields as `pub`, including
`host_ptr: *mut u8` and `dev_ptr: CUdeviceptr`. External code accessed these directly,
bypassing any encapsulation.

**Fix**:
- Changed all 9 fields from `pub` to `pub(crate)` in `hostcall.rs`
- Added 9 public accessor methods: `host_ptr()`, `dev_ptr()`, `size()`, `num_packets()`,
  `num_shards()`, `pkts_per_shard()`, `sideband_host_ptr()`, `sideband_dev_ptr()`, `sideband_size()`
- Updated all 8 files with direct field access (test crates + integration tests) to use accessors

**Files changed**:
- `crates/core/gpu-host/src/hostcall.rs` — fields + accessors
- `crates/core/gpu-host/tests/gpu_integration.rs` — caller updates
- `crates/test/gpu-test-harness/src/tests_benchmark.rs` — caller updates
- `crates/test/gpu-test-harness/src/tests_hostcall.rs` — caller updates
- `crates/test/gpu-test-harness/src/tests_pipeline.rs` — caller updates
- `crates/test/gpu-test-harness/src/tests_scaling.rs` — caller updates
- `crates/test/gpu-test-harness/src/tests_search.rs` — caller updates
- `crates/test/gpu-test-harness/src/tests_std.rs` — caller updates
- `crates/test/gpu-test-harness/src/tests_warp.rs` — caller updates

### Issue 3: GpuVec + scheduler re-exports to async-gpu facade

**Problem**: `GpuVec` (high-level GPU buffer) and scheduler types (`Scheduler`, `CpuScheduler`,
`GpuScheduler`, `AutoScheduler`) were only accessible via `gpu_host::memory::GpuVec` and
`gpu_host::scheduler::*`. Users of the facade crate (`async-gpu`) couldn't access them.

**Fix**: Added re-exports to `crates/async-gpu/src/lib.rs`:
- `pub use gpu_host::GpuVec;`
- `pub use gpu_host::scheduler::{AutoScheduler, CpuScheduler, GpuScheduler, Scheduler};`

### Issues NOT Fixed (documented as tech debt)

1. **GpuHostError::Verification overload** — ~32 error modes funneled through one variant.
   Splitting requires touching every call site in gpu.rs. Separate task needed.
4. **onnx_rt missing docs** — Entire module behind `#[allow(missing_docs)]`. Large scope.
5. **No thiserror** — Hand-rolled Error impls are functional, consistent. Low priority.
6. **Pipeline hardcoded sleep** — 100ms sleep in Pipeline::run() needs investigation
   before removal (may be load-bearing for GPU synchronization).

## Verification

- `cargo check --workspace` — clean
- `cargo +stable fmt --check` — clean
- `cargo +stable clippy -- -D warnings` — clean
- `bash scripts/ci-lint.sh` — all checks passed
