# audit-api.1: Systematic API Surface Review

## Summary

Reviewed the public API surface across all 7 core crates (async-gpu, gpu-atomics, gpu-protocol,
gpu-libc, gpu-host, gpu-runtime) and their sub-modules. The API is remarkably clean: zero anyhow
usage, zero #[allow(dead_code)], consistent naming conventions, and well-structured error types.
The main issues found are: 4 `#[allow(unused_mut)]` annotations, 3 `#[allow(missing_docs)]`
on public modules, HostcallBuffer exposes raw pointer fields as `pub`, GpuHostError::Verification
is overloaded as a catch-all for ~32 distinct failure modes, and error types use hand-rolled
Display/Error impls instead of thiserror.

## Findings

### 1. Naming Consistency — PASS

- All public functions follow snake_case consistently.
- All types (structs, enums, traits) follow CamelCase consistently.
- Constants follow SCREAMING_SNAKE_CASE consistently.
- No naming anomalies detected across any crate.
- Module naming is consistent (snake_case, descriptive).

### 2. Error Handling

#### 2a. anyhow usage — PASS (zero occurrences)
No `anyhow` usage found in any `.rs` or `.toml` file across the entire `crates/` tree.

#### 2b. Error type structure — MINOR ISSUES

- **gpu-host**: `GpuHostError` (hand-rolled Display/Error) — well-structured but not using `thiserror`.
- **gpu-host::nn**: `NnError` (hand-rolled Display/Error) — same pattern.
- **gpu-host::hostcall**: `HostcallError` (hand-rolled Display/Error) — same pattern.
- **gpu-protocol**: `GpuError` — `#[repr(C)]` struct, not an enum. Correct design for GPU-side use.
- **gpu-runtime::executor**: `ExecutorError` — needs verification.
- `thiserror` is not used anywhere in the project. All error types are manually implemented.
  This is consistent but means boilerplate that `thiserror` would eliminate.

#### 2c. GpuHostError::Verification overload — DESIGN ISSUE

`GpuHostError::Verification { test, detail }` is used as a catch-all error for ~32 distinct
failure modes in `gpu.rs` alone. Examples:
- PTX loading failures
- Null byte in kernel names
- cuModuleLoadData failures
- cuLaunchKernel failures
- Synchronization failures
- Device-to-host copy failures

These are operationally distinct errors that get flattened into the same variant, making
programmatic error handling difficult. A downstream user cannot match on "PTX load failed"
vs "kernel launch failed" without string-parsing the `detail` field.

#### 2d. Error type fragmentation — INFO

- `HostcallBuffer::new()` returns `Result<Self, HostcallError>`.
- `GpuContext::launch()` returns `Result<GpuResult>` (using `GpuHostError`).
- `Pipeline::run()` uses `std::result::Result<(), crate::GpuHostError>` — fully qualified.
- `Pipeline::new()` uses `std::result::Result<Self, HostcallError>`.
- Mixing `HostcallError` and `GpuHostError` in the same module creates cognitive overhead.
  However, `From<HostcallError> for GpuHostError` exists, so `?` works across boundaries.

### 3. Import Paths / Re-exports — GOOD

#### 3a. Facade crate (async-gpu) — WELL DESIGNED

The `async_gpu` facade re-exports all key types users need:
- `gpu` module (one-liner API)
- `GpuHostError`, `Result`, `error_category_name`, `GpuKernelErrorInfo`
- `GpuRuntime`, `HostcallBuffer`, `HostcallSession`, `MappedBuffer`, `Pipeline`
- `HostcallError`, `FlightRecorder`
- `GpuStream`
- Feature-gated: `nn`, `async_rt`, `AsyncGpuRuntime`, `AsyncHostcallSession`, `GpuTask`, `HostcallEvent`

Users can do `use async_gpu::gpu;` and `use async_gpu::GpuRuntime;` without reaching into internals.

#### 3b. Missing from facade — MINOR GAP

- `GpuVec` is not re-exported from `async_gpu`. Users must go through `gpu_host::memory::GpuVec`.
- `Scheduler`, `CpuScheduler`, `GpuScheduler`, `AutoScheduler` are not re-exported.
- `CommandBuffer`, `Command` are not re-exported.
- `resource_report` types (`SmConfig`, `KernelResources`, etc.) are not re-exported.
  These may be intentionally internal, but `GpuVec` and the scheduler types seem user-facing.

### 4. Dead Code / Allow Annotations

#### 4a. #[allow(dead_code)] — PASS (zero occurrences)
No `#[allow(dead_code)]` found anywhere in the crate tree. Policy is upheld.

#### 4b. #[allow(unused_mut)] — 4 occurrences in gpu-runtime/src/warp.rs

Lines 7, 35, 63, 93 — on `reduce_sum_f32`, `reduce_sum_u32`, `reduce_max_f32`, `reduce_min_f32`.

These are justified: the `mut` is needed on nvptx64 (the value is modified inside the asm block)
but appears unused on non-nvptx targets (stub returns the input unchanged). The `#[allow]` is
necessary to suppress warnings on host-target doc builds. This is acceptable.

#### 4c. #[allow(missing_docs)] — 3 occurrences

1. `gpu-host/src/lib.rs:117` — on `pub mod onnx_rt;`. The ONNX runtime module lacks doc comments
   on its public API. This is a gap worth tracking.
2. `gpu-host/src/nn/autograd/tape.rs:48` — on a type in the autograd module.
3. `gpu-host/src/nn/models/resnet.rs:169,266` — on ResNet model internals.

The last two are in `#[cfg(any(test, feature = "demo"))]` modules, so they're not part of
the stable public API. The `onnx_rt` one is the only concern.

### 5. Visibility

#### 5a. HostcallBuffer pub fields — ENCAPSULATION ISSUE

`HostcallBuffer` exposes all its internal state as `pub` fields:
- `pub host_ptr: *mut u8`
- `pub dev_ptr: sys::CUdeviceptr`
- `pub size: usize`
- `pub num_packets: u16`
- `pub num_shards: u32`
- `pub pkts_per_shard: u32`
- `pub sideband_host_ptr: *mut u8`
- `pub sideband_dev_ptr: sys::CUdeviceptr`
- `pub sideband_size: usize`

Raw pointers are exposed publicly. While users rarely construct `HostcallBuffer` directly
(they use `HostcallSession`), this is an encapsulation gap. These should be `pub(crate)` or
have accessor methods.

#### 5b. GpuStream::new is pub(crate) — CORRECT

`GpuStream::new()` is correctly `pub(crate)` — users create streams via `GpuRuntime::create_stream()`.

#### 5c. fresh_module_name is pub(crate) — CORRECT

`gpu::fresh_module_name()` is correctly `pub(crate)` as it's an internal utility.

#### 5d. CommandBuffer fields are private — CORRECT

`CommandBuffer` fields are private with proper accessor methods.

### 6. Documentation

#### 6a. Module-level docs — EXCELLENT

All major modules have comprehensive `//!` module-level doc comments:
- `gpu-atomics`, `gpu-protocol`, `gpu-libc`, `gpu-host`, `gpu-runtime` all have
  architecture descriptions, usage examples, and key type listings.
- `async_gpu` facade has a Quick Start example and feature flag docs.

#### 6b. Public type/function docs — VERY GOOD

- All public types in `gpu-host` have doc comments (enforced by `#![warn(missing_docs)]`).
- `gpu-atomics` has thorough safety docs and PTX instruction documentation.
- `gpu-protocol` has doc comments with examples (tested via `///` doc tests).
- The `gpu-runtime` modules have extensive module-level and type-level docs.

#### 6c. Missing docs — MINOR

- `onnx_rt` module has `#[allow(missing_docs)]` — its public API lacks doc comments.

### 7. Clippy Allows — INFO

Found clippy allows across the codebase. These are generally justified:
- `too_many_arguments` (14 occurrences): Common in ML code with many hyperparameters.
- `type_complexity` (4 occurrences): Complex callback types in GPU launch API.
- `needless_range_loop` (6 occurrences): GPU code where index is the semantically meaningful value.
- `new_without_default` (3 occurrences): GPU executor types that require unsafe init.
- `declare_interior_mutable_const` (5 occurrences): GPU atomics in const context.
- `mut_from_ref` (1 occurrence): Intentional in `safety.rs` for unsafe shared memory access.
- `missing_safety_doc` (1 occurrence, crate-level): `gpu-atomics` — all fns are unsafe,
  module-level doc explains the safety contract.

### 8. Deprecated API

- `gpu::compute()` is correctly marked `#[deprecated(note = "use gpu::launch() instead")]`.
  Clean deprecation pattern.

## Unexpected Discoveries

1. **No thiserror anywhere**: Despite being the recommended approach per CLAUDE.md, all error
   types use hand-rolled `Display` and `Error` impls. This is consistent (all crates do it
   the same way) but represents an opportunity for cleanup.

2. **GpuVec::as_slice() is safe but has unsafe contract**: The method is not marked `unsafe`
   but its doc says "caller must ensure GPU has finished writing." This is a deliberate
   design choice (matching the standard library's approach to shared-memory views) but could
   surprise users.

3. **Pipeline::run() has a hardcoded sleep**: `std::thread::sleep(Duration::from_millis(100))`
   at line 1986 of hostcall.rs, before session shutdown. This looks like a workaround that
   should be investigated.

## Open Questions

1. Should `GpuVec` and scheduler types be re-exported from the `async_gpu` facade?
2. Should `GpuHostError::Verification` be split into more specific variants?
3. Should error types migrate to `thiserror` (as CLAUDE.md suggests)?

## Impact on Downstream Tasks

- **API consistency fixes** (audit-api.2+): Should address:
  - HostcallBuffer pub fields → pub(crate) + accessors
  - GpuVec facade re-export
  - onnx_rt missing docs
- **Error type cleanup**: Consider splitting Verification into typed variants (e.g.,
  PtxLoadFailed, KernelLaunchFailed, SyncFailed).
- **thiserror migration**: A mechanical refactor if the team decides to adopt it.
- All `#[allow(unused_mut)]` in warp.rs are justified — no action needed.
- All `#[allow(dead_code)]` violations: zero found — policy is clean.
