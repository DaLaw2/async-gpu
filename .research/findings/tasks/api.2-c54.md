# api.2: Implement gpu-runtime facade crate
**Cycle**: 54 | **Theme**: api | **Kind**: experiment | **Status**: done

## Summary
Created `crates/gpu-runtime` — a facade crate that re-exports gpu-protocol,
gpu-atomics, gpu-critical-section, and provides a consolidated `hostcall` module
with all GPU-side hostcall helpers. Kernel authors now depend on one crate instead
of four, and get a `prelude` module for ergonomic imports.

## Findings

### Q: Does a single re-export crate simplify kernel authoring?
A: **Yes.** Before: kernel needed `gpu-protocol`, `gpu-atomics`, `gpu-critical-section`
as direct dependencies plus copy-pasted hostcall helpers. After: depend on `gpu-runtime`
only, use `gpu_runtime::prelude::*` for all imports.

**Confidence**: high (crate compiles and re-exports correctly)

### Q: Can we hide gpu-protocol/gpu-atomics behind a clean API?
A: **Partially.** The prelude re-exports the most-used items from both crates.
Advanced users can still access `gpu_runtime::gpu_protocol` and `gpu_runtime::gpu_atomics`
directly for full API access. The hostcall module provides higher-level wrappers
(`gpu_hostcall_print`, `gpu_hostcall_request`) that hide the CAS/push/pop details.

**Confidence**: high

### Q: What ergonomic wrappers are needed for hostcall operations?
A: Three levels provided:
1. **High-level**: `gpu_hostcall_print(buf, msg, len)` — fire-and-forget print
2. **Mid-level**: `gpu_hostcall_request(buf, service, fill_fn)` — generic request/response
3. **Low-level**: `hc_pop_free`, `hc_push`, `gpu_hostcall_release` — manual protocol

**Confidence**: high

## Files Created
- `crates/gpu-runtime/Cargo.toml`
- `crates/gpu-runtime/.cargo/config.toml`
- `crates/gpu-runtime/src/lib.rs`

## Design Notes
- `gpu-critical-section` is linked via `extern crate` to ensure the critical-section
  implementation is available when Embassy executor is used. No re-export needed.
- The `prelude` module intentionally does NOT include Embassy types — those remain
  a separate dependency so non-async kernels don't pull in the executor.
- Hostcall helpers are `#[inline(always)]` for Fat LTO inlining.

## Impact on Downstream Tasks
- api.3 can now create an example using gpu-runtime as the sole GPU dependency
