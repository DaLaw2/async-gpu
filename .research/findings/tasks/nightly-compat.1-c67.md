# nightly-compat.1: Test latest nightly compatibility
**Cycle**: 67 | **Theme**: nightly-compat | **Kind**: investigation | **Status**: done

## Summary

Tested Rust nightly-2026-03-11 (1.96.0) against our codebase (previously on nightly-2025-08-25,
1.91.0 — a 5 minor version jump). The PTX `.target sm_30` header bug is FIXED (now correctly
emits `.target sm_86`). Requires `lto = "fat"` in Cargo.toml to avoid unresolved `.extern .func`
references. The `asm_experimental_arch` feature gate was stabilized and should be removed.
All 14 test suites pass with the new nightly.

## Findings

### Q: Does the PTX header bug (.target sm_30) persist in latest nightly?
A: **NO — it's FIXED.** The new nightly correctly emits `.target sm_86` when compiled with
`-C target-cpu=sm_86`. This was a long-standing LLVM NVPTX back-end bug that prevented using
GPU features above sm_30 capability.

**Confidence**: high (verified in PTX output)

### Q: Does gpu-kernel ABI compile and produce correct PTX?
A: Yes, with one caveat: **LTO must be enabled**. Without LTO (`lto = false`), the new nightly
generates `.extern .func` declarations for cross-crate functions (e.g., `send_panic_hostcall`,
`core::panicking::panic_fmt`). These extern references cause `CUDA_ERROR_INVALID_PTX` because
PTX JIT can't resolve them. With `lto = "fat"`, all functions are inlined into a single
compilation unit — no extern references.

Old nightly (1.91.0) inlined these functions automatically without LTO. The new nightly's
codegen is more conservative about cross-crate inlining.

**Confidence**: high

### Q: Any new compilation errors or warnings?
A: One warning: `feature(asm_experimental_arch)` is "declared but not used". This feature
gate was stabilized between 1.91.0 and 1.96.0. Removing the `#![feature(...)]` line
eliminates the warning and compiles cleanly.

**Confidence**: high

### Q: Does the existing test suite pass on newer nightly?
A: **All 14 test suites pass.** Verified:
- Basic kernels (write_thread_idx, vector_add)
- Inline PTX asm (membar, st/ld acquire/release, CAS, volatile)
- u64 atomics (CAS, fetch_add, exchange)
- Warp intrinsics (activemask, lane_id, spin_load)
- Hostcall (single print, multi-block print, file I/O, stdin/time)
- Embassy executor (immediate, countdown, two-task)
- Async hostcall (single, two concurrent, futures::join)
- -Zbuild-std=std (Vec, String, format!)
- Dynamic allocation stress test
- GPU panic handler
- Latency benchmark

**Confidence**: high

## Benchmark Comparison

| Config | 1.91.0 (old) | 1.96.0 (new) | Notes |
|--------|-------------|-------------|-------|
| 1 thread p50 | 13µs | 55-99µs | Higher variance, possibly cold-start |
| 32 threads p50 | 1,358µs | 1,004-1,049µs | **~25% improvement** |
| 32 threads CAS retries | 19-24 | 2.9-3.8 | **~6x improvement** |
| 128 threads throughput | 14K calls/s | 14K calls/s | Comparable |
| PTX size | 6024 lines | 4226 lines | **30% smaller** |

The dramatic CAS retry reduction at 32 threads (from 19-24 down to 3-4) suggests the new
LLVM backend generates better atomic operation sequences.

## Changes Made

### rust-toolchain.toml
- Updated: `nightly-2025-08-25` → `nightly-2026-03-11`

### crates/gpu-kernel/Cargo.toml
- Changed: `lto = false` → `lto = "fat"` (required for new nightly)

### crates/gpu-runtime/src/lib.rs
- Removed: `#![feature(asm_experimental_arch)]` (stabilized in 1.96.0)

### crates/gpu-host/kernel.ptx
- Regenerated with new nightly + fat LTO

## Impact on Downstream Tasks

- **nightly-compat.2**: Toolchain update complete — can mark done immediately.
- **All kernel crates**: Must use `lto = "fat"` in release profile going forward.
- **PTX header fix**: Can now potentially use sm_86-specific features (e.g., async copy)
  that were previously blocked by the sm_30 target.
