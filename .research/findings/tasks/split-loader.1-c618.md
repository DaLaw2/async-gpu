# split-loader.1: Per-crate PTX constants + backward-compat aliases

**Task**: Per-crate PTX constants with backward-compatible aliases
**Date**: 2026-06-05 | **Cycle**: 618

## Summary

Added per-crate PTX constants to `gpu-host` for all 4 kernel crates produced by
the kernel split (gpu-kernel-core, gpu-kernel-compute, gpu-kernel-io,
gpu-kernel-test). Backward-compatible aliases ensure zero call-site changes.

## Changes

### 1. Built all 4 kernel crate PTX files

All 4 crates build successfully to nvptx64 PTX:
- `gpu_kernel_core.ptx`    — 1.0 MB (basic ops, math)
- `gpu_kernel_compute.ptx` — 1.4 MB (GEMM, transformer, CNN, physics)
- `gpu_kernel_io.ptx`      — 1.8 MB (hostcall, pipeline)
- `gpu_kernel_test.ptx`    — 7.2 MB (std demos, warp tests, par_iter)

PTX files copied to `crates/core/gpu-host/kernel_{core,compute,io,test}.ptx`.

### 2. Updated build.rs

Replaced single-crate build with a loop over all 4 kernel crates using a
`KernelCrate` descriptor struct. Each crate is built independently; failures
fall back to existing PTX (same resilience as before). Backward-compat copies
`kernel_test.ptx → kernel.ptx` and `kernel_test.ptx → kernel_std.ptx` are
still produced so old code paths that `include_str!("../kernel.ptx")` or
`include_str!("../kernel_std.ptx")` continue to work during transition.

### 3. Updated ptx module (lib.rs)

Added 4 canonical per-crate constants:
- `ptx::KERNEL_CORE`
- `ptx::KERNEL_COMPUTE`
- `ptx::KERNEL_IO`
- `ptx::KERNEL_TEST`

Added backward-compatible aliases (not deprecated yet — Phase 1):
- `ptx::KERNEL     = KERNEL_COMPUTE` — all KernelRegistry / gpu.rs call sites use ML kernels
- `ptx::KERNEL_STD = KERNEL_TEST`    — gpu-test-macro and harness use test kernels

Legacy test PTX constants (EMBASSY_TEST, etc.) unchanged.

### 4. Verification

- `cargo build -p gpu-host` — compiles successfully
- `cargo check -p gpu-host --test gpu_integration` — passes (uses `ptx::KERNEL`)
- `cargo check -p gpu-test-harness` — passes (uses `ptx::KERNEL_STD`)
- `cargo check -p gpu-test-macro` — passes (generates `ptx::KERNEL_STD` references)

All existing call sites work with zero code changes.

## Design Decisions

1. **KERNEL aliases to KERNEL_COMPUTE** (not KERNEL_TEST): Every `ptx::KERNEL`
   usage in the codebase is for ML/compute kernels (KernelRegistry, gpu.rs
   get_kernel, integration tests). The compute module contains these functions.

2. **No deprecation warnings yet**: The design doc proposes `#[deprecated]` on
   aliases, but adding them now would produce warnings in ~30 call sites across
   the codebase. Better to do this in a dedicated migration pass (Phase 3).

3. **build.rs watches individual .rs files**: Instead of hard-coding specific
   source files, the build script enumerates `src/*.rs` for each kernel crate
   via `read_dir`. This is more maintainable as kernel crate sources grow.

## Files Changed

- `crates/core/gpu-host/build.rs` — multi-crate build loop
- `crates/core/gpu-host/src/lib.rs` — per-crate PTX constants + aliases
- `crates/core/gpu-host/kernel_core.ptx` — new (build artifact, gitignored)
- `crates/core/gpu-host/kernel_compute.ptx` — new (build artifact, gitignored)
- `crates/core/gpu-host/kernel_io.ptx` — new (build artifact, gitignored)
- `crates/core/gpu-host/kernel_test.ptx` — new (build artifact, gitignored)
