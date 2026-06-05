# split-loader.2: Host Loader Multi-Module Update

**Task**: Update host loader (gpu.rs, KernelRegistry) for multi-module PTX
**Date**: 2026-06-05 | **Cycle**: 619

## Summary

Updated the host-side loader to work with the 4-crate kernel split. Added `PtxModule` struct and `ALL` catalog to the `ptx` module for auto-discovery, and added `.module()` method to `CustomLaunchBuilder` for explicit module selection. All existing call sites continue working unchanged via backward-compatible aliases.

## Changes Made

### 1. `ptx` module (lib.rs) — Added `PtxModule` + `ALL` catalog

Added a `PtxModule` struct with `name` and `ptx` fields, and a `const ALL` array listing all 4 modules. This enables auto-discovery APIs and provides structured iteration over available modules.

Cubins are NOT embedded yet (only `kernel_std.cubin` exists as the monolithic binary). The `PtxModule` struct intentionally omits cubin fields — these will be added when per-crate cubins are built.

### 2. `CustomLaunchBuilder` (gpu.rs) — Added `.module()` method

New `.module(&PtxModule)` convenience method that sets the PTX source from a catalog entry. Example: `gpu::custom("my_io_kernel").module(&ptx::ALL[2]).prepare()`.

### 3. KernelRegistry — Verified, Zero Changes

`KernelRegistry::new()` takes `ptx_src: &str` and `init_default()` uses `crate::ptx::KERNEL` which aliases to `KERNEL_COMPUTE`. All ~60 ML_KERNELS entries are from compute files (gemm, transformer, cnn, physics, fused, backward). Confirmed: no changes needed.

### 4. gpu-test-macro — Verified, Zero Changes

The macro expands to `gpu_host::ptx::KERNEL_STD` (aliased to `KERNEL_TEST`) and loads cubin from `kernel_std.cubin`. Both aliases work correctly. All 3 `#[gpu_test]` functions pass.

### 5. GpuStdModule — Verified, Zero Changes

Already accepts `ptx_src: &str` and `cubin: &[u8]` as parameters. Callers select their module explicitly. No changes needed.

## Verification

- `cargo build -p gpu-host --features nn,cublas --lib` — compiles clean
- `cargo build -p gpu-test-harness` — compiles clean
- `cargo build -p gpu-test-macro` — compiles clean
- `cargo test -p gpu-host --features nn,cublas --lib` — 100/103 pass (3 pre-existing benchmark failures)
- `cargo test -p gpu-test-harness --test gpu_tests` — 5/5 pass (includes GPU kernel launches)

## Files Changed

- `crates/core/gpu-host/src/lib.rs` — added `PtxModule` struct, `ALL` catalog to `ptx` module
- `crates/core/gpu-host/src/gpu.rs` — added `.module()` to `CustomLaunchBuilder`
