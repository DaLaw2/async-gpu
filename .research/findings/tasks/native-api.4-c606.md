# native-api.4 — Migrate all kernels from extern "ptx-kernel" to extern "gpu-kernel"

**Kind**: experiment
**Status**: done
**Conversation**: c606

## Summary

Mechanical migration of all 213 `extern "ptx-kernel"` call sites across 34 kernel/test/example files to `extern "gpu-kernel"`, plus the `#![feature(abi_ptx)]` → `#![feature(abi_gpu_kernel)]` feature gate swap in 16 crate roots.

## What Changed

### Phase 1: Feature gate swap (16 files)
- All kernel, test, and example crate `lib.rs` files: `#![feature(abi_ptx)]` → `#![feature(abi_gpu_kernel)]`
- Doc comments in `gpu-runtime/src/lib.rs` and `gpu-runtime/src/thread.rs` updated

### Phase 2: ABI string replacement (34 files, 213 sites)
- All `extern "ptx-kernel"` → `extern "gpu-kernel"` in function signatures
- Comment references updated (e.g., basic.rs line 1)
- `warp-macro/src/lib.rs` codegen: quote! block now emits `extern "gpu-kernel"`
- `examples/std/thread-demo/src/main.rs` doc comment updated

### Phase 3: gpu_kernel_abi feature removal (3 files)
- `gpu-kernel/Cargo.toml`: removed `gpu_kernel_abi = []` feature
- `gpu-kernel/src/lib.rs`: removed `#![cfg_attr(feature = "gpu_kernel_abi", feature(abi_gpu_kernel))]`
- `gpu-kernel/src/thread_test.rs`: removed `#[cfg(feature = "gpu_kernel_abi")]` guard on `gpu_kernel_demo`

### Phase 4: Host-side cleanup (1 file)
- `gpu-host/src/main.rs`: `gpu_kernel_demo` test now asserts success (no longer gracefully skips)

## Verification

1. `cargo +nightly-2026-06-03 build --release -p gpu-kernel --target nvptx64-nvidia-cuda` — OK
2. `cd crates/kernel/gpu-kernel-std && cargo +nightly-2026-06-03 build --release` — OK
3. `AUTO_BUILD_KERNEL=0 cargo +stable build --release -p gpu-host` — OK
4. `AUTO_BUILD_KERNEL=0 ONLY_TEST=gpu_run cargo +stable run --release -p gpu-host` — PASSED (both thread_spawn_test and gpu_kernel_demo)
5. `bash scripts/ci-lint.sh` — All checks passed

## Key Finding

The `gpu_kernel_demo` kernel, previously gated behind `gpu_kernel_abi` feature and skipped at runtime, now compiles unconditionally and passes:
```
gpu::launch("gpu_kernel_demo", 2, 128)...
  result = [42, 99]
extern "gpu-kernel" ABI — PASSED
```

This confirms `extern "gpu-kernel"` is functionally identical to `extern "ptx-kernel"` on the NVPTX target, as predicted by the investigation (native-api.3).

## Files Changed (37 total)

- crates/kernel/gpu-kernel/Cargo.toml
- crates/kernel/gpu-kernel/src/lib.rs
- crates/kernel/gpu-kernel/src/{basic,compute_cnn,compute_demo,compute_fused,compute_gemm,compute_math,compute_mma,compute_persistent,compute_physics,compute_search,compute_transformer,hostcall_kernels,hybrid,pipeline,thread_test,warp}.rs
- crates/kernel/gpu-kernel-std/src/lib.rs
- crates/macro/warp-macro/src/lib.rs
- crates/core/gpu-host/src/main.rs
- crates/core/gpu-runtime/src/{lib,thread}.rs
- crates/test/{async-hostcall-test,async-pipeline-test,embassy-test,gpu-std-test,multi-warp-test,std-build-test}/src/lib.rs
- examples/hostcall/{async-io,async-pipeline,hello-gpu,parallel-search,tcp-echo,vector-math}/kernel/src/lib.rs
- examples/hostcall/warp-cooperative/src/lib.rs
- examples/std/thread-demo/src/main.rs
