## Current Focus
**Cycle 639 — gpu-panic + feature-audit activated** (2026-06-06). Four investigation/experiment tasks completed: std panic path traced, unwrap/expect/assert verified, build audit (42/43 pass), API audit (6 issues found). Next: dispatch experiment tasks for both stories.

## Recent Decisions
- 2026-06-06: Brainstorm bs126 — activated gpu-panic (high) and feature-audit (high). Created 6 features, 12 tasks. Priority gate strictly enforced: compile-time-cost (medium) blocked until high stories complete.
- 2026-06-06: Key finding — panic_handler!() macro is never used in practice. All kernel crates use `#![feature(restricted_std)]`, routing panics through patched std's `default_hook` → SERVICE_PRINT hostcall → trap. The no_std panic path is dead code.
- 2026-06-06: Key finding — 42/43 crates compile. Only std-build-test fails (linker collision: `gpu_stdin_read` #[no_mangle] defined in both std-build-test and gpu-runtime).

## Tried & Rejected
- MIR-only register estimation: >3x error margin vs physical registers
- PTX virtual register counting: 2-5x overcount vs ptxas allocation
- Channels for streaming pipeline: adds buffering, not zero-buffered
- Per-lane yield values: changes SIMT model
- Off-direction stories (hardware-intrinsics, actor-model, gpu-repl, gpu-hot-reload, cross-vendor): removed

## Active Constraints
- GTX 1660 (sm_75): 192 GB/s, 5 TFLOPS FP32, 48KB smem, 64K regs
- Max 2 concurrent heavy subagents
- kernel_test.ptx JIT takes 20+ min — need pre-compiled cubin for GPU runtime tests
- Device function register inflation: 34 kernels show 112 regs / 50% occ falsely
- Priority gate: gpu-panic + feature-audit (high) block compile-time-cost (medium)

## Key Metrics
- 808 tasks completed, 56 stories completed across project history
- 3 strategic epics active: lang-complete, invisible-exec, perf-transparent
- gpu-panic: 2/6 tasks done (panic-std-intercept.1, panic-unwrap-verify.1)
- feature-audit: 2/6 tasks done (audit-build.1, audit-api.1)
- API audit: 6 issues (GpuHostError overload, pub fields, missing re-exports, missing docs, manual Error impls, Pipeline sleep hack)

## Next
1. panic-std-intercept.2: modify default_hook to include GPU warp/thread metadata
2. panic-unwrap-verify.2: test unwrap/expect/assert in std kernel with patched std
3. audit-build.2: fix std-build-test linker collision
4. audit-api.2: fix 6 API inconsistencies (depends on audit-build.2)
