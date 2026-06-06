## Current Focus
**Cycle 640 — gpu-panic near completion, feature-audit progressing** (2026-06-06). Patched std now includes GPU block/warp/lane metadata in panic output. std-build-test linker collision fixed (43/43 compile). 3 features completed (panic-std-intercept, panic-unwrap-verify, audit-build). Remaining: gpu_assert deprecation + runtime/API audit.

## Recent Decisions
- 2026-06-06: Patched std default_hook modified for `target_os = "cuda"` — panic output now: `thread 'main' (block B, warp W, lane L) panicked at location:\nmessage`. Uses inline PTX `%ctaid.x` and `%tid.x`.
- 2026-06-06: std-build-test fixed — removed ~300 lines of duplicated `#[no_mangle]` hostcall code, delegated to gpu-runtime.
- 2026-06-06: panic_handler!() macro is dead code — all kernel crates use `#![feature(restricted_std)]`.
- 2026-06-06: Gap identified — std abort path doesn't call `set_warp_trapped()` or `write_panic_to_result()`. Needs follow-up.

## Tried & Rejected
- MIR-only register estimation: >3x error margin vs physical registers
- PTX virtual register counting: 2-5x overcount vs ptxas allocation
- Channels for streaming pipeline: adds buffering, not zero-buffered
- Per-lane yield values: changes SIMT model

## Active Constraints
- GTX 1660 (sm_75): 192 GB/s, 5 TFLOPS FP32, 48KB smem, 64K regs
- Max 2 concurrent heavy subagents
- kernel_test.ptx JIT takes 20+ min — need pre-compiled cubin for GPU runtime tests
- Priority gate: gpu-panic + feature-audit (high) block compile-time-cost (medium)

## Key Metrics
- 811 tasks completed, 56 stories completed
- gpu-panic: 4/6 tasks done, 2 remaining (deprecate gpu_assert)
- feature-audit: 3/6 tasks done, 3 remaining (runtime audit + API fixes)
- API audit findings: 6 issues (GpuHostError overload, pub fields, missing re-exports, missing docs, manual Error impls, Pipeline sleep)

## Next
1. panic-deprecate-gpu-assert.1: design migration plan from gpu_assert! to standard assert!
2. audit-runtime.1: run all compilable examples on GPU, catalog runtime failures
3. audit-api.2: fix 6 API inconsistencies (depends on audit-build.2 ✅)
