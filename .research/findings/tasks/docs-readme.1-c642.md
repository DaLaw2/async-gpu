# docs-readme.1 — README vs Actual Capabilities Audit

## Summary

Audited README.md (514 lines) against the codebase. The README covers the core
value proposition well but has significant gaps: **12 major capabilities are
absent or buried**, **3 factual errors**, and **2 stale counts**. The feature
matrix (L119-138) and crate map (L484-501) are the most outdated sections.

## Findings

### A. Capabilities NOT mentioned in the README

| # | Capability | Source | Notes |
|---|-----------|--------|-------|
| 1 | **GpuArray\<T\>** — transparent data with 4-state residency | `gpu-host/src/gpu_array.rs` | Zero-copy below 64KiB, auto host-device sync. Not mentioned anywhere. |
| 2 | **AutoTuner** — warmup-based block-size tuning + cache | `gpu-host/src/auto_tune.rs` | TuningKey, TuningCache, occupancy-aware candidate filtering. Not mentioned anywhere. |
| 3 | **AutoScheduler / CpuScheduler / GpuScheduler** — unified CPU/GPU work routing | `gpu-host/src/scheduler.rs` | `par_map` auto-routes by data size. Not mentioned anywhere. |
| 4 | **SharedRef / GlobalRef (tiered memory)** — address-space-aware GPU pointers | `gpu-runtime/src/tiered_mem.rs` | `GpuRef<'scope, T, Tier>` emitting `ld.shared`/`ld.global`. Not mentioned anywhere. |
| 5 | **Dynamic dispatch / Box\<dyn Trait\> / hashbrown on GPU** | `gpu-kernel-test/src/lib.rs:3894` | hashbrown HashMap/HashSet with internal `&dyn FnMut` working on GPU. Not mentioned. |
| 6 | **GpuHashMap** — fixed-capacity GPU hash map (CAS-based) | `gpu-runtime/src/collections.rs` | Lock-free concurrent insert/get. Not mentioned. |
| 7 | **FlightRecorder** — mapped-memory ring buffer for post-mortem tracing | `gpu-runtime/src/flight_recorder.rs`, `gpu-host/src/lib.rs:57` | Fire-and-forget GPU trace events. Not mentioned in README. |
| 8 | **#[gpu_test] macro** — GPU test framework | `crates/test/gpu-test-macro/` | `#[gpu_test]` proc macro with custom thread/grid config. Not mentioned. |
| 9 | **Gradient checkpointing** — trade compute for memory | `gpu-host/src/nn/autograd/checkpoint.rs` | Re-executes forward during backward to save memory. Not mentioned. |
| 10 | **Tape-level fusion detection** — autograd fusion plan | `gpu-host/src/nn/fusion.rs` | Greedy longest-match pattern detection (MatmulBiasGelu, ElemAddLayerNorm, MatmulBias). Only ONNX fusion mentioned, not autograd tape fusion. |
| 11 | **Unified channel API** — auto-selects shared vs global transport | `gpu-runtime/src/unified_channel.rs` | `ScopedOneshotSender/Receiver`, `ScopedMpscSender/Receiver`. Not mentioned. |
| 12 | **Cross-block work coordination** — GridScope work dispatch | `gpu-runtime/src/grid_work.rs` | Coordinator/worker pattern without cooperative launch. GridScope mentioned in feature matrix L128 but grid_work details absent. |

### B. Capabilities mentioned but INSUFFICIENTLY covered

| # | Capability | README location | Gap |
|---|-----------|-----------------|-----|
| 1 | **par_iter** (GPU iterators) | L159 only (inside collapsed details) | One line. Missing from feature matrix, no code example, no standalone example link. Task mentions "standalone examples for par_iter" as success criteria. |
| 2 | **Structured concurrency** | Feature matrix L128, code example L273-289 | `GridScope` work dispatch, unified channels, `DisjointSlice` + `WarpIndex` safety integration not shown. |
| 3 | **GPU coroutines** | Feature matrix L132 only | One line. No code example, no details section. |
| 4 | **Compile-time cost model** | Feature matrix L133 only | One line. No explanation of `KernelResources`, `SmConfig`, `OccupancyLevel`, `WarningConfig`, or `KernelWarning` diagnostics. |
| 5 | **GPU generics** | Feature matrix L130 only | One line "Full trait system on GPU." Missing `GpuReducible`, `GpuTransformable` traits. |
| 6 | **GPU panic** | Feature matrix L129 only | One line. No details about `GpuKernelResult`, block/warp/lane metadata format. |

### C. Factual Errors / Stale Information

| # | Location | Issue | Correct value |
|---|----------|-------|---------------|
| 1 | L131 | `ThreadIndex<'kernel>` | Type does not exist. Actual type is `WarpIndex` (in `gpu-runtime/src/safety.rs`). |
| 2 | L498 | "7 integration test crates" | Actually 9 crates: async-hostcall-test, async-pipeline-test, embassy-test, gpu-critical-section, gpu-std-test, gpu-test-harness, gpu-test-macro, multi-warp-test, std-build-test. |
| 3 | L143 | "130+ kernels total" | Actual count is 245 kernel entry points (`pub unsafe extern "gpu-kernel"`). |

### D. Missing structural elements

| # | Element | Status |
|---|---------|--------|
| 1 | **async-gpu facade crate** | Not in crate map (L484-501). `crates/async-gpu/` is the user-facing crate but unlisted. |
| 2 | **docs/ directory** | Contains ARCHITECTURE.md, CHANGELOG.md, VALIDATION.md, getting-started.md, DESIGN-executor.md. Not linked from README. |
| 3 | **GpuVec** | Listed in `gpu-host/src/lib.rs:54` as a key type. Not in README core types table (L405-411). |
| 4 | **Pipeline** | Listed in `gpu-host/src/lib.rs:56` as key type. Not in README core types table. |
| 5 | **FlightRecorder** (host-side) | Listed in `gpu-host/src/lib.rs:57` as key type. Not in README core types table. |

### E. ARCHITECTURE.md and CHANGELOG.md status

- **ARCHITECTURE.md** (`docs/ARCHITECTURE.md`): Covers hostcall protocol and MIR pass well, but MISSING kernel split (4 kernel crates), unified runtime (AutoScheduler), tiered memory (SharedRef/GlobalRef), GpuArray, AutoTuner, par_iter, nn module, ONNX runtime, autograd, and all v0.2+ features.
- **CHANGELOG.md** (`docs/CHANGELOG.md`): Covers through "Unreleased (v0.2.0 → current)" with 28 stories shipped. Appears reasonably complete but uses generic descriptions rather than linking to specific code.

## Open Questions

1. Should GpuArray and AutoTuner be promoted to the hero section or feature matrix? They represent the "transparent data" and "auto-tuning" breakthroughs.
2. Should the README link to `docs/ARCHITECTURE.md` and `docs/CHANGELOG.md`, or should those docs be at the repo root?
3. The README currently has 7 collapsible `<details>` sections. Adding more for missing features risks making it unwieldy. Should some features get their own docs?
4. The `GpuParallelIterator` name in L159 — is this the canonical name? The implementation uses `GpuSlice::par_iter()` returning adapter types. Clarify naming.
