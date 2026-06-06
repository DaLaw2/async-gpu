# docs-refresh.1: Audit ARCHITECTURE.md + CHANGELOG.md vs Current Project State

## Summary

All documentation lives in `docs/` which is **gitignored** (`docs/` on line 44 of `.gitignore`).
Nothing in `docs/` is tracked — it exists only on the local machine. There are no
ARCHITECTURE.md, CHANGELOG.md, DESIGN-executor.md, or VALIDATION.md at the repo root.

## Findings

### 1. ARCHITECTURE.md — Severely Outdated

File: `docs/ARCHITECTURE.md` (untracked, 348 lines)

**Crate Map is wrong.** The doc shows a flat structure (`gpu-host`, `gpu-runtime`,
`gpu-protocol`, `gpu-atomics`, `gpu-libc`, `warp-macro`, plus 4 examples). The actual
crate layout is:

| Documented | Actual |
|---|---|
| `gpu-host` | `crates/core/gpu-host` — exists, massively expanded |
| `gpu-runtime` | `crates/core/gpu-runtime` — exists, massively expanded |
| `gpu-protocol` | `crates/core/gpu-protocol` — exists |
| `gpu-atomics` | `crates/core/gpu-atomics` — exists |
| `gpu-libc` | `crates/core/gpu-libc` — exists |
| `warp-macro` | **MISSING** — no warp-macro crate found in repo |
| *(not documented)* | `crates/async-gpu` — facade crate (new) |
| *(not documented)* | `crates/kernel/gpu-kernel-core` — kernel split (new) |
| *(not documented)* | `crates/kernel/gpu-kernel-compute` — kernel split (new) |
| *(not documented)* | `crates/kernel/gpu-kernel-io` — kernel split (new) |
| *(not documented)* | `crates/kernel/gpu-kernel-test` — kernel split (new) |
| *(not documented)* | `crates/test/*` — 9 test crates (new) |
| 4 examples | 12 hostcall + 15 std examples = 27 total |

**Missing major subsystems** (all undocumented in ARCHITECTURE.md):

- **Kernel split**: 4 kernel crates (`gpu-kernel-{core,compute,io,test}`)
- **Unified runtime**: `AutoScheduler`, `CpuScheduler`, `GpuScheduler` in `gpu-host/src/scheduler.rs`
- **Tiered memory**: `SharedRef<T>`, `GlobalRef<T>`, `GpuRef<'scope, T, Tier>` in `gpu-runtime/src/tiered_mem.rs`
- **Auto-fusion pipeline**: `nn::fusion` module + `onnx_rt::fusion` in `gpu-host`
- **PTX auto-discovery**: `PtxModule` catalog + multi-cubin loader in `gpu-host`
- **GpuArray<T>**: Transparent host-device data type in `gpu-host/src/gpu_array.rs`
- **AutoTuner**: Kernel autotuning framework in `gpu-host/src/auto_tune.rs`
- **Neural network stack**: `nn/` module with autograd, layers, models, ops, fusion, tensor
- **ONNX runtime**: `onnx_rt` module with protobuf parser + graph executor
- **GPU iterators**: `par_iter` in `gpu-runtime`
- **Structured concurrency**: `scope.rs`, `block.rs` in `gpu-runtime`
- **GPU collections**: `collections.rs` in `gpu-runtime`
- **Channel primitives**: `channel.rs`, `block_channel.rs`, `unified_channel.rs` in `gpu-runtime`
- **GPU test framework**: `gpu-test-harness`, `gpu-test-macro` crates
- **Model inference**: `model.rs`, `model_generic.rs`, `model_yolo.rs`, `tokenizer.rs` in `gpu-host`
- **Resource reporting**: `resource_report.rs` in `gpu-host`
- **CUDA streams**: `streams.rs` in `gpu-host`

**Sections that are still accurate**:
- Hostcall protocol design (buffer layout, request lifecycle, lock-free stack)
- Warp-cooperative MIR pass explanation
- Platform adaptation layer (PAL) concept
- Key constants table

**Sections partially accurate**:
- Services table — missing services added post-v0.2.0 if any
- Build pipeline — toolchain date references `nightly-2026-03-11`, now `nightly-2026-06-03`
- PTX compilation flow — cubin path not documented

### 2. CHANGELOG.md — Missing Recent Activity

File: `docs/CHANGELOG.md` (untracked, 129 lines)

- **Latest entry**: "Unreleased (v0.2.0 → current)" covering cycles 309-638
- **Current cycle**: 642 (per state.toml and git log)
- **Gap**: Cycles 639-642 are not captured in the unreleased section
- The header claims "329 cycles (309→638)" but should now cover through 642
- Recent cycle 642 work includes: Conv2D Winograd optimization (54.8% peak), ownership-memory
  (SharedRef/GlobalRef), GpuArray<T>, auto-tuning, gpu-dyn-dispatch — some of these may
  already be summarized in the "Unreleased" section but the cycle range is stale

### 3. Stale Docs Assessment

| File | Status | Recommendation |
|---|---|---|
| `docs/DESIGN-executor.md` | **Stale design doc** — 637 lines describing a GPU-side task spawning executor that was never implemented. The actual executor is `block_on()` + `SpinExecutor` in `gpu-runtime/src/executor.rs`. | Remove or move to `docs/design/` as historical context |
| `docs/VALIDATION.md` | **Partially stale** — references correct examples but toolchain version is now `nightly-2026-06-03` (doc says `nightly-2026-06-03` on line 29, so this was updated). Example list is incomplete (missing 23 newer examples). | Update example list and validate flow |
| `docs/getting-started.md` | Not audited in detail — may have similar staleness | Needs separate audit |

### 4. docs/ Directory Git Status

- `docs/` is listed in `.gitignore` on line 44 with comment `# Local docs (not tracked)`
- **None of these files are tracked in git** — they exist only locally
- This means documentation is invisible to any collaborator or CI system
- Decision needed: should docs be tracked? Current `.gitignore` entry is intentional

## Open Questions

1. **Should `docs/` be un-gitignored and tracked?** The current policy means documentation
   is invisible to collaborators. If the intent is "local-only scratch docs," the investment
   in ARCHITECTURE.md and CHANGELOG.md quality seems wasted.

2. **Should DESIGN-executor.md be removed?** It describes an unimplemented design. If kept,
   it should be clearly labeled as a design proposal, not current architecture.

3. **What is the correct cycle range for the Unreleased changelog section?** The git log shows
   active work at cycle 642, but the changelog header says 638.

4. **Where is warp-macro?** ARCHITECTURE.md documents it, but no `warp-macro` crate exists.
   Was it removed, renamed, or never created? The `warp-cooperative` example exists at
   `examples/hostcall/warp-cooperative/` but that's an example, not the proc-macro crate.
