# lib-cleanup.6 — Verify cargo doc --no-deps produces clean API docs

Status: **done**

## Findings

### gpu-host public API (default features)

**Re-exports** (top-level convenience):
- `GpuHostError`, `Result`
- `HostcallBuffer`, `HostcallSession`, `Pipeline`
- `MappedBuffer`
- `GpuRuntime`

**Modules**:
- `error` — Error types for gpu-host
- `gpu` — One-liner GPU launch API (run, launch, custom)
- `hostcall` — Host-side hostcall listener and buffer management
- `memory` — RAII wrappers for pinned, device-mapped host memory
- `runtime` — High-level GPU runtime (CudaDevice wrapper)
- `streams` — CUDA stream overlap support

**Functions**: none (model_dir now hidden)

### async-gpu public API (default features)

**Modules**: `gpu`
**Structs**: `GpuKernelErrorInfo`, `GpuRuntime`, `GpuStream`, `HostcallSession`, `MappedBuffer`, `Pipeline`
**Enums**: `GpuHostError`
**Functions**: none (model_dir now hidden)
**Type Aliases**: `Result`

### Issues found and fixed

1. **`mapped_mem` module visible in gpu-host docs** — Low-level unsafe allocation functions
   (`alloc_mapped_u32`, `free_mapped_mem`, etc.) used only by internal code and test harness.
   Added `#[doc(hidden)]` since it must remain `pub` for cross-crate test access.

2. **`model_dir` function visible in both crates** — Workspace-path utility that walks up
   to find `Cargo.toml [workspace]`. Internal to repo structure, not useful for end users.
   Added `#[doc(hidden)]` in gpu-host and async-gpu.

3. **lib.rs doc "Optional modules" section listed demo modules** — Listed `model` and
   `tokenizer` (gpt2 feature) which are demo-gated. Replaced with the actual optional
   modules: `nn`, `async_rt`, `streams`.

4. **Broken intra-doc links for feature-gated modules** — `nn` and `async_rt` are only
   compiled with their respective features, so `[`nn`]` links broke under default features.
   Changed to plain backtick references.

5. **`gpu` module lacked doc comment at declaration site** — Added one-line doc comment.

### Items NOT appearing in docs (verified)

- `ptx` module — already `#[doc(hidden)]`
- `cubin` module — already `#[doc(hidden)]`
- `model`, `model_generic`, `model_yolo`, `tokenizer`, `yolo_backbone` — `pub(crate)` without `demo` feature
- `mapped_mem` — newly `#[doc(hidden)]`
- `model_dir` — newly `#[doc(hidden)]`

### Assessment

The API surface is clean and ready for a getting-started guide. Both crates show
only user-relevant types and modules. The `async-gpu` facade provides a minimal,
well-documented entry point. The `gpu-host` crate shows the full runtime API for
advanced users without leaking internals.

### Verification

- `cargo doc --no-deps -p gpu-host` — clean, 0 warnings
- `cargo doc --no-deps -p async-gpu` — clean, 0 warnings
- `scripts/ci-lint.sh` — all checks passed
