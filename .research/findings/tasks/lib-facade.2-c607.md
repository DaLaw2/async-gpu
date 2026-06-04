# lib-facade.2 — Experiment: create crates/async-gpu/ with curated re-exports

**Task**: Experiment — create async-gpu facade crate  
**Status**: done  
**Cycle**: 607  

---

## What Was Done

Created `crates/async-gpu/` as a pure re-export facade over `gpu-host`. The crate
exposes the user-facing API surface only — no internal types, no PTX constants, no
raw allocation helpers.

## Re-export Surface (verified against actual gpu-host exports)

### Always available (no features)
| Export | Source |
|--------|--------|
| `async_gpu::gpu` (module) | `gpu_host::gpu` — run, run_with_output, launch, custom |
| `async_gpu::GpuHostError` | `gpu_host::error::GpuHostError` |
| `async_gpu::GpuKernelErrorInfo` | `gpu_host::error::GpuKernelErrorInfo` |
| `async_gpu::Result` | `gpu_host::Result` (type alias) |
| `async_gpu::GpuRuntime` | `gpu_host::GpuRuntime` |
| `async_gpu::HostcallSession` | `gpu_host::HostcallSession` |
| `async_gpu::MappedBuffer` | `gpu_host::MappedBuffer` |
| `async_gpu::Pipeline` | `gpu_host::Pipeline` |
| `async_gpu::GpuStream` | `gpu_host::streams::GpuStream` |
| `async_gpu::model_dir` | `gpu_host::model_dir` |

### Feature-gated
| Feature | Export | Source |
|---------|--------|--------|
| `nn` | `async_gpu::nn` (module) | `gpu_host::nn` |
| `async` | `async_gpu::async_rt` (module) | `gpu_host::async_rt` |

## Verification

| Check | Result |
|-------|--------|
| `cargo +stable check -p async-gpu` | OK |
| `cargo +stable check -p async-gpu --features nn` | OK |
| `cargo +stable clippy -p async-gpu` | OK |
| `RUSTDOCFLAGS='-D missing_docs' cargo +stable doc -p async-gpu` | OK |
| `cargo +stable check --manifest-path examples/hostcall/hello-gpu/host/Cargo.toml` | OK |
| `bash scripts/ci-lint.sh` | All checks passed |

## Design Deviations from lib-facade.1

None significant. The design was accurate — all referenced types exist in gpu-host
with the expected paths and visibility.

## Files Changed

| File | Change |
|------|--------|
| `Cargo.toml` | Added `crates/async-gpu` to workspace members |
| `crates/async-gpu/Cargo.toml` | New — facade crate manifest |
| `crates/async-gpu/src/lib.rs` | New — curated re-exports with doc comments |
| `examples/hostcall/hello-gpu/host/Cargo.toml` | Migrated from `gpu-host` to `async-gpu` |
| `examples/hostcall/hello-gpu/host/src/main.rs` | Migrated imports from `gpu_host` to `async_gpu` |
| `scripts/ci-lint.sh` | Added async-gpu to fmt, clippy, doc, and check steps |

---

**STATUS**: done  
**SUMMARY**: Created `crates/async-gpu/` facade crate with curated re-exports of gpu-host's user-facing API surface. Migrated the hello-gpu example to depend on `async-gpu` instead of `gpu-host` directly. All CI lint checks pass including fmt, clippy, doc (-D missing_docs), and cargo check with both default and `nn` features.  
**FILES_CHANGED**: Cargo.toml, crates/async-gpu/Cargo.toml, crates/async-gpu/src/lib.rs, examples/hostcall/hello-gpu/host/Cargo.toml, examples/hostcall/hello-gpu/host/src/main.rs, scripts/ci-lint.sh
