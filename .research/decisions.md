# Architecture Decision Records

Record important technical decisions here as they emerge from research.

## Format

### ADR-{N}: {Title}
- **Date**: YYYY-MM-DD
- **Status**: proposed | accepted | rejected | superseded
- **Context**: Why this decision was needed
- **Decision**: What was chosen
- **Rationale**: Why this choice
- **Alternatives**: Options considered but not adopted

---

### ADR-1: Use ptx-kernel ABI on nvptx64-nvidia-cuda target
- **Date**: 2026-03-11
- **Status**: accepted
- **Context**: Need to choose between ptx-kernel (stable, NVPTX-only), gpu-kernel (nightly, cross-vendor), and rustc_codegen_nvvm for GPU kernel compilation.
- **Decision**: Use `extern "ptx-kernel"` on the upstream `nvptx64-nvidia-cuda` target with `-Zbuild-std`.
- **Rationale**: (1) gpu-kernel is already in nightly but functionally identical to ptx-kernel on NVPTX — same LLVM calling convention. (2) VectorWare's own async/await demo uses ptx-kernel, not gpu-kernel. (3) No custom rustc patch needed. (4) codegen_nvvm diverges from VectorWare's approach and has maintenance concerns.
- **Alternatives**: `extern "gpu-kernel"` (migrate later for AMDGPU portability), `rustc_codegen_nvvm` (fallback if LLVM PTX backend proves too buggy).
- **Sources**: toolchain.1-c1, toolchain.3-c1
- **Amendment (bs2)**: `core::sync::atomic` is PROHIBITED in any code path that crosses the GPU-CPU boundary. All GPU-CPU synchronization must use the `gpu-atomics` crate (system-scope intrinsics/inline PTX). Intra-GPU synchronization may use `core::sync::atomic`. When porting std, all multi-warp sync primitives must be replaced. Rationale: LLVM bug #173993 — atomics emit no scope qualifier, fences silently dropped.

### ADR-2: Port Embassy executor using arch-spin as base template
- **Date**: 2026-03-11
- **Status**: accepted
- **Context**: Need to decide whether to port Embassy's executor to GPU or build a custom one from scratch.
- **Decision**: Port Embassy using its `arch-spin` feature as the starting template. Fall back to custom executor if register pressure or warp performance is unacceptable.
- **Rationale**: (1) Embassy is ~90% GPU-compatible out of the box. (2) `arch-spin` already provides the spin-loop model needed for GPU. (3) No TLS, no heap usage, lock-free task queue. (4) VectorWare confirmed Embassy works on GPU. (5) Only 3 concrete changes needed: no-op __pender, disable integrated-timers, SM70+ target.
- **Alternatives**: Custom minimal executor from scratch (fallback option if Embassy port proves too heavy on registers).
- **Sources**: async-runtime.1-c1
