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

### ADR-3: ROCm-style lock-free two-stack hostcall protocol
- **Date**: 2026-03-11
- **Status**: accepted
- **Context**: Need to design a GPU-to-host RPC mechanism for libc/std operations. Investigated ROCm hostcall (two-stack pool), CUDA printf (FIFO ring), and double-buffering.
- **Decision**: Adopt ROCm's lock-free two-stack pool design with warp-granular packets (32 lanes × 8 u64 slots), tagged pointers for ABA prevention, doorbell counter for host notification, and adaptive-timeout host polling. Buffer allocated via `cuMemHostAlloc(DEVICEMAP|PORTABLE)`. All GPU-side synchronization uses `gpu-atomics` system-scope primitives.
- **Rationale**: (1) ROCm's design is proven in production. (2) Lock-free CAS allows multi-warp concurrency without barriers. (3) Warp-granular packets match NVIDIA's SIMT model. (4) LIFO ordering is acceptable for independent RPC requests. (5) CUDA printf is write-only (no response path). (6) Double-buffering stalls all warps (unacceptable).
- **Alternatives**: CUDA printf-style FIFO (no response path, rejected), double-buffer (stalls all warps, rejected), single global mutex (serializes all RPCs, rejected).
- **Sources**: hostcall.1-c1, hostcall.2-c1, atomics.3-c1, hostcall.3-c10

### ADR-4: Per-thread GPU executor with bitmask run queue
- **Date**: 2026-03-12
- **Status**: accepted
- **Context**: Need to decide executor granularity (per-thread vs per-warp vs per-block), memory layout, waker mechanism, and critical section strategy for running async/await on GPU.
- **Decision**: Each GPU thread (lane) runs its own independent `GpuExecutor` instance. Executor state lives in registers/local memory. Run queue is a plain `u32` bitmask (no atomics needed — single owner). Waker encodes a task index in the `RawWaker` data pointer and sets a bit via plain `OR`. Critical section is a no-op (no shared mutable state in per-thread executor). Hostcall is exposed as an async `HostcallFuture` that returns `Poll::Pending` while waiting for host response. Default max 4-8 tasks per executor. Kernel exit when all tasks complete. Idle poll uses PTX `nanosleep` instruction.
- **Rationale**: (1) VectorWare confirms per-thread executor works ("each GPU thread runs its own executor instance"). (2) Zero synchronization overhead — no atomics, no locks for executor internals. (3) Maps directly to Embassy's `arch-spin` model. (4) Register pressure is manageable (~20-30 registers for 2 tasks). (5) Warp divergence is a performance concern, not a correctness concern, and is inherent to async. (6) Async hostcall enables overlapping host RPCs with other GPU-side computation — the key value proposition.
- **Alternatives**: Per-warp executor (less register pressure but complex warp-level sync, serializes task execution to lane 0), per-block executor (requires shared-memory sync, introduces bottlenecks), hybrid per-warp-with-per-lane-pools (too complex for initial implementation).
- **Sources**: async-runtime.1-c1, async-runtime.1.1-c12, async-runtime.2-c16
