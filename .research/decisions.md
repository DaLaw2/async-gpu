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

### ADR-4: Per-thread GPU executor with Embassy-first approach
- **Date**: 2026-03-12
- **Status**: accepted (reworked 2026-03-12 per rv3)
- **Context**: Need to decide executor granularity (per-thread vs per-warp vs per-block), memory layout, waker mechanism, and critical section strategy for running async/await on GPU. Original design (async-runtime.2) reviewed in rv3; reworked to address 7 issues.
- **Decision**: Each GPU thread (lane) runs its own independent executor instance. **Primary path**: Embassy executor via `arch-spin` + `lto = "fat"` (confirmed working by async-runtime.1.2 — no fork needed). **Fallback**: stripped-down custom GpuExecutor only if Embassy register pressure exceeds 64 regs/thread (measured in Phase 1). Key design points: (1) Poll-all-tasks model — no self-waking; executor unconditionally polls all non-completed tasks each cycle (matches Embassy arch-spin). (2) Per-lane packets for async hostcall — each lane independently acquires its own packet; pool sized at 32x vs synchronous warp-cooperative mode. (3) Pool exhaustion returns Poll::Pending (back-pressure), not a hard error. (4) No CURRENT_EXECUTOR global — Embassy handles waker-to-task routing internally; custom fallback uses stack-local references. (5) No-op critical section with explicit documentation warning against inter-thread misuse. (6) Register pressure measured immediately after Phase 1 (Embassy integration).
- **Rationale**: (1) VectorWare confirms per-thread executor works. (2) Fat LTO resolves ALL Embassy cross-crate calls — no fork, no vendoring. (3) Embassy's type erasure (TaskPool<F, N>) solves heterogeneous future storage. (4) Poll-all is simpler and matches Embassy's actual arch-spin behavior; self-waking added overhead and made nanosleep dead code. (5) Per-lane packets are the only feasible async model (lanes reach hostcall at different times, can't cooperatively fill). (6) Pending-based back-pressure is more resilient than hard errors for pool exhaustion.
- **Alternatives**: Per-warp executor (complex sync, rejected), custom-executor-first (duplicates Embassy work, rejected per rv3), warp-cooperative async hostcall (requires barrier, deferred to Phase 4), self-waking model (overhead with no benefit, rejected per rv3).
- **Sources**: async-runtime.1-c1, async-runtime.1.1-c12, async-runtime.1.2-c16, async-runtime.2-c16, rv3-async-runtime.2, async-runtime.2.1-c19

### ADR-5: GPU panic handler via SERVICE_PANIC hostcall
- **Date**: 2026-03-12
- **Status**: accepted
- **Context**: GPU panic handler currently does `loop {}`, causing hard hangs with no diagnostic output. Need panic messages to reach the host for debugging.
- **Decision**: Add `SERVICE_PANIC = 10` opcode. Panic handler formats message into PanicBuf (56 bytes max), packs threadIdx.x + blockIdx.x into metadata slot, sends via hostcall, then executes `trap; exit;`. Global static `HOSTCALL_BUF` pointer initialized by each kernel at entry. Best-effort delivery (skip to trap on pool exhaustion or timeout).
- **Rationale**: (1) Reuses existing hostcall protocol — no new transport mechanism. (2) 56-byte message is sufficient for most panic messages. (3) `trap` instruction cleanly terminates the thread on SM70+. (4) Global static is the only way to pass buffer pointer to `#[panic_handler]` (fixed signature). (5) Best-effort avoids double-panic.
- **Alternatives**: Device-side printf (CUDA-specific, no custom formatting), shared memory flag (limited info, no message text), custom exception handler (no PTX support).
- **Sources**: gpu-panic.1-c61

### ADR-6: Host listener I/O thread separation
- **Date**: 2026-03-12
- **Status**: accepted
- **Context**: Blocking FILE I/O handlers (OPEN, WRITE, READ, CLOSE) and STDIN stall the entire listener thread, preventing timely processing of fast services (PRINT, PANIC). STDIN can block indefinitely. Code was duplicated across two listener methods (`listen` and `listen_with_stdin`).
- **Decision**: Split listener into fast-path (inline) and slow-path (I/O thread via `mpsc` channel). Fast services (NOP, PRINT, TIME, PANIC) handled inline. Slow services (FILE I/O, STDIN) offloaded to dedicated I/O thread. Unified both listener implementations via `StdinSource` trait with `RealStdin` and `CannedStdin` implementations. Uses `std::thread::scope` for safe I/O thread lifecycle.
- **Rationale**: (1) Host profiling (host-scaling.1) showed NOP processing takes ~2µs — host CPU is NOT the bottleneck. (2) Blocking I/O is the only service that stalls other packets. (3) Channel overhead (~100ns) is negligible vs FILE I/O cost (10-500µs). (4) No protocol changes needed — GPU doesn't know which host thread responded. (5) `StdinSource` trait eliminates 100+ lines of duplicated dispatch code.
- **Alternatives**: Multi-threaded dispatch (ready stack doesn't partition well), async runtime/tokio (over-engineering for CPU-bound fast path), per-warp packet pools (requires GPU protocol changes).
- **Sources**: host-scaling.1-c63, host-scaling.2-c64
