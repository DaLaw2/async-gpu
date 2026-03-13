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

### ADR-7: Sideband buffer for bulk data transfer
- **Date**: 2026-03-12
- **Status**: accepted
- **Context**: Hostcall packets have a 56-byte payload limit (7 slots × 8 bytes in lane 0). File I/O and future workloads need arbitrary-size data transfer between GPU and host. Three approaches considered: multi-packet chaining, sideband mapped buffer, enlarged packet slots.
- **Decision**: Allocate a separate CUDA mapped buffer ("sideband") alongside the hostcall buffer. GPU uses a bump allocator (atomic fetch_add) to reserve regions. Two new services: `SERVICE_BULK_WRITE` (11) for GPU→host file writes and `SERVICE_BULK_READ` (12) for host→GPU file reads. Hostcall packet carries `(fd, sideband_offset, length)` — data lives in the sideband buffer. Default sideband size: 1MB. Bump allocator is reset in bulk (at kernel start or after all pending ops complete).
- **Rationale**: (1) Zero changes to existing packet format — backward compatible. (2) No deadlock risk — one packet per request regardless of data size. (3) Existing CONTROL_FILLED/CONTROL_READY release-acquire fences provide all necessary GPU-host synchronization. (4) Bump allocator is simple and efficient for sequential I/O patterns. (5) Arbitrary data size up to sideband capacity.
- **Alternatives**: Multi-packet chaining (complex reassembly, deadlock risk if pool < data packets needed), enlarged packet slots (wastes memory for the common case of small messages, requires format change).
- **Sources**: large-payload.1-c69, large-payload.2-c70

### ADR-9: Warp-level Future — SIMT-convergent async on GPU
- **Date**: 2026-03-12
- **Status**: proposed
- **Context**: Current GPU async model uses per-thread Futures. Each lane has its own state machine; when lanes enter different enum variants, warp divergence occurs and SIMT throughput drops to 1/32. This is the normal outcome for any non-trivial async state machine on GPU, not an edge case. The fundamental conflict: GPU's SIMT model requires uniform control flow, but cooperative multitasking (async) naturally creates divergent control flow.
- **Decision**: Introduce a `WarpFuture` trait where one Future drives 32 lanes in lockstep. All lanes synchronously poll, yield, and resume together. State machine enum variant is uniform across all lanes; only data differs (SIMD semantics). Implementation in two phases: (1) Library + proc macro approach (no rustc changes) with custom `WarpFuture` trait, `#[warp_async]` proc macro, `warp_await!` macro with `__syncwarp()` barriers, and warp-aware executor. (2) If validated, upstream rustc proposal for target-specific async desugaring and SIMT-aware MIR pass.
- **Rationale**: (1) Warp-level Future is the only design that preserves both SIMT throughput and async ergonomics. (2) Hostcalls are a natural fit — warp collectively submits request via payload slots, collectively waits for response. (3) Phase 1 requires no compiler changes, allowing rapid prototyping. (4) The "no per-lane divergence" constraint aligns with GPU's SIMD philosophy — differences are in data, not control flow.
- **Alternatives**: (a) Accept warp divergence and rely on compiler/hardware to handle it (unacceptable: 1/32 throughput). (b) Only use per-block parallelism without warp-level async (limits concurrency). (c) Require rustc changes from the start (too slow, unvalidated design).
- **Sources**: User discussion 2026-03-12, VectorWare blog analysis

### ADR-10: Hybrid executor — per-thread compute blocks in WarpFuture
- **Date**: 2026-03-12
- **Status**: accepted
- **Context**: WarpFuture (ADR-9) requires all lanes to follow the same control flow. However, real GPU workloads often need per-lane divergent computation between I/O phases (e.g., read data → process per-thread → write results). Need a way to safely mix warp-cooperative I/O (WarpFuture) with per-thread computation in the same state machine.
- **Decision**: Allow per-thread compute blocks as states in a WarpFuture state machine. The invariant is: **per-thread blocks MUST NOT yield (return WarpPoll::Pending on a hostcall)**. They must be pure computation with no I/O. The pattern is:
  1. All lanes enter the COMPUTE state together (state is broadcast from lane 0)
  2. Each lane computes independently (may have different iteration counts, branches)
  3. `syncwarp(active_mask)` reconverges all lanes
  4. Lane 0 advances the state
  5. `syncwarp(active_mask)` ensures all lanes see the new state
  6. Return `WarpPoll::Pending` to transition to next state
- **Safety enforcement**: Documentation + code review. No compile-time enforcement in the current design.
  - **Why not a macro?** The pattern is 5-6 lines and self-explanatory. A `per_thread_block!` macro would save ~3 lines but add cognitive overhead. Deferred to the `#[warp_async]` proc macro work if a DSL emerges.
  - **Why not type state?** Would require splitting WarpFuture into `WarpFuture<Cooperative>` and `WarpFuture<PerThread>` modes. Heavy type machinery for a simple invariant.
  - **Why not runtime detection?** `activemask()` check after `syncwarp()` could detect if lanes dropped out (diverged due to yield). However, this only detects the bug post-hoc — the damage (warp deadlock) has already occurred. Not useful for prevention.
- **What happens if the invariant is violated**: If a lane yields (hostcall → WarpPoll::Pending) inside a per-thread block while other lanes continue computing, the yielding lane re-enters the WarpExecutor poll loop. On the next poll, `broadcast_u32` reads state from lane 0 — but lane 0 may be in a different state than the yielding lane expected. This causes undefined behavior: the yielding lane may execute the wrong state's code, corrupt data, or deadlock at the next `syncwarp()` where it expects lanes that are no longer convergent.
- **Validated by**: hybrid-executor.1 (basic PoC, 6 states) and hybrid-executor.2 (stress test: 9 states, 3100x lane duration variance, XOR-fold with branches). Both passed with all results correct.
- **Rationale**: (1) The pattern is simple and composes linearly — adding more I/O↔compute switching points is mechanical. (2) syncwarp() is a hardware barrier that handles arbitrary lane timing differences. (3) The switching overhead is negligible (~5-10 ns per syncwarp). (4) Documentation-based enforcement is appropriate at this maturity level — no external users yet.
- **Alternatives**: (a) Separate kernels for I/O and compute phases (context switch overhead, doesn't compose). (b) Always use per-thread futures (32x CAS overhead for I/O). (c) Compile-time enforcement via type system (premature, deferred).
- **Sources**: hybrid-executor.1-c87, hybrid-executor.2-c88, bs19

---

### ADR-011: MMA m16n8k16 Fragment Mapping — Column-Major B and Correct Output Indices

- **Decision**: For `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`, the B matrix must be stored in **column-major packed** format, and the D output mapping is `d0=D[group][lane*2]` (group=row, lane=column), not `D[lane*2][group]`.
- **Context**: The MMA instruction with `.col` on B means the B fragment uses column-major layout: thread (g=tid/4, l=tid%4) provides b0 = pack(B[l*2][g], B[l*2+1][g]) — two consecutive K-dimension rows from the same N-column g. Feeding row-major B causes the MMA to compute A × B^T instead of A × B. This bug was invisible in all prior tests because they used uniform B data (all 1s), where B = B^T.
- **What happens if violated**: Non-uniform B matrices produce wrong results (each output = K × mean(B_cols) instead of K × B_col_value). With row-major B, the MMA transposes B silently. Combined with the swapped output mapping (d0=D[l*2][g] vs correct d0=D[g][l*2]), results appear doubly transposed.
- **Validated by**: gemm-scale.1 with 5 test cases: uniform K=16, uniform K=32, A=1 B=nonunif, A=nonunif B=1, both nonunif. All pass with column-major B and corrected output mapping.
- **B memory format**: `B_cm[col][k_pair] = pack_f16x2(B[k_pair*2][col], B[k_pair*2+1][col])`, shape [N][K/2] u32.
- **Fragment read**: `b0 = B_smem[(warp_n*8+group)*8 + lane]`, `b1 = B_smem[(warp_n*8+group)*8 + lane + 4]`.
- **Output write**: `d0→D[warp_m*16+group][warp_n*8+lane*2]`, `d1→D[warp_m*16+group][warp_n*8+lane*2+1]`, `d2→D[warp_m*16+group+8][warp_n*8+lane*2]`, `d3→D[warp_m*16+group+8][warp_n*8+lane*2+1]`.
- **Alternatives**: (a) Keep row-major B and accept A×B^T semantics (confusing). (b) Transpose B in shared memory after loading (extra bar_sync + complexity). (c) Store B column-major globally and load directly (chosen — simplest and cleanest).
- **Sources**: gemm-scale.1-c1

### ADR-012: Weight Loading via Pre-allocated Device Buffers

- **Decision**: Model weights are pre-loaded to device memory via `cudarc::htod_sync_copy()` before kernel launch. Kernel receives weight pointers as launch parameters — no hostcall streaming.
- **Context**: For inference, weights are static and known before kernel execution. Pre-allocation is simpler, faster (no per-layer round-trip latency), and leverages existing cudarc infrastructure. Hostcall streaming would add unnecessary complexity for data that doesn't change during execution.
- **What happens if violated**: Hostcall-based weight loading would incur per-layer latency (shared memory protocol overhead) and complicate the kernel with I/O logic that belongs on the host side.
- **Trade-offs**: Pre-allocation is less "autonomous" than hostcall streaming, but for inference the weights are deterministic inputs, not runtime decisions. The kernel's autonomy is in its compute graph execution, not data loading.
- **Buffer layout**: Each weight tensor is a separate `CudaSlice<T>` on the host side, passed as raw pointer to the kernel. GEMM weights use column-major packed f16x2 format (per ADR-011). Bias/gamma/beta are plain f32 arrays.
- **Sources**: transformer-layer.5-c1
