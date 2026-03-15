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

### ADR-013: GPU→Host Error Propagation via Result Buffer

- **Date**: 2026-03-13
- **Status**: accepted (implemented in gpu-error-propagation.2-4)
- **Context**: GPU kernels currently have no way to report errors to the host except via panic (which traps and kills the CUDA context) or writing ad-hoc status values to output buffers. std::io returns Result, so proper error handling is needed before real std can work on GPU.
- **Decision**: Define a 64-byte `GpuKernelResult` struct passed as the last kernel parameter. TAG_OK/TAG_ERR/TAG_UNINIT tag field. GPU writes error info (category, errno, thread/block idx, message). Host reads after synchronization. `?` operator works through standard Rust Result mechanics with a wrapper function at kernel entry.
- **Rationale**: (1) Kernel parameter is explicit and composable — no global state. (2) No hostcall protocol change needed. (3) Works with both no_std and std kernels. (4) TAG_UNINIT (0xDEAD_BEEF) detects kernel crashes. (5) Standard Rust error handling patterns — no proc macro required.
- **Layout**: `[tag:u32][category:u16][errno:u16][thread_idx:u16][block_idx:u16][msg_len:u32][msg:48B]` = 64 bytes.
- **Alternatives**: (a) Embed in hostcall buffer header — breaks protocol, not composable. (b) Hostcall error channel — adds latency, trap still kills context. (c) Global error buffer — not composable with concurrent launches.
- **Sources**: gpu-error-propagation.1-c181

## ADR-011: gpu-host as library + binary with feature-gated model/tokenizer

- **Date**: 2026-03-14
- **Status**: accepted (implemented in host-sdk.2)
- **Context**: gpu-host was a binary-only crate with test runner. Users need it as a reusable library for kernel management, hostcall, and memory. Model/tokenizer are GPT-2-specific, not core SDK.
- **Decision**: gpu-host is both library (lib.rs) and binary (main.rs). Core modules (runtime, memory, hostcall, error) always available. model/tokenizer gated behind `feature = "gpt2"` (default-on). Convenience re-exports at crate root: `GpuRuntime`, `HostcallBuffer`, `MappedBuffer`, `GpuHostError`.
- **Rationale**: (1) Library consumers get lean SDK without GPT-2 deps. (2) Binary test runner still works with default features. (3) No breaking changes — existing usage unchanged. (4) Re-exports reduce import verbosity.
- **Alternatives**: (a) Separate crate — duplicates code, harder to maintain. (b) No feature gates — forces safetensors+tiktoken on all consumers.
- **Sources**: host-sdk.2-c193

### ADR-012: ci-lint.sh as single source of truth for CI
- **Date**: 2026-03-14
- **Status**: accepted (implemented in build-auto.2)
- **Context**: build.yml had ~60 lines of duplicated fmt/clippy/doc steps that drifted out of sync with ci-lint.sh. PTX stubs were hardcoded in both files.
- **Decision**: CI lint job calls `bash scripts/ci-lint.sh` directly. PTX stubs auto-discovered via `grep include_str!`. Separate build-ptx job removed (ci-lint.sh builds PTX).
- **Rationale**: (1) One script, two environments — no drift. (2) Developers verify locally what CI will check. (3) PTX stub list auto-maintained.
- **Alternatives**: (a) YAML-first CI with separate config — fragile, hard to test locally. (b) Makefile — heavier tooling for same benefit.
- **Sources**: build-auto.1-c199, build-auto.2-c201

### ADR-013: HostcallSession for multi-launch hostcall persistence
- **Date**: 2026-03-14
- **Status**: accepted (design in hc-session.1)
- **Context**: Each kernel launch creates a new HostcallBuffer + listener thread. For multi-launch pipelines, this overhead is unnecessary and prevents cross-launch fd sharing.
- **Decision**: HostcallSession wraps HostcallBuffer with persistent listener. `reinit_packets()` resets free/ready stacks between launches without reallocation. fd_table persists across launches.
- **Rationale**: (1) Zero allocation overhead between launches. (2) File handles persist — Kernel B can use Kernel A's fds. (3) Listener thread stays alive with adaptive sleep.
- **Alternatives**: (a) Recreate buffer per launch — wastes allocation + thread spawn. (b) Global persistent listener — less composable.
- **Sources**: hc-session.1-c217, bs54.md

### ADR-014: Host→GPU command buffer via mapped memory ring buffer
- **Date**: 2026-03-14
- **Status**: accepted (design in cmd-buffer.1)
- **Context**: Current protocol is GPU→host only. Multi-command kernel needs host→GPU command channel.
- **Decision**: Mapped memory ring buffer with write_idx (host, Release) / read_idx (GPU, Release). 64-byte header + 64-byte command slots. Command types: CMD_COMPUTE, CMD_PRINT, CMD_EXIT.
- **Rationale**: (1) Ring buffer is simpler than lock-free stack for single-producer. (2) FIFO ordering preserved. (3) Uses proven sys-scope atomics pattern. (4) Independent from hostcall buffer.
- **Alternatives**: (a) Repurpose hostcall buffer as bidirectional — overloads existing protocol. (b) Shared memory flag polling — no ordering guarantee.
- **Sources**: cmd-buffer.1-c218, bs54.md

### ADR-015: .await in #[warp_async] — proc macro transforms await to warp_poll_future
- **Date**: 2026-03-14
- **Status**: accepted (design in warp-async-v2.1)
- **Context**: Current #[warp_async] only recognizes explicit `warp_*!()` macro calls. Users want standard async fn syntax with `.await` expressions.
- **Decision**: The proc macro recognizes `expr.await` expressions in the function body and transforms each `.await` into a state that calls `warp_cooperative::warp_poll_future()`. The inner future is stored in the generated struct. Each `.await` becomes one state (not two like INIT+WAIT), because the inner future manages its own state machine.
- **Rationale**: (1) Phase 1 proved `warp_poll_future()` works — lane 0 polls, broadcasts via shfl.sync, all lanes converge. (2) Inner futures are standard `impl Future` — they have no warp awareness. (3) One state per `.await` is cleaner than two (INIT+WAIT), because the inner future handles the submit/wait internally. (4) The no-op Waker is created once per `poll_warp()` call and passed to the inner future.
- **Alternatives**: (a) Require users to write `warp_await!(expr)` instead of `expr.await` — less ergonomic. (b) Transform async fn into a real Rust coroutine — requires rustc changes (Phase 3).
- **Key design points**:
  - `.await` is parsed by syn as `ExprAwait { base, .. }` — easy to detect
  - Inner future type must be known at macro expansion time (stored as field in generated struct)
  - `let var = expr.await` captures the Output value; lane 0 writes to struct field, broadcasts
  - `expr.await?` combines .await with ? — broadcasts Ok/Err discriminant after the future resolves
  - Function must be marked `async` or the macro strips the `async` keyword (since it generates a WarpFuture, not an async fn)
- **Sources**: warp-future-bridge.2-c233, bs57.md

### ADR-016: Post-StateTransform MIR pass for warp-cooperative async codegen
- **Date**: 2026-03-14
- **Status**: accepted (design specification only — implementation deferred)
- **Context**: Standard `async fn` compiles to valid PTX on nvptx64 (proven by rustc-warp.1), but generates per-thread state machines. For warp-cooperative execution, the state machine must be transformed so that lane 0 polls exclusively while all 32 lanes stay convergent via `shfl.sync` broadcasts and `bar.warp.sync` barriers. Three insertion points were considered: (1) library-level wrapper, (2) custom MIR pass after StateTransform, (3) modified StateTransform pass, (4) custom codegen backend.
- **Decision**: A custom MIR pass (`WarpCooperativeTransform`) runs AFTER `coroutine::StateTransform` in the `mir_drops_elaborated_and_const_checked` pipeline. The pass: (a) identifies the coroutine dispatch switch on discriminant, (b) inserts `shfl.sync.idx.b32` to broadcast the discriminant from lane 0 to all lanes, (c) wraps inner future poll calls in leader-only predication, (d) broadcasts poll result discriminant and Ready payload fields via 32-bit shuffle decomposition, (e) inserts `bar.warp.sync` barriers before every `return`. The pass is gated by `#[warp_cooperative]` attribute and `target_arch = "nvptx64"`.
- **Rationale**: (1) Post-StateTransform is the least invasive insertion point — the state machine is already lowered to plain struct + switch MIR, so the pass operates on regular MIR without understanding coroutine semantics. (2) No changes needed to the LLVM NVPTX backend — the pass emits calls to intrinsics that lower to existing PTX instructions. (3) The transformation is provably correct for value-type state machines: all `switchInt` branches operate on broadcast values, ensuring warp convergence. (4) The existing `#[warp_async]` proc macro performs the same logical transformation at AST level, validating the algorithm.
- **Limitations**: (a) Self-referential borrows across yield points are unsound (followers would hold stale pointers into leader's struct). (b) Trait object futures (`dyn Future`) are rejected (payload type must be known at compile time for field broadcasting). (c) Maximum broadcast size of ~256 bytes per yield point (64 shuffle operations). (d) Panics during leader-only poll cause warp deadlock at the next barrier (moot on GPU where panic = trap).
- **Alternatives**: (a) Library-level wrapper — cannot modify state machine structure, only wraps poll dispatch (insufficient for field broadcasting). (b) Modified StateTransform — more invasive, requires understanding coroutine lowering internals, higher maintenance cost. (c) Custom codegen backend — maximum control but massive maintenance burden, no benefit over MIR-level approach. (d) Proc macro only — already works for 90%+ of practical patterns, but cannot handle arbitrary control flow (loops with .await, deeply nested async).
- **Sources**: rustc-warp.2-c241, rustc-survey.1-c232, bs57.md, bs58.md

### ADR-017: Thread-ID-indexed ThreadLocal replacement for GPU multi-thread
- **Date**: 2026-03-14
- **Status**: accepted
- **Context**: The patched std routes `target_os = "cuda"` to the `no_threads` thread-local backend, which uses plain statics shared across all GPU threads. This causes data races when launching kernels with more than one thread. `#[thread_local]` (native TLS via LLVM) is NOT supported on nvptx64 (LLVM lacks `GlobalTLSAddress` selection). The project already uses thread-ID-indexed arrays for per-thread errno in `gpu-libc/src/errno.rs`, proving the pattern works.
- **Decision**: Create a new `gpu_threads.rs` module in `patched-std/library/std/src/sys/thread_local/` that replaces `no_threads.rs` for CUDA targets. The module provides `EagerStorage<T>`, `LazyStorage<T>`, and `LocalPointer` backed by `[T; MAX_GPU_THREADS]` arrays indexed by flat thread ID (`threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y`). MAX_GPU_THREADS = 1024 (one full block). Array initialization uses `MaybeUninit::uninit().assume_init()` for non-Copy types. The `thread_local_inner!` macro is modified so EagerStorage returns a per-thread reference via `get()` instead of `&value`.
- **Rationale**: (1) Reuses the proven errno pattern — minimal novelty risk. (2) Drop-in replacement — same public API (EagerStorage, LazyStorage, LocalPointer, thread_local_inner), no changes needed to thread_local consumers. (3) Memory overhead is ~62-78 KB total for all std-internal thread_locals — negligible for GPU. (4) No destructor changes needed — GPU kernels don't have thread exit, matching the existing `guard::enable()` no-op.
- **Sources**: std-multithread.1-c242, gpu-tls.1-c207, bs58.md

### ADR-018: Host-side async/await via spawn_blocking + hostcall event stream
- **Date**: 2026-03-14
- **Status**: proposed
- **Context**: The gpu-host SDK is fully synchronous — `dev.synchronize()` blocks, and `HostcallBuffer::listen()` runs a blocking polling loop. Modern Rust programs expect tokio-compatible async APIs. Host programs cannot composably `.await` GPU kernel completion alongside other async work.
- **Decision**: Three-tier approach: (1) `AsyncGpuRuntime` wraps `launch_kernel().await` via `tokio::task::spawn_blocking(dev.synchronize())` — zero-risk, works with cudarc 0.12. (2) `HostcallEventStream` bridges the existing listener thread to `tokio::sync::mpsc`, exposing print/file/trace events as an async `Stream`. (3) CUDA event-based polling (cuEventCreate/cuEventQuery) deferred — requires raw driver API, premature complexity. `tokio` dependency feature-gated behind `feature = "async"`.
- **Rationale**: spawn_blocking is the idiomatic tokio pattern for wrapping blocking operations. The listener thread already runs independently — bridging it to a channel is minimal work. CUDA events would avoid the blocking thread pool but cudarc 0.12 doesn't expose them, and spawn_blocking scales well for typical GPU workloads (1-4 concurrent kernels).
- **Sources**: host-async.1-c248, bs61.md

### ADR-019: WarpCooperativeTransform MIR pass — detailed design specification
- **Date**: 2026-03-14
- **Status**: proposed (design specification only — implementation deferred)
- **Context**: ADR-016 established the high-level decision to use a post-StateTransform MIR pass for warp-cooperative async codegen. This ADR records the detailed design choices made during the full specification (rustc-warp-async.2).
- **Decision**: The `WarpCooperativeTransform` MIR pass uses: (1) Inline PTX assembly (not LLVM intrinsics) for `shfl.sync.idx.b32`, `bar.warp.sync`, `activemask`, and `%laneid` — matching the project's existing `gpu-atomics` approach and avoiding broken NVVM intrinsics. (2) Leader predication via MIR-level `switchInt(_is_leader)` branches rather than PTX `@p` predicated instructions, keeping the transformation entirely at MIR level. (3) Tiered type broadcasting: u32 = 1 shuffle, u64 = 2 shuffles (lo/hi decomposition), small structs (<=256B) = N/4 shuffles (field decomposition), large types = shared memory fallback. (4) Explicit rejection of self-referential borrows across yield points, `dyn Future` trait objects, and `Drop`-implementing output types as unsound in warp-cooperative context. (5) `activemask()` for dynamic lane membership instead of hardcoded `0xFFFFFFFF`.
- **Rationale**: (1) Inline PTX is the only reliable path — NVVM intrinsics are broken on nightly LLVM (project technical notes). (2) MIR-level predication avoids introducing PTX-specific concepts before codegen, keeping the pass target-agnostic in structure. (3) Tiered broadcasting minimizes shuffle count for common types while maintaining correctness for arbitrary sizes. (4) Soundness restrictions prevent silent data corruption — follower lanes cannot hold valid pointers into leader's coroutine struct, and Drop on broadcast copies would double-free. (5) activemask handles partial warps from non-multiple-of-32 launch configurations.
- **Alternatives**: (a) LLVM intrinsics for shuffles — rejected, broken on current LLVM. (b) PTX predicated instructions — rejected, would require codegen-level pass instead of MIR-level. (c) Shared memory for all broadcasts — rejected, unnecessary overhead for small types. (d) Allow self-referential borrows with pointer fixup — rejected, complexity outweighs benefit for initial version.
- **Sources**: rustc-warp-async.1-c260, rustc-warp-async.2-c261

### ADR-18: Tokio bridge — keep listener as std::thread, add GpuTask orchestrator
- **Date**: 2026-03-15
- **Status**: accepted
- **Context**: Need to integrate GPU kernel execution with tokio-based host applications. The hostcall listener uses spin-polling with adaptive sleep for low-latency GPU doorbell monitoring.
- **Decision**: (1) Keep the hostcall listener as a `std::thread` (not a tokio task). Connect it to tokio via `tokio::sync::mpsc` channel for event streaming. (2) Add `GpuTask` struct to `async_rt.rs` that orchestrates `AsyncGpuRuntime` + `AsyncHostcallSession` — provides `launch().await` and `next_event().await`. (3) Use `spawn_blocking` for kernel launch and synchronize calls.
- **Rationale**: (1) Converting the listener to a tokio task would add scheduling jitter to latency-sensitive doorbell polling, and `spawn_blocking` would just pin a thread anyway. The hybrid design (std::thread + tokio channel) is correct. (2) `GpuTask` provides the ergonomic `gpu_spawn()`-like API users expect without requiring changes to the hostcall core. (3) The existing `AsyncGpuRuntime` and `AsyncHostcallSession` already handle 2/4 tokio-bridge epic criteria.
- **Alternatives**: (a) Full tokio-native listener with `tokio::time::sleep` — rejected, adds latency. (b) CUDA events/interrupts instead of polling — possible future optimization, but doorbell polling works well. (c) Single monolithic `gpu_spawn()` function — rejected in favor of `GpuTask` struct for more flexibility (reuse across launches).
- **Sources**: tokio-investigate.1-c337, tokio-investigate.2-c338

### ADR-20: Two-tier CUDA stream model — compute streams vs hostcall default stream
- **Date**: 2026-03-15
- **Status**: accepted
- **Context**: Production workloads need overlapping GPU compute for pipelining. cudarc 0.12.1 has `CudaStream`, `launch_on_stream()`, and `fork_default_stream()`. But the hostcall safety model requires device-level sync (`cuCtxSynchronize`) before packet reset.
- **Decision**: (1) Add `GpuStream` wrapper in `streams.rs` for pure compute kernels — wraps `fork_default_stream()` + `launch_on_stream()`. (2) Hostcall kernels stay on default stream via `GpuTask` — device-level sync + `reinit_packets()` between launches. (3) `AsyncGpuStream` for tokio integration via `spawn_blocking`. (4) Two tiers never mix: compute streams don't touch hostcall buffers.
- **Rationale**: Separating compute streams from hostcall kernels preserves the device-idle safety invariant without restricting compute overlap. cudarc already provides the full stream API (`launch_on_stream`, `wait_for`, `fork_default_stream`). Forward-compatible with per-stream hostcall buffers if needed later.
- **Alternatives**: (a) Per-stream hostcall buffers — correct but complex (multiple listener threads, buffer routing). Deferred. (b) Single stream for everything — simple but loses overlap benefit. (c) Full device sync after every launch — defeats purpose of streams.
- **Sources**: cuda-streams.1-c342, cuda-streams.2-c345
