# Brainstorm 87 — Proposer: Bold New Directions
**Cycle**: 303 | **Date**: 2026-03-15 | **Level**: Deep (Proposer)

## Project State Assessment

The project stands at **303 cycles, 323 completed tasks, 11 completed feature epics**, 43k+ lines of Rust across 7 crates. Every major thesis has been proven: Rust async/await on GPU, warp-cooperative MIR pass, real std on GPU, file I/O, TCP networking, error propagation, debugging/tracing, GPT-2 inference. Safety audit (83 SAFETY comments) and module docs are done. Two evergreen epics remain active but nearly satisfied.

This is no longer a research prototype — it is a demonstrated capability seeking impact. The question is: **what transforms async_gpu from "impressive proof-of-concept" to "tool people actually use"?**

---

## Active Epics Assessment

### codebase-health (evergreen)
**Status**: 4/5 criteria met. The remaining gap — scripted automation for "new crate → CI update" — is marginal value given the project's current stability. Effectively complete.

### public-api (evergreen)
**Status**: 4/4 criteria met (after bs86 criteria update). gpu-host has module docs, 7 examples with READMEs, single-command build. Effectively complete.

**Verdict**: Both evergreen epics are satisfied to the extent possible without GPU hardware. New work should come from new strategic directions, not squeezing the last 5% from evergreen items.

---

## Technical Direction Analysis

### 1. Multi-GPU Coordination Protocol

**Concept**: Design a protocol for GPU-to-GPU messaging via host relay. GPU A sends a message to GPU B by issuing a hostcall; the host routes it to GPU B's hostcall listener.

**Feasibility without GPU**: **Medium-High**. The protocol design, service IDs, host relay logic, and GPU-side Future types can all be implemented and compile-tested. The existing TCP service pattern is directly reusable — inter-GPU messaging is structurally identical to TCP but with device buffers as endpoints.

**Value**: **High**. Multi-GPU is the natural next step for any serious GPU computing system. No other Rust GPU framework offers this. The hostcall architecture is uniquely suited — it already has a host relay, fd namespace, and async Future pattern.

**Complexity**: **Medium**. New service IDs (SERVICE_GPU_SEND, SERVICE_GPU_RECV), a device registry on the host, and buffer routing. Core mechanism is proven by TCP.

**Concrete deliverables**:
- New service IDs in gpu-protocol (SERVICE_GPU_SEND, SERVICE_GPU_RECV, SERVICE_GPU_BROADCAST)
- Host-side device registry + message router in gpu-host
- GPU-side GpuSendFuture / GpuRecvFuture in gpu-runtime
- Demo kernel: two-GPU pipeline (GPU 0 computes → sends to GPU 1 → GPU 1 processes)

**Assessment**: High value, medium effort, naturally extends the proven hostcall pattern. **Recommend as P2.**

---

### 2. GPU-Side Task Spawning (On-GPU Async Executor)

**Concept**: Design an on-GPU executor that can spawn and schedule async tasks dynamically, rather than having a fixed task-per-warp model. A warp finishes its current Future and picks up the next from a shared work queue.

**Feasibility without GPU**: **Medium**. The executor design and work-stealing queue can be implemented in pure Rust. The lock-free queue uses the same CAS pattern already proven in the hostcall protocol. However, correctness of warp-level scheduling is untestable without hardware.

**Value**: **Very High**. This is the missing piece for real-world GPU async. Current model: one Future per kernel launch. With task spawning: launch a kernel, it spawns work dynamically based on data. This enables GPU-driven servers, GPU-side map-reduce, and GPU-autonomous pipelines that adapt at runtime.

**Complexity**: **High**. Requires:
- Lock-free work queue in GPU shared/global memory
- Warp-level work stealing (lane 0 dequeues, broadcasts to warp)
- Lifetime management for spawned Futures (they must be `'static` or pinned)
- Integration with `#[warp_cooperative]` MIR pass

**Concrete deliverables**:
- `GpuExecutor` type with `spawn()` and `block_on()` methods
- Lock-free MPMC queue in gpu-atomics (CAS-based, warp-cooperative dequeue)
- Example: data-dependent pipeline where one task spawns follow-up tasks
- Design doc covering scheduling policy, memory layout, warp assignment

**Assessment**: Highest technical ambition. The design is feasible; correctness verification requires hardware. **Recommend as P3 (design doc now, implementation when GPU available).**

---

### 3. Formal Verification of CAS Protocol (TLA+)

**Concept**: Model the lock-free hostcall CAS protocol in TLA+ or a similar formal verification language. Verify liveness (no deadlock), safety (no data races), and progress guarantees.

**Feasibility without GPU**: **Perfect**. Formal verification is purely mathematical — no hardware needed. This is one of the few directions that is *more* valuable without GPU access, because it provides correctness confidence when testing is impossible.

**Value**: **Very High**. The CAS protocol is the correctness foundation of the entire system. It has:
- Free-stack and ready-stack with CAS-based push/pop
- GPU→host signaling via atomic status word
- Acquire/release ordering assumptions
- Warp-level broadcast of results

A TLA+ model would either prove correctness or find subtle bugs — both are extremely valuable outcomes.

**Complexity**: **Medium**. The CAS protocol is small enough to model completely (~200 lines of critical path). TLA+ is well-suited to this class of problem.

**Concrete deliverables**:
- TLA+ model of packet lifecycle: FREE → FILLING → READY → PROCESSING → DONE → FREE
- Invariant verification: no double-allocation, no lost packets, progress under contention
- Liveness check: every submitted request eventually gets a response
- Findings document with any discovered edge cases

**Assessment**: Perfect fit for the current constraint (no GPU). Highest confidence-per-effort ratio. **Recommend as P1.**

---

### 4. Cross-Compilation Improvements

**Concept**: Make the build pipeline more user-friendly. Currently requires: specific nightly, nvptx64 target, rust-src component, CUDA driver, manual PTX post-processing, patched toolchain for std.

**Feasibility without GPU**: **High**. Build tooling improvements are purely host-side.

**Value**: **High for adoption, medium for the project itself**. The build complexity is the #1 barrier to adoption. A new user must: install specific nightly, add nvptx64 target, add rust-src, have CUDA driver, understand `-Zbuild-std`. If we could reduce this to `cargo install async-gpu-tools && async-gpu build`, adoption would increase dramatically.

**Complexity**: **Medium**. Concrete improvements:
- Cargo xtask pattern replacing shell scripts
- Automatic PTX post-processing in build.rs (already partially done)
- Pre-built PTX artifacts for CI (no GPU needed to compile kernels)
- Docker-based build environment

**Concrete deliverables**:
- `cargo xtask build-kernel` replacing `scripts/build-toolchain.ps1` / `.sh`
- Unified build config (toolchain version, target triple, features) in workspace Cargo.toml metadata
- `cargo xtask new-example <name>` scaffolding tool
- CI improvements: matrix build across nightly versions

**Assessment**: High adoption value but somewhat pedestrian engineering. **Recommend as P4 (do when motivated, not as research priority).**

---

### 5. GPU-Native Data Structures

**Concept**: Implement concurrent HashMap, Channel, Queue using warp-cooperative patterns. These would use the CAS primitives from gpu-atomics and integrate with the async executor.

**Feasibility without GPU**: **Medium**. Can implement and compile-test. Correctness requires GPU testing. The CAS pattern is proven but data structure composition introduces new edge cases.

**Value**: **High**. Concurrent data structures are what make GPU async practical beyond simple I/O. A GPU-side Channel enables producer-consumer patterns across warps. A concurrent HashMap enables GPU-side caching (key for inference, graph algorithms, etc.).

**Complexity**: **High**. Lock-free concurrent data structures are notoriously hard to get right. On GPU, additional constraints: no thread-local storage, warp-level operations needed for efficiency, memory model differences.

**Concrete deliverables**:
- `GpuChannel<T>` — SPMC or MPMC channel using CAS-based queue
- `GpuMutex<T>` — warp-cooperative mutex (lane 0 acquires, all lanes access)
- `GpuHashMap<K, V>` — open-addressing hash map with CAS-based insert
- Integration with async executor: `channel.recv().await` yields warp

**Assessment**: High value but very hard to verify without hardware. **Recommend as P5 (design doc now, defer implementation).**

---

### 6. Async Trait Integration

**Concept**: Make GPU Futures compatible with `async_trait` or design a GPU-specific async trait pattern that allows abstraction over I/O operations.

**Feasibility without GPU**: **High**. This is purely type-system work.

**Value**: **Medium**. The current concrete Future types (GpuOpenFuture, GpuReadFuture, etc.) work well. Trait abstraction would enable generic GPU code that works with any I/O backend, but the I/O backends are fixed (hostcall only). Over-abstraction risk is high.

**Complexity**: **Low-Medium**. The main challenge: `async_trait` uses `Box<dyn Future>` which requires allocation. On GPU, allocation goes through hostcall. This is feasible but adds round-trip latency to every trait-dispatched call.

**Assessment**: Low priority. The concrete types work well and abstraction adds overhead without clear benefit. **Skip.**

---

### 7. Profiling / Instrumentation

**Concept**: Add timing instrumentation to hostcall round-trips. Measure: packet acquisition time, host processing latency, response delivery time.

**Feasibility without GPU**: **Medium-High**. The instrumentation code can be written and compiled. Host-side timing is immediately measurable. GPU-side timing requires `clock64()` intrinsic and hardware.

**Value**: **High**. Performance profiling is essential for optimization and for users to understand bottlenecks. The hostcall round-trip is the critical path — knowing whether latency comes from CAS contention, PCIe transfer, or host processing guides optimization.

**Complexity**: **Low**. Add timestamp fields to packet header, record timestamps at key points, expose via gpu_trace!() or dedicated profiling API.

**Concrete deliverables**:
- Timestamp fields in HostcallPacket (submit_time, ack_time, complete_time)
- Host-side latency histogram per service type
- GPU-side `clock64()` based timestamps (compile-tested, verified on hardware later)
- Profiling report format (JSON or structured trace)

**Assessment**: Low effort, high diagnostic value. **Recommend as P3 (alongside executor design).**

---

### 8. Plugin / Extension System

**Concept**: Let users register custom hostcall services. Instead of hardcoded SERVICE_* handlers, provide a registration API: `runtime.register_service(42, |packet| { ... })`.

**Feasibility without GPU**: **High**. This is host-side refactoring.

**Value**: **Very High**. This transforms async_gpu from a fixed framework into an extensible platform. Users could implement: database queries, HTTP requests, IPC, custom hardware control — all as hostcall services without forking the codebase.

**Complexity**: **Low-Medium**. The host listener already dispatches on service ID. Refactoring from a match statement to a registry + trait is straightforward.

**Concrete deliverables**:
- `HostcallService` trait with `fn handle(&self, packet: &mut HostcallPacket) -> Result<()>`
- `HostcallSession::register_service(id: u32, handler: impl HostcallService)`
- Default implementations for all existing services (print, file I/O, TCP)
- Example: custom "key-value store" service registered by user code
- GPU-side generic `GpuCustomFuture<const SERVICE_ID: u32>`

**Assessment**: High value, low-medium effort. Transforms the project's extensibility story. **Recommend as P1.**

---

### 9. Tutorial Series

**Concept**: Step-by-step guide for newcomers. From "what is hostcall?" to "write your first GPU kernel with async I/O."

**Feasibility without GPU**: **High**. Pure documentation.

**Value**: **High for adoption**. The README is good but dense. A progressive tutorial series would dramatically lower the entry barrier: (1) Hello GPU, (2) Understanding hostcall, (3) Async I/O, (4) Warp-cooperative async, (5) TCP networking.

**Complexity**: **Low**. Writing, not coding.

**Assessment**: High adoption value. **Recommend as P4 (after technical work).**

---

### 10. Benchmarking Framework

**Concept**: Design a standardized framework for measuring hostcall latency, throughput, warp utilization. Even without GPU, design the harness, metrics, and reporting format.

**Feasibility without GPU**: **Medium**. Framework design and host-side components can be built. Actual measurements require hardware.

**Value**: **Medium**. Essential for optimization but the project isn't at the optimization stage — it needs hardware testing first.

**Complexity**: **Low-Medium**.

**Assessment**: Premature without hardware. **Skip for now.**

---

### 11. Comparison Document

**Concept**: Detailed technical comparison with CUDA C++, HIP, SYCL, Triton, CuPy, etc. Positioning document for the project.

**Feasibility without GPU**: **High**. Research and writing.

**Value**: **High for positioning**. Potential users/contributors need to understand: why async_gpu vs. just writing CUDA? The answer (GPU autonomy, Rust safety, async/await ergonomics, hostcall I/O) is compelling but not documented in comparative form.

**Complexity**: **Low**. A few days of research and writing.

**Assessment**: High positioning value. **Recommend as P4.**

---

### 12. RFC for rustc Upstream

**Concept**: Draft an RFC proposing the warp-cooperative MIR pass for inclusion in rustc. This would make `#[warp_cooperative]` a first-class Rust feature.

**Feasibility without GPU**: **High**. RFC writing is pure documentation + design.

**Value**: **Potentially transformative**, but premature. The MIR pass works but has never been tested on hardware. Submitting an RFC for an untested compiler feature would damage credibility.

**Complexity**: **Medium** (RFC writing is non-trivial — requires clear motivation, detailed design, alternatives analysis).

**Assessment**: **Premature. Park until hardware testing validates the approach.** The risk of submitting an RFC based on compile-only verification is too high.

---

## Consolidated Recommendations

### Priority Ordering

| Priority | Direction | Epic | Rationale |
|----------|-----------|------|-----------|
| **P1** | Plugin/Extension System | **New: extensibility** | Low-medium effort, transforms the project from fixed framework to extensible platform. Directly implementable and testable without GPU. |
| **P1** | Formal Verification (TLA+) | **New: formal-verification** | Perfect fit for no-GPU constraint. Provides correctness confidence for the CAS protocol, the system's foundation. High signal for academic credibility. |
| **P2** | Multi-GPU Coordination Protocol | **New: multi-gpu** | Natural extension of proven hostcall pattern. Differentiating feature — no other Rust GPU framework has this. |
| **P3** | GPU-Side Task Spawning | **Design doc only** | Highest technical ambition but needs hardware for verification. Design doc captures the architecture. |
| **P3** | Profiling/Instrumentation | **codebase-health** | Low effort, high diagnostic value. Timestamp infrastructure is useful even before GPU testing. |
| **P4** | Cross-Compilation Improvements | **codebase-health** | High adoption value but engineering work, not research. |
| **P4** | Tutorial Series | **public-api** | High adoption value, pure writing. |
| **P4** | Comparison Document | **public-api** | High positioning value, pure research/writing. |
| **P5** | GPU-Native Data Structures | **Design doc only** | Very hard to verify without hardware. |
| **Skip** | Async Trait Integration | — | Over-abstraction, adds overhead without clear benefit. |
| **Skip** | Benchmarking Framework | — | Premature without hardware. |
| **Park** | RFC for rustc Upstream | — | Needs hardware validation first. |

---

## Proposed New Epics

### Epic: extensibility (medium priority)
**Title**: Plugin/Extension System — user-defined hostcall services
**Success criteria**:
1. `HostcallService` trait with `handle()` method defined in gpu-host
2. `HostcallSession::register_service()` API for user-defined handlers
3. All existing services (print, file I/O, TCP) refactored to use the trait
4. Example demonstrating a custom user-defined hostcall service
5. GPU-side `GpuCustomFuture<const SERVICE_ID: u32>` for generic service calls

**Why**: The hostcall system is the project's core innovation. Making it extensible transforms async_gpu from a demo into a platform. Users can add database, HTTP, IPC, or domain-specific services without forking.

### Epic: formal-verification (medium priority)
**Title**: Formal Verification — TLA+ model of CAS hostcall protocol
**Success criteria**:
1. TLA+ model covers packet lifecycle (FREE → FILLING → READY → PROCESSING → DONE → FREE)
2. Safety invariants verified: no double-allocation, no lost packets, no data races
3. Liveness verified: every request eventually receives a response (under fairness)
4. Model covers multi-warp contention (at least 2 warps + 1 host thread)
5. Findings documented with any discovered edge cases or protocol improvements

**Why**: The CAS protocol is the correctness foundation. Formal verification is uniquely suited to the no-GPU constraint — it provides mathematical confidence that testing cannot (and that compile-only verification certainly cannot).

### Epic: multi-gpu (low priority)
**Title**: Multi-GPU Coordination — GPU-to-GPU messaging via host relay
**Success criteria**:
1. SERVICE_GPU_SEND / SERVICE_GPU_RECV / SERVICE_GPU_BROADCAST defined in gpu-protocol
2. Host-side device registry and message router implemented
3. GPU-side GpuSendFuture / GpuRecvFuture implemented
4. Demo: two-device pipeline (GPU 0 computes → sends → GPU 1 processes)

**Why**: Multi-GPU is the natural scaling path. The hostcall architecture is uniquely suited — the host already acts as a relay, and the fd namespace can be extended to device endpoints.

---

## Key Insight

The project has proved its thesis. Now it needs to prove its **utility**. The two highest-value directions are:

1. **Extensibility** (plugin system) — because a platform beats a framework. Users who can add custom services will actually adopt this.
2. **Formal verification** (TLA+) — because this is the rare situation where formal methods add genuine value: the protocol is small enough to model completely, hardware testing is unavailable, and the CAS protocol is the single point of failure for the entire system.

Both are perfectly suited to the no-GPU constraint. Together, they move async_gpu from "impressive demo" toward "trustworthy, extensible GPU async runtime."
