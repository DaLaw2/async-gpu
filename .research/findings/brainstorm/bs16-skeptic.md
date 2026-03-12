# BS16 Skeptic — Warp-level Future Critique
**Date**: 2026-03-12
**Brainstorm seq**: 16
**Role**: Skeptic
**Level**: deep (proposer + skeptic)
**Target**: ADR-9 / warp-future theme — WarpFuture trait + proc macro for SIMT-convergent async

---

## Executive Summary

The WarpFuture proposal addresses a real but poorly-quantified problem by introducing significant new complexity (custom trait, proc macro, warp-aware executor) without first exhausting simpler alternatives. Several critical claims are unsupported by evidence from this project's actual codebase. The proposal conflates two distinct problems — warp divergence from async state machines and warp-cooperative hostcall throughput — and tries to solve both simultaneously, increasing risk. I recommend a cheaper empirical investigation before committing to the full WarpFuture architecture.

---

## Challenge 1: Is the Problem Real?

### 1.1 The "1/32 throughput" claim is worst-case theoretical, not measured

ADR-9 states: "when lanes enter different enum variants, warp divergence occurs and SIMT throughput drops to 1/32."

**Challenge**: This assumes all 32 lanes are in 32 different states simultaneously and that the hardware serializes execution across all 32 paths. In practice:

- NVIDIA hardware since Volta (SM 7.0+) has **independent thread scheduling** — each lane has its own program counter. Divergent lanes don't fully serialize; the hardware interleaves instruction issues from active lanes. The performance impact is reduced warp-level IPC, not a 32x throughput cliff.
- The SM 8.6 hardware in this project (RTX 3070 or similar) has even better divergence handling than the theoretical SIMT model suggests.
- **No one has actually measured** how much divergence occurs with the current per-thread Embassy executor in this project. The async-hostcall-test kernels run with thread 0 only (`if global_idx != 0 { return; }`). The multi-warp-test uses synchronous spin-wait, not async futures. There is literally zero empirical data on warp divergence with per-thread async.

**What should be done first**: Run a benchmark where 32 threads in one warp each run their own Embassy executor with a HostcallPrintFuture. Measure SIMT efficiency using `nvprof --metrics warp_execution_efficiency` or `ncu`. Only then will we know if the problem is worth a new architecture.

### 1.2 The current async executor doesn't even run multi-lane

Looking at the actual test kernels:

- `async_hostcall_single_kernel`: thread 0 only
- `async_hostcall_two_kernel`: thread 0 only
- `pipeline_kernel`: thread 0 only
- `multi_block_async_kernel`: 4 blocks x 1 thread (still one thread per block)

**No existing kernel runs per-thread async futures across a full 32-thread warp.** The warp divergence problem is entirely hypothetical in this project. Before designing a solution (WarpFuture), the project should first build the thing it claims is broken (multi-lane per-thread async) and measure the actual damage.

### 1.3 How many async states actually exist?

The HostcallPrintFuture has 3 states: Init, WaitingResponse, Done. The PipelineFuture has effectively 9 states (4 steps x {submit, wait} + done). In a typical workload where all lanes start at the same time doing the same operation:

- All lanes enter Init together (convergent)
- All lanes submit packets and move to WaitingResponse roughly together (convergent)
- Some lanes get host responses before others (divergent for a brief period)
- All lanes reach Done (convergent again)

The actual window of divergence is narrow: it's the skew between the fastest and slowest host response for packets submitted nearly simultaneously. With per-block sharding, packets from the same warp go to the same shard and are processed sequentially by the single host listener thread — meaning responses arrive in order, minimizing divergence.

**The natural execution pattern may already be mostly convergent**, making WarpFuture an expensive solution to a small problem.

---

## Challenge 2: Is the Solution Sound?

### 2.1 `__syncwarp()` at every yield point is a deadlock factory

The proposal calls for `__syncwarp()` (or `bar.sync` equivalent) at every `warp_await!` point to force reconvergence. This is dangerous:

**Scenario: conditional hostcall**
```rust
#[warp_async]
async fn conditional_work(buf: *mut u8, lane: u32) {
    if lane % 2 == 0 {
        warp_await!(hostcall_print(buf, b"even lane"));  // syncwarp here
    }
    warp_await!(hostcall_print(buf, b"all lanes"));     // syncwarp here
}
```

If `__syncwarp()` is inserted at both yield points, the first one deadlocks because odd lanes never reach it. The proposal must either:

1. **Forbid conditional yield** — but this makes the async abstraction nearly useless for real workloads
2. **Use `__syncwarp(mask)` with a dynamically computed mask** — but computing the correct mask at every yield point requires tracking which lanes are still active, which is itself a complex synchronization problem
3. **Require the user to manually manage masks** — defeating the "ergonomic async" goal

This is not a minor issue. It is a fundamental tension between conditional control flow and SIMT convergence that no proc macro can paper over.

### 2.2 Early returns and error handling break convergence

Consider real-world async GPU code:
```rust
#[warp_async]
async fn file_pipeline(buf: *mut u8) -> Result<(), GpuError> {
    let fd = warp_await!(hostcall_open(buf, path))?;  // Some lanes fail: ERR_NOT_FOUND
    let data = warp_await!(hostcall_read(buf, fd))?;   // Only successful lanes reach here
    warp_await!(hostcall_write(buf, data))?;
    Ok(())
}
```

The `?` operator causes early return for error lanes. After the first `?`, the warp is permanently diverged — some lanes returned early, others continue. No amount of `__syncwarp()` can fix this because the early-returned lanes are gone.

The proposal would need to either:
- Ban `?` / `Result` in warp-async functions (unacceptable for Rust ergonomics)
- Convert errors to per-lane predicated execution (all lanes continue but some are "inactive") — this is essentially reinventing SIMT predication in software, duplicating what the hardware already does

### 2.3 The proc macro cannot handle arbitrary Rust control flow

The proposal suggests a `#[warp_async]` proc macro that generates warp-convergent state machines. But Rust async functions can contain:

- `loop { ... .await ... break condition ... }` — variable iteration count per lane
- `match result { Ok(x) => x.await, Err(e) => fallback.await }` — different futures per lane
- `select!` / `join!` combinators — runtime-determined execution paths
- Nested function calls that themselves contain `.await`
- `while let Some(x) = stream.next().await { ... }` — data-dependent loop count

A proc macro operates on syntax trees. It cannot, in general, prove that all control flow paths reach the same yield points in the same order. The only way to guarantee convergence is to restrict the language to a subset where all lanes execute the same sequence of yield points — but this is so restrictive that it no longer resembles Rust async.

### 2.4 Nested WarpFuture composition

If `WarpFuture` is a separate trait from `Future`, it cannot compose with the existing `futures_util` ecosystem (`join`, `select`, `FuturesUnordered`). The project already demonstrated `futures_util::future::join` on GPU (futures_join_kernel). WarpFuture would break this interop.

Either WarpFuture replaces the entire Future ecosystem (massive scope) or it coexists but can't use any existing combinators (limited utility).

---

## Challenge 3: Are There Simpler Alternatives?

### 3.1 Warp-cooperative scheduling in the existing executor

Instead of a new Future trait, modify the Embassy executor's poll loop:

```
// Current: each lane polls independently
lane_executor.poll();  // divergent: each lane in different future state

// Alternative: round-robin per-warp
for task_idx in 0..num_tasks {
    __syncwarp();  // converge before each task
    if lane_has_task(task_idx) {
        lane_executor.poll_task(task_idx);  // brief divergence within poll
    }
    __syncwarp();  // reconverge after
}
```

This keeps per-thread Futures (full Rust async compatibility) while adding convergence points between task polls. The divergence window is limited to individual Future::poll calls, which are typically short (check one condition, return Pending or Ready).

**This is strictly simpler and preserves full Future compatibility.**

### 3.2 Warp-cooperative hostcall without changing the Future model

The biggest source of warp-level inefficiency in the current design isn't async state divergence — it's that 32 lanes independently do CAS on the free stack (per ADR-4, per BS14 analysis showing 49 CAS retries/call at 128 threads).

A warp-cooperative hostcall that elects one lane to do CAS and broadcasts the result via `shfl.sync` would:
- Reduce CAS contention by 32x
- Naturally create convergence (the election + broadcast is a synchronization point)
- Work with the existing per-thread Future model
- Not require a proc macro, new trait, or new executor

This was already identified in BS14 (item #10) but deferred. It addresses the actual measured bottleneck (CAS contention) rather than the unmeasured one (async state divergence).

### 3.3 Accept hardware predication

On SM 7.0+ (independent thread scheduling), the hardware already handles divergent warps via predication. The lanes that are in a different state simply have their execution predicated off. The penalty is reduced instruction throughput (not all lanes are active in every instruction), but:

- For hostcall-heavy workloads, the vast majority of time is spent spin-waiting for host response. During this wait, all lanes execute the same tight loop (load-acquire + check). Divergence only occurs at the brief submit and release phases.
- The actual compute cost of Future::poll is tiny compared to the hostcall round-trip latency (~13us per ADR-6 profiling). Even at 1/32 SIMT efficiency during the poll, the poll itself is <1% of total time.

**The divergence penalty may be negligible compared to the hostcall latency**, making WarpFuture optimization of a non-bottleneck.

### 3.4 Literature on SIMT-aware cooperative multitasking

The proposal does not cite any academic or industry precedent for warp-level futures. Relevant prior work:

- **Merrill & Grimshaw (2011)**: "Revisiting Sorting on GPUs" — warp-cooperative patterns with `__ballot` and `__shfl` for intra-warp communication, but for data-parallel algorithms, not task-parallel async
- **NVIDIA Cooperative Groups**: Provides explicit warp/block synchronization primitives, but assumes data-parallel (same function, different data), not task-parallel (different coroutine states)
- **GPU tasking systems (Tzeng et al., 2010)**: Task parallelism on GPU via persistent threads with work queues — closer to the per-thread executor model, not warp-level coroutines

The absence of prior art for "warp-level cooperative multitasking" is a warning sign. Either the idea is novel (high risk) or there's a reason nobody does it (because hardware predication already handles divergence adequately for the workloads where it matters).

---

## Challenge 4: Implementation Risks

### 4.1 Register pressure

A WarpFuture state machine must store state for 32 lanes' worth of data in a uniform structure. If each lane has, say, 8 bytes of per-lane state (a packet pointer + a status flag), the warp-level state is 256 bytes — stored where? Options:

- **Registers**: 256 bytes = 64 x 32-bit registers just for state, plus control flow variables. The project already identified register pressure as a concern (ADR-4 mentions 64-reg threshold). WarpFuture likely blows this budget.
- **Shared memory**: Viable, but now state access is via memory loads/stores instead of register access — slower and requires manual management
- **Local memory (spill)**: Already observed in current async kernels (benchmark.3). WarpFuture would increase spilling.

### 4.2 The proc macro ships with all the complexity but none of the compiler's optimization

Rust's built-in async desugaring benefits from LLVM optimization passes that understand the resulting state machine (enum layouts, dead branch elimination, move coalescing). A proc macro that generates a different state machine structure (WarpFuture instead of Future) loses all of this:

- No guaranteed enum niche optimization for WarpFuture states
- No compiler-level understanding of warp convergence — LLVM sees ordinary code
- No integration with borrowck's lifetime analysis for the custom state machine

### 4.3 Interaction with per-block sharding

The current hostcall protocol uses per-block sharding (ADR-3). Each block has its own free/ready stacks. WarpFuture's warp-cooperative hostcall would need to interact with the shard assigned to its block. This is straightforward for single-warp-per-block launches, but with multiple warps per block, multiple WarpFutures would compete on the same shard's free stack — reintroducing contention at the warp level instead of the lane level.

### 4.4 Fallback to Phase 2 (rustc changes) indicates Phase 1 insufficiency

The proposal explicitly plans a Phase 2 involving rustc changes (target-specific async desugaring, SIMT-aware MIR pass). This is effectively an admission that the proc macro approach (Phase 1) will hit fundamental limitations. If the plan already anticipates needing compiler changes, why invest in a proc macro that will be replaced?

---

## Concrete Recommendations

### Before proceeding with WarpFuture:

1. **Measure the problem** (1-2 days): Create a benchmark kernel where 32 threads in one warp each run a per-thread Embassy executor with HostcallPrintFuture. Measure SIMT efficiency with NVIDIA profiling tools. If efficiency is >50%, WarpFuture may not be justified.

2. **Try warp-cooperative scheduling first** (2-3 days): Modify the executor poll loop to add `__syncwarp()` convergence points between task polls. Measure whether this recovers most of the SIMT efficiency without a new trait or proc macro.

3. **Try warp-cooperative hostcall allocation** (2-3 days): Implement lane-0-elect + shfl broadcast for free-stack CAS. This is the proven technique from warp-coop (BS14 #10) and addresses the measured CAS contention bottleneck.

### If proceeding despite the above:

4. **Start with a hand-written WarpFuture for one specific case** (the hostcall print) — not a proc macro. Prove the concept compiles to convergent PTX before investing in code generation.

5. **Define the exact subset of Rust that `#[warp_async]` supports** before implementing it. If the subset is too restrictive (no conditionals, no error handling, no loops), acknowledge that it's a DSL rather than Rust async.

6. **Do not create ADR-9 as "accepted"** — it should remain "proposed" until empirical data from recommendations 1-3 is available. The current "proposed" status is appropriate.

---

## Risk Matrix

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| WarpFuture solves unmeasured problem | High | Medium | Measure divergence first (rec #1) |
| Proc macro cannot handle real Rust control flow | High | High | Define supported subset explicitly (rec #5) |
| `__syncwarp()` causes deadlocks with conditional yields | Critical | High | Must solve before any implementation |
| Register pressure exceeds GPU limits | Medium | Medium | Prototype and measure before architecture |
| Simpler alternatives achieve same benefit | High | Medium | Try alternatives first (rec #2, #3) |
| Phase 1 proc macro replaced by Phase 2 compiler changes | Medium | High | Skip Phase 1 if Phase 2 is the real goal |

---

## Verdict

**Conditional PROCEED — but only after empirical measurement (est. 1-2 days).**

The WarpFuture concept has intellectual merit, but the proposal jumps to a complex solution without evidence that the problem is significant in this project's actual workloads. The project should:

1. First measure warp divergence with per-thread futures (does the problem exist?)
2. Then try cheap alternatives (warp-cooperative scheduling, warp-cooperative CAS)
3. Only then, if neither solves it, proceed to WarpFuture — starting with a hand-written proof-of-concept, not a proc macro

Do not invest in `#[warp_async]` proc macro until a hand-written WarpFuture demonstrates measurable improvement on hardware.
