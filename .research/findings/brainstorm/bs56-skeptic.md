# BS56 — Skeptic Challenge: Aligning `#[warp_async]` with Rust Native `async/await`

**Role**: Skeptic
**Epic**: gpu-autonomous v3 — Native async/await on GPU
**Date**: 2026-03-14

---

## 1. Assumption Challenges

### Approach A: Wrapper Layer (Async Hostcall I/O with Standard Future)

**"This assumes single-thread GPU async is useful — but what about real GPU workloads?"**

The proposer frames Approach A as "2-3 weeks for basic async hostcall + executor." But the result is a system that runs one thread on one SM, doing I/O one operation at a time. This is strictly worse than the synchronous `gpu-kernel-std` path, which already provides `File::create()`, `println!()`, etc. What does `.await` buy you when there is only one task and no concurrency?

The `async-hostcall-test` crate already demonstrates exactly this: `HostcallPrintFuture` implements `core::future::Future`, Embassy polls it, and it works. The proposer acknowledges this as "PROVEN." But the key finding from that test is also present: Embassy's `__pender` is a no-op. The Waker does nothing. `wake_by_ref()` just re-enqueues — there is no actual wakeup mechanism. This means:

- Every `Poll::Pending` results in a busy-loop re-poll
- There is zero benefit over a synchronous spin-wait unless you have multiple tasks
- Multiple tasks on a single thread means time-slicing I/O waits — but all hostcall packets go to the same listener anyway, so the throughput is identical

**"The `HostcallPrintFutureB` duplication reveals a deeper problem."**

The `async-hostcall-test` crate had to create a *duplicate struct* (`HostcallPrintFutureB`) with identical logic just because Embassy's `TaskStorage` is generic over the Future type and each static needs a unique type. This is a fundamental ergonomic failure that `.await` syntax does not solve. The async-io theme would face the same problem unless it abandons Embassy's static task storage model — but then what executor does it use?

**"This would break if we try multi-thread."**

The proposer admits "Needs multi-thread fix" for Approach A. But the entire value proposition of `.await` is composability with ecosystem tools (`select!`, `join!`, `FuturesUnordered`). These tools assume independent tasks with independent memory. On a GPU with 32 threads in a warp all executing the same instruction, `FuturesUnordered` would have 32 copies of the same Future polling the same hostcall buffer with CAS contention — exactly the anti-pattern that `WarpFuture` was designed to avoid.

---

### Approach B: WarpFuture <-> Future Bridge

**"This assumes `.await` syntax is the bottleneck — but is it?"**

The proposer's B2 sub-approach replaces `warp_print!(buf, b"msg")` with `gpu_print(buf, b"msg").await`. This is literally a search-and-replace at the macro level. The `.await` is intercepted by the proc macro and discarded. No Future is created. No Poll is returned. The compiler never sees an async fn.

What does the user gain? Autocomplete? No — the marker types returned by `gpu_print()` are not real Futures, so rust-analyzer will complain. Familiarity? Arguably — but the user still needs to understand warp convergence, lane 0 leadership, and SIMT constraints. The `.await` sugar hides these realities, making debugging harder.

**"This would break with any non-trivial composition."**

If the user writes:
```rust
let (a, b) = futures::join!(gpu_print(buf, b"hello"), gpu_open(buf, b"file.txt", 1)).await;
```
The proc macro cannot handle this. `futures::join!` produces a real Future combinator. The proc macro would need to understand arbitrary Future combinators, which is impossible. So the "bridge" approach immediately falls apart when users try to use it like real async.

---

### Approach C: rustc Modification

**"This assumes the Rust team would accept GPU-specific codegen — but they won't."**

The Rust compiler team has consistently rejected target-specific language semantics. The `nvptx64` target is a Tier 3 target with no CI, no guarantees, and known breakage. Proposing a warp-cooperative coroutine lowering pass for a Tier 3 target is dead on arrival for upstream. And a fork means maintaining patches against a compiler that ships new nightlies every day.

**"This assumes LLVM's nvptx backend can emit warp-cooperative code — but it can't."**

The proposer mentions "LLVM backend changes" casually. The LLVM nvptx backend emits per-thread PTX. It has no concept of warps at the IR level. `shfl.sync`, `bar.warp.sync`, and lane-conditional execution are inline assembly or intrinsics — they are invisible to LLVM's optimization passes. A compiler-generated warp-cooperative state machine would need to survive LLVM's optimization pipeline without LLVM understanding what it is. LLVM could:
- Hoist the `shfl.sync` out of a loop
- Sink the `syncwarp()` past a conditional
- Merge two separate lane-guarded regions
Any of these would break warp convergence silently.

**"This assumes `File::create().await` makes semantic sense — but it doesn't."**

`std::fs::File::create()` returns `io::Result<File>`, not a Future. Making it return a Future on GPU would mean changing the return type of a fundamental std function based on the target. This violates Rust's core principle that the type system is target-independent. The alternative — `AsyncFile::create()` — is exactly what Approach A proposes, just reached via a longer path.

---

## 2. Fundamental Tensions

### Warp-Cooperative (SIMT) vs Per-Thread (CPU Async): Is This Gap Bridgeable?

No. These are fundamentally different execution models:

| Property | WarpFuture (SIMT) | core::future::Future (CPU) |
|----------|-------------------|----------------------------|
| Execution unit | 32 lanes, lockstep | 1 task, independent |
| State ownership | Lane 0 owns, broadcasts | Each task owns its state |
| Side effects | Lane 0 only | Any task |
| Synchronization | Implicit (SIMT) + syncwarp | Explicit (Mutex, channel) |
| Divergence | Forbidden (deadlock) | Expected (independent progress) |
| Memory model | Registers + shfl.sync | Heap-allocated Future state |

The proposer's Approach A acknowledges this by abandoning warp cooperation entirely. Approach B acknowledges it by using cosmetic syntax over WarpFuture. Approach C proposes to bridge it via compiler magic. But compiler magic cannot change the hardware: SIMT requires lockstep, and `Future::poll()` assumes independent progress. These are axioms, not implementation details.

**The honest answer**: The gap is not bridgeable at the language level without fundamentally changing what `async fn` means. What we have — `#[warp_async]` as a domain-specific async paradigm — is the correct abstraction for SIMT hardware.

### Spin-Poll vs Waker-Based Wake: Does This Matter?

Yes, but not in the way the proposer frames it. The proposer correctly notes that `__pender` is a no-op on GPU. But the deeper issue is: **the entire Waker mechanism is wasted code on GPU**. Every `Poll::Pending` allocates or references a Waker that does nothing. Embassy's executor still checks the run queue, iterates task states, and calls wake_by_ref — all for a spin-poll that `nanosleep + loop` achieves in 3 instructions.

The WarpExecutor is 30 lines of code. Embassy's executor on GPU is hundreds of lines of code doing the same thing with more overhead. The proposer's "PROVEN: Embassy executor already compiles and runs on GPU" is true, but "compiles and runs" is not the same as "is the right tool."

### Single-Thread std vs Multi-Thread Warp: Which Model Wins?

Neither wins universally. But the proposer's analysis reveals a critical insight they underplay: **these two models serve fundamentally different use cases**.

- Single-thread std (with async): Good for sequential I/O-heavy workloads where the GPU is used as a general-purpose processor. This is a niche use case — if you want sequential I/O, use the CPU.
- Multi-thread warp (WarpFuture): Good for GPU-native workloads where 32 lanes cooperate on compute + I/O. This is the actual GPU value proposition.

The proposer recommends both layers. But maintaining two parallel async systems (`async-io` for single-thread, `#[warp_async]` for warp) doubles the API surface, doubles the documentation burden, doubles the testing matrix, and confuses users about which to use.

---

## 3. The "Do We Even Want This?" Challenge

### Is aligning with async/await actually valuable, or just aesthetically pleasing?

Let me be direct: **it is primarily aesthetic**.

The current `#[warp_async]` system already provides:
- Sequential code style (no manual state machines)
- Control flow: if/else, match, loop with break
- Variable capture across yield points
- Error propagation (via hostcall result codes)
- Composable pipelines (see `autonomous_pipeline`)

What `.await` adds:
- Familiar syntax for Rust developers
- Theoretical composability with ecosystem (which doesn't work on GPU anyway)
- IDE support (which also doesn't work for custom proc macros on nvptx64)

What `.await` costs:
- A second async system to maintain
- Confusion about when to use which
- No warp cooperation (or fake warp cooperation via proc macro tricks)
- The Waker/Context overhead for a spin-poll executor

### What does the user ACTUALLY gain from `.await` vs `warp_print!`?

The user writing `gpu_print(buf, b"hello").await` vs `warp_print!(buf, b"hello")` gains:
1. Five fewer characters to type
2. A `.await` keyword that suggests interruptibility (which is technically true but pragmatically irrelevant on a spin-poll executor)
3. A false sense that this is "normal" async Rust (it is not — there are no wakers, no tasks, no runtime)

The user loses:
1. The explicit signal that this is a warp-cooperative operation
2. Clarity about what is happening (32 lanes cooperating vs 1 thread spinning)
3. The ability to mix with real Rust async (because it isn't real async)

### Is the real value in the syntax, or in something deeper?

The proposer hints at "composability" and "ecosystem" as deeper values. Let me challenge each:

**Composability**: What would you compose? GPU I/O futures with... what? There is no `tokio::net` on GPU. There is no `reqwest` on GPU. There is no `sqlx` on GPU. The only futures are hostcall I/O futures that we write ourselves. Composing our own futures with our own executor using our own protocol. `select!` and `join!` are available — but `join!` of two hostcall prints is already demonstrated by `futures_join_kernel` in `async-hostcall-test`. It works, but the value is marginal.

**Ecosystem**: The async ecosystem assumes `std`, `alloc`, networking, timers, and OS threads. GPU has none of these. The "ecosystem" argument is hollow.

---

## 4. Alternative Perspectives

### What if the right answer is NOT alignment, but doubling down on WarpFuture?

Consider: `#[warp_async]` is already a sophisticated code generator that handles if/else, match, loop, variable capture, and multi-service hostcall composition. Instead of building a parallel `async-io` system, invest in making `#[warp_async]` better:

- **Add `?` operator support**: `let fd = warp_open!(buf, b"data.txt", 0)?;` — early return on error, propagate to caller
- **Add nested function calls**: `warp_call!(other_warp_fn, buf)` — compose WarpFutures
- **Add compute blocks**: Arbitrary Rust code between hostcall points, not just `if`/`match`
- **Add warp-cooperative iterators**: `for item in warp_iter!(buf, fd) { ... }`

This extends the existing, working, warp-native system rather than building a parallel, warp-incompatible system.

### What if the right answer is abandoning warp-cooperative for single-thread async?

This is the path the proposer's Approach A takes, though they don't frame it this way. If single-thread Embassy + `core::future::Future` is "good enough," then:

- Delete `WarpFuture`, `WarpExecutor`, `WarpContext`, `warp_macro`
- Standardize on single-thread Embassy executor for all GPU async
- Accept the 32x inefficiency for I/O operations (only 1 of 32 lanes does useful work)
- Accept that GPU compute cannot be async (only I/O can yield)

This is a valid simplification — but it means giving up the GPU's most distinctive feature (SIMT parallelism) for async I/O. Is async I/O on GPU important enough to justify this?

### What if the right answer is something nobody proposed?

**A hostcall-as-a-syscall model**: Instead of thinking about async/await, think about hostcalls as GPU "system calls." The GPU executes synchronously. When it needs I/O, it issues a hostcall (like a CPU issuing `syscall`). The hostcall blocks the warp (like a CPU syscall blocks the thread). Other warps continue executing (like other CPU threads continue during a syscall). No async needed — the "concurrency" comes from warp scheduling, which CUDA already provides.

This is actually what the synchronous `gpu-kernel-std` path already does, minus the multi-warp part. If multi-warp synchronous hostcall works, then:
- No executor needed (not Embassy, not WarpExecutor)
- No Future trait needed (not WarpFuture, not core::future::Future)
- No proc macro needed (not `#[warp_async]`, not `#[gpu_async]`)
- Code is just regular Rust: `let f = File::create("data.txt")?;`

The only missing piece is multi-warp synchronous hostcall, which requires the hostcall buffer to handle concurrent requests from multiple warps — **which it already does** (sharded buffers, per-block free stacks, ABA-tagged CAS).

Has this path been fully explored? The multi-block async tests (`multi_block_async_kernel`, `warp_scale_async_kernel`) use Embassy, but has anyone tried multi-block synchronous hostcall with blocking?

---

## 5. Risk Assessment

### Approach A (Wrapper Layer): What's the worst case?

**Worst case**: We spend 2-3 weeks building `GpuFile`, `GpuPrint`, etc., write an Embassy-based executor, get it working for single-thread kernels, and then discover:
1. Nobody uses it because single-thread GPU is not useful enough
2. The API is incompatible with warp-cooperative kernels, so performance-critical code still uses `#[warp_async]`
3. We now maintain two async systems forever

**Probability**: High. The `async-hostcall-test` crate already demonstrates this approach and has not been extended or reused since its creation. This suggests the approach was tried, validated, and found insufficient.

### Approach B (WarpFuture Bridge): What's the worst case?

**Worst case**: We spend 2-3 weeks making `.await` work in `#[warp_async]`, and the result is indistinguishable from the current macro system to users. The cosmetic improvement does not attract new users or simplify existing code.

**Probability**: Very high. The syntax difference between `warp_print!(buf, b"msg")` and `gpu_print(buf, b"msg").await` is negligible. The mental model (warp-cooperative state machine) is unchanged.

### Approach C (rustc Modification): What's the worst case?

**Worst case**: We spend 3-6 months studying rustc internals, write a design document, attempt a fork, and discover that LLVM's nvptx backend cannot correctly optimize warp-cooperative code generated by rustc's coroutine transform. The project is abandoned, having consumed significant resources.

**Probability**: Medium-high for the fork attempt, near-certain for upstream acceptance.

### The Recommended Hybrid (A + B): What's the worst case?

**Worst case**: We build two new async layers (async-io + syntax sugar), adding ~2000 lines of code, and the result is a confusing API surface with three ways to do I/O on GPU:
1. Synchronous `std`: `File::create("x")` (single-thread, blocking)
2. Async wrapper: `GpuFile::create("x").await` (single-thread, Embassy)
3. Warp async: `warp_open!(buf, b"x", 1)` (warp-cooperative, WarpFuture)

Users ask "which one do I use?" The answer is always "it depends" — which means the abstraction has failed.

### Minimum Viable Experiments

Before committing to any approach, these experiments would resolve the key uncertainties:

1. **Multi-warp synchronous hostcall test**: Launch a kernel with 4 warps, each doing a synchronous `println!()` via `gpu-kernel-std`. If this works without modification, the entire async discussion may be moot — synchronous blocking + warp scheduling gives you concurrency for free.

2. **WarpFuture `?` operator prototype**: Implement error propagation in `#[warp_async]` using `?` syntax. If this works, it closes the biggest ergonomic gap between `#[warp_async]` and real async Rust without building a new system.

3. **Measure the actual overhead**: Compare register usage and execution time between:
   - Synchronous hostcall (current `gpu-kernel-std`)
   - `WarpFuture` hostcall (current `#[warp_async]`)
   - Embassy `core::future::Future` hostcall (current `async-hostcall-test`)

   If Embassy adds 50%+ register pressure, that alone may disqualify Approach A for any workload beyond trivial demos.

---

## Summary Verdict

The proposer frames this as a quest for "dream syntax" — making GPU async look like CPU async. But the skeptic's challenge is: **the GPU is not a CPU, and making it look like one is a category error**.

1. **Approach A** (Wrapper Layer) is proven but irrelevant: single-thread async on GPU solves a problem nobody has. The `async-hostcall-test` crate already demonstrates it and has not been reused.

2. **Approach B** (WarpFuture Bridge) is a cosmetic rename that adds no capability. The `.await` keyword is intercepted and discarded. The user still writes warp-cooperative code.

3. **Approach C** (rustc Modification) is a multi-year research project with near-zero probability of upstream acceptance and significant risk of LLVM-level breakage.

4. **The recommended hybrid** (A + B + research C) creates three async systems where one already works, adding maintenance burden without clear user benefit.

**Counter-recommendation**: Instead of chasing alignment with `async/await`, invest in:
1. Error propagation (`?`) in `#[warp_async]`
2. WarpFuture composition (nested calls, warp-cooperative iterators)
3. Multi-warp synchronous hostcall (if this works, async is unnecessary)
4. If a new epic is needed, focus on typed cross-launch pipelines (compile-time safe multi-kernel workflows) — genuinely novel, not achievable with current CUDA, and directly useful
