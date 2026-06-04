# sc-design.1 — Structured Concurrency Models → GPU Hardware Mapping

## Status: done
## Summary: Surveyed five scope-based concurrency models (Kotlin, Swift, Trio, Rayon, crossbeam) and mapped their core patterns to GPU hardware hierarchy (warp/block/grid). The key finding is that Rayon's `scope()` model maps most naturally to GPU structured concurrency: a scope borrows data, spawns work with lifetime-bounded parallelism, and joins implicitly at scope exit. BlockScope should own shared memory and use `bar.sync` for join; GridScope should use global memory with atomic completion counters. Cancellation should be cooperative (flag-based), not preemptive, matching GPU execution constraints.

## 1. Survey of Concurrency Models

### 1.1 Kotlin coroutineScope / supervisorScope

**Core abstraction:** `coroutineScope { }` creates a scope that suspends until all child coroutines complete. Children are launched with `launch { }` or `async { }` inside the scope. The scope itself is a suspend function — the caller is suspended, not blocked.

**Resource lifecycle:** The scope's `CoroutineContext` carries a `Job` hierarchy. Child jobs are automatically registered with the parent scope's job. When the scope's block completes, it awaits all children. Cancellation of the scope cascels all children and their children recursively.

**Cancellation:** Bidirectional. Parent cancellation propagates downward (all children cancelled). Child failure propagates upward — in `coroutineScope`, any child exception cancels all siblings and the parent. In `supervisorScope`, child failure does NOT cancel siblings (supervisor pattern). Cancellation is cooperative: coroutines check `isActive` at suspension points.

**Error handling:** `coroutineScope` re-throws the first child exception after cancelling all siblings. `supervisorScope` lets each child fail independently — the parent only sees exceptions explicitly awaited via `async { }.await`.

**GPU relevance:** The job hierarchy maps well to scope nesting (grid → block → warp). The cooperative cancellation model (check a flag at suspension points) is the only viable approach on GPU since there's no preemption. `supervisorScope` maps to independent block execution where one block's failure shouldn't kill others.

### 1.2 Swift TaskGroup (Structured Concurrency)

**Core abstraction:** `withTaskGroup(of: T.self) { group in group.addTask { ... } }`. The scope (`withTaskGroup`) doesn't return until all added tasks complete or are cancelled. `async let` provides implicit scoping — the binding's lifetime IS the scope.

**Resource lifecycle:** Tasks inherit the parent's task-local values and priority. The task group itself is the scope boundary — it cannot escape the `withTaskGroup` closure. Swift's type system enforces this: `TaskGroup` is `~Copyable` and non-escaping.

**Cancellation:** Cooperative and hierarchical. `Task.checkCancellation()` throws `CancellationError`. Parent cancellation propagates to all children. Cancellation sets a flag; tasks must check it. Unique feature: `withThrowingTaskGroup` auto-cancels remaining tasks when any task throws.

**Error handling:** In `withThrowingTaskGroup`, the first thrown error causes remaining tasks to be cancelled, and the error propagates to the parent. Results can also be collected iteratively via `for await result in group`.

**GPU relevance:** The `async let` pattern (implicit scope = variable lifetime) is compelling for GPU: imagine `let result = block_scope.spawn(...)` where the scope is tied to the enclosing block. The auto-cancel-on-error behavior maps to "if one warp traps, set a block-level error flag." The non-escaping constraint is critical — GPU shared memory MUST NOT escape block scope.

### 1.3 Python Trio Nurseries

**Core abstraction:** `async with trio.open_nursery() as nursery: nursery.start_soon(fn, args)`. The `async with` block is the scope boundary. The nursery doesn't exit until all tasks started in it have completed.

**Resource lifecycle:** The nursery IS the scope. All tasks must complete before the `async with` block exits. This is strictly enforced — there's no way to "fire and forget." Resources opened in the nursery block are guaranteed to outlive all nursery tasks.

**Cancellation:** Nursery cancellation scope wraps all tasks. If any task raises an exception, the nursery cancels all other tasks, waits for them to finish cleanup, then re-raises. "Shield" scopes can protect subtrees from parent cancellation. Cancel scopes can have deadlines.

**Error handling:** Strict. Any unhandled exception in a nursery task cancels all siblings and propagates as `MultiError` (now `ExceptionGroup` in Python 3.11+). No silent failures. The nursery guarantees you see every error.

**GPU relevance:** The strict "no fire-and-forget" rule is perfect for GPU: every spawned warp/block MUST be joined before the scope exits, because GPU resources (shared memory, synchronization barriers) are scope-bound. The MultiError pattern is interesting — if multiple warps fail, we need a way to collect all errors, not just the first. On GPU this could be a bitmask of failed warps.

### 1.4 Rust Rayon Scopes

**Core abstraction:** `rayon::scope(|s| { s.spawn(|s| { ... }); })`. The scope closure receives a `Scope<'scope>` handle. `spawn()` takes a closure bounded by `'scope` lifetime. The scope blocks until all spawned tasks complete.

**Resource lifecycle:** The `'scope` lifetime is the key innovation. Spawned closures can borrow stack data with `'scope` lifetime — the compiler guarantees the scope (and thus the borrowed data) outlives all tasks. No `'static` requirement, unlike `std::thread::spawn`.

**Cancellation:** No built-in cancellation. Rayon uses work-stealing and doesn't provide cancellation tokens. Panic propagation: if a spawned task panics, the panic is caught and re-thrown when the scope exits. Other tasks may continue executing until the scope tries to join.

**Error handling:** Panics in spawned tasks are caught and re-raised at scope exit. Only one panic is propagated (the first one encountered during join). No multi-error.

**GPU relevance:** **This is the closest model to what we need.** The lifetime-bounded scope is exactly right: `block_scope(|s| { s.spawn(...) })` where the scope guarantees shared memory is valid for all spawned work. Rust's borrow checker can enforce that shared memory references don't escape the scope. The missing cancellation is fine — GPU tasks generally run to completion anyway.

### 1.5 Crossbeam Scoped Threads

**Core abstraction:** `crossbeam::scope(|s| { s.spawn(|_| { ... }); }).unwrap()`. Very similar to Rayon scopes but maps to OS threads instead of a thread pool. The scope blocks the calling thread until all spawned threads join.

**Resource lifecycle:** Identical to Rayon: `'scope` lifetime bounds allow borrowing stack data without `'static`. The scope destructor joins all threads, ensuring cleanup.

**Cancellation:** None. Threads run to completion. If a thread panics, the panic is caught and the scope returns `Err`.

**Error handling:** `scope()` returns `Result<T, E>` — `Err` if any spawned thread panicked. The scope always joins all threads before returning, even on panic.

**GPU relevance:** Simpler model than Rayon (no work-stealing), which actually maps better to GPU blocks — each block is a fixed unit of execution, not a work-stealing pool. The explicit thread → warp mapping is natural. The `Result` return is clean for error propagation from block-level execution.

## 2. GPU Hardware Hierarchy Mapping

### 2.1 Warp Level (32 threads, registers + shuffle)

**What exists today:** The codebase has warp-level reductions (`warp::reduce_sum_f32`, `reduce_max_f32`), shuffle operations (`shfl_bfly_u32`, `shfl_down_u32`, `shfl_up_u32`), vote operations (`ballot`, `all`, `any`), and cooperative future polling (`warp_poll_future`, `warp_run_future`). All 32 lanes execute in SIMT lockstep.

**Structured concurrency at warp level:** Not applicable in the traditional sense. Warp lanes are not independent tasks — they execute the same instruction with different data (SIMD). There is no "spawn a task on lane 5." The warp is the atomic unit of scheduling.

**What a WarpScope would mean:** Not a concurrency scope but a **data-parallel scope** — all 32 lanes process elements in lockstep using shuffle-based communication. Communication cost: ~1 cycle (register shuffle). Synchronization: implicit (SIMT lockstep on Volta+, `syncwarp` for explicit convergence).

**Recommendation:** No WarpScope in the structured concurrency sense. Instead, warp-level operations remain as primitives (`warp::reduce`, `warp::shuffle`) that scopes at higher levels can use internally.

### 2.2 Block Level (up to 1024 threads = 32 warps, shared memory 48KB, bar.sync)

**What exists today:** `block::sync()` (bar.sync 0), `block::shared_mem_ptr()`, `block::shared_mem_at::<T>()`, block-level reductions (`reduce_sum_f32`, `reduce_max_f32`, `reduce_min_f32`). The thread pool (`thread.rs`) operates within a single block: warp 0 runs main, warps 1..N-1 are workers.

**Structured concurrency at block level:** This is the sweet spot. Within a block:
- **Spawn:** Wake a parked warp (existing mechanism in `thread::spawn`)
- **Join:** Wait for warp completion (existing `JoinHandle::join`)
- **Shared resources:** 48KB shared memory, accessible by all warps in the block
- **Synchronization:** `bar.sync` is cheap (~4 cycles latency), available to all warps
- **Scope boundary:** All warps must reach `bar.sync` before any can proceed — natural scope join point

**BlockScope design implications:**
- Scope allocates a region of shared memory at entry, deallocates (logically) at exit
- Spawned tasks can borrow shared memory with `'scope` lifetime
- Scope exit requires all spawned warps to complete (bar.sync or poll-based join)
- Maximum parallelism: ~32 warps per block (hardware limit)
- Maximum shared memory: 48KB on SM75 (our GTX 1660)

### 2.3 Grid Level (multiple blocks, global memory, atomics only)

**What exists today:** The executor (`GpuExecutor`) operates in global memory with system-scope atomics. Channels (oneshot, MPSC) use global memory with acquire/release semantics. The `Mutex` in `sync.rs` uses CAS-based spin-lock in global memory.

**Structured concurrency at grid level:** Challenging. Blocks are independently scheduled — there is no `grid.sync` on SM75 (cooperative groups with grid sync require SM70+ cooperative launch, which has significant limitations). Key constraints:
- **No barrier:** Blocks cannot barrier-sync with each other (on SM75 without cooperative launch)
- **Communication:** Global memory only, via atomics (~100 cycles for system-scope CAS)
- **Scheduling:** Blocks may not all be resident simultaneously — the GPU scheduler may context-switch blocks
- **No preemption:** A block cannot be interrupted or cancelled externally

**GridScope design implications:**
- Scope cannot use `bar.sync` — must use atomic completion counters in global memory
- Spawned blocks communicate via global memory channels (existing oneshot/MPSC)
- Scope exit waits for all blocks' completion counters, polled from host or a persistent block
- Resource cleanup: global memory allocations freed when scope's completion counter reaches N
- Grid sync possible via cooperative launch (`cudaLaunchCooperativeKernel`) but requires all blocks resident simultaneously — limits grid size

## 3. Pattern Translation Analysis

### 3.1 Direct translations

| CPU Pattern | GPU Equivalent | Notes |
|------------|---------------|-------|
| Rayon `scope(\|s\| s.spawn(...))` | `block_scope(\|s\| s.spawn_warp(...))` | Lifetime-bounded, shared memory access |
| `bar.sync` at scope exit | `block::sync()` | Already exists, cheap |
| Crossbeam `scope().unwrap()` | `grid_scope().unwrap()` | Returns Result for error propagation |
| Trio nursery (no fire-and-forget) | Implicit in both scopes | GPU MUST join all work before scope exit |
| Swift `async let` (implicit scope) | `let h = scope.spawn(...)` | Handle must be joined before scope drops |
| Kotlin cooperative cancellation | Atomic flag checked at yield points | GPU can't preempt; polling is natural |

### 3.2 GPU-specific adaptations needed

**1. Memory tier selection (the North Star):** No CPU model has this concept. The scope must determine which memory tier to use:
- BlockScope → allocator carves from 48KB shared memory
- GridScope → allocator uses global device memory (via existing hostcall or device-side allocation)
- Channel selection: BlockScope channels use shared memory atomics (~2 cycles); GridScope channels use global memory atomics (~100 cycles)

**2. SIMT divergence management:** CPU threads are independent; GPU warps have 32 lanes in lockstep. A scope spawn doesn't spawn a single thread — it wakes an entire warp. The scope must ensure all lanes of the spawned warp converge at expected points. The existing thread pool handles this correctly (only lane 0 manages closure data; all lanes execute the trampoline).

**3. Fixed parallelism:** CPU scopes (Rayon) can spawn arbitrary numbers of tasks into a work-stealing pool. GPU blocks have a fixed number of warps (set at launch time). BlockScope parallelism is bounded by `blockDim.x / 32`. GridScope parallelism is bounded by `gridDim.x * gridDim.y * gridDim.z`. Neither can grow dynamically.

**4. Two-phase scope exit:** CPU scopes simply join all threads. GPU BlockScope needs two phases: (a) all warps reach a sync point, (b) shared memory is logically freed. This maps to `bar.sync` followed by scope destructor resetting the shared memory allocator watermark.

**5. No stack unwinding:** CPU structured concurrency often relies on exception/panic unwinding for cleanup. GPU has no stack unwinding. Cleanup must be explicit: set an error flag, let all warps poll it at their next checkpoint, then join. This is closer to Kotlin's cooperative cancellation than Trio's exception-based model.

### 3.3 Key design tensions

**Tension 1: bar.sync join vs. polling join.** The existing thread pool uses polling (spin-wait on WARP_STATUS). `bar.sync` is faster (~4 cycles) but requires ALL threads in the block to participate — if any warp diverges or exits early, the block deadlocks. The current architecture handles this well for `cooperative()` but `spawn()` uses polling because spawned tasks have dynamic lifetimes. **Resolution:** BlockScope join should use bar.sync when ALL warps participate (cooperative mode), polling when only some warps are active (spawn mode).

**Tension 2: Scope nesting depth.** GPU has limited shared memory (48KB). Each nested BlockScope that allocates shared memory reduces the pool. Deep nesting can exhaust shared memory. **Resolution:** Limit BlockScope nesting to 2-3 levels. Use a watermark allocator: each scope pushes a watermark, allocations grow upward, scope exit pops back. Similar to a stack allocator.

**Tension 3: GridScope completion detection.** Without grid-level barriers, detecting that all blocks in a GridScope have completed requires either: (a) a persistent "monitor" block that polls atomic counters, (b) host-side polling of a mapped memory flag, or (c) cooperative launch with grid sync. Option (b) is simplest and works with the existing hostcall infrastructure. Option (c) limits grid size but provides true grid sync. **Resolution:** Start with (b) — host-poll completion via mapped memory. The host already polls for hostcall responses; adding a scope-completion poll is natural.

**Tension 4: Cancellation granularity.** Cancelling a single warp in a block is cheap (set its status flag). Cancelling a block in a grid requires that block to check a global memory flag. Cancelling a grid requires host intervention. **Resolution:** Each scope level has a cancellation flag at the appropriate memory tier: BlockScope flag in shared memory (read by all warps), GridScope flag in global memory (read by all blocks). Tasks check the flag at yield/poll points.

**Tension 5: Error collection.** If multiple warps fail, how does the scope collect errors? Shared memory has limited space. **Resolution:** Use a bitmask (one bit per warp/block) plus a single error slot for the first error's details. For BlockScope: a u32 bitmask in shared memory (covers 32 warps). For GridScope: a u32 atomic counter in global memory + first-error slot.

## 4. Recommendation

### BlockScope Design

```
block_scope(shared_mem_bytes: usize, |scope| {
    // scope.alloc::<T>(count) → &'scope mut [T]  (from shared memory)
    // scope.spawn(|| { ... })  → JoinHandle (wakes a parked warp)
    // scope.cooperative(|| { ... })  (all warps execute in lockstep)
    // implicit join + dealloc at scope exit
})
```

**Implementation approach:**
- Build on the existing `thread::cooperative()` and `thread::spawn()` infrastructure
- Add a shared memory watermark allocator (push watermark on scope entry, pop on exit)
- `scope.alloc()` bumps the watermark, returns `&'scope mut [T]` — Rust lifetime prevents escape
- Scope exit: poll-join all spawned warps, then `bar.sync` as final barrier, then pop watermark
- Cancellation: u32 flag in shared memory; warps check on yield
- Error: u32 error bitmask in shared memory; first error detail in a shared slot

### GridScope Design

```
grid_scope(|scope| {
    // scope.alloc::<T>(count) → DevicePtr<T>  (global memory)
    // scope.spawn_block(|| { ... })  (launch a block's worth of work)
    // implicit join + dealloc at scope exit
})
```

**Implementation approach:**
- GridScope is fundamentally a host-side coordination primitive
- Allocations use global device memory (via the existing memory subsystem)
- Block completion tracked by atomic counter in global memory (mapped for host visibility)
- Scope exit: host polls completion counter until it reaches expected block count
- Cancellation: global memory flag; blocks check at checkpoints
- Error: atomic counter for failed blocks + first-error slot in global memory
- Channels between blocks: use existing MPSC/oneshot in global memory (already system-scope)

### Channel Auto-Selection

The North Star says "channels pick fastest path automatically." Implementation:
- If both endpoints are in the same BlockScope → shared memory channel (fast path)
- If endpoints are in different BlockScopes or in GridScope → global memory channel (slow path)
- Detection at channel creation: check if both endpoint warp IDs share the same block ID
- This can be encoded in the channel type: `BlockChannel<T>` vs `GridChannel<T>`, with a unified `Channel<T>` enum that dispatches

### Recommended Implementation Order

1. **Shared memory watermark allocator** — foundation for BlockScope
2. **BlockScope** with spawn + cooperative + join — builds on existing thread pool
3. **BlockScope channels** using shared memory — fast intra-block communication
4. **GridScope** with host-side completion tracking — extends to multi-block
5. **Channel auto-selection** — unified API over block/grid channels

## Files Changed: none (investigation only)
