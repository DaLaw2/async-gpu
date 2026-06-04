# sc-runtime.1 — Warp Scheduling Model for Nested Scopes

## Status: done
## Summary: Fork/join with warp-0-only spawning is the correct model for async-gpu. Work-stealing adds complexity and overhead (atomic CAS contention on SM75) for a problem — nested spawning from non-zero warps — that is both rare and better solved by design constraints. The recommended approach is: (1) keep the current warp-0-as-manager model, (2) nested `block_scope` calls work only from warp 0 (which is already the case), (3) `spawn_all` in nested scopes always operates on the full worker pool, and (4) spawned warps that need sub-parallelism use `spawn_all`-within-`spawn_all` via a callback pattern rather than direct nested spawning.

## 1. Current Scheduling Model Analysis

The current model is a **centralized manager/worker thread pool**:

- **Warp 0** is the sole manager: it runs `gpu_main`, calls `spawn()`, and joins results
- **Warps 1..N-1** are workers: they spin-poll `WARP_STATUS[wid]` in `worker_loop()`
- States: `IDLE(0) → ASSIGNED(1) → RUNNING(2) → DONE(3)` for spawn, `IDLE(0) → COOPERATIVE(5) → RUNNING(2) → DONE(3)` for spawn_all
- `spawn()` does a linear scan for IDLE warps, writes closure+trampoline to per-warp `SCRATCH[wid]`, then sets `STATUS_ASSIGNED` with release ordering
- Workers see the status change, call the trampoline, set `STATUS_DONE`
- Only lane 0 of each warp manages closure data; other lanes participate via SIMT lockstep

**Key invariant**: Only warp 0 ever writes to another warp's `WARP_STATUS` (the ASSIGNED/COOPERATIVE transitions). Workers only write their own status (RUNNING, DONE). This single-writer design eliminates all contention on status flags.

**Strengths**:
- Zero contention: no CAS, no compare-and-swap retry loops
- Simple correctness: deterministic assignment, no ABA problems
- Low overhead: each spawn is O(N) scan + 1 release store
- Matches GPU hardware: GPU SMs are designed for bulk SIMD, not fine-grained work stealing

**Weaknesses**:
- Only warp 0 can spawn — serializes all task creation
- Workers cannot spawn sub-tasks (they run closures but cannot call `spawn()`)
- Nested scopes from non-zero warps are structurally impossible

## 2. Fork/Join vs Work-Stealing Comparison

### 2.1 Fork/Join (current approach, enhanced)

**Model**: Warp 0 forks work, workers execute, warp 0 joins. Nested scopes are entered only by warp 0 — they push/pop the allocator watermark and spawn from the same single manager.

**Scheduling cost**:
- Spawn: 1 linear scan (O(N) worst case, N ≤ 32) + 1 atomic store (release)
- Join: 1 atomic load poll loop per warp
- No contention on shared data structures
- Total: ~10-50 cycles per spawn on SM75

**Nested scope behavior**:
```
block_scope(|outer| {                    // warp 0 enters
    outer.spawn_all(|wid, nw| { ... }); // all warps participate
    // all warps return to IDLE

    block_scope(|inner| {                // warp 0 enters (nested)
        inner.spawn_all(|wid, nw| { ... }); // all warps participate again
    });
});
```

The allocator watermark stack (depth 4) already supports this. Each nested `block_scope` pushes a new watermark. Workers go IDLE between scopes, then get re-assigned.

### 2.2 Work-Stealing (alternative)

**Model**: Shared work queue in shared memory or global memory. Any warp can enqueue tasks. Idle warps dequeue from the queue head.

**Scheduling cost**:
- Enqueue: 1 atomic CAS on queue tail pointer
- Dequeue: 1 atomic CAS on queue head pointer (contention when many warps idle)
- CAS retry loop: 5-15 cycles per failed attempt on SM75 (no dedicated CAS unit)
- With 8 idle warps competing: average ~3 retries = ~30-75 cycles per dequeue
- Queue bookkeeping: ring buffer in shared memory (~128 bytes for 32-slot queue)

**Nested scope behavior**:
```
block_scope(|outer| {
    let h = outer.spawn(|| {
        // warp 3 running
        block_scope(|inner| {
            inner.spawn(|| { ... }); // warp 3 enqueues to shared work queue
        });
    });
});
```

**Advantages**:
- Any warp can spawn sub-tasks
- Load balancing: idle warps pull work dynamically
- Nested spawning from worker warps

**Disadvantages**:
- **Contention**: Multiple warps CASing the same queue pointers. SM75 has no hardware CAS — it's emulated via `atom.cas`, which serializes on the memory controller
- **Complexity**: Work queue needs overflow handling, memory ordering, ABA prevention
- **Deadlock risk**: If the queue is full and all warps are waiting to enqueue, deadlock. Requires either unbounded queue (memory pressure) or backpressure (complexity)
- **Memory overhead**: Per-task descriptors in shared memory (~16 bytes each: fn ptr + data ptr). With 32 warps and 32 queue slots, that's 512 bytes of shared memory for the queue alone
- **Debugging difficulty**: Non-deterministic task execution order makes reasoning about correctness harder
- **Diminishing returns**: With only 4-8 warps typical, the "load balancing" benefit is minimal — static partitioning via spawn_all is nearly optimal

### 2.3 Verdict

Work-stealing is a poor fit for intra-block GPU scheduling because:
1. The warp count is small (4-32), so load imbalance is small
2. SM75 atomic CAS is expensive relative to the work granularity
3. The fork/join model already achieves near-perfect utilization via `spawn_all`
4. Nested spawning from workers is a rare pattern that has simpler solutions (see Section 4)

**Recommendation: Keep fork/join.**

## 3. Nested Scope Scheduling Challenges

### 3.1 The core problem

Consider this scenario with 8 warps (warp 0 = manager, warps 1-7 = workers):

```rust
block_scope(|outer| {
    let h = outer.spawn(|| {
        // Runs on warp 3 (assigned by warp 0)
        block_scope(|inner| {
            inner.spawn(|| { /* ??? */ }); // Which warp runs this?
        });
    });
    h.join();
});
```

When warp 3 enters `block_scope`, it calls the `block_scope()` function. The current code has `debug_assert_eq!(warp_id(), 0, "block_scope() must be called from warp 0")` — so this will panic (or be a no-op in release mode with UB).

**Why it's hard**: `block_scope` assumes it is warp 0 because:
1. Only warp 0 can modify the `SharedMemAllocator` (not thread-safe)
2. Only warp 0 scans for idle warps in `spawn()`
3. The `WARP_STATUS` single-writer invariant would be violated if warp 3 tried to write `STATUS_ASSIGNED` to another warp's slot

### 3.2 What happens with spawn_all in nested scopes?

```rust
block_scope(|outer| {
    // Warps 1-7 are IDLE (available)
    let h1 = outer.spawn(|| { /* warp 1 */ });
    let h2 = outer.spawn(|| { /* warp 2 */ });
    // Warps 3-7 still IDLE; warps 1,2 are ASSIGNED/RUNNING

    // Now warp 1 wants to do parallel work:
    // PROBLEM: warp 1 cannot call spawn_all — it's not warp 0
    // PROBLEM: if it somehow could, warps 3-7 are idle but warp 1 can't manage them
});
```

The interaction between `spawn` and `spawn_all` in nested contexts is problematic because:
- `spawn_all` wakes ALL workers (warps 1..N-1) with `STATUS_COOPERATIVE`
- But some workers might already be RUNNING/ASSIGNED from an `outer.spawn()` call
- The current `spawn_all` code doesn't check for already-busy warps — it unconditionally sets `STATUS_COOPERATIVE` for warps 1..N-1

This is already a latent bug in the current scope.rs: if you call `spawn_all` while some warps are still running from a previous `spawn()`, you'd overwrite their status.

**The current code is safe** only because `spawn_all` is called from warp 0 and blocks until all warps are DONE. Sequential usage (spawn+join, then spawn_all) avoids the conflict. But interleaving spawn and spawn_all within the same scope is unsafe.

### 3.3 Warp exhaustion

With 8 warps total (typical 256-thread launch):
- Outer scope spawns 4 tasks → uses warps 1,2,3,4. Warps 5,6,7 idle.
- If warp 1 could nest-spawn, it has access to warps 5,6,7 = 3 workers.
- If warp 1 spawns 3 tasks and warp 2 also tries to nest-spawn → 0 warps available.
- Deadlock: warp 2 spin-waits for an idle warp that will never become idle.

With 4 warps (128-thread launch, common for small workloads):
- Outer scope spawns 2 tasks → warps 1,2 busy. Warp 3 idle.
- One nesting level already exhausts available warps.

**Conclusion**: With typical warp counts (4-8), nested spawning exhausts the warp pool after 1-2 levels. This makes deep nesting impractical regardless of the scheduling model.

## 4. Warp Partitioning Strategies

### Option A: Shared pool (all idle warps available to any scope)

Any scope's `spawn()` can claim any IDLE warp. First-come-first-served.

**Pros**: Maximum flexibility, no wasted warps
**Cons**: Requires making `spawn()` callable from any warp (breaks single-writer invariant), deadlock risk from warp exhaustion, non-deterministic behavior

**Assessment**: Requires fundamental redesign of the status flag protocol. High complexity, dubious benefit.

### Option B: Partitioned warps (outer scope gives inner scope a subset)

When spawning a closure that will enter a nested scope, the outer scope assigns a partition of warps:

```rust
outer.spawn_with_pool(warps: &[usize], || {
    // This warp is now the "manager" of its partition
    block_scope_partitioned(warps, |inner| {
        inner.spawn(|| { ... }); // Uses only warps from the partition
    });
});
```

**Pros**: No contention — each nested scope has exclusive access to its warp subset. Deterministic. No deadlock (partition is pre-allocated).
**Cons**: Requires API changes. Warp partitioning is wasteful — an idle warp in one partition can't help a busy partition. Adds significant implementation complexity (per-partition status arrays, or bitmask-based scanning).

**Assessment**: Elegant in theory but impractical. With 8 warps, partitioning leaves 2-3 warps per scope — barely useful. The complexity isn't justified by the benefit.

### Option C: Warp 0 only spawns; workers use spawn_all only (recommended)

Only warp 0 can call `scope.spawn()`. Workers cannot spawn sub-tasks. For data-parallel work within a spawned closure, the pattern is:

1. **Sequential-then-parallel**: The spawned closure does sequential setup, then calls back to warp 0 (via a result + re-spawn pattern) for parallel work
2. **Pre-partitioned spawn_all**: Warp 0 launches `spawn_all` with pre-computed partitions. Workers do their partition and return. No nesting needed.
3. **Cooperative nesting**: Only warp 0 can enter nested `block_scope`. The pattern is always: `spawn_all` → join → nested `block_scope` → `spawn_all` → join → ...

**Pros**: No changes to the scheduling model. Zero additional complexity. Matches the hardware reality (GPU favors bulk SIMD over nested task parallelism). Simple mental model for users.
**Cons**: Cannot nest spawn from worker warps. Users must structure code as sequential phases separated by parallel bulk operations.

**Assessment**: This is the right choice. See Section 6 for detailed justification.

## 5. spawn_all in Nested Contexts

### 5.1 Current behavior

`spawn_all` wakes warps 1..N-1 unconditionally. In a nested `block_scope`, this is exactly what you want — the nested scope uses the same full warp pool. The watermark allocator stack handles shared memory isolation.

```rust
block_scope(|outer| {
    let buf1 = outer.alloc::<f32>(256);
    outer.spawn_all(|wid, nw| { /* uses buf1, all warps */ });
    // All warps DONE → IDLE

    block_scope(|inner| {
        // Nested: watermark pushed
        let buf2 = inner.alloc::<f32>(128);
        inner.spawn_all(|wid, nw| {
            // Uses buf1 (still valid, outer scope) AND buf2 (inner scope)
            // ALL warps participate — same pool
        });
        // All warps DONE → IDLE
    });
    // Watermark popped: buf2 is reclaimed, buf1 still valid
});
```

This works correctly today because:
1. `spawn_all` always operates on the full warp pool (1..N-1)
2. It blocks until all warps complete before returning
3. Between `spawn_all` calls, all workers are IDLE
4. Nested `block_scope` is entered only by warp 0

### 5.2 What about spawn_all from a spawned warp?

If warp 3 (spawned by outer scope) calls `spawn_all`, it would try to wake warps 1..N-1. But:
- Warp 0 might be doing something (it's the manager)
- Warps 1,2,4..7 might be running other spawned tasks
- Writing `STATUS_COOPERATIVE` to a warp that's `STATUS_RUNNING` corrupts its state

**Conclusion**: `spawn_all` from a non-zero warp is unsafe with the current protocol. This is another reason to keep the warp-0-only-manages constraint.

### 5.3 Recommended pattern: phased execution

The idiomatic pattern for nested parallelism should be **phased execution**:

```rust
block_scope(|scope| {
    // Phase 1: parallel initialization
    let data = scope.alloc::<f32>(1024);
    scope.spawn_all(|wid, nw| {
        // All warps initialize their partition
    });

    // Phase 2: sequential analysis (warp 0 only)
    let pivot = find_pivot(data);

    // Phase 3: parallel computation based on analysis
    let results = scope.alloc::<f32>(1024);
    scope.spawn_all(|wid, nw| {
        // All warps compute using pivot value
    });

    // Phase 4: sequential reduction (warp 0 only)
    let total = reduce(results);
});
```

This alternation between `spawn_all` (parallel) and warp-0-sequential code is the natural GPU programming pattern. It maps directly to how CUDA kernels are structured (grid-stride loops with barriers).

## 6. Recommendation

### Primary recommendation: Keep fork/join with warp-0-only management

**Do not change the scheduling model.** The current centralized manager/worker design is correct for the hardware and use case.

**Specific decisions**:

1. **`block_scope()` remains warp-0-only**: The `debug_assert_eq!(warp_id(), 0)` check should become a hard `assert!` in release mode. Calling `block_scope` from a worker warp is a programming error, not a feature gap.

2. **`spawn()` remains warp-0-only**: Same reasoning. Only the manager warp can assign work.

3. **`spawn_all()` remains warp-0-only**: It wakes all workers, which only makes sense when the caller knows no other spawned tasks are running.

4. **Nested `block_scope` calls work via watermark stacking**: Already implemented. Warp 0 can enter nested scopes freely (up to depth 4). Each level pushes/pops the shared memory watermark. All workers participate in `spawn_all` at each level.

5. **No work-stealing queue**: The overhead (atomic CAS contention) exceeds the benefit (marginal load balancing) given the small warp counts (4-32) and the availability of `spawn_all` for bulk parallelism.

6. **No warp partitioning**: The warp pool is too small to partition effectively. The phased execution pattern (spawn_all → sequential → spawn_all) provides all the nested parallelism that practical GPU programs need.

### Latent safety issue to fix

The interaction between `spawn()` and `spawn_all()` within the same scope should be documented and guarded:

- `spawn_all()` currently sets STATUS_COOPERATIVE for warps 1..N-1 unconditionally
- If any warps are still RUNNING from a prior `spawn()` call, this corrupts their state
- **Fix**: `spawn_all()` should either (a) assert that `self.spawned_warps == 0` (all spawned tasks must be joined before calling spawn_all), or (b) only wake warps that are in the IDLE state

Option (a) is simpler and prevents misuse. Add to `spawn_all()`:
```rust
assert_eq!(
    self.spawned_warps, 0,
    "scope.spawn_all: all spawned tasks must be joined before calling spawn_all"
);
```

### Why not nested spawning from workers?

The argument against is both practical and principled:

1. **Practical**: With 4-8 warps typical, nesting exhausts the pool in 1 level. The overhead of enabling nested spawn (work queue, partitioning, contention management) is not justified by 1 extra level of parallelism with 2-3 available warps.

2. **Principled**: GPU programming is fundamentally about bulk data parallelism, not task parallelism. The `spawn_all` + phased pattern maps perfectly to the GPU execution model. Nested task spawning from workers is a CPU pattern that doesn't translate well to GPU hardware.

3. **User-facing**: A user who needs nested parallelism from a worker warp should restructure their algorithm into phases. This is actually better for GPU performance (coalesced memory access, fewer divergent warps) than ad-hoc nested spawning.

### Future consideration: spawn_all_subset

If use cases emerge where a subset of warps needs to do cooperative work (e.g., "warps 0-3 do task A while warps 4-7 do task B"), this could be added as:

```rust
scope.spawn_all_subset(warp_mask: u32, |wid, nw| { ... });
```

This would be a warp-0-managed operation (warp 0 decides the partition and wakes only the specified warps). It does not require changes to the scheduling model — it's just a filtered version of `spawn_all`. This can be deferred until a concrete use case demands it.

## Files Changed: none
