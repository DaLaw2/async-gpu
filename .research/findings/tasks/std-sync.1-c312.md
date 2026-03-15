# std-sync.1: GPU sync primitive feasibility
**Cycle**: 312 | **Theme**: std-sync | **Kind**: investigation | **Status**: done

## Summary
GPU Mutex is feasible across warps/blocks using `sys_cas_u32` spin-lock, and intra-warp
mutex is also possible on SM70+ (Volta) thanks to Independent Thread Scheduling (ITS).
However, intra-warp locks should be avoided by design — warp-cooperative patterns
(lane 0 does work, broadcast result) are always superior. RwLock, Barrier, and HashMap
are all feasible with varying complexity.

## Findings

### Q: Can GPU atomics implement Mutex correctly across warps/blocks?
A: **Yes.** The project already has all required primitives in `gpu-atomics`:

- `sys_cas_u32(ptr, 0, 1)` — atomic compare-and-swap for lock acquisition
- `sys_exchange_u64(ptr, 0)` — (or a u32 variant) for unlock via atomic exchange
- `sys_spin_load_acquire_u32` — spin-loop-safe acquire load (includes `nanosleep.u32 64`)
- `membar_sys()` — system-scope memory barrier for ordering

A minimal Mutex implementation:
```
lock:   while sys_cas_u32(&lock_word, 0, 1) != 0 { nanosleep; }
unlock: sys_store_release_u32(&lock_word, 0)
```

Cross-warp/cross-block mutexes work because different warps can be at different
instruction pointers. The existing Treiber stack in `gpu-runtime` already uses
ABA-tagged CAS loops (sys_cas_u64) for lock-free concurrent access — proving
the atomics work correctly for cross-warp synchronization.

The allocator (`std-patches/sys_alloc_cuda.rs`) also uses `compare_exchange_weak`
CAS loops on `AtomicU32` bitmaps for concurrent slab allocation, further confirming
lock-free patterns work on this target.

**Confidence**: high

### Q: What are the deadlock risks with warp-level lock acquisition?
A: There are two distinct scenarios:

**1. Cross-warp (safe):** Different warps have independent program counters. Warp A
can hold a lock while Warp B spins — Warp A will eventually release. No deadlock.

**2. Intra-warp (conditionally safe on SM70+):**
- **Pre-Volta (SM < 7.0):** FATAL deadlock. All 32 lanes share one program counter.
  If lane 0 acquires the lock and lane 1 spins, lane 0 cannot advance to release
  because the warp is stuck at the spin instruction. Classic SIMT deadlock.
- **Volta+ (SM >= 7.0, including our target SM86/RTX 3060):** Independent Thread
  Scheduling (ITS) gives each thread its own program counter and call stack. The
  scheduler can execute lane 0 (which holds the lock) forward to the unlock while
  lane 1 remains in the spin loop. This makes intra-warp locks *technically* safe.

**However, intra-warp locks should be avoided by design:**
- ITS progress guarantees are implementation-defined, not architecturally guaranteed
- Performance is terrible — serializes what should be parallel SIMT execution
- The project's warp-cooperative pattern (lane 0 acts, `shfl.sync` broadcasts) is
  strictly superior for all use cases encountered so far

**Recommended invariant:** `Mutex<T>` should document that it is designed for
cross-warp/cross-block synchronization. Intra-warp use is "works but don't."

**Confidence**: high

### Q: Should Mutex use spin-lock or hostcall-based sleep?
A: **Spin-lock with nanosleep yield**, for multiple reasons:

**Spin-lock advantages:**
- Simple, proven — the project already uses spin-poll extensively (hostcall protocol,
  `SpinExecutor`, `block_on`, `WarpFuture` poll loops)
- `sys_spin_load_acquire_u32` already includes `nanosleep.u32 64` to yield the warp
  scheduling slot, preventing the spin from starving other warps
- Low latency for short critical sections (expected use case)
- No host round-trip overhead

**Hostcall-based sleep disadvantages:**
- Round-trip to host CPU adds microseconds of latency per lock attempt
- Requires a free hostcall packet (can fail under contention)
- Over-engineered for the expected use case (protecting small shared data structures)

**Hybrid approach (future optimization):**
- Spin for N iterations (e.g., 1000)
- If still contended, yield via hostcall (prevents livelock under extreme contention)
- Similar to Linux `mutex_lock` adaptive spinning

**Recommended:** Start with pure spin-lock using `sys_cas_u32` + `nanosleep`. The
project already validates this pattern in `GPU_MAX_SPIN` (10M iteration timeout).
Add a configurable spin limit and panic/trap on timeout for debugging.

**Confidence**: high

### Q: RwLock, Barrier, HashMap feasibility?
A:

**RwLock — Feasible but complex:**
- Use a single `u32` state word: bits 0-30 = reader count, bit 31 = writer flag
- Read lock: CAS to increment reader count (fail if writer flag set)
- Write lock: CAS to set writer flag (fail if any readers or writer)
- Unlock: CAS to decrement/clear appropriately
- Same intra-warp caveats as Mutex — cross-warp only
- **Risk:** Reader starvation or writer starvation depending on policy
- **Recommendation:** Defer until a concrete use case emerges. Most GPU shared state
  is better served by lock-free patterns (atomics, Treiber stacks).

**Barrier — Feasible, two levels:**
1. **Intra-block:** Already available via `bar.sync` PTX instruction (`__syncthreads()`
   equivalent). Could expose as `gpu_barrier_block()`. Simple and hardware-supported.
2. **Cross-block:** Must use atomic counter + spin-wait. Pattern:
   `fetch_add(&counter, 1)` → if result == num_blocks-1, all arrived, release;
   else spin on a "generation" flag. Known to work (published research) but requires
   knowing total block count at barrier creation time.
   **Risk:** If any block finishes early and its SM is reused for a new block, the
   new block never reaches the barrier → deadlock. Only safe if grid is persistent
   (num_blocks <= SM count).

**HashMap — Feasible with bump allocator:**
- Open-addressing hash map with fixed-size bucket array
- Allocate bucket array from bump allocator at init time
- Use `sys_cas_u32/u64` for atomic insert (CAS empty slot → key)
- Linear or quadratic probing for collision resolution
- Lock-free reads (atomic loads), lock-free inserts (CAS), deletes need tombstones
- **Constraint:** No resizing (bump allocator doesn't support realloc). Must size
  the table generously at creation. Load factor > 0.7 → severe probe chain lengths.
- **Alternative:** Per-warp hash maps (no contention, each warp owns its table) with
  a merge phase. Better performance but more complex API.

**Confidence**: medium (RwLock/Barrier well-understood in theory; HashMap on GPU
has performance unknowns that need benchmarking)

## Unexpected Discoveries

1. **The project already has extensive spin-loop infrastructure.** The `sys_spin_load_acquire_u32`
   primitive with `nanosleep` yield, the `GPU_MAX_SPIN` timeout pattern, and the
   `SpinExecutor` all provide a proven foundation for Mutex implementation. Very
   little new code is needed.

2. **The critical-section crate (`gpu-critical-section`) is a no-op.** It explicitly
   documents that GPU threads cannot be preempted and each has its own executor.
   This means Mutex is NOT needed for executor internals — only for user-facing
   shared data structures across warps/blocks.

3. **The allocator already proves lock-free CAS patterns work.** Both the slab bitmap
   allocator (`compare_exchange_weak` loops) and the Treiber stack (`sys_cas_u64`
   with ABA tagging) confirm that atomic operations work correctly for concurrent
   GPU access patterns.

## Open Questions

1. **Should Mutex be system-scope or GPU-scope?** System-scope (`.sys`) is needed
   for GPU↔CPU shared memory. For GPU-only shared data, `.gpu` scope would be
   faster but `gpu-atomics` currently only exposes `.sys` scope. Consider adding
   `.gpu` scope variants.

2. **Poisoning semantics?** std::sync::Mutex poisons on panic. GPU panics trap the
   thread — should Mutex detect this? Probably not in v1 (keep it simple).

3. **Should we expose `__syncthreads()` as `Barrier::block_sync()`?** It's trivial
   to implement and useful, but it's a block-level primitive, not a general barrier.

4. **HashMap probing strategy?** Robin Hood hashing gives better worst-case but
   requires more CAS operations per insert. Needs benchmarking.

## Impact on Downstream Tasks

- **std-sync.2 (Mutex implementation):** Can proceed — all primitives exist. Design:
  `Mutex<T>` with `lock() -> MutexGuard<T>`, spin-lock via `sys_cas_u32`, RAII unlock.
  Estimated effort: small (< 200 LOC).
- **std-sync.3 (Barrier):** Block-level barrier is trivial. Cross-block barrier
  should be deferred (complex, niche use case).
- **std-sync.4 (HashMap):** Feasible but should wait for a concrete use case to
  drive the API design. The bump allocator's no-realloc constraint shapes the design
  significantly.
- **async-std theme:** Mutex/RwLock can integrate with the warp-cooperative async
  runtime — `lock().await` could yield the warp instead of spinning.
