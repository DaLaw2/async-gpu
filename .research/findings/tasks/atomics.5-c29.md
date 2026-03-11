# atomics.5: Audit .sys scope inline PTX correctness vs VectorWare Relaxed approach
**Date**: 2026-03-12
**Cycle**: 29
**Theme**: atomics
**Kind**: investigation
**Status**: done
**Spawned by**: user

## Summary

**CRITICAL UPDATE: `core::sync::atomic` now works correctly on nvptx64 with LLVM 21.**

The project's nightly Rust (rustc 1.91.0-nightly, 2025-08-25) ships with **LLVM 21.1.0**, which
contains a completely reworked NVPTX atomic lowering. The LLVM NVPTX backend now correctly maps
LLVM IR atomic orderings and syncscopes to PTX scoped instructions. This overturns the central
finding of `atomics.1` (which was based on LLVM 19.1.7).

Empirically verified PTX output on SM 86 / PTX 7.1:

| Rust code | PTX emitted (LLVM 21) | PTX emitted (LLVM 19, atomics.1) |
|-----------|----------------------|----------------------------------|
| `store(42, Release)` | `st.release.sys.global.b32 [ptr], 42` | (not testable, would be wrong) |
| `load(Acquire)` | `ld.acquire.sys.global.b32 %r, [ptr]` | (not testable, would be wrong) |
| `store(1, Relaxed)` | `st.relaxed.sys.global.b32 [ptr], 1` | `st.volatile` |
| `load(Relaxed)` | `ld.relaxed.sys.global.b32 %r, [ptr]` | `ld.volatile` |
| `fence(SeqCst)` | `fence.sc.sys` | (crash or silently dropped) |
| `fetch_add(1, SeqCst)` | `atom.global.add.u32` (NO scope/sem) | `atom.global.add.u32` (same) |

**Key findings:**

1. Atomic loads and stores now emit correct `.sys` scope + `.sem` ordering on LLVM 21.
2. `fence(SeqCst)` now correctly emits `fence.sc.sys` (previously crashed or was dropped).
3. **`atomicrmw` (fetch_add, CAS, etc.) is STILL broken** -- emits bare `atom.global.add.u32`
   without scope or ordering. The `atomicrmw` lowering path has NOT been fixed.
4. Our inline PTX in `gpu-atomics` remains necessary for RMW operations (`atom.cas.sys`,
   `atom.add.sys`, `atom.exch.sys`).
5. For plain loads/stores, `core::sync::atomic` is now a viable alternative to inline PTX.
6. VectorWare's `Ordering::Relaxed` is likely intentional, not a bug.

---

## Q1: What PTX instructions does core::sync::atomic generate on nvptx64?

### Empirical results (LLVM 21.1.0, SM 86, PTX 7.1)

**Atomic stores:**
```rust
AtomicU32::store(42, Ordering::Release)
// Emits: st.release.sys.global.b32 [%rd2], 42;

AtomicU32::store(1, Ordering::Relaxed)
// Emits: st.relaxed.sys.global.b32 [%rd2], 1;
```

**Atomic loads:**
```rust
AtomicU32::load(Ordering::Acquire)
// Emits: ld.acquire.sys.global.b32 %r1, [%rd2];

AtomicU32::load(Ordering::Relaxed)
// Emits: ld.relaxed.sys.global.b32 %r1, [%rd2];
```

**Fence:**
```rust
core::sync::atomic::fence(Ordering::SeqCst)
// Emits: fence.sc.sys;
```

**Atomic RMW (fetch_add):**
```rust
AtomicU32::fetch_add(1, Ordering::SeqCst)
// Emits: atom.global.add.u32 %r1, [%rd2], 1;   ← STILL BROKEN
```

### What changed between LLVM 19 and LLVM 21?

The LLVM NVPTX backend was significantly reworked. Key code in
`NVPTXISelDAGToDAG.cpp` (current `main` branch):

1. **`NVPTXScopes` class** maps LLVM syncscopes to PTX scopes:
   - `""` (default/system) → `NVPTX::Scope::System` → `.sys`
   - `"device"` → `NVPTX::Scope::Device` → `.gpu`
   - `"block"` → `NVPTX::Scope::Block` → `.cta`
   - `"cluster"` → `NVPTX::Scope::Cluster` → `.cluster`

2. **`getOperationOrderings()`** maps LLVM orderings to PTX semantics:
   - `Monotonic` → `.relaxed` (on SM 70+) or `.volatile` (on SM < 70)
   - `Acquire` → `.acquire`
   - `Release` → `.release`
   - `SequentiallyConsistent` → `fence.sc.<scope>` + `.acquire`/`.release`

3. **`hasMemoryOrdering()`** requires SM >= 70 AND PTX >= 60.
   **`hasAtomScope()`** requires SM >= 60.

4. **`getOperationScope()`** reads `N->getSyncScopeID()` and maps through `NVPTXScopes`.
   The default LLVM syncscope `""` maps to `System`, which emits `.sys`.

5. **Atomic loads/stores** go through `tryLoad()`/`tryStore()` which call
   `insertMemoryInstructionFence()` and correctly propagate scope+ordering.

6. **`atomicrmw`** appears to still use a different lowering path that does NOT
   go through `getOperationOrderings()`/`getOperationScope()`. This is why
   `fetch_add(SeqCst)` still emits bare `atom.global.add.u32`.

### Timeline of LLVM NVPTX atomic fixes

| Date | Change |
|------|--------|
| 2024-late | Fence crash fixed; scoped fence intrinsics work |
| 2025-07-15 | Commit 0f1b16dd: syncscope support for cmpxchg (#140812) |
| 2025-11-03 | Commit 5d9d8909: better error messages for invalid syncscope |
| Current | Loads/stores: fully fixed. Fences: fixed. RMW: still broken. |

The project's nightly (built 2025-08-25) includes the July 2025 cmpxchg fix but
may or may not include full atomicrmw ordering. The empirical test shows fetch_add
is still broken, confirming that atomicrmw lowering remains incomplete.

---

## Q2: Is our inline PTX .sys scope correct/necessary?

### For loads/stores: No longer strictly necessary (but still recommended)

`core::sync::atomic` on LLVM 21 now emits the exact same PTX instructions as our
inline PTX wrappers:

| Operation | Our inline PTX | core::sync::atomic (LLVM 21) |
|-----------|---------------|------------------------------|
| Release store | `st.release.sys.global.u32` | `st.release.sys.global.b32` |
| Acquire load | `ld.acquire.sys.global.u32` | `ld.acquire.sys.global.b32` |

The only difference is `.u32` vs `.b32` type suffix, which is functionally identical
for 32-bit integers.

**However, our inline PTX wrappers remain valuable because:**

1. **RMW operations are still broken in core::sync::atomic.** Our `sys_cas_u32`,
   `sys_cas_u64`, `sys_fetch_add_u32/u64`, `sys_exchange_u64` are the ONLY way to
   get system-scope RMW operations. These are critical for the hostcall protocol's
   lock-free stacks.

2. **Spin-loop safety.** Our `sys_spin_load_acquire_u32/u64` variants omit
   `options(readonly)` to prevent LLVM LICM hoisting, and include `nanosleep`.
   `core::sync::atomic::load()` does not provide these guarantees.

3. **Stability.** The LLVM NVPTX backend is actively being reworked. Relying on
   inline PTX for critical synchronization primitives insulates us from potential
   regressions in future LLVM versions.

### For RMW (CAS, fetch_add, exchange): Absolutely necessary

`atom.cas.sys.global.b32/b64` and `atom.add.sys.global.u32/u64` cannot be obtained
through `core::sync::atomic` on any current nightly. Our inline PTX is the only path.

### Recommendation

Keep the `gpu-atomics` crate as the canonical API for GPU-CPU synchronization.
For code that only needs loads/stores (e.g., flag polling), `core::sync::atomic`
is now a viable alternative on LLVM 21+, but using `gpu-atomics` consistently
avoids the risk of accidentally using a broken RMW path.

---

## Q3: Is VectorWare's Ordering::Relaxed a bug or intentional?

### Analysis of VectorWare's code

From the "Async/await on GPU" blog post, VectorWare uses `Ordering::Relaxed` in:

```rust
self.iteration_counter.fetch_add(1, Ordering::Relaxed);
self.shared.last_activity.fetch_add(1, Ordering::Relaxed);
if self.shared.stop_flag.load(Ordering::Relaxed) != 0 { ... }
```

### Assessment: Likely intentional, not a bug

**Reasoning:**

1. **Intra-GPU-only atomics.** The `iteration_counter`, `last_activity`, and
   `stop_flag` appear to be GPU-internal coordination state, not GPU-CPU
   synchronization variables. They are used to track activity within the GPU
   async runtime, not to signal the host CPU.

2. **Relaxed is correct for counters.** `fetch_add(1, Relaxed)` on a counter
   that is only used for monitoring/statistics does not need ordering guarantees.
   The counter value will eventually converge; no other data depends on the
   counter's ordering relative to other stores.

3. **stop_flag with Relaxed is sufficient for termination signals.** When the host
   wants to stop the GPU executor, it sets `stop_flag = 1`. The GPU polls this with
   `Relaxed`. Since Relaxed on LLVM 21 emits `ld.relaxed.sys.global.b32` (system-scope),
   the GPU will eventually see the flag. There is no data that needs to be ordered
   relative to the stop signal -- the GPU just needs to stop.

4. **VectorWare likely uses proper ordering for actual hostcall protocol.** The blog
   post says "We use standard GPU programming techniques such as double-buffering and
   atomic operations and take care to avoid data tearing and ensure memory consistency."
   The Relaxed usage in the example code is for monitoring counters, not for the
   hostcall synchronization path itself.

5. **Performance.** `Relaxed` atomics avoid the overhead of acquire/release barriers.
   On GPU, this matters because fences can stall the entire warp pipeline.

### Critical caveat about scope

On LLVM 19 (when VectorWare may have developed their code), `Relaxed` load/store
would emit `ld.volatile`/`st.volatile`, which per PTX ISA has `.relaxed.sys`
semantics -- system-scope visible. So even on the "broken" LLVM, Relaxed
load/store was accidentally correct for system-scope visibility.

On LLVM 21, `Relaxed` emits `ld.relaxed.sys.global.b32` / `st.relaxed.sys.global.b32`,
which is explicitly system-scope. So VectorWare's code is correct on both LLVM versions.

**The one scenario where VectorWare's Relaxed would be wrong:** If they used
`fetch_add(1, Relaxed)` for GPU-CPU synchronization (e.g., doorbell counter).
On LLVM 21, `atomicrmw` still emits bare `atom.global.add.u32` without scope.
But their counters appear to be GPU-internal, so this is not a problem.

---

## Q4: Are there scenarios where .sys scope is unnecessary?

Yes, but not for our use case:

1. **Intra-GPU atomics** (thread-to-thread within same device): `.gpu` scope suffices.
   Example: warp-level coordination, block-level barriers.

2. **Unified Memory with coherent caching** (SM 90+ Hopper, with `cudaMemAdvise`):
   The hardware may provide automatic coherence, reducing the need for explicit
   `.sys` scope. However, this is SM 90+ only and configuration-dependent.

3. **Same-GPU-only workloads** (no host polling): If the host only reads results
   after `cudaStreamSynchronize()`, `.gpu` scope atomics are sufficient because
   the synchronize call provides the necessary fence.

**For our hostcall protocol:** `.sys` scope is **mandatory** because the GPU and CPU
concurrently access shared memory without an intervening stream synchronization.
The GPU writes a packet and signals the host; the host polls and responds. This
requires system-scope visibility for both sides.

---

## Impact on the project

### What this means for gpu-atomics crate

The `gpu-atomics` crate remains the correct approach, but for different reasons
than originally thought:

**Original rationale (atomics.1):** "core::sync::atomic is completely broken on
nvptx64, all orderings collapse to unscoped instructions."

**Updated rationale (atomics.5):** "core::sync::atomic now works for loads/stores/fences
on LLVM 21, but RMW operations (CAS, fetch_add, exchange) remain broken. The
gpu-atomics crate is necessary specifically for system-scope RMW operations used
in the hostcall protocol's lock-free data structures."

### What could change in the future

If LLVM fixes `atomicrmw` lowering to include scope/ordering (which is being
actively worked on -- see commit 0f1b16dd for cmpxchg progress), then
`core::sync::atomic` could become a complete replacement for our inline PTX.
However:
- Our inline PTX would still work correctly (it emits the same instructions)
- The spin-loop safety variants (`sys_spin_load_acquire`) have no equivalent in
  `core::sync::atomic`
- Switching to `core::sync::atomic` would lose explicit control over scope

### Recommendation

1. **Keep gpu-atomics as-is** for the hostcall protocol and all GPU-CPU synchronization
2. **Document the LLVM 21 improvement** so future developers know that
   `core::sync::atomic` loads/stores are now viable for simpler use cases
3. **Do NOT switch existing code** from gpu-atomics to core::sync::atomic --
   the inline PTX is more explicit, more portable across LLVM versions, and
   provides spin-loop safety features
4. **Monitor LLVM atomicrmw progress** -- when RMW is fixed, evaluate whether
   to deprecate some gpu-atomics functions

---

## Summary table: LLVM 19 vs LLVM 21 vs our inline PTX

| Operation | LLVM 19 (atomics.1 era) | LLVM 21 (current nightly) | Our inline PTX |
|-----------|------------------------|--------------------------|----------------|
| `store(Release)` | Broken (no scope/sem) | `st.release.sys.global.b32` | `st.release.sys.global.u32` |
| `load(Acquire)` | Broken (no scope/sem) | `ld.acquire.sys.global.b32` | `ld.acquire.sys.global.u32` |
| `store(Relaxed)` | `st.volatile` (.relaxed.sys by PTX spec) | `st.relaxed.sys.global.b32` | N/A |
| `load(Relaxed)` | `ld.volatile` (.relaxed.sys by PTX spec) | `ld.relaxed.sys.global.b32` | N/A |
| `fence(SeqCst)` | Crash or silently dropped | `fence.sc.sys` | `membar.sys` |
| `fetch_add(SeqCst)` | `atom.global.add.u32` (broken) | `atom.global.add.u32` (STILL broken) | `atom.add.sys.global.u32` |
| `compare_exchange` | `atom.global.cas.b32` (broken) | Unknown (may be fixed per #140812) | `atom.cas.sys.global.b32` |
