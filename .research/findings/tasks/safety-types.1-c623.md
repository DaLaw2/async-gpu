# safety-types.1 — cuda-oxide DisjointSlice + ThreadIndex: Adaptation to async-gpu

**Task**: Investigation — cuda-oxide DisjointSlice + ThreadIndex design, adapt to async-gpu model
**Status**: Complete
**Cycle**: 623

## Summary

cuda-oxide (NVlabs, May 2026) introduces a 3-tier safety model where Tier 1 achieves
race-freedom by construction through two key types: `DisjointSlice<T, IndexSpace>` and
`ThreadIndex<'kernel, IndexSpace>`. async-gpu's warp-cooperative execution model (where
warps are logical threads, not SIMT lanes) requires significant adaptation of this pattern.
The core insight is transferable but the indexing granularity changes from per-lane to
per-warp, and the async/await model introduces temporal concerns that cuda-oxide's
synchronous model does not address.

## cuda-oxide Analysis

### DisjointSlice<T, IndexSpace>

- A slice-like type that provides **exclusive mutable access** to exactly one element
  per hardware thread via a compile-time-checked indexing scheme.
- `get_mut(idx: ThreadIndex<'k, IS>) -> Option<&mut T>` — returns `None` for OOB.
- The `IndexSpace` parameter (e.g., `Index1D`, `Index2D<STRIDE>`) creates a compile-time
  constraint: a `DisjointSlice<T, Index2D<128>>` only accepts `ThreadIndex<'_, Index2D<128>>`.
  Stride mismatches are compile errors.
- No `unsafe` needed for the common case (one thread writes one element).

### ThreadIndex<'kernel, IndexSpace>

- Opaque witness type wrapping a `usize`. **No public constructor** — can only be obtained
  from trusted `thread::index_1d()` / `thread::index_2d::<S>()` functions that read hardware
  built-in variables (`threadIdx`, `blockIdx`, `blockDim`).
- `'kernel` lifetime ties the witness to a stack-local scope injected by the `#[kernel]` macro,
  preventing index laundering (storing in shared memory for later reuse).
- Marked `!Send + !Sync + !Copy + !Clone`:
  - `!Copy + !Clone`: Forces single ownership — no accidental duplication.
  - `!Send + !Sync`: Prevents passing across thread boundaries via shared memory.
- `index_1d()` returns `ThreadIndex<'kernel, Index1D>` using
  `blockIdx.x * blockDim.x + threadIdx.x` — unique per lane across the grid.
- `index_2d::<S>()` returns `Option<ThreadIndex<'kernel, Index2D<S>>>` with const stride S.
- `unsafe index_2d_runtime(s)` for runtime strides, caller must assert all threads use same s.

### 3-Tier Safety Model

| Tier | Scope           | What's covered                                    | unsafe needed? |
|------|-----------------|---------------------------------------------------|----------------|
| 1    | Safe by default | DisjointSlice + ThreadIndex — one thread/one elem | No             |
| 2    | Scoped unsafe   | Shared memory, warp shuffles, atomics, barriers   | Yes            |
| 3    | Raw hardware    | TMA, WGMMA, tcgen05, cluster programming         | Yes            |

### Shared Memory (Tier 2)

- `SharedArray<T, N>` declared as `static mut`, uses `UnsafeCell`, is `!Sync` by design.
- All access requires `unsafe` blocks + explicit `sync_threads()` barriers.
- DisjointSlice is for **global memory only** — shared memory access is inherently Tier 2.

### Warp Operations (Tier 2)

- Shuffles, ballot, vote are all `unsafe` with documented safety contracts.
- cuda-oxide's warp ops are lane-level SIMT — all 32 lanes must participate.
- No type-level integration between warp ops and DisjointSlice.

## async-gpu Current Model

### Execution Architecture (Key Difference)

**cuda-oxide**: Traditional SIMT — each lane is an independent "thread", `threadIdx.x`
is the fundamental unit. All 32 lanes execute in lockstep but are logically independent.

**async-gpu**: Warp-cooperative model — each **warp** (32 lanes) acts as a single logical
"thread". Warp 0 (lane 0) runs the main function. Other warps park until `spawn()` assigns
work. Lane 0 of each warp executes closure logic; other lanes participate in warp-level ops.

This means:
- async-gpu's "thread index" is **warp_id**, not `threadIdx.x`.
- Parallelism is at warp granularity (up to 32 warps per block).
- Within a warp, lane 0 controls execution (no intra-warp divergence for user code).

### cooperative() API

`thread::cooperative(f: &F)` — wakes all worker warps to execute the same closure.
Each warp uses `current_id()` (warp index) and `available_parallelism()` to determine
its data partition. Safety contract: caller must ensure no data races (manual partitioning).
This is currently `unsafe` because the compiler cannot verify partitioning correctness.

### BlockScope / GridScope

- `BlockScope<'scope>` — structured concurrency with HRTB lifetime bound (`for<'scope>`).
  Allocates from shared memory, spawns closures to warps, joins all before scope exit.
  Already uses `PhantomData<&'scope mut &'scope ()>` for invariance.
- `spawn_all(f: Fn(warp_id, n_warps))` — cooperative data-parallel across all warps.
  Closures receive `(warp_id, n_warps)` for manual partitioning.
- `GridScope<'scope>` — same pattern over global memory with system-scope atomics.

### par_iter (GpuParallelIterator)

- Rayon-like lazy iterator chains dispatched via `BlockScope::spawn_all`.
- `GpuSlice<T>` / `GpuSliceMut<T>` carry raw pointer + length (no lifetime).
- Terminal methods (`for_each`, `collect_into`, `fold`) use warp round-robin partitioning.
- `SendPtr<T>` / `SendPtrMut<T>` wrappers make raw pointers `Send + Sync`.
- **No compile-time race prevention** — correctness relies on partitioning logic in
  terminal implementations being correct (and it is, by round-robin striding).

### Existing Safety Gaps

1. `cooperative()` is `unsafe` with no compile-time verification of non-overlapping access.
2. `GpuSliceMut<T>` is `Copy + Send + Sync` — can be aliased freely.
3. `BlockScope::spawn_all` closures receive `(warp_id, n_warps)` as plain `u32` — no type
   witness prevents using wrong values or indexing into the wrong buffer.
4. Raw pointer casts (`SendPtr`) bypass the borrow checker entirely.

## Adaptation Design

### WarpIndex<'scope> — async-gpu's ThreadIndex

Since async-gpu's unit of parallelism is the **warp** (not the lane), the analog of
cuda-oxide's `ThreadIndex` should be `WarpIndex<'scope>`:

```rust
/// Opaque witness proving this code runs on a specific warp within a scope.
/// Cannot be constructed, copied, sent, or stored.
pub struct WarpIndex<'scope> {
    warp_id: u32,
    n_warps: u32,
    _not_send: PhantomData<*const ()>,       // !Send
    _scope: PhantomData<&'scope mut &'scope ()>, // invariant lifetime
}

// Explicitly NOT implementing Send, Sync, Copy, Clone
```

- Constructed **only** by `BlockScope::spawn_all` — the closure receives
  `WarpIndex<'scope>` instead of `(u32, u32)`.
- `'scope` lifetime ties it to the enclosing BlockScope/GridScope.
- `!Send + !Sync + !Copy + !Clone` — cannot be stored, shared, or duplicated.
- Provides `.warp_id() -> u32` and `.n_warps() -> u32` for partitioning.

### DisjointSlice<'scope, T> — Warp-Disjoint Access

```rust
/// A slice where each warp gets exclusive access to its own partition.
/// Access requires a WarpIndex witness — compile-time proof of ownership.
pub struct DisjointSlice<'scope, T: Copy> {
    ptr: *mut T,
    len: usize,
    _scope: PhantomData<&'scope mut [T]>,
}
```

Key API:
- `get_mut(idx: &WarpIndex<'scope>) -> &mut [T]` — returns this warp's partition
  (round-robin or contiguous, depending on `PartitionStrategy`).
- `get(idx: &WarpIndex<'scope>, i: usize) -> Option<&T>` — bounds-checked read.
- Constructed only from scope-allocated memory: `scope.disjoint_slice::<T>(buf, len)`.
- `!Send + !Sync` — cannot escape the scope.

### Integration with BlockScope

```rust
block_scope(|scope| {
    let input = scope.alloc::<f32>(1024);
    let output = scope.disjoint_slice::<f32>(1024); // NEW: returns DisjointSlice

    // spawn_all now provides WarpIndex instead of (u32, u32)
    scope.spawn_all_safe(|widx: WarpIndex<'scope>| {
        let out_partition = output.get_mut(&widx);
        for (i, slot) in out_partition.iter_mut().enumerate() {
            let global_i = widx.global_index(i);
            *slot = input[global_i] * 2.0;
        }
    });
});
```

### Integration with par_iter

The iterator terminals already partition correctly, but we can make this type-safe:

```rust
data.par_iter()
    .map(|x| x * 2.0)
    .collect_into_disjoint(output);  // DisjointSlice guarantees no overlap
```

### Async/Await Interaction

**Key challenge**: In async-gpu, warp-cooperative futures use `shfl.sync` to broadcast
poll results across lanes. The `WarpIndex` must be valid across `.await` points.

**Solution**: `WarpIndex<'scope>` is scoped to `BlockScope`'s lifetime, which already
enforces that all spawned work completes before scope exit. The async executor runs
within a single scope entry, so the index remains valid. The `!Send` constraint prevents
the index from being moved across `.await` boundaries in a multi-warp executor.

### Warp-Level Operations (Tier 2)

Warp shuffles, ballot, and reductions are inherently cooperative (all lanes participate).
They should remain `unsafe` but can be enhanced with a `WarpHandle` witness:

```rust
/// Witness that all lanes are active and converged.
/// Constructed only by trusted entry points (spawn_all, warp_cooperative).
pub struct WarpHandle<'scope> {
    mask: u32,
    _scope: PhantomData<&'scope ()>,
}

impl<'scope> WarpHandle<'scope> {
    pub fn reduce_sum_f32(&self, val: f32) -> f32 {
        // SAFE: WarpHandle guarantees all lanes are active
        unsafe { crate::warp::reduce_sum_f32(val) }
    }
}
```

This lifts shuffle/ballot/reduce from `unsafe` to safe when called through WarpHandle.

## Design Questions (Answered)

### Where should DisjointSlice live?

**In gpu-runtime** — alongside scope.rs, par_iter.rs, and thread.rs. It's a core safety
primitive that integrates deeply with BlockScope/GridScope. No new crate needed.

### How to integrate with cooperative()?

Replace `unsafe cooperative(f: &F)` with a safe `cooperative_safe(f: impl Fn(WarpIndex))`.
The unsafe version remains for backward compatibility. `spawn_all_safe` on BlockScope
is the primary entry point.

### Can the 3-tier model map to Rust's type system?

Yes, with async-gpu's granularity:

| Tier | async-gpu mechanism                              | unsafe? |
|------|--------------------------------------------------|---------|
| 1    | DisjointSlice + WarpIndex — per-warp exclusivity | No      |
| 1    | par_iter terminals — partitioning built-in        | No      |
| 2    | WarpHandle — safe warp ops via witness            | No*     |
| 2    | Shared memory via BlockScope::alloc               | No*     |
| 2    | Atomics, sync_threads via WarpHandle              | Yes     |
| 3    | Raw PTX asm, TMA, tensor cores                   | Yes     |

*Safe through scope/witness API, raw access remains unsafe.

### Shared memory: DisjointSlice or separate pattern?

Shared memory should use a **separate SharedSlice<'scope, T>** type:
- Shared memory is inherently block-scoped (all warps share it).
- DisjointSlice partitions are per-warp — shared memory has overlapping access.
- SharedSlice should require `sync_threads()` between write and read phases.
- Reads are always safe; writes require either DisjointSlice proof or `unsafe`.

## Risks

1. **Ergonomics overhead** — Adding WarpIndex to all cooperative closures changes the
   signature of `spawn_all`. Must provide both safe (new) and unsafe (old) variants to
   avoid a massive breaking change. The parallel iterator API already hides partitioning,
   so users of par_iter see no change.

2. **Warp-granularity limitation** — cuda-oxide's DisjointSlice works at lane granularity
   (one element per lane). async-gpu's WarpIndex works at warp granularity (one partition
   per warp). This is coarser — a single warp processes multiple elements in a loop.
   The DisjointSlice must partition into **sub-slices**, not single elements.

3. **Lifetime complexity** — `WarpIndex<'scope>` tied to BlockScope's `for<'scope>` HRTB
   already works correctly, but nested scopes and GridScope add complexity. Need to verify
   that lifetime invariance prevents escaping across scope boundaries (the existing
   `PhantomData<&'scope mut &'scope ()>` pattern handles this).

4. **No intra-warp safety** — Within a single warp, lane 0 controls execution. There's no
   intra-warp race in async-gpu's model (unlike cuda-oxide where all 32 lanes execute
   independently). This is actually a simplification — fewer race surfaces to protect.

5. **Performance** — The WarpIndex and DisjointSlice types should be zero-cost (ZSTs or
   small values, all methods inlined). Bounds checking in `get_mut` should optimize to
   a single comparison. No runtime overhead expected.

## Sources

- [cuda-oxide Safety Model](https://nvlabs.github.io/cuda-oxide/gpu-safety/the-safety-model.html)
- [cuda-oxide Memory and Data Movement](https://nvlabs.github.io/cuda-oxide/gpu-programming/memory-and-data-movement.html)
- [cuda-oxide GitHub Repository](https://github.com/NVlabs/cuda-oxide)
- [cuda-oxide Shared Memory](https://nvlabs.github.io/cuda-oxide/advanced/shared-memory-and-synchronization.html)
- [cuda-oxide Warp-Level Programming](https://nvlabs.github.io/cuda-oxide/advanced/warp-level-programming.html)
