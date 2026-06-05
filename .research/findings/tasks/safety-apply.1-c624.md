# safety-apply.1 — Enhance BlockScope/GridScope with DisjointSlice + ThreadIndex

**Task**: safety-apply.1  
**Cycle**: 624  
**Status**: complete  

## What was done

### 1. `alloc_disjoint<T>()` on BlockScope (`scope.rs`)

Added convenience method that combines `alloc()` + `disjoint_slice()`:

```rust
pub fn alloc_disjoint<T: Copy>(&self, count: usize) -> DisjointSlice<'scope, T>
```

Users no longer need the two-step pattern. Single call allocates shared memory
AND returns a ready-to-use DisjointSlice.

### 2. `cooperative_indexed()` in `thread.rs`

Added a safe variant of the unsafe `cooperative()`:

```rust
pub fn cooperative_indexed<F>(f: &F)
where
    F: for<'coop> Fn(WarpIndex<'coop>, WarpHandle<'coop>) + Sync,
```

Key design decisions:
- Uses `for<'coop>` HRTB to create a fresh lifetime, preventing WarpIndex/WarpHandle
  from escaping the cooperative call — same pattern as `block_scope`.
- Internally calls `unsafe { cooperative(...) }` but constructs WarpIndex + WarpHandle
  for each warp, making the call site fully safe.
- The `Sync` bound (not `Send + Sync`) matches the fact that `cooperative()` shares
  the closure reference across warps rather than moving it.

### 3. DisjointSlice trait impl fixes (`safety.rs`)

Fixed three issues that made the existing type-safe API unusable in practice:

- **`unsafe impl Send for DisjointSlice`** — Needed so `move` closures in
  `spawn_all_indexed` can capture DisjointSlice by value. Sound because
  mutable access is gated by WarpIndex, not by which warp holds the slice.

- **`unsafe impl Sync for DisjointSlice`** — Needed so `&DisjointSlice` references
  in `cooperative_indexed` closures satisfy the `Sync` bound. Sound because
  `get_mut()` returns disjoint partitions per warp via the WarpIndex witness.

- **`derive(Copy, Clone)` on DisjointSlice** — Enables capturing by value in
  `move` closures while still using the slice afterward for verification.
  Sound because DisjointSlice is a thin wrapper around `(*mut T, usize)` —
  no owned resources. Safety is enforced by WarpIndex uniqueness, not
  DisjointSlice uniqueness.

- **`get_mut()` now accepts `WarpIndex<'_>`** (any lifetime) instead of requiring
  `WarpIndex<'scope>`. This allows a `DisjointSlice<'scope>` from a BlockScope
  to work with a `WarpIndex<'coop>` from `cooperative_indexed()`. The safety
  doesn't depend on lifetime matching — it depends on WarpIndex being a valid
  warp-identity witness.

### 4. Test kernel (`gpu-kernel-test/src/lib.rs`)

Added `test_gpu_type_safe_cooperative` with three sub-tests:
1. `alloc_disjoint` + `spawn_all_indexed` — fill + verify per-warp writes
2. `alloc_disjoint` + `cooperative_indexed` — same pattern, different API
3. `DisjointSlice::get()` — immutable reads + bounds checking

Registered in the `#[gpu_test]` harness (`gpu_tests.rs`).

### 5. GridScope analysis

GridScope does NOT need `alloc_disjoint` or `cooperative_indexed` because:
- GridScope coordinates across blocks, not warps within a block
- WarpIndex/DisjointSlice are warp-level primitives (intra-block)
- GridScope's work distribution uses BlockWorkSlot + completion counters,
  which operate at a different granularity
- A block inside a GridScope can still use BlockScope + DisjointSlice
  for its intra-block work (the two compose naturally)

## Build verification

- `cargo check` on gpu-runtime (nvptx64): PASS (no errors)
- `cargo clippy` on gpu-runtime (nvptx64): PASS (no warnings)
- `cargo check` on gpu-kernel-test (nvptx64): PASS (only pre-existing warnings)
- `cargo check` on gpu-test-harness (host): PASS

## Files changed

- `crates/core/gpu-runtime/src/safety.rs` — Send/Sync/Copy for DisjointSlice, relaxed get_mut lifetime
- `crates/core/gpu-runtime/src/scope.rs` — alloc_disjoint method
- `crates/core/gpu-runtime/src/thread.rs` — cooperative_indexed function
- `crates/kernel/gpu-kernel-test/src/lib.rs` — test_gpu_type_safe_cooperative kernel
- `crates/test/gpu-test-harness/tests/gpu_tests.rs` — #[gpu_test] entry
