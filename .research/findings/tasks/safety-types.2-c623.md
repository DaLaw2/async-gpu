# safety-types.2 — DisjointSlice<T> + WarpIndex<'scope> + WarpHandle<'scope>

**Task**: Experiment — implement type-level safety primitives for race-free GPU programming
**Status**: Complete
**Cycle**: 623

## Summary

Implemented three type-level safety primitives in `gpu-runtime` that enable
compile-time race prevention for warp-cooperative GPU programming, inspired by
cuda-oxide's safety model and adapted for async-gpu's warp-as-logical-thread
execution model.

## Files Changed

- **NEW** `crates/core/gpu-runtime/src/safety.rs` — WarpIndex, DisjointSlice, WarpHandle
- **EDIT** `crates/core/gpu-runtime/src/scope.rs` — added `spawn_all_indexed`, `disjoint_slice`
- **EDIT** `crates/core/gpu-runtime/src/lib.rs` — added `pub mod safety;` with doc comment
- **EDIT** `crates/core/gpu-runtime/src/prelude.rs` — re-exports DisjointSlice, WarpHandle, WarpIndex

## API Surface

### WarpIndex<'scope> (safety.rs)

Opaque witness proving this code runs on a specific warp within a scope.

```rust
pub struct WarpIndex<'scope> { /* private, !Send, !Sync, !Copy, !Clone */ }

impl<'scope> WarpIndex<'scope> {
    pub fn warp_id(&self) -> u32;
    pub fn n_warps(&self) -> u32;
    pub fn global_index(&self, local_i: usize) -> usize;
}
```

- `pub(crate) fn new(warp_id, n_warps)` — only `scope.rs` can construct
- `PhantomData<*const ()>` for `!Send + !Sync`
- `PhantomData<&'scope mut &'scope ()>` for invariant lifetime
- `global_index()` converts partition-local index to global index (round-robin formula)

### DisjointSlice<'scope, T: Copy> (safety.rs)

A slice where each warp gets exclusive access to its own contiguous partition.

```rust
pub struct DisjointSlice<'scope, T: Copy> { /* private, !Send, !Sync */ }

impl<'scope, T: Copy> DisjointSlice<'scope, T> {
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn get_mut(&self, idx: &WarpIndex<'scope>) -> &mut [T];
    pub fn get(&self, global_index: usize) -> Option<&T>;
    pub unsafe fn raw_parts(&self) -> (*mut T, usize);
}
```

- `pub(crate) unsafe fn new(ptr, len)` — only `scope.rs` can construct
- `get_mut` takes `&self` (interior mutability pattern, `#[allow(clippy::mut_from_ref)]`)
- **Contiguous partitioning** (not round-robin) — enables returning `&mut [T]` without gather/scatter

### WarpHandle<'scope> (safety.rs)

Witness that all 32 lanes are active and converged. Lifts warp ops from unsafe to safe.

```rust
pub struct WarpHandle<'scope> { /* private, !Send, !Sync, !Copy, !Clone */ }

impl<'scope> WarpHandle<'scope> {
    pub fn active_mask(&self) -> u32;
    pub fn reduce_sum_f32(&self, val: f32) -> f32;
    pub fn reduce_sum_u32(&self, val: u32) -> u32;
    pub fn reduce_max_f32(&self, val: f32) -> f32;
    pub fn reduce_min_f32(&self, val: f32) -> f32;
    pub fn shfl_bfly_u32(&self, val: u32, offset: u32) -> u32;
    pub fn shfl_down_u32(&self, val: u32, delta: u32) -> u32;
    pub fn shfl_up_u32(&self, val: u32, delta: u32) -> u32;
    pub fn ballot(&self, predicate: bool) -> u32;
    pub fn all(&self, predicate: bool) -> bool;
    pub fn any(&self, predicate: bool) -> bool;
}
```

### BlockScope extensions (scope.rs)

```rust
impl<'scope> BlockScope<'scope> {
    /// Cooperative spawn with type-safe WarpIndex + WarpHandle
    pub fn spawn_all_indexed<F>(&mut self, f: F)
    where F: Fn(WarpIndex<'scope>, WarpHandle<'scope>) + Send + Sync + 'scope;

    /// Create a DisjointSlice from a scope-allocated buffer
    pub fn disjoint_slice<T: Copy>(&self, buf: &'scope mut [T]) -> DisjointSlice<'scope, T>;
}
```

## Design Decisions

### 1. Contiguous partitioning instead of round-robin

The investigation design mentioned round-robin striding, but returning `&mut [T]`
requires contiguous elements. Round-robin produces scattered indices that cannot
form a contiguous slice without a temporary buffer (violating zero-cost).

**Decision**: Use contiguous partitioning (divide-and-remainder), which gives each
warp a contiguous sub-slice. The `global_index()` helper on WarpIndex preserves
the round-robin formula for cases where the user needs it.

### 2. `get_mut` takes `&self` (interior mutability)

DisjointSlice uses the interior mutability pattern: `get_mut(&self, &WarpIndex)`.
This is necessary because the DisjointSlice is captured by shared reference in
the closure passed to `spawn_all_indexed` (closures are `Fn`, not `FnMut`).
Safety is enforced by the WarpIndex witness — each warp gets a different partition.
The `#[allow(clippy::mut_from_ref)]` attribute documents this intentional pattern.

### 3. `spawn_all_indexed` delegates to `spawn_all`

Rather than duplicating the cooperative dispatch logic, `spawn_all_indexed`
wraps the user's `Fn(WarpIndex, WarpHandle)` closure into a `Fn(u32, u32)`
closure and delegates to the existing `spawn_all`. This ensures the new API
has identical behavior to the established code path.

### 4. WarpHandle wraps all existing warp operations

Every `pub unsafe fn` in `warp.rs` gets a safe wrapper in WarpHandle. The mask
is always `0xFFFF_FFFF` (full warp), matching the `spawn_all` contract where all
32 lanes execute the trampoline.

### 5. Naming: `spawn_all_indexed` not `spawn_all_safe`

The original design suggested `spawn_all_safe`, but this implies the existing
`spawn_all` is "unsafe" (it is not — it is safe Rust, just less type-checked).
`spawn_all_indexed` describes what it does: provides indexed/typed witnesses.

## Build Verification

- `cargo +stable clippy -- -D warnings`: PASS (zero warnings)
- `cargo +stable fmt -- --check`: PASS
- `cargo +stable doc --no-deps`: PASS (warnings are pre-existing, unrelated)
- `cargo +nightly-2026-06-03 check --target nvptx64-nvidia-cuda -Zbuild-std=core,alloc`: PASS

## Integration Points for Future Work

1. **par_iter integration**: Terminal methods (`for_each`, `collect_into`, `fold`)
   could accept `DisjointSlice` as output, making the pipeline fully type-safe.
2. **GridScope**: `disjoint_slice` and `spawn_all_indexed` could be added to
   `GridScope` for cross-block type-safe partitioning.
3. **cooperative()**: The `thread::cooperative()` function could get a safe
   variant that provides WarpIndex, replacing the current `unsafe` contract.
