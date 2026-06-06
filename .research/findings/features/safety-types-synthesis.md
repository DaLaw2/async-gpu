# safety-types — Theme Synthesis

## Core Insight
cuda-oxide's DisjointSlice + ThreadIndex achieves compile-time race-freedom
for SIMT lane-level parallelism. async-gpu must adapt this to warp-level
parallelism: `WarpIndex<'scope>` (not ThreadIndex) + `DisjointSlice<'scope, T>`
that partitions into per-warp sub-slices (not per-lane elements).

## Key Design Decisions
1. **WarpIndex<'scope>**: !Send !Sync !Copy, constructed only by spawn_all.
   Replaces the `(warp_id, n_warps)` tuple with a type-safe witness.
2. **DisjointSlice<'scope, T>**: scope-allocated, returns &mut [T] partition
   per warp. Bounds-checked. Lives in gpu-runtime alongside scope.rs.
3. **WarpHandle<'scope>**: witness for warp convergence, lifts shuffle/ballot
   from unsafe to safe. Constructed by trusted cooperative entry points.
4. **SharedSlice<'scope, T>**: separate type for shared memory (Tier 2),
   requires sync_threads between write/read phases.

## Integration Points
- BlockScope::spawn_all_safe — new safe variant providing WarpIndex
- par_iter terminals — already safe; add collect_into_disjoint for explicitness
- cooperative() — add cooperative_safe(Fn(WarpIndex)) alongside unsafe version

## Risk: Ergonomics vs Safety tradeoff
Both old (unsafe, flexible) and new (safe, typed) APIs must coexist.
The par_iter API is unaffected — safety is internal.
