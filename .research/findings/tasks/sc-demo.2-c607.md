# sc-demo.2 — Multi-Block Reduce with GridScope

## Status: done
## Summary:

Added Demo 5 (`sc_grid_reduce`) to `sc_demo.rs` — a multi-block parallel
sum reduction using `GridScope` for grid-level structured concurrency.

Since SM75 lacks cooperative launch (no guarantee multiple blocks run
simultaneously), the demo uses a single block where warps act as "virtual
blocks":

1. Warp 0 (coordinator) enters `grid_scope` with a host-provided global
   memory pool.
2. GridScope allocates global memory for input data (128 u32 values) and
   partial sums (one per worker warp).
3. Coordinator fills input: `data[i] = i + 1`.
4. Worker warps (1..N-1) each compute a partial sum of their data
   segment via `block_scope` + `spawn_all`, write to the
   GridScope-allocated partial_sums array, and atomically increment
   the GridScope completion counter via `sys_fetch_add_u32`.
5. Coordinator calls `gscope.wait_for_completions(n_workers)` and then
   reduces the partial sums to produce the final result.

Expected output: `sum(1..=128) = 8256`, completions = 3 (for 4-warp
launch), success flag = 1.

The demo exercises:
- `grid_scope()` entry with pool-based global memory allocation
- `GridScope::alloc::<T>()` for global memory bump allocation
- `GridScope::set_expected_completions()` and `wait_for_completions()`
- `GridScope::completion_counter_ptr()` with `sys_fetch_add_u32`
- Composition of GridScope (global memory) with BlockScope (shared memory)

## Implementation

Kernel `sc_grid_reduce(pool, pool_size, result)`:
- Launch config: 1 block × 128 threads (4 warps), 2048 bytes shared memory
- Pool: host-allocated device memory (>= 2048 bytes)
- Output: `result[0]` = final sum, `result[1]` = completion count,
  `result[2]` = success flag

## Files Changed:
- `crates/kernel/gpu-kernel-std/src/sc_demo.rs` — added Demo 5 (sc_grid_reduce)
