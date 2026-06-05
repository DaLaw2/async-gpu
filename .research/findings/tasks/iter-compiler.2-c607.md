# iter-compiler.2 — filter() + fold() with warp ballot compaction + shuffle reduction

## Status: done

## Summary

Implemented `filter()` for `GpuParallelIterator` with full terminal method support: `for_each`, `fold`, `collect_into`, `sum`, `product`, `min`, `max`, `count`. Also implemented fused `GpuFilterMap` for `filter().map()` chains.

## Design Decision: Non-Indexed Adapter

`GpuFilter` does NOT implement `GpuParallelIterator` because filter is fundamentally non-indexed: the output length is data-dependent, so `len()` and `get_unchecked(i)` cannot be satisfied. Instead, `GpuFilter` provides its own terminal methods directly. This is the correct design — Rayon makes the same distinction between `IndexedParallelIterator` and `ParallelIterator`.

The `filter()` method is added to the `GpuParallelIterator` trait as an adapter method that returns `GpuFilter<Self, F>`.

## Implementation Details

### GpuFilter<I, F>

- `inner: I` — the source indexed iterator
- `predicate: F` — the filter predicate `Fn(&Item) -> bool + Copy + Send + Sync`
- `#[derive(Clone, Copy)]` — GPU-safe, no heap

### Terminal Methods

**for_each / fold**: Each warp iterates its partition (round-robin by warp ID) and simply skips elements that fail the predicate. No compaction needed — this is the simple case. Cross-warp reduction for fold uses the existing `WARP_RESULT` slots mechanism.

**collect_into**: Uses Approach B (atomic counter) for cross-warp output coordination:
1. `WARP_RESULT[0]` is repurposed as an atomic output counter (reset to 0 before launch)
2. Each warp evaluates the predicate for its assigned elements
3. For each matching element, `fetch_add(1, AcqRel)` atomically reserves one output slot
4. The matching element is written to `output[reserved_index]` via `write_volatile`
5. After all warps complete, `WARP_RESULT[0].load()` gives the total written count
6. Returns `usize` (number of elements written)
7. Output buffer must be at least as large as input (worst case: all elements pass)

**count**: Each warp counts matches locally, stores to `WARP_RESULT[wid]`. Warp 0 sums all partial counts.

**sum/product/min/max**: Built on fold with appropriate identity elements.

### GpuFilterMap<I, P, M>

Created by `GpuFilter::map()`. Fuses the filter predicate with a map function — the map is applied only to elements that pass the predicate, in a single pass with no intermediate buffer. Provides the same terminal methods as `GpuFilter`, but the output type is `B` (the map's output type) rather than `I::Item`.

### Why Not Warp Ballot + Popcount for MVP?

The task brief suggested using `warp::ballot()` + `popcount` for intra-warp compaction. We have `ballot()` in `warp.rs` but each warp in this architecture runs only lane 0 for closure logic (see `spawn_all_trampoline` — `if lid == 0 { ... }`). Warp ballot requires all lanes to participate, which is incompatible with the current single-lane execution model.

The atomic counter approach (Approach B) is simpler and correct for the current architecture. If/when the runtime evolves to use all 32 lanes per warp for data-parallel work, ballot-based compaction can be added as an optimization without changing the API.

### Prelude Export

`GpuFilter` and `GpuFilterMap` are exported from `gpu_runtime::prelude`.

## Verification

`bash scripts/ci-lint.sh` — all checks pass (host checks, PTX compilation, examples).

## Files Changed

- `crates/core/gpu-runtime/src/par_iter.rs` — added `GpuFilter`, `GpuFilterMap`, `filter()` method on trait
- `crates/core/gpu-runtime/src/prelude.rs` — export new types

## API Example

```rust
// Filter + fold (sum only elements > 5.0)
let result = data.par_iter()
    .map(|x| x * 2.0)
    .filter(|x| *x > 5.0)
    .sum();

// Filter + collect (compact matching elements)
let written = data.par_iter()
    .filter(|x| *x > 0.0)
    .collect_into(output_buf);
// `written` is the number of elements stored

// Filter + map + fold (fused)
let result = data.par_iter()
    .filter(|x| *x > 0.0)
    .map(|x| x * x)
    .sum();

// Count matches
let n_positive = data.par_iter()
    .filter(|x| *x > 0.0)
    .count();
```
