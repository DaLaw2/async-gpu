# iter-compiler.1 — GpuParallelIterator trait with map/for_each/collect_into

## Status: done

## Summary

Implemented the `GpuParallelIterator` trait and all supporting types in `crates/core/gpu-runtime/src/par_iter.rs`. The module provides a Rayon-like lazy parallel iterator API for GPU kernels, with compile-time fusion via Rust monomorphization.

## Files Changed

- **`crates/core/gpu-runtime/src/par_iter.rs`** (new) — Full iterator module
- **`crates/core/gpu-runtime/src/lib.rs`** — Added `pub mod par_iter;` declaration
- **`crates/core/gpu-runtime/src/prelude.rs`** — Added re-exports for key types

## What Was Built

### Core Types
- `GpuSlice<T>` — immutable fat pointer (ptr + len) to GPU global memory, `Copy`
- `GpuSliceMut<T>` — mutable fat pointer for output buffers, `Copy`
- `SendPtr<T>` / `SendPtrMut<T>` — raw pointer wrappers with `Send + Sync` for closure capture

### Identity Traits
- `GpuZero` / `GpuOne` — additive/multiplicative identity (implemented for f32, f64, u32, u64, i32, i64, usize)
- `GpuMaxValue` / `GpuMinValue` — extremal values (implemented for same types)

### GpuParallelIterator Trait
- Supertraits: `Sized + Copy + Send + Sync + 'static`
- `type Item: Copy + Send + Sync + 'static`
- Required: `fn len(&self)`, `unsafe fn get_unchecked(&self, i: usize) -> Self::Item`
- Adapters (lazy): `map`, `enumerate`, `zip`
- Terminals (eager, via `spawn_all`): `for_each`, `fold`, `collect_into`
- Convenience terminals: `sum`, `product`, `min`, `max`, `count`

### Adapter Types
- `GpuParIter<T>` — base iterator over `GpuSlice`, reads via `read_volatile`
- `GpuMap<I, F>` — lazy map, fuses via monomorphization
- `GpuEnumerate<I>` — pairs elements with indices
- `GpuZip<A, B>` — pairs elements from two iterators

### Terminal Implementations
- `default_for_each` — `block_scope` + `spawn_all`, warp-striped round-robin
- `default_fold` — per-warp sequential fold, cross-warp combine via `WARP_RESULT` slots (Item must fit in 8 bytes)
- `default_collect_into` — warp-striped writes via `write_volatile`

### Entry Point
- `pub fn par_iter<T>(slice: &GpuSlice<T>) -> GpuParIter<T>`

## Key Design Decisions

1. **`'static` bound on trait and Item**: Required because `block_scope` uses HRTB (`for<'scope>`), so closures passed to `spawn_all` must satisfy any lifetime — effectively `'static`. Since GPU iterator types contain raw pointers (no references), this is always satisfied in practice.

2. **`Send + Sync` on Item**: GPU global memory is accessible from all warps. The `Send + Sync` bounds on `Item` allow the fold identity value to be captured in `Fn` closures (which borrow via `&self`, requiring `&T: Send` i.e. `T: Sync`).

3. **`move` closures for `block_scope` + `spawn_all`**: Both the outer closure (to `block_scope`) and inner closure (to `spawn_all`) use `move` to copy all `Copy` values into the closure, avoiding lifetime issues with the HRTB pattern.

4. **Revised fold signature (MVP)**: Uses `fn fold<F>(self, identity: Self::Item, fold_op: F) -> Self::Item` where `fold_op: Fn(Item, Item) -> Item`. Simpler than the dual-type `Fn(B, Item) -> B` approach — the accumulator and element are the same type. This covers all practical cases (sum, product, min, max).

## Verification

`bash scripts/ci-lint.sh` — all checks pass (fmt, clippy, doc, check, PTX kernel builds).
