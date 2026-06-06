# om-borrow-safety.2 — Valid SharedRef/GlobalRef patterns on GPU

**Kind**: experiment
**Status**: DONE
**Date**: 2026-06-06

## Summary

Created and verified two GPU kernel tests (`test_gpu_shared_ref_valid_patterns`, `test_gpu_global_ref_valid_patterns`) that exercise the complete SharedRef/GlobalRef API surface. Both compile to nvptx64 PTX containing address-space-specific instructions (`ld.shared`/`st.shared` for SharedRef, `ld.global.u32`/`st.global.u32`/`ld.global.u64`/`st.global.u64` for GlobalRef). GPU hardware execution confirmed both tests pass.

## Findings

### SharedRef valid patterns (5 sub-tests)

| Test | Pattern | Result |
|------|---------|--------|
| A | `alloc_shared` + sequential write/read 16 u32 elements | PASS |
| B | `sub_ref` for tiling: two non-overlapping 32-element tiles, write-through verification | PASS |
| C | Pass `SharedRef` to helper functions by reference within scope | PASS |
| D | Multiple `SharedRef` allocations in one scope, interleaved reads, helper accumulation | PASS |
| E | f32 type via `SharedRef<f32>`, write + read + floating-point sum | PASS |

### GlobalRef valid patterns (4 sub-tests)

| Test | Pattern | Result |
|------|---------|--------|
| A | `alloc_global` + sequential write/read 16 u32 elements | PASS |
| B | `sub_ref` for partitioning, write-through verification | PASS |
| C | Cross-warp use via `as_global_mut_ptr()` + `SendPtrMut` + `spawn_all` | PASS |
| D | u64 type via `GlobalRef<u64>`, pattern values including MAX and 0 | PASS |

### PTX instruction verification

- SharedRef test region: 54 `ld.shared`/`st.shared` instructions (e.g., `st.shared.u32 [%rd83], %r19;`)
- GlobalRef test region: 216 `ld.global`/`st.global` instructions, including `ld.global.u64`/`st.global.u64` for test D

### Design constraint discovered: GlobalRef lifetime vs spawn_all

GlobalRef is `Send + Sync` by design, but the `for<'scope>` HRTB on `grid_scope()` prevents moving a `GlobalRef<'scope, T>` into a nested `block_scope`'s `spawn_all` closure. The compiler emits E0521 because the grid_scope's lifetime cannot satisfy the `'static` bound required by the spawn_all closure.

**Correct cross-warp pattern**: Extract raw pointer via `gref.as_global_mut_ptr()`, wrap in `SendPtrMut`, and pass to `spawn_all`. This mirrors the existing `sc_grid_reduce` demo. The coordinator then reads back via the original `GlobalRef`, proving the data flows correctly.

This is not a bug — it's the HRTB doing its job. GlobalRef's Send+Sync enables cross-block use (multiple blocks each receive their own GlobalRef to the same data), not cross-scope nesting.

### SharedRef !Send confirmed

SharedRef is correctly `!Send`, preventing it from being passed to `spawn_all`. The SharedRef tests use warp-0-only patterns (alloc, fill, read, verify) which is the intended usage. For multi-warp shared memory access, use `scope.alloc()` (returns `&'scope mut [T]`) or `scope.alloc_disjoint()` (returns `DisjointSlice`).

## Open Questions

1. **Static pool for GridScope**: The zero-param test uses a static `[AtomicU8; 2048]` as the GridScope pool. This works but is not the recommended pattern for production kernels (which should receive the pool as a kernel parameter from the host).

2. **Dynamic shared memory size**: The `#[gpu_test]` macro launches with `shared_mem_bytes = 0` in the CUDA launch config. The tests pass because the `init_shared_mem_allocator(4096)` tracks offsets logically but the actual shared memory is provided by the `.extern .shared .b8 dynamic_smem[]` declaration. On the test GPU, this appears to work without explicit dynamic shared memory allocation in the launch config.
