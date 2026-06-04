# coop-api.2: cooperative_reduce + cooperative_map_with_params + unified_io_compute rewrite

**Status**: done
**Kind**: experiment
**Theme**: coop-api — Cooperative API ergonomics

## Summary

Implemented `cooperative_reduce()` for multi-warp reductions and `cooperative_map_with_params()`
for parameterized data-parallel maps. Rewrote `unified_io_compute` (the North Star demo) to use
`cooperative_map`, eliminating all 3 global atomics (`UIC_IN_PTR`, `UIC_OUT_PTR`, `UIC_LEN`).
All tests verified on GPU hardware (GTX 1660, sm_75).

## cooperative_reduce

Design: each warp runs the user function on its partition and returns a `u64` partial result.
Worker warps store their partial in `WARP_RESULT[wid]`. Warp 0 computes its own partial,
then sequentially collects from all workers and sums.

```rust
let total = thread::cooperative_reduce(
    data.as_ptr() as *const u8,
    data.len(),
    |args| -> u64 {
        let src = args.src as *const u64;
        let mut sum = 0u64;
        let mut i = args.warp_id as usize;
        while i < args.len { sum += read(src, i); i += args.n_warps as usize; }
        sum
    },
);
```

Key decisions:
- Uses `WARP_RESULT` slots (already in thread.rs) — no new globals needed for per-warp storage
- Warp 0 does sequential collection after barrier — simpler than tree reduction, sufficient for
  4-32 warps (not a bottleneck vs atomic add)
- Returns `u64` — users cast to f64/u32/etc. as needed. u64 chosen because it's the widest
  native type and WARP_RESULT is already AtomicU64

Test: sum(0..256) = 32640, 4 warps. PASSED.

## cooperative_map_with_params

Adds `CoopMapExtArgs` with a `params: [u64; 4]` field for extra user-defined parameters.
Uses a separate global arg block (`COOP_MAP_EXT_ARGS: [AtomicU64; 8]`) to avoid clobbering
`cooperative_map`'s `COOP_MAP_ARGS`.

Use cases: scalar multiplier, matrix dimensions (M, K, N), stride values, iteration count.

```rust
thread::cooperative_map_with_params(
    src, dst, len,
    [7, 0, 0, 0],   // params[0] = scalar
    |args| {
        let scale = args.params[0] as u32;
        // ... each element *= scale
    },
);
```

Test: 256 elements × 7, 4 warps. PASSED.

## unified_io_compute rewrite

Before (cooperative + 3 global atomics):
```rust
static UIC_IN_PTR: AtomicU64 = AtomicU64::new(0);
static UIC_OUT_PTR: AtomicU64 = AtomicU64::new(0);
static UIC_LEN: AtomicU32 = AtomicU32::new(0);
// ... 6 atomic ops + unsafe block
unsafe { thread::cooperative(&|| { /* load atomics, compute */ }); }
```

After (cooperative_map, zero global atomics):
```rust
thread::cooperative_map(
    data.as_ptr() as *const u8,
    output.as_mut_ptr() as *mut u8,
    n,
    |args| { /* pure compute with args.src, args.dst */ },
);
```

Eliminated: 3 static declarations, 3 store ops, 3 load ops, 1 unsafe block.
The kernel_std PTX confirms UIC_* symbols are fully removed.

## Test Results

- `cooperative_debug`: PASSED (baseline, unchanged)
- `cooperative_compute_test`: PASSED (baseline, unchanged)
- `cooperative_map_test`: PASSED (baseline, unchanged)
- `cooperative_reduce_test`: PASSED (sum(0..256) = 32640, 4 warps)
- `cooperative_map_ext_test`: PASSED (256 elements × 7, 4 warps)
- CI lint: all checks pass

## Files Changed

- `crates/core/gpu-runtime/src/thread.rs` — added `cooperative_reduce()`, `CoopReduceArgs`,
  `cooperative_map_with_params()`, `CoopMapExtArgs`
- `crates/kernel/gpu-kernel/src/thread_test.rs` — added `cooperative_reduce_test`,
  `cooperative_map_ext_test` kernels
- `crates/kernel/gpu-kernel-std/src/lib.rs` — rewrote `unified_io_compute` to use
  `cooperative_map`, removed `UIC_IN_PTR`/`UIC_OUT_PTR`/`UIC_LEN` globals
- `crates/test/gpu-test-harness/src/main.rs` — added host-side verification for both new tests
- `crates/core/gpu-host/kernel.ptx` — rebuilt with new kernels
- `crates/core/gpu-host/kernel_std.ptx` — rebuilt with rewritten unified_io_compute
