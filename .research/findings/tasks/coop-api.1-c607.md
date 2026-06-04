# coop-api.1: cooperative() data passing without global atomics

**Status**: done
**Kind**: investigation
**Theme**: coop-api — Cooperative API ergonomics

## Summary

Designed and implemented `cooperative_map()` — a higher-level API that eliminates global
atomics boilerplate from cooperative compute. Verified on GPU hardware (GTX 1660, sm_75)
with a 256-element elementwise x2 test using 4 warps.

## Root Cause: Why Closure Captures Fail

Each warp's stack lives in **per-thread local memory** (PTX `.local` address space).
When `cooperative()` copies a closure to another warp's SCRATCH buffer, the closure's
*code* is reproduced correctly, but any **captured references** still point to the
original warp's local memory. When the destination warp dereferences those pointers,
it accesses a different address space → `ILLEGAL_ADDRESS`.

Heap-allocated data (`Vec`, `Box`, etc.) lives in the **global address space** and is
accessible from all warps. The problem is purely about getting the *pointer* across
warp boundaries, not the data itself.

## Solution: `cooperative_map(src, dst, len, fn)`

Instead of closure captures, pass data via explicit parameters stored in a global
static argument block (`COOP_MAP_ARGS: [AtomicU64; 4]`):

1. Warp 0 writes `(src_ptr, dst_ptr, len, fn_ptr)` to `COOP_MAP_ARGS`
2. A trampoline function reads from `COOP_MAP_ARGS`, constructs a `CoopMapArgs` struct
   with `(src, dst, len, warp_id, n_warps)`, and calls the user function
3. The user function takes `fn(&CoopMapArgs)` — a plain function pointer, not a closure
4. No `unsafe` needed at the call site

### API comparison

**Before** (cooperative + global atomics):
```rust
static IN_PTR: AtomicU64 = AtomicU64::new(0);
static OUT_PTR: AtomicU64 = AtomicU64::new(0);
static LEN: AtomicU32 = AtomicU32::new(0);

IN_PTR.store(data.as_ptr() as u64, Ordering::Release);
OUT_PTR.store(output.as_mut_ptr() as u64, Ordering::Release);
LEN.store(n as u32, Ordering::Release);

unsafe {
    thread::cooperative(&|| {
        let src = IN_PTR.load(Ordering::Acquire) as *const f32;
        let dst = OUT_PTR.load(Ordering::Acquire) as *mut f32;
        let len = LEN.load(Ordering::Acquire);
        // ... partition and compute ...
    });
}
```

**After** (cooperative_map):
```rust
thread::cooperative_map(
    data.as_ptr() as *const u8,
    output.as_mut_ptr() as *mut u8,
    n,
    |args| {
        let src = args.src as *const f32;
        let dst = args.dst as *mut f32;
        let mut i = args.warp_id as usize;
        while i < args.len {
            // ... compute ...
            i += args.n_warps as usize;
        }
    },
);
```

Eliminates: 3 global statics, 3 atomic stores, 3 atomic loads, `unsafe` block.

## Design Decisions

1. **`fn` pointer vs closure**: Using `fn(&CoopMapArgs)` instead of `impl Fn` makes
   the "no captures" constraint visible at the type level. A closure that accidentally
   captures a local variable is a compile error, not a runtime ILLEGAL_ADDRESS.

2. **Single global arg block vs per-warp copy**: Since all warps read the same
   `(src, dst, len, fn)` values, one global `COOP_MAP_ARGS` suffices. Only `warp_id`
   differs, and that's computed from hardware registers.

3. **`CoopMapArgs` struct**: Bundles all partition info into one argument. Users don't
   need to call `current_id()` / `available_parallelism()` manually.

## Test Results

- `cooperative_debug`: PASSED (baseline, 4 warps write [100..103])
- `cooperative_compute_test`: PASSED (baseline, 256 elements via old API)
- `cooperative_map_test`: PASSED (256 elements x2, zero global atomics, 4 warps)

## Files Changed

- `crates/core/gpu-runtime/src/thread.rs` — added `CoopMapArgs`, `cooperative_map()`
- `crates/kernel/gpu-kernel/src/thread_test.rs` — added `cooperative_map_test` kernel
- `crates/core/gpu-host/src/main.rs` — added host-side verification for cooperative_map
