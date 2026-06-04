# sc-resource.2 — BlockScope Shared Memory Allocator

## Status: done
## Summary: Implemented SharedMemAllocator, BlockScope, ScopeJoinHandle, and block_scope() entry function. All code compiles clean for nvptx64-nvidia-cuda target via gpu-kernel-std.

## Implementation

### Files created
- `crates/core/gpu-runtime/src/scope.rs` — new module with all scope types

### Files modified
- `crates/core/gpu-runtime/src/thread.rs` — made internal constants, statics, and helpers `pub(crate)` so the scope module can access warp management infrastructure: `MAX_WARPS`, `STATUS_IDLE`, `STATUS_ASSIGNED`, `STATUS_DONE`, `STATUS_COOPERATIVE`, `WARP_STATUS`, `WARP_FN`, `WARP_DATA`, `WARP_RESULT`, `SCRATCH`, `SCRATCH_SIZE`, `NUM_WARPS`, `warp_id()`, `lane_id()`, `nanosleep_short()`
- `crates/core/gpu-runtime/src/lib.rs` — added `pub mod scope;` with doc comment

### Key design decisions

1. **`&mut BlockScope` instead of `&BlockScope`**: The `block_scope` closure receives `&mut BlockScope` so that `spawn()` can track spawned warps via the mutable `spawned_warps` bitmask without needing UnsafeCell in BlockScope itself. The `alloc` methods still take `&self` since they only mutate the global ALLOCATOR (via UnsafeCell). `spawn_all` also takes `&self` since it doesn't track spawned warps (it's synchronous — joins before returning).

2. **AllocatorCell wrapper**: Rather than making SharedMemAllocator itself implement Sync, used a private `AllocatorCell` wrapper with `UnsafeCell<SharedMemAllocator>` and `unsafe impl Sync`, matching the pattern used by `Mutex<T>` in `sync.rs`.

3. **Drop as safety net**: `BlockScope` implements `Drop` to join remaining warps and pop the watermark. The primary cleanup path is explicit in `block_scope()` (join_all then let Drop pop), but Drop handles early-return or panic cases.

4. **spawn reuses thread.rs trampoline pattern**: `scope.spawn()` follows the exact same trampoline/scratch-buffer pattern as `thread::spawn()`, but with `'scope` lifetime bounds instead of `'static`. This ensures compatibility with the existing worker loop in thread.rs.

5. **spawn_all reuses cooperative pattern**: `scope.spawn_all()` follows the same pattern as `thread::cooperative()` — copies closure to each worker's scratch buffer, wakes all warps with STATUS_COOPERATIVE, warp 0 also participates, then joins all workers.

6. **Cancellation via volatile reads/writes**: `cancel()` and `is_cancelled()` use `write_volatile`/`read_volatile` on a shared-memory flag, matching the pattern used throughout the codebase for cross-warp signaling.

### Deviations from design spec

- `block_scope` takes `FnOnce(&mut BlockScope<'scope>)` instead of `FnOnce(&BlockScope<'scope>)`. This is needed so `spawn()` can mutate the spawned_warps bitmask. The lifetime safety guarantee from `for<'scope>` HRTB is preserved.
- `saved_watermark` field was removed from BlockScope — the watermark is managed by the allocator's internal stack, and the pop is done in Drop. No need to duplicate it.
- `GridScope`, `GridScopeJoinHandle`, and `grid_scope()` are not implemented — the task scope was BlockScope only.

## Testing Notes

### How to compile-test
```bash
cd crates/kernel/gpu-kernel-std && cargo build
```

### How to functionally test
A kernel that uses `block_scope` would look like:
```rust
use gpu_runtime::scope::{block_scope, init_shared_mem_allocator};

// In kernel entry, after gpu_main:
unsafe { init_shared_mem_allocator(shared_mem_bytes); }
block_scope(|scope| {
    let buf = scope.alloc::<f32>(64);
    scope.spawn_all(|wid, nw| {
        let mut i = wid as usize;
        while i < 64 { buf[i] = i as f32; i += nw as usize; }
    });
});
```

### Known limitations
1. `init_shared_mem_allocator()` must be called before any `block_scope()` — the capacity is not auto-discovered.
2. Max nesting depth is 4 (panics on overflow).
3. Panics in spawned warps (`trap;`) will deadlock the scope — same as existing `thread::spawn` + `join()`. Deferred to error-bitmask work.
4. `scope.alloc()` from non-warp-0 is a debug_assert (compiled out in release).

## Files Changed
- `crates/core/gpu-runtime/src/scope.rs` (new)
- `crates/core/gpu-runtime/src/thread.rs` (visibility changes)
- `crates/core/gpu-runtime/src/lib.rs` (added pub mod scope)
