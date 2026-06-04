# sc-runtime.4 — Scope + Cooperative Bridge

## Status: done
## Summary: BlockScope composes cleanly with cooperative_map/cooperative_reduce/cooperative_map_with_params. No code changes needed — no conflicts exist when the documented sequencing rule is followed (join all spawned tasks before calling cooperative APIs). Added documentation to both scope.rs and thread.rs explaining the composition model, pointer compatibility, and the preference for spawn_all inside scopes.

## Analysis

### 1. Warp lifecycle trace — cooperative_map

`cooperative_map(src, dst, len, fn)` in `thread.rs`:

1. Warp 0 stores args to `COOP_MAP_ARGS` global statics (4 × AtomicU64)
2. For each worker warp i in 1..N-1: sets `WARP_FN[i]`, `WARP_DATA[i]` (unused, set to 0), then `WARP_STATUS[i] = STATUS_COOPERATIVE` (release store)
3. Warp 0 executes its own partition inline
4. Warp 0 spin-waits for each worker: `STATUS_DONE → reset to STATUS_IDLE`

**After return**: All warps 1..N-1 are in `STATUS_IDLE`. No residual state.

### 2. Warp lifecycle trace — scope.spawn_all

`BlockScope::spawn_all(f)` in `scope.rs`:

1. Asserts `self.spawned_warps == 0` (no in-flight spawn tasks)
2. Copies closure to each worker's `SCRATCH[i]` buffer, sets `WARP_FN[i]`, `WARP_DATA[i]`, then `WARP_STATUS[i] = STATUS_COOPERATIVE`
3. Warp 0 executes its own partition inline
4. Warp 0 spin-waits for each worker: `STATUS_DONE → reset to STATUS_IDLE`, records `STATUS_TRAPPED` warps in `error_mask`

**After return**: All warps 1..N-1 are in `STATUS_IDLE` (or trapped). `spawned_warps` remains 0 (spawn_all does not set it).

### 3. Composition analysis

**Scenario A — cooperative_map inside block_scope, no prior spawns:**

```rust
block_scope(|scope| {
    let buf = scope.alloc::<f32>(256);
    // All warps are IDLE (no spawn() calls)
    cooperative_map(buf.as_ptr() as _, dst, 256, my_fn);
    // All warps back to IDLE — scope state is clean
});
```

**Result**: Works correctly. `cooperative_map` does not touch `scope.spawned_warps` or `scope.error_mask`. All warps return to IDLE. The scope's allocator state is unaffected.

**Scenario B — cooperative_map after spawn_all:**

```rust
block_scope(|scope| {
    let buf = scope.alloc::<f32>(256);
    scope.spawn_all(|wid, nw| { /* init buf */ });
    // All warps IDLE after spawn_all returns
    cooperative_map(buf.as_ptr() as _, dst, 256, my_fn);
    // All warps IDLE — clean
});
```

**Result**: Works correctly. `spawn_all` joins synchronously, leaving all warps IDLE. `cooperative_map` then runs its own cooperative pass cleanly.

**Scenario C — cooperative_map after spawn (DANGER):**

```rust
block_scope(|scope| {
    let h = scope.spawn(|| 42u32); // warp 1 is ASSIGNED/RUNNING
    cooperative_map(src, dst, len, my_fn); // OVERWRITES warp 1's status!
    h.join(); // UB: warp 1 state corrupted
});
```

**Result**: UNSAFE. `cooperative_map` unconditionally writes `STATUS_COOPERATIVE` to warps 1..N-1, overwriting warp 1's `STATUS_RUNNING`. This is the same hazard that `spawn_all` already guards against with `assert!(self.spawned_warps == 0)`.

**Key difference**: `spawn_all` has the assertion guard. `cooperative_map` does not, because it is a free function unaware of the scope. This is a documentation issue, not a code bug — the user must join all spawned tasks before calling any cooperative API.

### 4. Pointer compatibility

`scope.alloc::<T>(n)` returns `&'scope mut [T]` backed by shared memory (via `block::shared_mem_at`). The underlying pointer is a valid GPU address into the block's dynamic shared memory region.

`cooperative_map` accepts `*const u8` / `*mut u8`. Converting via `.as_ptr() as *const u8` / `.as_mut_ptr() as *mut u8` produces valid shared memory pointers that all warps can read/write.

**No issues**: Shared memory is visible to all threads in the block. The cooperative function pointers run on warps within the same block, so they can access shared memory without any special synchronization beyond the implicit barrier of the cooperative join.

### 5. spawn_all vs cooperative_map — when to use which

| Feature | spawn_all | cooperative_map |
|---------|-----------|-----------------|
| Argument passing | Closure captures (can reference scope allocs) | Function pointer + raw pointers via global statics |
| Error tracking | error_mask records trapped warps | No error detection (warp trap = hang) |
| Safety guard | Asserts no in-flight spawns | No guard (caller responsibility) |
| Extra params | Via closure captures | cooperative_map_with_params: 4×u64 |
| Scope integration | Full (tracks cooperative lifecycle) | None (standalone) |

**Recommendation**: `spawn_all` is strictly preferred inside scopes. `cooperative_map` remains useful for scope-free code (e.g., inside `gpu_main` without a scope) or when a function-pointer-based API is needed for interop.

### 6. No helper needed

No `scope.cooperative_map()` wrapper is needed because:
- The existing `cooperative_map` works correctly when called after joining all spawned tasks
- `spawn_all` already provides a superior alternative for scope-internal cooperative work
- Adding a wrapper would be redundant API surface with no correctness benefit

## Files Changed

- `crates/core/gpu-runtime/src/scope.rs` — Added module-level "Composing with cooperative APIs" documentation section (4 rules). Added composition example and safety note to `block_scope()` function docs.
- `crates/core/gpu-runtime/src/thread.rs` — Added "Composing with BlockScope" documentation section to `cooperative_map()` explaining pointer compatibility, sequencing prerequisite, and spawn_all preference.
