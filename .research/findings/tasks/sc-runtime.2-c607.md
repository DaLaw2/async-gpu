# sc-runtime.2 — BlockScope Warp Fork/Join

## Status: done
## Summary: Fixed spawn_all safety issue, added STATUS_TRAPPED detection to prevent join deadlocks, made join_all() public for mid-scope synchronization, added error_mask tracking, and updated the panic handler to signal trapped warps.

## Implementation

### 1. STATUS_TRAPPED constant (thread.rs)
Added `pub(crate) const STATUS_TRAPPED: u32 = 6` alongside the existing status constants. This value is set by the panic handler before calling `trap;`, allowing join loops to detect dead warps.

### 2. spawn_all safety guard (scope.rs)
Changed `spawn_all` signature from `&self` to `&mut self` and added an assert at the top:
```rust
assert!(
    self.spawned_warps == 0,
    "scope.spawn_all: all spawned tasks must be joined before calling spawn_all"
);
```
This prevents the latent bug where spawn_all would overwrite STATUS_COOPERATIVE on warps still RUNNING from prior spawn() calls. The spawn_all join loop also now detects STATUS_TRAPPED.

### 3. join_all() made public with trap detection (scope.rs)
The private `join_all` method is now `pub fn join_all(&mut self)`. The spin-wait loop now matches on both STATUS_DONE (normal) and STATUS_TRAPPED (dead warp). Trapped warps are recorded in `error_mask` and NOT reset to STATUS_IDLE (they are permanently dead). Users can call this mid-scope to synchronize all spawned warps and free them for reuse.

### 4. error_mask field and accessors (scope.rs)
Added `error_mask: u32` to BlockScope, initialized to 0. Provides:
- `pub fn error_mask(&self) -> u32` — bitmask of trapped warps
- `pub fn has_errors(&self) -> bool` — convenience check

### 5. ScopeJoinHandle::join trap detection (scope.rs)
Updated `ScopeJoinHandle::join()` to detect STATUS_TRAPPED and panic with a clear message instead of spinning forever.

### 6. Drop safety net (scope.rs)
The Drop impl calls `join_all()` which already handles STATUS_TRAPPED. No additional changes needed — the existing `if self.spawned_warps != 0 { self.join_all(); }` path inherits the trap detection.

### 7. Panic handler integration (lib.rs, panic.rs)
Added `pub unsafe fn set_warp_trapped()` in panic.rs — sets `WARP_STATUS[wid]` to `STATUS_TRAPPED` via release store (lane 0 only). Called from the `panic_handler!()` macro just before `trap;`.

## Files Changed
- `crates/core/gpu-runtime/src/thread.rs` — added STATUS_TRAPPED constant
- `crates/core/gpu-runtime/src/scope.rs` — error_mask field, spawn_all guard, public join_all with trap detection, ScopeJoinHandle trap detection, error_mask/has_errors accessors
- `crates/core/gpu-runtime/src/panic.rs` — set_warp_trapped() function
- `crates/core/gpu-runtime/src/lib.rs` — panic_handler! macro calls set_warp_trapped() before trap
