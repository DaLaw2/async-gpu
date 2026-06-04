# sc-runtime.3 — GridScope Cross-Block Coordination

## Status: done

## Summary

Implemented cross-block work coordination utilities for GridScope. Added a new
`grid_work` module with `BlockWorkSlot` work descriptors, coordinator dispatch
functions, and worker block polling loops. All atomics are system-scope for
cross-block visibility on SM75 without cooperative launch.

## Implementation

### BlockWorkSlot

`#[repr(C)]` struct with status (u32 atomic), work_fn (u64 function pointer),
args ([u64; 4]), and result (u64). Status state machine:
IDLE(0) → WORK_AVAILABLE(1) → COMPLETED(2), with CANCELLED(3) as terminal.

### Coordinator API (grid_work module)

- `init_work_slots(slots)` — zero-init + system-scope release store per slot
- `dispatch_work(slot, fn, args)` — write descriptor + release-store WORK_AVAILABLE
- `cancel_slot(slot)` — release-store CANCELLED
- `load_slot_status(slot)` — one-shot acquire load
- `poll_slot_status(slot)` — spin-safe acquire load with nanosleep
- `wait_slot_done(slot)` — spin until COMPLETED or CANCELLED
- `read_result(slot)` — read result after completion
- `reset_slot(slot)` — reset to IDLE for reuse

### Worker API (grid_work module)

- `grid_worker_loop(slot)` — single-shot: poll, execute one work item, return
- `grid_worker_loop_continuous(slot)` — persistent: execute items until CANCELLED

### GridScope integration (scope.rs)

- `GridScope::alloc_work_slots(count)` — allocate + init from pool
- `GridScope::dispatch_work_to_slot(slot, fn, args)` — convenience wrapper

## Files Changed

- `crates/core/gpu-runtime/src/grid_work.rs` — new module (all coordination types + functions)
- `crates/core/gpu-runtime/src/scope.rs` — added `alloc_work_slots()` and `dispatch_work_to_slot()` to `GridScope`
- `crates/core/gpu-runtime/src/lib.rs` — registered `grid_work` module with doc comment
