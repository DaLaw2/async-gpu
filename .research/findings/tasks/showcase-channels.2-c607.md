# showcase-channels.2: GPU Executor Example — Spawn 4+ Async Tasks Dynamically

## Status: DONE

## Finding

The GPU channels example at `examples/hostcall/gpu-channels/` already contains
a comprehensive executor demo (Demo 3: "Async Executor") that satisfies all
requirements:

### Requirements Check

| Requirement | Status | Evidence |
|---|---|---|
| Spawn 4+ async tasks | COVERED | 8 tasks: 4 WriteValueFuture + 4 CounterFuture |
| Tasks communicate via channels | COVERED | Demo 1 (oneshot) + Demo 2 (MPSC) in same example |
| Dynamic scheduling | COVERED | CounterFuture yields (Pending) on first poll, re-scheduled by executor on second poll |
| Await results from completed tasks | COVERED | Host reads results after executor.run() completes; executor tracks spawned vs completed counts |

### Dynamic Scheduling Detail

The executor demo already demonstrates dynamic scheduling through the
`CounterFuture` type, which has a two-phase lifecycle:

1. **First poll**: returns `Poll::Pending` (yields), task transitions QUEUED -> RUNNING -> PARKED
2. **Re-schedule**: executor re-enqueues parked task, PARKED -> QUEUED
3. **Second poll**: increments counter, returns `Poll::Ready`, task completed

This is genuine dynamic scheduling — the executor must interleave execution of
immediate tasks (WriteValueFuture) with yielding tasks (CounterFuture) across
multiple scheduling rounds.

### What Changed

Enhanced documentation and output in `examples/hostcall/gpu-channels/src/main.rs`
to better communicate the dynamic scheduling aspect that was already present:

- Updated module-level doc comment to describe task lifecycle states
- Updated Demo 3 section header to "Dynamic Scheduling"
- Added per-task-type output showing immediate vs yield-and-reschedule behavior
- Added explicit "Dynamic scheduling" line in verification output
- Expanded kernel-side comments to describe task state transitions

### Uses async-gpu Facade

Yes — `Cargo.toml` uses `async-gpu = { path = "../../../crates/async-gpu" }`.

### CI Lint

`bash scripts/ci-lint.sh` passes all checks including `check gpu-channels`.

## Files Modified

- `examples/hostcall/gpu-channels/src/main.rs` — enhanced docs + output
