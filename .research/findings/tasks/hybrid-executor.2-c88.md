# hybrid-executor.2: Variable-duration + multi-switch stress test
**Cycle**: 88 | **Theme**: hybrid-executor | **Kind**: experiment | **Status**: done

## Summary
Verified the hybrid executor pattern under stress: 9-state machine with 3 I/O phases and 2 per-thread compute phases. COMPUTE1 has ~3100x lane duration variance (1 vs 3101 iterations), COMPUTE2 uses XOR-fold with lane-dependent iteration counts. All 64 results (32 per phase) verified correct. syncwarp() correctly handles extreme lane divergence timing. Total kernel time ~1ms.

## Findings

### Q: Does syncwarp() correctly synchronize lanes with 100x duration variance?
A: **Yes.** Tested with up to 3100x variance: lane 0 does 1 iteration of sum, lane 31 does 3101 iterations. All lanes reconverge correctly at the syncwarp() after the compute block, and the subsequent WarpFuture I/O state executes correctly. This is the SIMT contract — syncwarp() is a hardware barrier, it blocks until all specified lanes reach it regardless of how long each took.
**Confidence**: high

### Q: Does the state machine remain clean with 10+ states (multiple I/O + compute phases)?
A: **Yes.** The 9-state machine (3 INIT/WAIT pairs + 2 COMPUTE + DONE) is straightforward. The `hybrid_warp_print_init()` and `hybrid_warp_wait()` helper functions factor out the repeated I/O pattern, so each I/O phase is just 2 match arms calling helpers. Each compute phase is a single match arm with the per-lane logic. The pattern composes linearly — adding more phases is mechanical.
**Confidence**: high

### Q: Any PTX codegen issues with complex per-thread blocks (loops, branches)?
A: **No issues.** The COMPUTE1 block contains a while loop with wrapping_add, and COMPUTE2 contains a while loop with XOR-shift operations. Both compile to clean PTX with the expected loop structure. No unexpected spilling, no codegen issues.
**Confidence**: high

## Implementation

### GPU kernel: `HybridStressFuture`
- 9-state machine: INIT1(0)→WAIT1(1)→COMPUTE1(2)→INIT2(3)→WAIT2(4)→COMPUTE2(5)→INIT3(6)→WAIT3(7)→DONE(8)
- COMPUTE1: `sum = wrapping_add(1..=(lane_id*100+1))` → results[lane_id]
- COMPUTE2: XOR-fold `(lane_id+1)*50` iterations → results[32+lane_id]
- Reuses `hybrid_warp_print_init()` and `hybrid_warp_wait()` helpers from .1

### Host: `run_hybrid_stress_test()`
- Allocates 65 mapped u32 values: 64 results + 1 status
- Verifies COMPUTE1: `results[i] == n*(n+1)/2` where `n = i*100+1`
- Verifies COMPUTE2: recomputes XOR-fold on host and compares
- Verifies 3 messages in order: "stress: phase1", "phase2", "phase3"

## Impact on Downstream Tasks
- All technical risk for hybrid-executor resolved
- Variable-duration per-thread work: proven
- Multiple switching points: proven
- Next: hybrid-executor.3 (safety ADR) is a pure design task
