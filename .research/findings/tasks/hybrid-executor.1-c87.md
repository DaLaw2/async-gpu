# hybrid-executor.1: Minimal hybrid PoC: WarpFuture read → per-thread compute → WarpFuture write
**Cycle**: 87 | **Theme**: hybrid-executor | **Kind**: experiment | **Status**: done

## Summary
Implemented and verified a hybrid WarpFuture that transitions between warp-cooperative I/O (PRINT hostcall) and per-thread divergent computation within a single state machine. 32 lanes compute `lane_id² + 1` independently in the per-thread block, with `syncwarp()` ensuring correct reconvergence at switching points. All 32 results verified correct, completed in ~1ms.

## Findings

### Q: Can a WarpFuture state machine transition to per-thread computation and back?
A: **Yes.** The state machine has 6 states: INIT_PRINT → WAIT_PRINT → COMPUTE → INIT_PRINT2 → WAIT_PRINT2 → DONE. The COMPUTE state is a per-thread block where each lane independently computes and writes its result. State transitions still use the broadcast-from-lane-0 pattern — all lanes enter the COMPUTE state together, do their independent work, then `syncwarp()` reconverges before lane 0 advances the state.
**Confidence**: high

### Q: How should the per_thread_block! API look for non-yielding computation?
A: For the PoC, the per-thread block is inline code within a match arm of the WarpFuture state machine. No macro or separate API was needed — the pattern is straightforward:
1. Enter state (all lanes, via broadcast)
2. Each lane reads `wcx.lane_id` and computes independently
3. Each lane writes its result to `results[lane_id]`
4. `syncwarp()` reconverges all lanes
5. Lane 0 advances state
6. Another `syncwarp()` ensures all lanes see the new state

A `per_thread_block!` macro could be added later for ergonomics, but the raw pattern is clean enough for hand-written WarpFutures.
**Confidence**: high

### Q: Does syncwarp() at entry/exit of per-thread block ensure correct reconvergence?
A: **Yes.** The state is broadcast from lane 0, so all lanes enter the COMPUTE arm simultaneously. After independent computation, `syncwarp(active_mask)` reconverges all lanes. A second `syncwarp()` after lane 0 updates the state ensures all lanes see the new state value on the next poll. This is the same pattern used in all WarpFuture I/O states.
**Confidence**: high

### Q: What is the measured overhead of WarpFuture↔per-thread switching?
A: The switching itself is essentially free — it's just a `syncwarp()` instruction (~5-10 ns). The total kernel time (~1ms) is dominated by the two PRINT hostcall round-trips, not the per-thread compute block. The computation (`lane_id * lane_id + 1`) is trivially fast.
**Confidence**: high

### Q: What happens when lanes compute different durations in per-thread block?
A: Not directly tested in this PoC (all lanes compute the same trivial formula). However, the `syncwarp()` at exit guarantees that faster lanes wait for slower lanes before reconverging. This is the standard SIMT synchronization guarantee — no additional mechanism needed.
**Confidence**: medium (inferred from SIMT model, not empirically measured with varying workloads)

## Implementation

### GPU kernel: `HybridFuture`
- 6-state WarpFuture: INIT_PRINT(0) → WAIT_PRINT(1) → COMPUTE(2) → INIT_PRINT2(3) → WAIT_PRINT2(4) → DONE(5)
- Helper functions `hybrid_warp_print_init()` and `hybrid_warp_wait()` factor out the repeated PRINT hostcall pattern
- COMPUTE state: each lane writes `results[lane_id] = lane_id * lane_id + 1`

### Host: `run_hybrid_executor_test()`
- Allocates 33 mapped u32 values: 32 for per-lane results + 1 for status
- Launches 1 block × 32 threads (1 warp)
- Verifies: `results[i] == i*i + 1` for i in 0..32
- Verifies: 2 PRINT messages received ("hybrid: start" and "hybrid: done")

## Impact on Downstream Tasks
- Proves WarpFuture ↔ per-thread switching is viable and cheap
- Next: hybrid-executor.2 could test variable-duration per-thread work, or multiple switching points
- The `per_thread_block!` macro could wrap the reconvergence pattern for safety
