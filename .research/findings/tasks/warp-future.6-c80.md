# warp-future.6: Multi-Hostcall WarpFuture — 3 Sequential PRINT Calls
**Cycle**: 80 | **Theme**: warp-future | **Kind**: experiment | **Status**: done

## Summary
Implemented and verified a 7-state WarpMultiPrintFuture state machine that performs 3 sequential PRINT hostcalls while maintaining full warp convergence. All 3 messages received in correct order: "WarpMulti[1/3]", "[2/3]", "[3/3]". Total round-trip: 0.802ms for 3 calls (vs 0.373ms for single call in warp-future.4). PTX contains 11x `shfl.sync.idx.b32` and 15x `bar.warp.sync`, confirming convergence across all state transitions.

## Findings

### Q: Can a WarpFuture state machine compose multiple sequential hostcalls?
A: **Yes.** The 7-state machine (INIT1→WAIT1→INIT2→WAIT2→INIT3→WAIT3→DONE) correctly chains 3 independent hostcall cycles. Each cycle performs the full pop→fill→push→doorbell→spin-wait→release sequence. The key pattern: after each WAIT state receives CONTROL_READY, lane 0 transitions to the next INIT state (not DONE), and the executor re-polls into the next hostcall. All 3 messages were received in exact order.
**Confidence**: high

### Q: Does the multi-state machine maintain warp convergence across all transitions?
A: **Yes.** Every state transition broadcasts the discriminant from lane 0 via `shfl.sync.idx.b32`, so all lanes always enter the same match arm. The PTX shows no lane-predicated divergence across any of the 7 states. The `bar.warp.sync` barriers at each state transition ensure convergence even when lane 0 performs asymmetric work (packet management).
**Confidence**: high

### Q: What is the register pressure of a 3-call state machine vs single-call?
A: The compiler effectively shared the init and wait helper functions across all 3 hostcall stages (`warp_multi_init_hostcall` and `warp_multi_wait_hostcall`). The state machine fields are minimal: `buf`, `state`, `pkt_idx`, `calls_completed` (4 registers). The helper function approach means the PTX body size scales with O(N_helpers) not O(N_states).
**Confidence**: medium (register count not directly measured, inferred from PTX structure)

### Q: Is the PTX output convergent across all state transitions?
A: **Yes.** 11x `shfl.sync.idx.b32` (state broadcast at entry + pkt_idx broadcasts per init/wait) and 15x `bar.warp.sync` (convergence barriers at payload writes + state transitions). This is roughly 3x the single-call counts (4+5), confirming linear scaling with hostcall count.
**Confidence**: high

## Implementation Details

### State Machine Design
```
WMP_INIT1 (0) → WMP_WAIT1 (1) → WMP_INIT2 (2) → WMP_WAIT2 (3) → WMP_INIT3 (4) → WMP_WAIT3 (5) → WMP_DONE (6)
```

Factored into two shared helper functions:
- `warp_multi_init_hostcall()`: pop packet, cooperative payload write, lane 0 submits
- `warp_multi_wait_hostcall()`: convergent spin-wait, lane 0 releases, state transition

The `poll_warp()` dispatch is a simple match on broadcast state → delegate to appropriate helper.

### Messages
1. "WarpMulti[1/3]: HELLO_FROM_32_LANES!!" (36 bytes)
2. "WarpMulti[2/3]: SECOND_CALL_WORKING!" (36 bytes)
3. "WarpMulti[3/3]: PIPELINE_COMPLETE!!" (35 bytes)

### Hardware Verification
- Target: SM86 (RTX 3060)
- Launch: 1 block × 32 threads
- All 3 messages received in order
- Result: 1 (success)
- Elapsed: 0.802ms (3 calls)
- Per-call average: ~0.267ms

### PTX Verification
- 11x `shfl.sync.idx.b32` — state + pkt_idx broadcasts across 7 states
- 15x `bar.warp.sync` — convergence barriers at all transition points
- Linear scaling: ~3x single-call instruction counts

## Open Questions
1. How does register pressure scale with >3 hostcalls? (e.g., 10-call pipeline)
2. Can the helper function pattern be made generic enough for proc macro generation?
3. What happens when backpressure occurs mid-pipeline? (currently retries forever in INIT)

## Impact on Downstream Tasks
- **warp-future.5** (proc macro): Now has TWO reference implementations (single + multi-call). The helper function pattern (shared init/wait) is mechanical and suitable for code generation.
- **api-cleanup**: The multi-call pattern exposed that `hc_pop_free`, `hc_push`, `pkt_offset` etc. are the critical composable primitives — API cleanup should ensure these are ergonomic.
