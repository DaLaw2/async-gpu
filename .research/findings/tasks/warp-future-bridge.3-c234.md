# warp-future-bridge.3: Two sequential warp-cooperative Future polls
**Cycle**: 234 | **Theme**: warp-future-bridge | **Kind**: experiment | **Status**: done

## Summary
Successfully chained two standard `GpuPrintFuture`s in a warp-cooperative state machine with sequential "await" points. `warp_sequential::warp_run_two_futures()` polls F1 to completion, issues `syncwarp()` barrier, then polls F2 to completion. All 32 lanes stay converged throughout. Both messages received in order.

## Findings
### Q: Can multiple sequential `.await` points maintain warp convergence?
A: Yes. The `syncwarp(mask)` barrier between the two polling loops ensures all lanes converge before moving to the second future. This is exactly what `#[warp_cooperative]` would generate: a syncwarp at each state machine transition.

**Confidence**: high (verified on GPU hardware)

### Key implementation
- `warp_run_two_futures(f1, f2)` — two independent polling loops with a `syncwarp` between them
- Each loop: lane 0 polls, encodes result, broadcasts via shfl.sync, all lanes decode
- Convergence model: lanes never diverge because all control flow depends on broadcast values

### Test results
- `warp_cooperative_two_futures_kernel`: 32 threads (1 warp)
  - Message 1: "warp-coop sequential 1" ✓
  - Message 2: "warp-coop sequential 2" ✓
  - result = 2 ✓
  - All 32 lanes wrote lane_id ✓

## Unexpected Discoveries
- The sequential model maps cleanly to the state machine pattern: each "await" is a polling loop, each transition is a syncwarp. This is exactly what rustc's StateTransform generates, just with warp barriers added.

## Open Questions
- None

## Impact on Downstream Tasks
- **warp-future-bridge.4**: Error broadcasting — add `Result<T, E>` to the broadcast protocol
- Phase 1 almost complete — only error handling remains
