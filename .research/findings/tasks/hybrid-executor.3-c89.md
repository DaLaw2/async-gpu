# hybrid-executor.3: Safety API design note (ADR): no yield in per-thread block
**Cycle**: 89 | **Theme**: hybrid-executor | **Kind**: design | **Status**: done

## Summary
Documented the per-thread compute block safety contract as ADR-10. The key invariant is: per-thread blocks MUST NOT yield (no hostcall, no WarpPoll::Pending on I/O). Evaluated four enforcement approaches: documentation, macro, type state, runtime detection. Recommended documentation + code review as the appropriate enforcement level for current project maturity.

## Findings

### Q: What happens if someone accidentally yields in a per-thread block?
A: **Warp deadlock or state corruption.** If lane X yields (returns Pending on a hostcall) while other lanes continue, lane X re-enters the WarpExecutor poll loop. On the next poll, `broadcast_u32` reads state from lane 0 — but lane 0 may have advanced to a different state. Lane X then executes code for the wrong state, causing data corruption or deadlocking at the next `syncwarp()` where convergence is expected but absent.
**Confidence**: high (analysis + SIMT model reasoning)

### Q: Can we detect this at compile time (type state, macro)?
A: **Not practically at this stage.**
- **Type state**: Would need `WarpFuture<Mode>` where `Mode ∈ {Cooperative, PerThread}`. The per-thread mode would need to restrict the set of operations — but Rust's type system can't easily express "this closure doesn't call any function that returns Poll::Pending." Would need a custom lint or unsafe marker trait.
- **Macro**: A `per_thread_block!` macro could wrap the boilerplate (syncwarp, state advance) but cannot prevent the body from calling hostcall functions — it's still regular Rust code inside.
- **Proc macro**: The `#[warp_async]` proc macro could potentially parse the body and reject hostcall calls, but it currently only supports `warp_print!()` calls and doesn't support per-thread blocks at all.

**Conclusion**: Compile-time detection requires either a custom lint or significant proc macro expansion. Not worth the complexity for 0 external users.
**Confidence**: high

### Q: Is runtime detection feasible (check active_mask after per-thread block)?
A: **Partially.** After `syncwarp()`, calling `activemask()` tells you which lanes are actually present. If `activemask() != expected_mask`, a lane diverged. However, this only detects the bug after `syncwarp()` returns — which means either (a) the divergent lane already corrupted state, or (b) `syncwarp()` deadlocked waiting for the missing lane. In case (b), the detection code never runs. In case (a), the detection fires too late. **Not useful for prevention**, only for post-mortem debugging.
**Confidence**: high

### Q: What is the recommended enforcement approach?
A: **Documentation + code review.**
1. ADR-10 clearly states the invariant and consequences of violation
2. The pattern is simple enough (5-6 lines) that it's visually obvious
3. Per-thread blocks should never call any function with "hostcall" in the name
4. Future: when `#[warp_async]` proc macro supports per-thread blocks, it can reject hostcall calls in per-thread arms at macro expansion time

**Confidence**: high

## Impact on Downstream Tasks
- ADR-10 written to decisions.md
- hybrid-executor theme success criteria all met
- Theme can be marked completed
