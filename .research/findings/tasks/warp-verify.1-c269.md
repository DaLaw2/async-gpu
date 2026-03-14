# warp-verify.1: Multi-await async fn — verify shfl.sync broadcast in PTX
**Cycle**: 269 | **Theme**: warp-verify | **Kind**: experiment | **Status**: done

## Summary
Fixed `find_dispatch_switch` heuristic that failed to detect the coroutine dispatch switch (was checking `discriminant(_1)` but actual MIR has `discriminant((*_47))` due to Pin unwrap). Multi-await async fn now correctly produces PTX with `activemask.b32` + `shfl.sync.idx.b32` + `bar.warp.sync` + `brx.idx` instructions.

## Findings

### Q: Why did the original heuristic fail?
A: After `StateTransform`, the async fn coroutine receives `Pin<&mut CoroutineState>` as `_1`. The dispatch block first unwraps Pin: `_47 = copy (_1.0: &mut ...)`, then takes `_46 = discriminant((*_47))`. The original `is_discriminant_of_self` required `discriminant(place)` where `place.local == Local::from(1u32)` (i.e., directly `_1`), which doesn't match the indirect pattern through `_47`.

**Fix**: Simplified detection to look for ANY `SwitchInt` whose scrutinee is assigned from a `Discriminant` rvalue in the same block, with target values >= 3 (suspension states). No longer requires the discriminant source to be specifically `_1`.

**Confidence**: high

### Q: Does multi-await async fn produce shfl.sync broadcast?
A: **Yes.** The PTX contains the full warp-cooperative pattern:
```ptx
activemask.b32 %r3;                         // get active lanes
shfl.sync.idx.b32 %r4, %r2, 0, 31, %r3;    // broadcast discriminant from lane 0
brx.idx %r4, $L_brx_0;                      // dispatch on broadcast value
```
This pattern repeats for each poll iteration (LLVM unrolls the loop).

**Confidence**: high

### Q: How does LLVM handle the coroutine + warp instructions?
A: LLVM unrolls the polling loop (max 10 iterations in `kernel_entry`) and for each unrolled iteration, preserves the full `activemask → shfl.sync → brx.idx` dispatch sequence followed by per-branch `activemask → bar.warp.sync` barriers. The coroutine state machine is preserved (not optimized away like the trivial case), showing states 0 (Unresumed), 1 (Returned), 3 (Suspend0), 4 (Suspend1).

**Confidence**: high

## Unexpected Discoveries
- LLVM uses `brx.idx` (indirect branch by index) for the dispatch, which is an efficient PTX instruction for computed jumps.
- The loop unrolling creates a large function (~630 lines PTX) but each iteration is structurally identical — a real GPU executor would use a smaller loop.

## Open Questions
- Does this work correctly on actual GPU hardware? (warp-verify.2)
- What happens with divergent lanes (different coroutine states)?

## Impact on Downstream Tasks
- warp-verify.1 SUCCESS — shfl.sync broadcast verified in PTX
- Next: warp-verify.2 (actual GPU execution)
