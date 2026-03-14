# rustc-build.2: Test patched compiler — compile #[warp_cooperative] async fn and inspect PTX
**Cycle**: 268 | **Theme**: rustc-build | **Kind**: experiment | **Status**: done

## Summary
Successfully compiled a `#[warp_cooperative] async fn` to PTX using the patched stage1 compiler. The MIR pass correctly identifies the async fn's coroutine body (via parent def_id lookup) and inserts `activemask.b32` + `bar.warp.sync` instructions before the return.

## Findings

### Q: Does the patched compiler emit warp-cooperative PTX instructions for `#[warp_cooperative] async fn`?
A: **Yes.** The generated PTX for `kernel_entry` (which polls `cooperative_poll`) contains:
```ptx
activemask.b32 %r1;      // get active lane mask
bar.warp.sync %r1;        // warp barrier before return
```
The `shfl.sync` instruction is NOT emitted because this simple async fn has no `.await` points — there's only one discriminant state (immediate Ready), so no broadcast is needed. The MIR pass diagnostic correctly reports: `0 yield(s), 0 poll(s), 0 suspension(s), 1 return(s)`.

**Confidence**: high

### Q: Does LLVM optimize away the inline asm instructions?
A: No. Both `activemask.b32` and `bar.warp.sync` survive through LLVM optimization at `-O2` (release profile). The `options(nostack, nomem)` on activemask and `options(nostack)` on bar.warp.sync correctly prevent LLVM from eliminating them.

**Confidence**: high

### Q: Does the async fn poll to Ready correctly?
A: Yes. The PTX shows `add.s32 %r3, %r2, 1` (the `x + 1` computation) followed by `st.param.b32 [func_retval0], %r3` and `ret`. The future is fully inlined — no coroutine state machine overhead for this trivial case.

**Confidence**: high

## Unexpected Discoveries
- LLVM fully inlines the async fn poll into `kernel_entry` — the coroutine state machine, waker creation, and poll dispatch are all optimized away, leaving only the computation + warp instructions. This is excellent for performance.

## Open Questions
- Need a test with actual `.await` points to verify `shfl.sync` discriminant broadcast works
- Need to verify behavior with multiple suspension points (multi-state coroutine)

## Impact on Downstream Tasks
- rustc-build theme SUCCESS — both criteria met (compiler builds + PTX verified)
- Next: test with multi-await async fn to verify shfl.sync broadcast
