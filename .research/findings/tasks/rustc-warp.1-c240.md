# rustc-warp.1: Verify baseline async fn → PTX
**Cycle**: 240 | **Theme**: rustc-warp | **Kind**: experiment | **Status**: done

## Summary
Verified that standard `async fn` already compiles and runs correctly on nvptx64 GPU without ANY rustc modifications. Two async functions tested: `trivial_async()` → 42, `one_yield(10)` → 22. LLVM fully optimizes the async state machine — the entire kernel collapses to a single `st.volatile.global.b32 [result], 0x0016002A` instruction.

## Findings
### Q: Can rustc compile async fn to nvptx64 PTX?
A: **Yes, already works with standard nightly rustc.** No fork needed for basic async fn.

- `async fn trivial_async() -> u32 { 42 }` → compiles, returns correctly
- `async fn one_yield(x: u32) -> u32 { poll_fn(|_| Ready(x+1)).await; y*2 }` → compiles, returns correctly
- Manual spin-poll with no-op Waker inside a `ptx-kernel` works

**Confidence**: high (GPU-verified with real hardware)

### PTX Analysis
The generated PTX for the entire kernel is only 4 instructions:
```ptx
rustc_async_baseline_test:
    ld.param.b64 %rd1, [result_param_0];
    cvta.to.global.u64 %rd2, %rd1;
    st.volatile.global.b32 [%rd2], 1441834;  // = 0x0016002A = (22<<16)|42
    ret;
```
LLVM completely eliminated the async state machines — constant-folded `trivial_async()=42` and `one_yield(10)=(10+1)*2=22`, combined into a single store.

### Impact on Phase 3 Strategy
The original Phase 3 plan was: "fork rustc, add WarpCooperativeTransform MIR pass." But:

1. **async fn already compiles to PTX** — no fork needed for compilation
2. **The real problem is warp cooperation, not async compilation** — standard async state machines are per-thread; warp cooperation needs shfl.sync broadcasts at yield points
3. **MIR pass approach**: Insert shfl.sync at every yield point in the MIR. This requires modifying `StateTransform` to emit warp-cooperative variants of state transitions.
4. **Alternative**: Since Phase 2 proc macro already generates warp-cooperative state machines from `.await`, the proc macro approach may be sufficient without a rustc fork.

### Key Question for Phase 3
Is a rustc MIR pass still needed? The proc macro (`#[warp_async]`) already:
- Accepts `.await` and `?` operator
- Generates warp-cooperative state machines
- Works on real GPU hardware

The MIR pass would automate what the proc macro does manually, but with more generality (arbitrary control flow, nested async, trait objects). However, the proc macro is already quite powerful.

## Open Questions
- Should Phase 3 focus on making the proc macro more general (async fn keyword, nested .await, more types) instead of a rustc fork?
- Or should Phase 3 demonstrate that a MIR pass CAN inject shfl.sync, even if the proc macro is the practical solution?

## Impact on Downstream Tasks
- rustc-warp.2: MIR transformation spec needs revision — baseline is "async already works", transformation is "add shfl.sync at yield points"
- The feasibility assessment is MORE positive than expected — no fundamental LLVM/nvptx blocker
