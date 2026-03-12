# gpu-compute.2: Prototype GPU-driven multi-step compute without host orchestration
**Cycle**: 121 | **Theme**: gpu-compute | **Kind**: experiment | **Status**: done

## Summary

Built and verified a GPU autonomous multi-step compute pipeline using `#[warp_async]` with full control flow (match + if/else + sequential hostcalls). Three pipeline modes demonstrate GPU-driven file I/O with conditional branching based on hostcall results — all without host orchestration between steps. The `#[warp_async]` macro replaced what previously required 150+ lines of hand-written state machine code.

## Findings

### Q: Can the GPU autonomously decide what to compute next based on intermediate results?
A: Yes. The `autonomous_pipeline` kernel demonstrates three patterns:
1. **Sequential pipeline** (Mode 0): create file → write data → close. GPU executes 3 hostcall steps autonomously.
2. **Conditional pipeline** (Mode 1): open file → read → close → branch on bytes_read. GPU reads 21 bytes, evaluates `n > 10` on lane 0, broadcasts decision to all 32 lanes, takes "large-payload" path. Zero host involvement in the decision.
3. **Roundtrip verification** (Mode 2): write file → close → reopen → read → verify. GPU creates a file with "verify-me" (9 bytes), re-opens it, reads back, confirms `nb > 0`, reports "roundtrip-ok". 6 hostcall steps with GPU-decided verification.

The key pattern: hostcall return values (`fd`, `bytes_read`) are stored as struct fields by the proc macro, available in subsequent decision states. The GPU evaluates conditions on lane 0 and broadcasts via `shfl.sync.idx.b32`.
**Confidence**: high

### Q: How does warp-cooperative async enable autonomous compute pipelines?
A: `#[warp_async]` with match/if-else/loop generates a WarpFuture state machine where:
- **match** on mode → GPU selects pipeline at launch (MATCH_DECISION state)
- **if/else** on hostcall result → GPU branches based on dynamic data (DECISION state)
- **Sequential hostcalls** within match arms → GPU drives multi-step I/O
- All 32 warp lanes maintain convergence via broadcast from lane 0

The proc macro handles variable scoping: each `let x = warp_open!(...)` becomes a struct field accessible in later states. Cross-arm variable uniqueness is required (`fd` in arm 0, `rfd` in arm 1).

Ergonomic comparison:
- Hand-written BranchingPipelineFuture: 15 states, 150+ lines, manual state constants
- `#[warp_async]` autonomous_pipeline: ~40 lines, auto-generated state machine

**Confidence**: high

## Changes Made
- **crates/gpu-kernel/src/lib.rs**: Added `autonomous_pipeline` kernel using `#[warp_async]` with match (3 arms) + if/else + 13 total hostcall steps
- **crates/gpu-host/src/main.rs**: Added `run_autonomous_pipeline_test()` with 3 modes, file content verification
- **crates/gpu-host/kernel.ptx**: Rebuilt with new kernel
- **.gitignore**: Added test file patterns (gpu_autonomous.txt, gpu_roundtrip.txt, branch_test.txt)

## Verification
- All 3 modes pass on GPU hardware (RTX 3090)
- Mode 0: File content verified ("GPU-autonomous-output")
- Mode 1: GPU correctly classifies 21 bytes as "large-payload" (> 10)
- Mode 2: Roundtrip write-then-read verified as "roundtrip-ok"
- clippy + fmt pass for gpu-host
- gpu-kernel compiled successfully for nvptx64

## Open Questions
1. Can `#[warp_async]` support mutable local state across warp calls? Currently only function params and warp-call return values persist across states.
2. Performance comparison: generated state machine vs hand-written. Register pressure?
3. Can this pattern scale to 100+ state machines (e.g., transformer layer pipeline)?
