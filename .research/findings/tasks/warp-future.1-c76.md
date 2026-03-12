# warp-future.1: Feasibility Analysis — WarpFuture Trait + Proc Macro
**Cycle**: 76 | **Theme**: warp-future | **Kind**: investigation | **Status**: done

## Summary
Deep brainstorm (bs16, proposer + skeptic) confirmed that WarpFuture is architecturally feasible without rustc changes. The approach uses a custom `WarpFuture` trait with uniform state discriminant broadcast via `shfl.sync.idx.b32`, `bar.warp.sync` for convergence guarantees, and a simple spin-poll executor. Key constraint: the programming model restricts control flow to SIMD-style (same path, different data). The problem's severity is currently unmeasured — empirical validation is required before full investment.

## Findings

### Q: Can Rust async state machines be made warp-convergent without rustc changes?
A: **Yes, with constraints.** Standard `async/await` syntax cannot be used directly (compiler-generated state machines embed per-instance discriminant). Instead, a custom `WarpFuture` trait with manual or proc-macro-generated state machines is needed. Lane 0 drives state transitions; all lanes receive the discriminant via `shfl.sync.idx.b32`. This is a syntax ergonomics tradeoff, not a fundamental impossibility.
**Confidence**: high

### Q: What does the WarpFuture trait signature look like concretely?
A: `unsafe trait WarpFuture { type Output; fn poll_warp(&mut self, wcx: &mut WarpContext) -> WarpPoll<Self::Output>; }` — where `WarpContext` provides `lane_id` and `active_mask` (no Waker needed), and `WarpPoll` is `Ready(T) | Pending`. The trait is `unsafe` because implementations must maintain warp convergence.
**Confidence**: high

### Q: How does `__syncwarp()` interact with async yield points?
A: `bar.warp.sync mask` ensures all lanes reach the same point before proceeding. Cost: 0-2 cycles when already converged (effectively free for WarpFuture since convergence is maintained by construction). **Critical constraint**: `__syncwarp()` in conditional branches causes deadlock if not all lanes enter the branch. WarpFuture forbids per-lane control flow divergence — conditional operations must use predication.
**Confidence**: high

### Q: What are the constraints on sub-futures for warp convergence?
A: All sub-futures must maintain the SIMD contract: same control flow path, different data. No per-lane `if/else` around yield points. Error handling (`?` operator) that causes per-lane early return is incompatible. These constraints make WarpFuture suitable for data-parallel GPU workloads but not arbitrary Rust async patterns.
**Confidence**: high

### Q: How does the WarpExecutor differ from per-thread Embassy executor?
A: Dramatically simpler. No run queue (single WarpFuture per warp), no waker infrastructure (synchronous spin-poll), no critical section, no TaskStorage. Just a loop calling `poll_warp()` until `Ready`. All 32 lanes participate in every poll. Expected register savings: 5-15 registers per thread vs Embassy.
**Confidence**: high

## Unexpected Discoveries
1. The existing hostcall packet layout (32 lanes x 8 slots x 8 bytes = 2048 bytes) is a perfect fit for warp-level hostcalls — each lane writes its own slot region with coalesced memory access.
2. WarpFuture reduces free-stack CAS contention by 32x (one allocation per warp vs per lane).
3. The skeptic raised valid concerns: no existing kernel in this project actually runs per-thread async across a full warp, so the divergence problem is entirely hypothetical and unmeasured.

## Open Questions
1. Do `shfl.sync.idx.b32` and `bar.warp.sync` work correctly from inline PTX asm on nvptx64? (Must-test prerequisite)
2. What is the actual SIMT efficiency of per-thread futures across a full warp? (Determines whether WarpFuture is justified)
3. Would simpler alternatives (warp-cooperative CAS, `__syncwarp()` between executor polls) capture most of the benefit?

## Impact on Downstream Tasks
- **warp-future.2** (design): Unblocked. Focus on minimal trait + executor + warp-cooperative hostcall submit.
- **warp-future.3** (implementation): Must first verify `shfl.sync` and `bar.warp.sync` intrinsics in gpu-atomics crate.
- **warp-future.4** (proc macro): Deferred — only after hand-written PoC demonstrates measurable improvement.
- **ADR-9**: Remains "proposed" until empirical validation.
