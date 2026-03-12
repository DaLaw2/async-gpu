# async-pipeline.3: Branching Pipeline — Conditional State Transitions
**Cycle**: 106 | **Theme**: async-pipeline | **Kind**: experiment | **Status**: done

## Summary
Implemented and hardware-verified a 14-state WarpFuture demonstrating conditional state transitions — the key pattern that `#[warp_async]` cannot express. The GPU tries to open a file; if it exists, takes the CLOSE+PRINT branch; if not, takes the CREATE+WRITE+CLOSE+PRINT branch. Both branches verified on GPU hardware in two runs.

## Findings

### Q: How to implement conditional state transitions in WarpFuture?
A: Override `self.state` in the WAIT state based on the hostcall response value. Since state is broadcast from lane 0 via `shfl.sync.idx.b32`, and lane 0 makes the branching decision, all 32 lanes automatically agree on the branch. No special convergence logic needed.

Key code pattern:
```rust
if wcx.is_leader() {
    if has_error {
        self.state = CREATE_BRANCH;
    } else {
        self.fd = fd;
        self.state = EXISTS_BRANCH;
    }
}
```
**Confidence**: high (hardware verified)

### Q: How to detect file-not-found on GPU?
A: Cannot use `FILE_ERROR_SENTINEL` — the host uses `encode_error(category, raw_errno)` which produces small values, NOT u64::MAX. Instead, check the `CONTROL_ERROR` bit in the control word: `(ctrl & CONTROL_ERROR) != 0`. This requires inlining the wait logic instead of using `warp_hostcall_wait_u64`, which hides the control word.
**Confidence**: high (hardware verified)

### Q: Does conditional state transition work correctly with shfl.sync broadcast?
A: Yes. The state is stored by lane 0 and broadcast at the top of each `poll_warp()` call. All lanes see the same state, so they execute the same match arm. Divergence happens only in leader-guarded blocks, which is already the WarpFuture pattern.
**Confidence**: high

## Unexpected Discoveries
- `FILE_ERROR_SENTINEL` (u64::MAX) is NOT what the host returns on error. The host uses `encode_error()` with structured error categories. The GPU-side branching must check `CONTROL_ERROR` bit, not compare payload values. This is an important pattern for any error-handling WarpFuture.

## Open Questions
- Should `warp_hostcall_wait_u64` have a variant that also returns the CONTROL_ERROR flag? This would make error-checking easier without inlining.

## Impact on Downstream Tasks
- Documents a pattern template for any WarpFuture that needs branching
- Confirms `warp_hostcall_submit`/`wait` from gpu-runtime work correctly in hand-written WarpFutures
