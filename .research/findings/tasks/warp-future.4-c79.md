# warp-future.4: Hand-Written WarpFuture PoC — Single PRINT Hostcall
**Cycle**: 79 | **Theme**: warp-future | **Kind**: experiment | **Status**: done

## Summary
Implemented and verified a complete WarpFuture stack: `WarpPoll` enum, `WarpContext` struct, `WarpFuture` trait, `WarpExecutor`, and a hand-written `WarpPrintFuture` state machine. All 32 lanes cooperatively build a 44-byte message ("WarpFuture: ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`") via the hostcall protocol. Verified on SM86 hardware — message received correctly, 0.373ms round-trip. The PTX contains 4x `shfl.sync.idx.b32` and 5x `bar.warp.sync`, confirming SIMT convergence by construction.

## Findings

### Q: Can a hand-written WarpFuture state machine compile to convergent PTX?
A: **Yes.** The 3-state machine (INIT→WAIT→DONE) compiles to a `match` on the broadcast discriminant. All branches are warp-uniform because the discriminant is distributed via `shfl.sync.idx.b32` from lane 0. PTX output shows no lane-predicated divergence in the state machine dispatch — all lanes take the same branch.
**Confidence**: high

### Q: Does WarpExecutor correctly drive 32 lanes through yield/resume?
A: **Yes.** The executor's poll loop calls `poll_warp()` on all 32 lanes simultaneously, handles `Pending` with `nanosleep` yield, and converges lanes with `bar.warp.sync` after each iteration. The WarpPrintFuture transitions INIT→Pending→WAIT→(spin)→Ready in exactly 2 polls (first poll does INIT, subsequent polls spin in WAIT until host responds).
**Confidence**: high

### Q: Is there measurable throughput improvement vs per-thread Future?
A: **The comparison is nuanced.** WarpFuture round-trip: 0.373ms. Per-thread hostcall (single thread): ~1ms. But this isn't a fair comparison — WarpFuture uses 1 packet for 32 lanes vs 1 packet for 1 thread. The real advantage is:
- **1 CAS operation instead of 32** (lane 0 handles all packet management)
- **Coalesced payload writes** (32 lanes write simultaneously to contiguous memory)
- **1 doorbell ring** instead of 32
- **Zero divergence** during spin-wait (all lanes read same control word)

For warp-future.2's measurement (32 threads doing independent hostcalls): 1040µs mean latency, 3.31 CAS retries/call. WarpFuture: 373µs total, 0 CAS retries for lanes 1-31 (only lane 0 does CAS). This is a **~3x improvement** in wall-clock time for the same amount of work (32 threads each sending a message).
**Confidence**: medium (needs more rigorous benchmarking for statistical significance)

## Implementation Details

### New module: `gpu_runtime::warp_future`
- `WarpPoll<T>`: `Ready(T) | Pending`
- `WarpContext`: `active_mask: u32, lane_id: u32, is_leader() -> bool`
- `WarpFuture` (unsafe trait): `poll_warp(&mut self, wcx: &mut WarpContext) -> WarpPoll<T>`
- `WarpExecutor::run<F: WarpFuture>(future: &mut F) -> F::Output`
- `broadcast_u32(mask, val) -> u32`: convenience wrapper for lane-0 broadcast

### WarpPrintFuture state machine
```
STATE_INIT (0):
  - lane 0: pop free packet
  - shfl broadcast pkt_idx to all lanes
  - all lanes write their character to payload (coalesced)
  - syncwarp
  - lane 0: fill header, FILLED, push ready, doorbell
  - lane 0: state = WAIT
  - syncwarp
  - return Pending

STATE_WAIT (1):
  - all lanes: spin_load_acquire control word (convergent — same address)
  - if READY: lane 0 releases packet, state = DONE
  - return Ready(true) or Pending

STATE_DONE (2):
  - return Ready(true)
```

### PTX verification
- 4x `shfl.sync.idx.b32` — state broadcast + pkt_idx broadcast (init + wait states)
- 5x `bar.warp.sync` — convergence barriers at state transitions and after payload writes
- No lane-divergent branches in state dispatch

### Hardware verification
- Target: SM86 (RTX 3060)
- Launch: 1 block × 32 threads
- Message received: "WarpFuture: ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`"
- Result: 1 (success)
- Elapsed: 0.373ms

## Open Questions
1. How does WarpFuture compose with multiple sequential hostcalls? (need multi-state PoC)
2. Can warp-cooperative CAS alone (without the full WarpFuture trait) achieve similar benefits?
3. What is the register pressure of WarpFuture state machines vs per-thread futures?

## Impact on Downstream Tasks
- **warp-future.5** (proc macro): Now has a concrete reference implementation to target. The state machine pattern (broadcast + match + per-state logic + syncwarp) is mechanical enough for proc macro generation.
- **ADR-9**: Can now be updated with empirical validation data. The WarpFuture concept is proven on hardware.
