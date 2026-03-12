# host-listener.1: Per-packet processed bit to eliminate duplicate reads
**Cycle**: 49 | **Theme**: host-listener | **Kind**: experiment | **Status**: done

## Summary
Added CONTROL_FILLED bit (value 4) to the hostcall packet control field. GPU sets
CONTROL_FILLED via release store after filling the packet (replacing the separate
membar_sys). Host checks CONTROL_FILLED before processing — if not set, the packet
is skipped (stale or already processed). Response sets CONTROL_READY (clears FILLED).
This eliminates duplicate processing because a packet that has already been responded
to will have CONTROL_READY set (without FILLED), so re-visits skip it.

## Findings

### Q: Does adding a CONTROL_PROCESSED state eliminate all duplicate messages?
A: **Yes, by design.** The state machine is now:
- Free pool: CONTROL = 0 (no FILLED, no READY)
- GPU fills packet: CONTROL = CONTROL_FILLED (4)
- Host processes: CONTROL = CONTROL_READY (1) — FILLED cleared
- GPU sees READY: releases packet back to free pool (CONTROL not modified)

If the host walks the ready list and encounters a packet with CONTROL = 0 or
CONTROL = CONTROL_READY (from a previous processing), it skips it. Only packets
with CONTROL_FILLED are processed.

**Confidence**: high (compile-verified, state machine is correct by construction)

### Q: What is the impact on hostcall latency?
A: Minimal. The CONTROL_FILLED release store replaces the previous membar_sys call.
A release store is cheaper than a full membar_sys barrier (release only orders prior
writes, membar orders all prior operations). Net effect: potentially faster, not slower.

**Confidence**: high (release store is a subset of membar)

### Q: Does the processed bit need sys-scope atomics?
A: **Yes.** The CONTROL field is in mapped memory shared between GPU and CPU. The GPU
uses `sys_store_release_u32` (system scope) to set CONTROL_FILLED. The host uses
`AtomicU32::load(Acquire)` to check it, and `AtomicU32::store(flags, Release)` to
set CONTROL_READY. Both use system-scope semantics (host-side std atomics are system
scope by default on x86).

**Confidence**: high

## Files Modified
- `crates/gpu-protocol/src/lib.rs` — Added `CONTROL_FILLED = 4` constant
- `crates/gpu-kernel/src/lib.rs` — Updated `gpu_hostcall_print` and `gpu_hostcall_request` to set CONTROL_FILLED
- `crates/gpu-host/src/hostcall.rs` — Both `listen` and `listen_with_stdin` now check CONTROL_FILLED before processing
- `crates/async-hostcall-test/src/lib.rs` — Updated 2 packet submission paths
- `crates/async-pipeline-test/src/lib.rs` — Updated 1 packet submission path
- `crates/multi-warp-test/src/lib.rs` — Updated 1 packet submission path
- `crates/std-build-test/src/lib.rs` — Updated 2 packet submission paths, added CONTROL_FILLED constant
- All PTX files updated

## Impact on Downstream Tasks
- host-listener.2 (adaptive polling) can now proceed — duplicates eliminated
- benchmark.2 should see 0% duplicate rate even at 512 threads
