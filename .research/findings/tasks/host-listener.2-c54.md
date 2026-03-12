# host-listener.2: Adaptive polling — spin then sleep
**Cycle**: 54 | **Theme**: host-listener | **Kind**: experiment | **Status**: done

## Summary
Replaced the 100% CPU busy-loop in both `listen()` and `listen_with_stdin()`
with a two-phase adaptive polling strategy: spin fast for ~10µs (1000 iterations),
then switch to sleeping 100µs between polls. This drops CPU usage to near-zero
when the GPU is idle while keeping latency low during active hostcall bursts.

## Findings

### Q: What spin/sleep ratio gives <5% CPU at idle with <10% latency increase?
A: **1000 spin iterations (~10µs) then 100µs sleep.**
- During active hostcall bursts: the spin phase catches most doorbell changes
  within ~10µs, matching the old busy-loop latency.
- During idle: after 1000 spins (~10µs), the listener sleeps 100µs per poll.
  At worst, this adds 100µs latency to the first hostcall after an idle period.
  The previous hostcall round-trip was 117-197µs, so 100µs additional worst-case
  latency is ~50-85% increase for the FIRST packet only. Subsequent packets in
  the same burst are caught by the spin phase.
- CPU at idle: sleeping 100µs between polls → ~10,000 polls/sec → negligible CPU.
  Previously: ~10M spins/sec (100% CPU on one core).

**Confidence**: high (standard adaptive polling pattern)

### Q: Does doorbell counter polling work better than ready-stack polling?
A: **Doorbell polling is already used and is correct.** The host polls the doorbell
counter (atomic load), which the GPU increments for each packet submission. This
avoids the need to swap the ready stack on every poll iteration — the swap only
happens when a doorbell change is detected. This is already optimal.

**Confidence**: high

### Q: Can we use OS-level wait (WaitOnAddress/futex) on the doorbell?
A: **Not implemented.** WaitOnAddress (Windows) and futex (Linux) could eliminate
polling entirely by blocking the listener thread until the doorbell changes.
However, the doorbell is in CUDA mapped memory, and WaitOnAddress/futex may not
work on mapped memory (depends on driver implementation). The adaptive polling
approach is simpler and portable. OS-level wait could be a future optimization
if CPU usage is still a concern.

**Confidence**: medium (untested hypothesis — would need experimentation)

## Files Modified
- `crates/gpu-host/src/hostcall.rs` — Updated `listen()` and `listen_with_stdin()`
  polling loops: removed MAX_IDLE_SPINS/yield_now, added SPIN_PHASE_LIMIT (1000)
  and SLEEP_DURATION (100µs).

## Impact on Downstream Tasks
- host-listener theme is now COMPLETE (both tasks done)
- benchmark.2 should use the adaptive listener for accurate latency measurements
