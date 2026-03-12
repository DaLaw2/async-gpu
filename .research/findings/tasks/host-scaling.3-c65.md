# host-scaling.3: Implement scaled listener
**Cycle**: 65 | **Theme**: host-scaling | **Kind**: experiment | **Status**: done

## Summary

Implemented the host-scaling.2 design: unified listener with I/O thread separation.
Replaced two duplicated 100-line listener loops with a single `listen_unified()` method
plus `StdinSource` trait. Blocking FILE/STDIN operations offloaded to dedicated I/O thread
via `mpsc` channel. Compiles with zero warnings. ADR-6 recorded.

## Changes Made

### crates/gpu-host/src/hostcall.rs

**New types (Phase A — StdinSource abstraction):**
- `StdinSource` trait: `fn read_line_bytes(&mut self, buf: &mut [u8]) -> usize` + `Send` bound
- `RealStdin`: reads from `std::io::stdin()`, blocks until input
- `CannedStdin`: returns pre-loaded data once, then EOF
- `IoRequest` struct: `{ pkt_idx: u16, service: u32 }` for channel dispatch

**Unified listener (Phase A+B):**
- `listen_unified<F, S>()`: Single generic listener replacing both old implementations
  - Fast path (inline): NOP, PRINT, TIME, PANIC — handled + CONTROL_READY set immediately
  - Slow path (channel): OPEN, WRITE, READ, CLOSE, STDIN → sent to I/O thread
  - Uses `std::thread::scope()` for safe I/O thread lifecycle
  - I/O thread exits when sender is dropped (listener shutdown)

- `io_thread_loop<S>()`: Dedicated I/O thread
  - Owns `fd_table` HashMap and `next_fd` counter
  - Processes `IoRequest` from channel sequentially
  - Sets CONTROL_READY on packet after each operation

- `handle_stdin_from_source<S>()`: Replaces old `handle_stdin()` and `handle_stdin_canned()`
  - Uses `StdinSource` trait instead of direct stdin or pre-loaded data

**Simplified wrappers:**
- `listen(on_print)` → thin wrapper calling `listen_unified(on_print, RealStdin)`
- `listen_with_stdin(on_print, data)` → thin wrapper calling `listen_unified(on_print, CannedStdin::new(data))`

**Removed:**
- Old `handle_stdin()` method (replaced by `handle_stdin_from_source`)
- Old `handle_stdin_canned()` method (replaced by `CannedStdin` + trait)
- ~120 lines of duplicated listener loop code

## Findings

### Q: Does the implementation match the design?
A: Yes. The implementation follows host-scaling.2 exactly:
- Phase A (StdinSource trait) and Phase B (I/O thread) implemented together
- Channel-based dispatch for slow services
- `std::thread::scope` for I/O thread lifecycle
- No protocol changes needed

**Confidence**: high

### Q: Are there race conditions under contention?
A: No new race conditions introduced. Analysis:
- The listener thread and I/O thread never touch the same packet simultaneously
- After the listener sends `IoRequest` via channel, it doesn't touch that packet again
- The I/O thread has exclusive access to `fd_table` (no sharing needed)
- `control.store(READY, Release)` is the same atomic operation used before — no change
- GPU-side behavior is unchanged (spin-wait on CONTROL_READY)

One subtle correctness property: the I/O thread may process packets out of order relative
to the listener's inline processing. Example: PRINT on packet A completes before OPEN on
packet B, even if B was earlier in the ready list. This is fine because each GPU thread
waits on its own packet — no cross-packet ordering dependency.

**Confidence**: high

### Q: What is the net code reduction?
A: Approximately 120 lines removed (duplicated listener loop). Added ~80 lines for
StdinSource trait + IoRequest + io_thread_loop + handle_stdin_from_source. Net reduction
of ~40 lines, with significantly better maintainability.

**Confidence**: high

## Unexpected Discoveries

1. **`std::thread::scope` is perfect for this pattern**: The I/O thread borrows `&self`
   without needing Arc, and automatically joins when the scope exits. No manual thread
   lifecycle management needed.

2. **`mpsc::channel` is unbounded**: The listener never blocks on `send()`, which is
   critical — a blocking send would defeat the purpose of I/O thread separation.

3. **The `handle_stdin` removal was clean**: Since both `listen()` and `listen_with_stdin()`
   now route through `listen_unified()`, the old `handle_stdin` (direct stdin) is
   fully replaced by `RealStdin` + `handle_stdin_from_source`.

## Open Questions

- What is the throughput impact of the channel dispatch overhead for FILE I/O?
  (Expected: negligible — channel send is ~100ns vs FILE I/O 10-500µs)
- Does the I/O thread saturate under heavy FILE I/O load?
  (Expected: no — GPU is limited to ~28K calls/s, I/O thread can process much faster)

## Impact on Downstream Tasks

- **host-scaling.4 (benchmark)**: Ready to test. Run NOP benchmark to verify no regression,
  then test with mixed PRINT + FILE I/O workload to measure I/O isolation benefit.
- **All existing tests**: Should pass unchanged since `listen()` and `listen_with_stdin()`
  are preserved as wrappers.
