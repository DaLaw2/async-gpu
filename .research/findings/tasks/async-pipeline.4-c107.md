# async-pipeline.4: Pipelined I/O + Compute — Overlap Hostcall with FMA Work
**Cycle**: 107 | **Theme**: async-pipeline | **Kind**: experiment | **Status**: done

## Summary
Implemented and hardware-verified an 8-state WarpFuture that overlaps hostcall I/O with FMA compute. While waiting for a PRINT hostcall response, the GPU executes FMA iterations (100 per poll). Hardware test confirmed ~17,000 iterations completed during a single hostcall round-trip, proving that WarpFuture can productively use I/O latency for compute.

## Findings

### Q: How to overlap hostcall I/O with compute in WarpFuture states?
A: Use a dedicated PP_COMPUTE_WHILE_IO state that performs bounded FMA work (100 iterations per poll), then does a **non-blocking** check of CONTROL_READY. If the host has not responded, return WarpPoll::Pending and re-enter the compute state on next poll. If CONTROL_READY is set, transition to the next I/O state.

Key pattern:
```rust
PP_COMPUTE_WHILE_IO => unsafe {
    // Bounded FMA work — 100 iterations per poll
    let mut i = 0u32;
    while i < 100 {
        self.acc = self.acc * 1.00001f32 + 0.00001f32;
        i += 1;
    }
    if wcx.is_leader() {
        self.iterations += 100;
    }
    // Non-blocking I/O completion check
    let pkt = self.buf.add(pkt_off);
    let ctrl = sys_load_acquire_u32(pkt.add(PKT_OFF_CONTROL) as *const u32);
    if ctrl & CONTROL_READY != 0 {
        // I/O done — release packet, transition to next state
        gpu_runtime::hostcall::gpu_hostcall_release(self.buf, pkt);
        self.state = PP_PRINT_DONE;
    }
    WarpPoll::Pending
},
```
**Confidence**: high (hardware verified — ~17,000 iterations during round-trip)

### Q: What are the sideband coherence constraints for pipelined operations?
A: None beyond the existing hostcall protocol. The sideband buffer is only used during bulk_read/bulk_write operations. PRINT uses packet payload only. For pipelined bulk I/O + compute, the sideband must not be reused until the I/O completes (check CONTROL_READY before starting new bulk operation on same sideband region).
**Confidence**: high

### Q: What latency improvement does pipelining achieve vs sequential?
A: Not a latency improvement per se — the hostcall round-trip is the same. The improvement is **throughput**: ~17,000 FMA iterations that would otherwise be wasted on spin-waiting are now productive. This is the GPU equivalent of CPU instruction-level parallelism — hiding I/O latency with compute.
**Confidence**: high (hardware verified)

## Unexpected Discoveries
- 100 FMA iterations per poll is a good balance: small enough to keep I/O completion latency low (sub-microsecond), large enough to amortize the poll overhead. The GPU executed ~170 polls during one hostcall round-trip.
- The FMA accumulator must be stored in the WarpFuture struct (`self.acc`) to persist across polls. This is the natural "register" for pipelined state.

## Open Questions
- What is the optimal iterations-per-poll count for different compute/IO ratios?
- Can multiple sideband regions be used for double-buffered pipelining?

## Impact on Downstream Tasks
- Documents the pipelining pattern for WarpFuture
- Shows that WarpFuture polling model naturally supports compute/IO overlap
