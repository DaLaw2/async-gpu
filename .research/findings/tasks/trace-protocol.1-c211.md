# trace-protocol.1: Define trace event format + SERVICE_TRACE in gpu-protocol
**Cycle**: 211 | **Theme**: trace-protocol | **Kind**: design | **Status**: done

## Summary
Defined structured trace event protocol with SERVICE_TRACE (13) and SERVICE_ASSERT (14) in gpu-protocol. Implemented GPU-side `gpu_hostcall_trace()`, `gpu_hostcall_assert()`, `gpu_trace!()` macro, and `gpu_assert!()` macro in gpu-runtime. Implemented host-side `handle_trace()` and `handle_assert()` in gpu-host.

## Findings

### Q: What trace event format enables structured GPU observability?
A: Two-service design:

**SERVICE_TRACE (13)** — Structured trace events:
- Slot 0: metadata (threadIdx:16 | blockIdx:16 | level:8 | msg_len:8 | lane_id:16)
- Slot 1: `%clock64` timestamp (u64)
- Slots 2-7: message bytes (up to 48 bytes)
- Fire-and-forget: host ACKs but no response data used

**SERVICE_ASSERT (14)** — Assertion failure diagnostic:
- Slot 0: metadata (same as PANIC: threadIdx:16 | blockIdx:16 | msg_len:16)
- Slots 1-7: message bytes (up to 56 bytes)
- After host ACK, GPU executes PTX `trap` (divergent `-> !`)

**Confidence**: high

### Q: How do the GPU-side macros work?
A: Both macros use `PanicBuf` (stack-allocated 128-byte formatter) for message formatting:
- `gpu_trace!(buf, LEVEL, "fmt", args...)` — calls `gpu_hostcall_trace`, returns `Result`
- `gpu_assert!(buf, cond, "fmt", args...)` — on failure calls `gpu_hostcall_assert` which traps
- `gpu_assert!(buf, cond)` — simple form without format args

**Confidence**: high

### Q: How does the host display trace/assert events?
A: Color-coded ANSI output to stderr:
- TRACE: `[GPU DEBUG/INFO/WARN/ERROR] B{block}.T{thread} @{timestamp}: {msg}`
- ASSERT: `[GPU ASSERT FAILED] block={block} thread={thread}: {msg}`
- Colors: DEBUG=cyan, INFO=green, WARN=yellow, ERROR=bold red

**Confidence**: high

## Files Modified
- `crates/gpu-protocol/src/lib.rs` — SERVICE_TRACE, SERVICE_ASSERT, encode/decode functions
- `crates/gpu-runtime/src/lib.rs` — gpu_hostcall_trace, gpu_hostcall_assert, gpu_trace!, gpu_assert! macros
- `crates/gpu-host/src/hostcall.rs` — handle_trace, handle_assert handlers

## Open Questions
- Event ordering across SMs: `%clock64` is per-SM, not globally ordered. May need `%globaltimer` for cross-SM ordering.
- Backpressure: if trace events flood the hostcall pool, threads will spin-wait. May need drop policy.
