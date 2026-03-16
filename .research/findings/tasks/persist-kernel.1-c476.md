# persist-kernel.1: Persistent kernel design
**Cycle**: 476 | **Theme**: persist-kernel | **Kind**: investigation | **Status**: done

## Summary
Existing HostcallSession already supports persistent kernel patterns. Lock-free Treiber stacks
with ABA-tagged pointers handle concurrent work dispatch. TDR avoided via periodic SERVICE_NOP
heartbeats. Design: simple mapped memory work queue with GPU polling.

## Findings
### Q: How to implement persistent kernel work dispatch?
A: Use mapped memory work queue (not hostcall protocol):
- Host writes `WorkItem { fn_id: u32, arg: f32, status: AtomicU32 }` to mapped buffer
- GPU block polls `status` for READY flag, executes work, writes result, sets DONE flag
- Host reads result, resets status for next item
- Simple producer-consumer with atomic status flags
**Confidence**: high

### Q: TDR avoidance?
A: On Linux, TDR timeout is very long. Embed periodic SERVICE_NOP heartbeat for Windows compat.
GPU spin with nanosleep (64ns) between polls. Existing `gpu_hostcall_request` already handles
10M spin limit (~640ms).
**Confidence**: high

## Design Decision
Mapped memory work queue with 3 states: FREE → READY → DONE.
Host pushes items (READY), GPU pops (DONE), host reads result (FREE).
Each work item: 64 bytes (fn_id, n_args, args[8], result, status).
