# async-pipeline.1: Warp-level hostcall wrappers
**Cycle**: 90 | **Theme**: async-pipeline | **Kind**: experiment | **Status**: done

## Summary
Implemented two general-purpose warp-cooperative hostcall helpers (`warp_hostcall_submit` and `warp_hostcall_wait_u64`) that generalize the existing PRINT-only helpers to support any hostcall service. These form the building blocks for the file transform pipeline demo.

## Findings
### Q: Can we factor out reusable warp-cooperative hostcall patterns?
A: Yes. Two helpers cover the full lifecycle:
1. `warp_hostcall_submit(buf, wcx, service, fill_payload, next_state, state_cell, pkt_idx_cell)` — lane 0 pops packet, fills payload via closure, submits to ready stack. All lanes participate in broadcast.
2. `warp_hostcall_wait_u64(buf, wcx, pkt_idx, next_state, state_cell)` — spin on control word, read u64 response from payload slot 0 (broadcast as two u32 halves), release packet.

**Confidence**: high (verified on hardware via file transform pipeline)

### Q: Memory ordering for warp-cooperative sideband access?
A: Requires care:
- After host writes sideband (BULK_READ): GPU's `sys_spin_load_acquire_u32` on CONTROL_READY ensures visibility. Mapped memory with `read_volatile` is safe.
- After all lanes write sideband (compute): `membar_sys()` per-lane + `syncwarp()` before lane 0 submits BULK_WRITE ensures all writes are globally visible via the `.sys` release on CONTROL_FILLED.

**Confidence**: high

## Open Questions
None — helpers are proven via the 16-state file transform pipeline.

## Impact on Downstream Tasks
- async-pipeline.2 uses these helpers directly (completed together)
