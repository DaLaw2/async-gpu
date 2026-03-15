# warp-batch-io.1: ADR-10 design + current hostcall Future patterns
**Cycle**: 350 | **Theme**: warp-batch-io | **Kind**: investigation | **Status**: done

## Summary

Investigated the existing warp-batching design, current hostcall Future implementations,
and warp-cooperative primitives. The protocol is already designed to support 32-lane
batching — packet layout has 32 lanes x 8 slots. The change is in the Future poll
implementation: lane 0 submits/spins, others wait for shfl.sync broadcast.

## Findings

### Q: What does the ADR-10 warp-batching design specify?
A: ADR-10 is actually the Hybrid Executor pattern (WarpFuture state machines), NOT
warp-batching. The warp-batch-io optimization is a separate parked theme. The design
concept is: lane 0 submits one hostcall packet for all 32 lanes, reducing CAS contention
by 32x on uniform I/O workloads.
**Confidence**: high

### Q: How do current Futures submit hostcall packets?
A: Located in `crates/kernel/gpu-runtime/src/std_future.rs`. Each Future type follows
this pattern:
1. **Init state**: Call `gpu_hostcall_request()` which does CAS on Treiber stack to pop
   a free packet. Fill packet header (active_mask, service, control) and payload. Push
   to ready stack. Ring doorbell.
2. **Waiting state**: Spin on `control` word via `sys_spin_load_acquire_u32()` checking
   for CONTROL_READY. Read response from payload slots.
3. **Done state**: Return result.

Currently EVERY lane independently pops a packet = 32 CAS operations per warp.
**Confidence**: high

### Q: Where is lane-0 election and shfl broadcast inserted?
A: The primitives already exist in `crates/kernel/gpu-runtime/src/warp_cooperative.rs`:
- `lane_id()` → PTX `%laneid`
- `activemask()` → PTX `activemask.b32`
- `shfl_sync_idx_u32(mask, val, src_lane)` → PTX `shfl.sync.idx.b32`
- `syncwarp(mask)` → warp barrier

There's also `warp_poll_future()` which already implements the lane-0-polls pattern.

For batched hostcall, the insertion points are:
1. **Init**: `if lid == 0 { pop_free + fill_packet }` then `shfl_sync(pkt_idx)`
2. **Waiting**: `if lid == 0 { spin on control }` then `shfl_sync(result_fields)`
3. **Cleanup**: `if lid == 0 { push_free }` then `syncwarp(mask)`
**Confidence**: high

### Q: What changes in gpu-protocol?
A: NONE needed. The packet layout already supports 32-lane payloads. The protocol
is backward compatible — the host side doesn't need to know whether 1 lane or 32
lanes submitted the packet. The `active_mask` field already indicates which lanes
participated.
**Confidence**: high

## Key Design Decisions for Implementation

1. **Uniform-only batching**: Start with the case where all 32 lanes want the same
   operation (e.g., all open the same file). Divergent operations fall back to per-lane.
2. **Broadcast via shfl**: Use `shfl_sync_idx_u32` for each u32 of the response. For
   a file descriptor (i32), one shfl is enough. For larger responses (read data), need
   multiple shuffles.
3. **Error handling**: Lane 0 broadcasts error code. All lanes see the same error.
4. **Packet reuse**: The packet header's `active_mask` reflects which lanes are active.
   For batched mode, set to `activemask()` value.

## Impact on Downstream Tasks

- **warp-batch-io.2**: Clear implementation path — modify GpuOpenFuture first (simplest
  case: all lanes open same file, returns i32 fd)
- **warp-batch-io.3**: Extend to read/write/close with data broadcast
- **warp-batch-io.4**: Benchmark by comparing single-warp I/O throughput
