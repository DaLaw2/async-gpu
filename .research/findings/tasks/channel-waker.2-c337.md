# channel-waker.2: Implement waker-based wake-on-send for channels
**Cycle**: 337 | **Theme**: channel-waker | **Kind**: experiment | **Status**: done

## Summary
Implemented real GPU waker integration using `RawWakerVTable`. The executor now creates per-task Wakers that pack `(WorkQueue*, task_id)` into a single pointer. When a sender writes to an MPSC channel, it calls `wake_consumer()` which reconstructs the Waker and re-enqueues the parked task. Added SLOT_PARKED state for tasks that returned Pending, and a backward-compatible fallback for legacy futures (oneshot) that don't store wakers.

## Findings
### Q: How does sender wake the receiver task?
A: Sender calls `channel.wake_consumer()` after successful `try_send()`. This copies the 16-byte Waker from channel's `waker_bytes` field, reconstructs it, and calls `.wake()`. The wake implementation (`gpu_waker_wake_impl`) unpacks the WorkQueue pointer and task_id from the data pointer, then CAS-transitions the task from PARKED→QUEUED and enqueues it.
**Confidence**: high

### Q: How is waker data packed into a single pointer?
A: Bottom 8 bits store task_id (0-255), upper bits store WorkQueue pointer. CUDA guarantees 256-byte alignment for allocations, so bottom 8 bits are always zero. `pack_waker_data(queue_ptr, task_id) = queue_ptr | task_id`. `unpack_waker_data(data) = (data & !0xFF, data & 0xFF)`.
**Confidence**: high

### Q: How to handle legacy futures that don't store wakers?
A: Fallback re-poll: when executor's queue is empty but tasks remain active (spawned != completed), it iterates all slots and CAS-transitions any PARKED→QUEUED tasks back into the queue. This ensures oneshot receiver futures (which return Pending without storing a waker in the channel) eventually get re-polled.
**Confidence**: high

## Unexpected Discoveries
- `RawWaker` is NOT `#[repr(C)]` — cannot cast to extract fields. Solution: copy full 16-byte Waker via `core::ptr::copy_nonoverlapping` into channel storage.
- NOOP_VTABLE and noop_waker became dead code after switching to per-task wakers — removed.
- Single poll per task (no inner spin-loop) works correctly with the PARKED state.

## Open Questions
- Could waker-based wake eliminate the fallback entirely if oneshot also stores wakers? (Low priority — fallback works fine)

## Impact on Downstream Tasks
- **channel-demo.1**: Unblocked. Can now build producer-consumer demo using waker-driven MPSC.
- **channel-mpsc theme**: Effectively complete (MpscRecvFuture is waker-aware).
- **channel-waker theme**: Both success criteria met.
