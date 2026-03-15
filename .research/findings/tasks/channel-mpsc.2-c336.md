# channel-mpsc.2: Implement Mpsc<T> sender + receiver on GPU
**Cycle**: 336 | **Theme**: channel-mpsc | **Kind**: experiment | **Status**: done

## Summary
Implemented MPSC (multi-producer, single-consumer) channel for GPU async tasks.
Lock-free ring buffer with CAS-based head reservation, sequence-number publication,
and single-consumer tail advancement. Verified on real GPU with 3 producers + 1 consumer.

## Implementation

### Data Structure
- `MpscChannel<T, N>`: ring buffer with `head` (CAS-contended), `tail` (consumer-only), `closed` flag
- `MpscSlot<T>`: per-slot sequence number for publication ordering + value storage
- N must be power of 2 (bitmask modulo)

### Protocol
- **Producer**: load head → check full → CAS head → write value → release-store sequence
- **Consumer**: load tail → check sequence → read value → recycle slot → advance tail
- **Atomics**: all system-scope (cross-SM visibility)

### API
- `MpscChannel::try_send(value)` — direct send on channel (used by kernel futures)
- `MpscChannel::try_recv()` — direct receive on channel
- `MpscSender` / `MpscReceiver` wrapper types with `mpsc()` factory
- `MpscRecvFuture` — Future adapter for receiver
- `MpscSendError::Closed` / `MpscSendError::Full` error variants

### Test Results
```
spawned=4 completed=4 tasks_executed=4 polls_total=4
received: sum=312 count=12  success=1
```
- 3 producers each sent 4 values: [10,20,30,40], [11,21,31,41], [12,22,32,42]
- Consumer received all 12 values, sum = 312 (correct)

### Key Design Decisions
1. **Direct channel methods**: Added `try_send`/`try_recv` on `MpscChannel` itself, not just on sender/receiver wrappers. This allows kernel code to use the channel directly via raw pointer without needing to construct wrapper types.
2. **Spawn order matters**: With noop waker executor, consumer must be spawned AFTER producers. The executor's FIFO dequeue + MAX_POLLS_PER_TASK limit means early-spawned tasks that return Pending too many times get dropped.
3. **Capacity 16**: Small ring buffer (16 slots) sufficient for demo; parameterized via const generic.

## Impact on Downstream Tasks
- channel-waker.2 can now add waker integration to make spawn order irrelevant
- channel-demo.1 can build producer-consumer pipeline using MPSC
