# channel-waker.1: Waker integration — how sender wakes receiver task
**Cycle**: 327 | **Theme**: channel-waker | **Kind**: investigation | **Status**: done

## Summary
Designed a real waker mechanism for the GpuExecutor that replaces `noop_waker()` with
task-aware wakers. When a future returns `Poll::Pending`, the executor parks the task
(removes it from the WorkQueue). When a sender writes a value, it calls the stored waker
which re-enqueues the task into the existing WorkQueue via `sys_cas_u64`. This eliminates
the current spin-poll loop that wastes GPU cycles on `nanosleep.u32 1000` between polls.

## Findings

### Q: How does send() notify the executor that the receiver task is ready?
A: **The sender calls `Waker::wake()`, which re-enqueues the task's slot index into the
executor's WorkQueue.** The flow:

1. Receiver's `poll()` returns `Pending` — it stores `cx.waker().clone()` into the channel slot
2. Executor sees `Pending` and does NOT re-enqueue the task (parks it)
3. Sender calls `send(value)` — writes value, transitions state to SENT, then loads
   the stored waker and calls `waker.wake()`
4. `wake()` implementation calls `WorkQueue::enqueue(task_slot_index)` using the same
   tagged-CAS MPMC enqueue path that `spawn()` uses
5. Next time any warp's `run()` loop dequeues, it picks up the re-enqueued task and polls it
6. This poll sees SENT state and returns `Poll::Ready(Ok(value))`

The waker carries two pieces of data: a pointer to the executor's WorkQueue, and the
task's slot index. These are packed into the `RawWaker`'s data pointer.

**Confidence**: high — this is the standard executor pattern (tokio, embassy, etc.)
adapted for GPU constraints.

### Q: Can we reuse the executor's WorkQueue to re-enqueue woken tasks?
A: **Yes, the existing WorkQueue is a perfect fit.** Key properties that make this work:

1. **MPMC**: Multiple senders can wake different tasks concurrently — the tagged-CAS
   enqueue handles contention correctly
2. **Lock-free**: No mutexes needed — `sys_cas_u64` on the tail pointer is sufficient
3. **Bounded**: MAX_TASKS=256 slots, and each task can only be in the queue once
   (a task is either parked or queued, never both), so the queue cannot overflow
   from waker re-enqueues alone
4. **Same dequeue path**: The executor's `run()` loop already dequeues from this queue —
   woken tasks are picked up with zero changes to the dequeue logic

The only addition needed: `wake()` must call `WorkQueue::enqueue()`, which requires
the waker to hold a pointer to the executor (or specifically to its `work_queue` field).

One subtlety: the current `run()` loop exits when the queue is empty (`head_idx == tail_idx`).
With real wakers, a warp might see an empty queue while tasks are parked waiting for
wakers. The `run()` loop needs a new exit condition: exit when `shutdown` flag is set
AND queue is empty, rather than just when queue is empty. Alternatively, use a
`tasks_active` counter — exit when `tasks_active == 0`.

**Confidence**: high

### Q: Waker storage: where does the receiver store its waker for the sender to call?
A: **Store the waker in the channel slot itself, alongside the value and state fields.**
For the oneshot channel, extend `OneshotSlot<T>`:

```rust
#[repr(C)]
pub struct OneshotSlot<T> {
    state: UnsafeCell<u32>,          // EMPTY=0, SENT=1, CLOSED=2
    waker_data: UnsafeCell<u64>,     // packed (work_queue_ptr_lo, task_slot_idx)
    waker_vtable: UnsafeCell<u64>,   // pointer to RawWakerVTable (or 0 = no waker)
    value: UnsafeCell<MaybeUninit<T>>,
}
```

However, storing a full `Waker` (which is `RawWaker` = data pointer + vtable pointer = 16 bytes)
is wasteful when we know the vtable is always the same. **Better approach: store only the
task slot index (u32) as a "waker token".**

```rust
#[repr(C)]
pub struct OneshotSlot<T> {
    state: UnsafeCell<u32>,           // EMPTY / SENT / CLOSED
    waker_task_id: UnsafeCell<u32>,   // slot index of the waiting task (0xFFFF = none)
    value: UnsafeCell<MaybeUninit<T>>,
}
```

The receiver's `poll()` stores its task slot index (extracted from the waker's data pointer)
into `waker_task_id`. The sender's `send()` reads `waker_task_id`, and if it's not
`0xFFFF`, it calls `work_queue.enqueue(waker_task_id)`.

This avoids storing a full `Waker` object (which would require 16 bytes and a vtable call).
Instead, the channel directly re-enqueues using the task ID. This is safe because:
- The vtable is always `GPU_WAKER_VTABLE` (there's only one executor)
- The "wake" action is always "enqueue task_id into work_queue"
- We just need to know WHICH task to wake (the slot index)

**But the channel needs a pointer to the WorkQueue.** Two options:
1. Store the WorkQueue pointer in the slot too (8 bytes extra)
2. Use a global executor pointer that all channels reference

Option 2 is cleaner: a `static` pointer to the executor, set during `init()`. Since there's
only one `GpuExecutor` per kernel launch, a global is safe. Alternatively, the receiver's
`poll()` can pass the executor pointer when constructing the channel — but that couples the
channel to the executor API.

**Recommended**: Store just `waker_task_id: u32` in the slot. The sender retrieves the
executor reference from a well-known global or from a field stored during channel creation.

**Confidence**: high

### Q: Atomics needed: state transitions + waker handoff without races?
A: **The critical race is between receiver storing the waker and sender calling wake.**
The state machine must ensure exactly one of these outcomes:

1. Receiver stores waker BEFORE sender sends → sender sees waker, calls wake (normal path)
2. Sender sends BEFORE receiver polls → receiver sees SENT on first poll, never stores waker

The existing `state` field already prevents races IF we define the protocol carefully:

**Receiver::poll():**
```
1. Load state (acquire)
2. If SENT → read value, return Ready  (sender already wrote, no waker needed)
3. If CLOSED → return Ready(Err)
4. If EMPTY:
   a. Store waker_task_id (release)     ← "I'm waiting"
   b. Re-load state (acquire)           ← double-check sender didn't race
   c. If still EMPTY → return Pending   ← safe to park
   d. If SENT → read value, return Ready ← sender raced, we caught it
```

**Sender::send(value):**
```
1. Write value (volatile)
2. Store state = SENT (release)         ← value is visible
3. Load waker_task_id (acquire)         ← check if receiver is waiting
4. If waker_task_id != 0xFFFF → enqueue(waker_task_id) into WorkQueue
```

The key insight is the **double-check after storing the waker**. This is the same
"compare-and-check" pattern used in Linux futexes and parking_lot. Without the
double-check, this race exists:

```
Receiver:                    Sender:
  load state → EMPTY
                             write value
                             store state = SENT
                             load waker_task_id → 0xFFFF (no waker yet!)
  store waker_task_id = 42
  return Pending             ← DEADLOCK: sender already checked, won't wake us
```

With the double-check:
```
Receiver:                    Sender:
  load state → EMPTY
                             write value
                             store state = SENT
                             load waker_task_id → 0xFFFF
  store waker_task_id = 42
  re-load state → SENT       ← caught the race!
  read value, return Ready   ← no park, no deadlock
```

Atomics required:
- `sys_store_release_u32` for state transitions and waker_task_id writes
- `sys_load_acquire_u32` for state reads and waker_task_id reads
- No CAS needed on the oneshot slot itself (single producer, single consumer)
- `sys_cas_u64` only needed by the waker to enqueue into WorkQueue (reuses existing code)

For MPSC channels (future work), the waker storage becomes more complex because multiple
senders might race. But for oneshot, the above is sufficient.

**Confidence**: high — double-check pattern is well-established and has been formally
verified in other contexts (futex, parking_lot, crossbeam).

## Design Summary

### RawWakerVTable for GPU

```rust
/// Data pointer layout: bits 63-32 = reserved, bits 31-0 = task slot index
const GPU_WAKER_VTABLE: RawWakerVTable = RawWakerVTable::new(
    |data| RawWaker::new(data, &GPU_WAKER_VTABLE),  // clone
    gpu_waker_wake,         // wake (consumes)
    gpu_waker_wake_by_ref,  // wake_by_ref
    |_| {},                 // drop (no-op, no heap)
);

unsafe fn gpu_waker_wake(data: *const ()) {
    gpu_waker_wake_by_ref(data);
}

unsafe fn gpu_waker_wake_by_ref(data: *const ()) {
    let task_id = data as u32;
    // Re-enqueue task into the executor's WorkQueue
    let executor = /* global executor pointer */;
    let _ = executor.work_queue.enqueue(task_id);
}
```

The `data` pointer is actually a `u32` task slot index cast to `*const ()`.
No heap allocation, no pointer dereference — just an integer.

### Modified Executor Run Loop

```
loop {
    task_id = work_queue.dequeue();
    if task_id == EMPTY {
        if tasks_active == 0 || shutdown {
            break;  // all done
        }
        nanosleep;  // yield, wait for wakers to enqueue work
        continue;
    }

    // Build waker with this task's slot index
    let waker = Waker::from_raw(RawWaker::new(task_id as *const (), &GPU_WAKER_VTABLE));
    let mut cx = Context::from_waker(&waker);

    let result = poll_fn(future_ptr, &mut cx);
    match result {
        Poll::Ready(()) => { recycle slot; tasks_active -= 1; }
        Poll::Pending   => { /* do NOT re-enqueue — task is parked */ }
    }
}
```

Key changes from current executor:
1. **No inner re-poll loop** — a Pending task is parked, not spun on
2. **Waker carries task_id** — built per-task, not a global noop
3. **Empty queue != exit** — must also check `tasks_active` counter
4. **`nanosleep` on empty queue** — brief yield while waiting for wakers

### Backward Compatibility

Futures that don't store the waker (existing code using `noop_waker`) will still work:
- If a future ignores `cx.waker()` and returns Pending, it gets parked permanently
- This is actually correct behavior — a future that returns Pending without arranging
  for a wake-up is a bug in the future, not in the executor
- Existing hostcall futures DO complete (they spin internally via `sys_spin_load_acquire`),
  so they return Ready, never Pending — they are unaffected
- The oneshot receiver is the first future that genuinely returns Pending

For a transition period, the executor could have a `MAX_PARK_DURATION` after which it
re-polls parked tasks as a safety net. But this should be removed once all futures
properly use wakers.

### TaskSlot State Machine Extension

Current states: `FREE=0, QUEUED=1, RUNNING=2`

Add: `PARKED=3`

```
FREE ──spawn()──► QUEUED ──dequeue()──► RUNNING ──poll()=Pending──► PARKED
  ▲                  ▲                                                 │
  │                  └────────────wake()────────────────────────────────┘
  │
  └──────────────── poll()=Ready ◄── RUNNING
```

When the executor sees `Poll::Pending`, it transitions the slot to `PARKED`.
When `wake()` fires, it transitions `PARKED → QUEUED` and enqueues. The state
transition uses `sys_cas_u32(PARKED, QUEUED)` to prevent double-enqueue if
`wake()` is called multiple times.

## Open Questions

1. **Global executor pointer**: How to set it? A `#[no_mangle] static` in gpu-runtime
   that `init()` writes to? Or pass the executor pointer through the waker's data field
   (would need to pack both pointer and task_id into 64 bits — feasible since task_id
   is only 8 bits and pointers are 48-bit on GPU)?

2. **Waker cloning cost**: `clone()` is trivial (copy the data pointer), but the Waker
   type itself is 16 bytes (data + vtable). On GPU, every byte in a TaskSlot matters.
   Should the channel store just the task_id (4 bytes) instead of a full Waker (16 bytes)?
   Recommendation: yes, store just task_id.

3. **Multiple wakers per task**: If a task awaits two channels sequentially, the second
   `.await` will store a new waker (same task_id). This is fine — the task_id doesn't
   change. But if a task awaits two channels concurrently (e.g., `select!`), both channels
   need the same waker. Since our waker is just a task_id integer, cloning is free.

4. **MPSC waker list**: For MPSC channels with multiple receivers (future work), the
   sender needs to wake ALL waiting receivers. This requires a list of waker_task_ids,
   not just one. Defer to channel-mpsc theme.

5. **`tasks_active` atomicity**: The counter must be updated atomically. Use
   `sys_fetch_add_u32` for increment (on spawn) and decrement (on completion).
   Currently it uses non-atomic `read_volatile` + `write_volatile` which races.

## Impact on Downstream Tasks
- **channel-waker.2**: Implement `GPU_WAKER_VTABLE` and `gpu_waker_wake()` in gpu-runtime
- **channel-waker.3**: Modify `GpuExecutor::run()` to use real wakers + PARKED state
- **channel-waker.4**: Update `OneshotSlot` with `waker_task_id` field + double-check protocol
- **channel-waker.5**: Integration test — oneshot send/receive without spin-polling
- **channel-mpsc.1**: MPSC channel design will build on this waker mechanism
