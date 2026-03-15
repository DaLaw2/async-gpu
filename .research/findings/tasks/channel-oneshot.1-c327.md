# channel-oneshot.1: Oneshot channel design for GPU async tasks
**Cycle**: 327 | **Theme**: channel-oneshot | **Kind**: investigation | **Status**: done

## Summary
Designed a GPU-native oneshot channel (single-value, single-use) for inter-task communication
within the GpuExecutor. Uses atomic state machine + inline value storage in mapped memory.
Compatible with existing spin-polling executor model.

## Findings

### Q: What memory model for oneshot on GPU? Mapped memory slot with atomic state?
A: **Shared slot in mapped memory with atomic u32 state field.** The slot must reside in global
memory (mapped or device) visible to all SMs. Since GpuExecutor already allocates its arena
in mapped memory, oneshot slots can be part of the executor's memory region or a separate pool.

The slot layout:
```rust
#[repr(C)]
pub struct OneshotSlot<T> {
    state: UnsafeCell<u32>,     // atomic state: EMPTY=0, SENT=1, CLOSED=2
    _pad: u32,                   // alignment padding
    value: UnsafeCell<MaybeUninit<T>>,  // inline storage for T
}
```

State transitions (atomic, system-scope):
- `EMPTY → SENT`: sender writes value, then `sys_store_release_u32(&state, SENT)`
- `EMPTY → CLOSED`: sender dropped without sending
- `SENT → read`: receiver loads with `sys_load_acquire_u32`, reads value

The acquire/release pair ensures value write is visible before state transition is observed.
**Confidence**: high (matches proven hostcall packet pattern)

### Q: How to handle Sender/Receiver lifetime without heap allocation?
A: **Slot pool approach.** Pre-allocate a fixed array of `OneshotSlot<T>` in the executor's
mapped memory region. `oneshot()` returns a (Sender, Receiver) pair that both hold the slot
index. This mirrors how the hostcall protocol uses a packet pool.

```rust
pub struct OneshotSender<T> {
    slot_ptr: *mut OneshotSlot<T>,   // raw pointer to slot in mapped memory
}

pub struct OneshotReceiver<T> {
    slot_ptr: *const OneshotSlot<T>, // raw pointer (read-only view)
}

pub fn oneshot<T>(slot: &mut OneshotSlot<T>) -> (OneshotSender<T>, OneshotReceiver<T>) {
    // Initialize slot state to EMPTY
    unsafe { sys_store_release_u32(slot.state.get() as *mut u32, EMPTY); }
    (
        OneshotSender { slot_ptr: slot as *mut _ },
        OneshotReceiver { slot_ptr: slot as *const _ },
    )
}
```

No heap allocation needed. The caller provides the slot (from a pre-allocated pool or stack).
**Confidence**: high

### Q: What state machine: Empty → Sent → Received? Atomic u32 with acquire/release?
A: **Three states, two transitions:**

```
EMPTY (0) ──send()──► SENT (1) ──poll() Ready──► [value consumed]
    │
    └──drop sender──► CLOSED (2)
```

- `EMPTY = 0`: Initial state. No value written yet.
- `SENT = 1`: Value written. Receiver can read.
- `CLOSED = 2`: Sender dropped without sending. Receiver gets error.

**Sender::send(value: T):**
1. Write value to `slot.value` via `core::ptr::write_volatile`
2. `sys_store_release_u32(&slot.state, SENT)` — release ensures value is visible

**Receiver::poll():**
1. `sys_load_acquire_u32(&slot.state)` — acquire ensures value read sees sender's write
2. If SENT → read value via `core::ptr::read_volatile`, return `Poll::Ready(Ok(value))`
3. If CLOSED → return `Poll::Ready(Err(Closed))`
4. If EMPTY → return `Poll::Pending`

No CAS needed — single producer, single consumer. Plain load/store with acquire/release
ordering is sufficient (no contention on state).
**Confidence**: high

### Q: How does Receiver::poll interact with executor waker mechanism?
A: **Currently: noop waker, spin-polling.** The existing executor uses `noop_waker()` — calling
`cx.waker().wake()` does nothing. The executor's `run()` loop polls each task up to
`MAX_POLLS_PER_TASK` (1000) times with `nanosleep.u32 1000` between polls.

This means a oneshot receiver future will be polled repeatedly by the executor until the
sender writes the value. This works but wastes GPU cycles spinning.

**Future improvement (channel-waker theme):** Replace noop waker with a real waker that
re-enqueues the task into the WorkQueue when `wake()` is called. Then:
1. Receiver returns `Poll::Pending`, executor removes task from run queue
2. Sender calls `send()`, writes value, calls stored waker
3. Waker re-enqueues task into WorkQueue
4. Executor picks up task, polls again, gets `Poll::Ready`

This requires modifying the executor to support "park/unpark" semantics — a task returning
Pending should be removed from the active poll loop and only re-added when woken. This is
the channel-waker theme's scope.

For the initial implementation (channel-oneshot.2), we'll use the spin-polling approach
which works correctly with the existing executor.
**Confidence**: high

## Design Summary

### OneshotSlot<T> (in gpu-runtime)
```rust
const ONESHOT_EMPTY: u32 = 0;
const ONESHOT_SENT: u32 = 1;
const ONESHOT_CLOSED: u32 = 2;

#[repr(C)]
pub struct OneshotSlot<T> {
    state: UnsafeCell<u32>,
    _pad: u32,
    value: UnsafeCell<MaybeUninit<T>>,
}

pub struct OneshotSender<T> { slot: *mut OneshotSlot<T> }
pub struct OneshotReceiver<T> { slot: *const OneshotSlot<T> }
```

### API
```rust
// Create channel from pre-allocated slot
fn oneshot<T>(slot: &mut OneshotSlot<T>) -> (OneshotSender<T>, OneshotReceiver<T>);

// Sender: write value (consumes sender)
impl<T> OneshotSender<T> {
    fn send(self, value: T);  // EMPTY → SENT
}

// Receiver: poll for value (implements Future)
impl<T> Future for OneshotReceiver<T> {
    type Output = Result<T, OneshotClosed>;
    fn poll(self: Pin<&mut Self>, _cx: &mut Context) -> Poll<Self::Output>;
}
```

### Constraints
- T must be `Copy` (or at least trivially movable) — no Drop on GPU
- T size must fit in OneshotSlot (compile-time check via const generic or assert)
- Slot must be in mapped memory for cross-warp visibility
- Single producer, single consumer — no Clone on Sender or Receiver

## Open Questions
- Should we support `OneshotSlot<T>` for dynamically-sized T? Probably not — fixed T with Copy bound.
- Pool management: who allocates/frees slots? Executor or caller?
  - Recommendation: caller provides slot reference (zero-overhead, no pool management)

## Impact on Downstream Tasks
- **channel-oneshot.2**: Implement the structs and methods described above
- **channel-oneshot.3**: Implement Future for OneshotReceiver (poll pattern above)
- **channel-waker.1**: Design real waker that re-enqueues tasks (extends this design)
- **channel-mpsc.1**: MPSC needs different approach (ring buffer, not single slot)
