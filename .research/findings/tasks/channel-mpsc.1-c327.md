# channel-mpsc.1: MPSC channel design — lock-free ring buffer on GPU
**Cycle**: 327 | **Theme**: channel-mpsc | **Kind**: investigation | **Status**: done

## Summary
Designed a GPU-native MPSC (multi-producer, single-consumer) channel using a fixed-capacity
lock-free ring buffer in mapped memory. Uses `sys_fetch_add_u32` for contention-free producer
slot reservation and a two-phase commit protocol (reserve → write → mark readable) to handle
the gap between slot reservation and value availability. Compatible with the existing
spin-polling executor.

## Findings

### Q: Ring buffer vs linked list for MPSC on GPU?
A: **Ring buffer, decisively.** A linked list requires per-node heap allocation, which is
unavailable on GPU. Even with a free-list approach, linked-list traversal has poor spatial
locality — each node can land on a different 128B L2 cache sector, causing cache thrashing
across SMs. A ring buffer is a flat, contiguous array with predictable access patterns.

Additional ring buffer advantages on GPU:
- **No pointer chasing**: array indexing is a single multiply-add, no dependent loads
- **Pre-allocated**: fits the "no heap on GPU" constraint naturally
- **Cache-friendly**: sequential producers hit adjacent cache sectors
- **Simple indexing**: `slot = head % capacity` — one AND if capacity is power-of-two

Linked lists are only preferable when unbounded growth is needed. Since GPU channels must
have fixed capacity anyway (pre-allocated mapped memory), a ring buffer is strictly superior.

**Confidence**: high

### Q: CAS-based enqueue with atomic head/tail pointers?
A: **fetch_add for producers, plain load for consumer — no CAS needed on the hot path.**

The key insight: in MPSC, multiple producers contend for write slots but only one consumer
reads. We can split the problem:

**Producer enqueue (multi-threaded):**
1. `ticket = sys_fetch_add_u32(&head, 1)` — atomically reserve a slot
2. `slot_index = ticket % capacity`
3. Write value to `slots[slot_index].value`
4. `sys_store_release_u32(&slots[slot_index].sequence, ticket + 1)` — mark slot readable

`fetch_add` is faster than CAS for contended reservation because it never retries — every
producer gets a unique ticket on the first attempt. CAS loops under contention (multiple
threads read the same `head`, only one CAS succeeds, others retry).

**Consumer dequeue (single-threaded):**
1. `seq = sys_load_acquire_u32(&slots[tail % capacity].sequence)`
2. If `seq == tail + 1` → value is ready, read it, advance `tail`
3. If `seq == tail` → slot not yet written, return `Poll::Pending`

The consumer never needs CAS because it is the only reader. A plain acquire load on the
sequence number suffices.

**Two-phase commit is essential.** Between step 1 (reserve) and step 3 (mark readable),
the producer is writing the value. Without the sequence number, the consumer might read a
partially-written slot. The per-slot sequence number acts as a publication flag, similar to
the oneshot channel's `EMPTY → SENT` transition.

**ABA prevention**: The monotonic ticket (fetch_add never wraps in practice for u32 with
reasonable message rates) combined with per-slot sequence numbers eliminates ABA. Unlike a
stack or free-list, ring buffer indices are derived from the ticket, not from CAS on a
shared pointer, so the classic ABA scenario does not arise.

**Confidence**: high

### Q: Fixed capacity — what size? How to handle full buffer (spin vs Pending)?
A: **Power-of-two capacity, recommend 64 slots as default. Return `Poll::Pending` when full.**

**Capacity sizing:**
- Must be power-of-two for fast modulo (`ticket & (capacity - 1)` instead of `%`)
- 16 slots: too small, contention under 32+ producers (full warp)
- 64 slots: good default — handles 2 full warps of producers without immediate backpressure
- 256 slots: generous, use for high-throughput scenarios
- Each slot is `size_of::<T>() + 8` bytes (4B sequence + 4B padding + T), so 64 slots of
  u64 values = 64 * 16 = 1024 bytes — fits in a single 128B-aligned region easily

**Full buffer handling — return `Pending`, do NOT spin:**
When `head - tail >= capacity`, the buffer is full. The producer should:
1. Read `tail` (acquire) to check available space
2. If full → return `Poll::Pending` (let executor poll again later)
3. Do NOT spin — GPU threads that spin block the warp scheduler

Spinning is bad on GPU because:
- A spinning thread holds its warp slot, preventing the scheduler from issuing other warps
- With 32 threads per warp, one spinning producer can stall 31 other lanes
- The executor already handles re-polling via its spin loop, so `Pending` is the natural fit

The producer side should be a Future too: `sender.send(value).await` returns `Pending` when
the buffer is full, and the executor re-polls it until space is available.

**Detecting full condition with fetch_add:**
There is a subtlety: `fetch_add` unconditionally increments `head`, so by the time we check
if the buffer is full, we already "reserved" a slot. Two approaches:

**(a) CAS loop for reservation (safe but slower):**
```rust
loop {
    let h = sys_load_acquire_u32(&head);
    let t = sys_load_acquire_u32(&tail);
    if h - t >= capacity {
        return Poll::Pending;  // full
    }
    if sys_cas_u32(&head, h, h + 1) == h {
        break; // reserved slot h
    }
    // CAS failed, retry
}
```

**(b) fetch_add with backoff (faster, slightly complex):**
```rust
let ticket = sys_fetch_add_u32(&head, 1);
let slot = ticket % capacity;
// Wait until consumer has consumed this slot from previous cycle
loop {
    let seq = sys_load_acquire_u32(&slots[slot].sequence);
    if seq == ticket {
        break; // slot is free (consumed or never used)
    }
    // Slot still occupied — could spin briefly or yield
}
// Write value, publish
```

**Recommendation: approach (b).** It keeps the fast path (fetch_add) and handles full
buffers by waiting on the per-slot sequence number. The sequence check naturally blocks until
the consumer has drained the slot from a previous ring cycle. This is the Dmitry Vyukov
bounded MPMC queue technique, adapted for MPSC (single consumer simplifies the dequeue side).

The consumer sets `slots[slot].sequence = tail + capacity` after reading, which tells the
next producer cycle that the slot is available.

**Confidence**: high

### Q: Memory layout for cache-line alignment on GPU (128B L2 sectors)?
A: **Align the control struct to 128B and separate head/tail into different cache sectors.**

GPU L2 cache operates on 128-byte sectors (on SM86 / RTX 3060). When multiple SMs write to
the same 128B sector simultaneously, the L2 must serialize these accesses. For MPSC:

- `head` is written by all producers (hot, contended)
- `tail` is written only by the consumer (cold from producers' perspective)
- Per-slot `sequence` fields are written by producers (write) and consumer (reset)

**Layout:**
```rust
#[repr(C, align(128))]
pub struct MpscRingBuffer<T, const N: usize> {
    // --- Cache sector 0: producer-hot ---
    head: UnsafeCell<u32>,          // offset 0: producers fetch_add this
    _pad0: [u8; 124],               // pad to 128B boundary

    // --- Cache sector 1: consumer-hot ---
    tail: UnsafeCell<u32>,          // offset 128: only consumer writes
    _pad1: [u8; 124],               // pad to 128B boundary

    // --- Cache sector 2+: slot array ---
    slots: [MpscSlot<T>; N],        // offset 256+
}

#[repr(C)]
pub struct MpscSlot<T> {
    sequence: UnsafeCell<u32>,      // publication sequence number
    _pad: u32,                       // align value to 8B
    value: UnsafeCell<MaybeUninit<T>>,
}
```

**Why separate cache sectors for head vs tail:**
- Producers atomically increment `head` — this causes L2 sector invalidation across SMs
- If `tail` is in the same sector, every producer fetch_add also invalidates the consumer's
  cached `tail` value, and vice versa — classic false sharing
- 128B padding between `head` and `tail` eliminates false sharing

**Slot alignment considerations:**
- Each `MpscSlot<T>` should ideally be a multiple of 8 bytes for natural alignment
- For small T (u32, u64), slots are 8-16 bytes — 8-16 slots fit per 128B sector
- Adjacent sequence numbers may share a cache sector, but this is acceptable: producers
  write different sequence numbers (different slots), so there is no true sharing conflict
- If T is large (>64B), consider aligning each slot to 128B to avoid cross-sector writes,
  but for typical GPU channel payloads (indices, small structs), this is unnecessary

**Capacity as const generic:**
Using `const N: usize` allows compile-time power-of-two verification:
```rust
const _: () = assert!(N.is_power_of_two(), "MPSC capacity must be power of two");
```

**Confidence**: high

## Design Summary

### Core Types (in gpu-runtime)
```rust
const MPSC_DEFAULT_CAPACITY: usize = 64;

#[repr(C, align(128))]
pub struct MpscChannel<T: Copy, const N: usize = 64> {
    head: UnsafeCell<u32>,
    _pad0: [u8; 124],
    tail: UnsafeCell<u32>,
    _pad1: [u8; 124],
    slots: [MpscSlot<T>; N],
}

#[repr(C)]
pub struct MpscSlot<T> {
    sequence: UnsafeCell<u32>,
    _pad: u32,
    value: UnsafeCell<MaybeUninit<T>>,
}

pub struct MpscSender<T: Copy, const N: usize> {
    channel: *mut MpscChannel<T, N>,
}

pub struct MpscReceiver<T: Copy, const N: usize> {
    channel: *mut MpscChannel<T, N>,
}
```

### Protocol (Vyukov bounded queue, adapted for MPSC)

**Initialization:**
```rust
fn init<T: Copy, const N: usize>(ch: &mut MpscChannel<T, N>) {
    sys_store_release_u32(ch.head.get(), 0);
    sys_store_release_u32(ch.tail.get(), 0);
    for i in 0..N {
        sys_store_release_u32(ch.slots[i].sequence.get(), i as u32);
    }
}
```
Each slot's sequence is initialized to its index. This means slot `i` is "available for
ticket `i`" — the first N producers can write immediately.

**Producer send (returns Poll):**
```rust
fn poll_send(&self, value: T) -> Poll<Result<(), SendError>> {
    let ticket = sys_fetch_add_u32(&ch.head, 1);
    let slot = &ch.slots[(ticket as usize) & (N - 1)];
    let seq = sys_load_acquire_u32(slot.sequence.get());
    if seq != ticket {
        // Slot not yet consumed from previous cycle — channel full.
        // NOTE: We already incremented head. Must "return" the ticket.
        // This is the complexity of fetch_add approach.
        // Alternative: use CAS loop (see below).
        return Poll::Pending;
    }
    core::ptr::write_volatile(slot.value.get() as *mut T, value);
    sys_store_release_u32(slot.sequence.get(), ticket + 1);
    Poll::Ready(Ok(()))
}
```

**Revised approach — CAS loop for send (recommended for correctness):**
```rust
fn poll_send(&self, value: T) -> Poll<Result<(), SendError>> {
    let head = sys_load_acquire_u32(&ch.head);
    let tail = sys_load_acquire_u32(&ch.tail);
    if head - tail >= N as u32 {
        return Poll::Pending;  // full
    }
    let old = sys_cas_u32(&ch.head, head, head + 1);
    if old != head {
        return Poll::Pending;  // lost race, executor re-polls
    }
    // Won slot `head`
    let slot = &ch.slots[(head as usize) & (N - 1)];
    core::ptr::write_volatile(slot.value.get() as *mut T, value);
    sys_store_release_u32(slot.sequence.get(), head + 1);
    Poll::Ready(Ok(()))
}
```
The CAS loop approach is simpler and avoids the "undo fetch_add" problem. Returning
`Pending` on CAS failure is fine — the executor re-polls, and the next attempt will succeed
if there is space.

**Consumer recv (single consumer, no CAS needed):**
```rust
fn poll_recv(&self) -> Poll<Option<T>> {
    let tail = sys_load_acquire_u32(&ch.tail);
    let slot = &ch.slots[(tail as usize) & (N - 1)];
    let seq = sys_load_acquire_u32(slot.sequence.get());
    if seq != tail + 1 {
        return Poll::Pending;  // no message ready
    }
    let value = core::ptr::read_volatile(slot.value.get() as *const T);
    // Reset sequence for next cycle: allow producer with ticket (tail + N) to use this slot
    sys_store_release_u32(slot.sequence.get(), tail.wrapping_add(N as u32));
    sys_store_release_u32(&ch.tail, tail + 1);
    Poll::Ready(Some(value))
}
```

### Available Atomics (from gpu-atomics)
All needed primitives are available:
- `sys_fetch_add_u32` — producer slot reservation (if using fetch_add approach)
- `sys_cas_u32` — producer slot reservation (CAS approach, recommended)
- `sys_load_acquire_u32` — reading sequence numbers, tail
- `sys_store_release_u32` — publishing values, advancing tail
- `sys_spin_load_acquire_u32` — for poll loops (includes nanosleep for warp yielding)

No new atomics are required. No u64 tagged CAS needed — unlike a free-list, the ring buffer
does not suffer from ABA (indices are deterministic from tickets, not reused pointers).

### Sender as Future
```rust
pub struct MpscSendFuture<'a, T: Copy, const N: usize> {
    sender: &'a MpscSender<T, N>,
    value: T,
}

impl<T: Copy, const N: usize> Future for MpscSendFuture<'_, T, N> {
    type Output = Result<(), SendError>;
    fn poll(self: Pin<&mut Self>, _cx: &mut Context) -> Poll<Self::Output> {
        self.sender.poll_send(self.value)
    }
}
```

### Receiver as Future (Stream-like)
```rust
impl<T: Copy, const N: usize> MpscReceiver<T, N> {
    pub fn recv(&self) -> MpscRecvFuture<'_, T, N> {
        MpscRecvFuture { receiver: self }
    }
}

impl<T: Copy, const N: usize> Future for MpscRecvFuture<'_, T, N> {
    type Output = Option<T>;  // None if channel closed
    fn poll(self: Pin<&mut Self>, _cx: &mut Context) -> Poll<Self::Output> {
        self.receiver.poll_recv()
    }
}
```

## Open Questions
- **Channel close semantics**: How does the consumer know all producers are done? Options:
  (a) separate atomic `closed` flag, (b) producer count + atomic decrement on drop,
  (c) application-level sentinel value. Recommendation: atomic `open_senders: u32` counter
  decremented by each sender on drop; consumer returns `None` when counter is 0 and buffer
  is empty.
- **Batch dequeue**: Consumer could drain multiple items per poll to amortize atomic
  overhead. Worth investigating if recv is a bottleneck.
- **Should MpscSender be Clone?** Yes — MPSC requires multiple senders. Since there is no
  Drop on GPU, "clone" is just copying the raw pointer. But we need the `open_senders`
  counter for close detection.
- **Capacity tuning**: 64 is a reasonable default, but workloads with bursty producers may
  benefit from 128 or 256. Expose as const generic.

## Impact on Downstream Tasks
- **channel-mpsc.2**: Implement `MpscChannel`, `MpscSlot`, `MpscSender`, `MpscReceiver`
  structs with the layout described above
- **channel-mpsc.3**: Implement `poll_send` (CAS approach) and `poll_recv` methods
- **channel-mpsc.4**: Implement `MpscSendFuture` and `MpscRecvFuture` for executor integration
- **channel-mpsc.5**: Test with multi-warp producers, verify no lost messages or duplicates
- **channel-waker.1**: When real wakers arrive, sender's `publish` step can call
  `waker.wake()` to re-enqueue the consumer task (eliminates spin-polling overhead)
