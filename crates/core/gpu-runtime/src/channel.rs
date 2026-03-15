use core::cell::UnsafeCell;
use core::future::Future;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::pin::Pin;
use core::task::{Context, Poll};
use gpu_atomics::{sys_cas_u32, sys_load_acquire_u32, sys_store_release_u32};

/// Oneshot channel states.
const ONESHOT_EMPTY: u32 = 0;
const ONESHOT_SENT: u32 = 1;
const ONESHOT_CLOSED: u32 = 2;

/// Error returned when a oneshot sender is dropped without sending.
#[derive(Debug, Clone, Copy)]
pub struct OneshotClosed;

/// A pre-allocated slot for a oneshot channel.
///
/// Must reside in global memory (device or mapped). The caller allocates
/// the slot and passes it to [`oneshot()`] to create a sender/receiver pair.
///
/// # Layout
///
/// - `state`: atomic u32 (EMPTY=0, SENT=1, CLOSED=2)
/// - `value`: inline storage for `T` (8-byte aligned)
#[repr(C)]
pub struct OneshotSlot<T: Copy> {
    state: UnsafeCell<u32>,
    _pad: u32,
    value: UnsafeCell<MaybeUninit<T>>,
}

// SAFETY: OneshotSlot is accessed via system-scope atomics.
// Cross-warp/cross-block access is synchronized by acquire/release.
unsafe impl<T: Copy + Send> Send for OneshotSlot<T> {}
unsafe impl<T: Copy + Send> Sync for OneshotSlot<T> {}

impl<T: Copy> Default for OneshotSlot<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy> OneshotSlot<T> {
    /// Create a new uninitialized slot.
    ///
    /// # Safety
    /// The slot must be placed in global memory before use.
    pub const fn new() -> Self {
        Self {
            state: UnsafeCell::new(ONESHOT_EMPTY),
            _pad: 0,
            value: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }

    /// Initialize slot state to EMPTY. Call before creating a channel pair.
    ///
    /// # Safety
    /// Must be called before any sender/receiver accesses the slot.
    #[inline(always)]
    pub unsafe fn reset(&self) {
        core::ptr::write_volatile(self.state.get(), ONESHOT_EMPTY);
    }

    /// Get raw pointer to the state field.
    ///
    /// Used for direct atomic access in kernel code.
    #[inline(always)]
    pub fn state_ptr(&self) -> *mut u32 {
        self.state.get()
    }

    /// Get raw pointer to the value storage.
    ///
    /// Used for direct volatile read/write in kernel code.
    #[inline(always)]
    pub fn value_ptr(&self) -> *mut MaybeUninit<T> {
        self.value.get()
    }
}

/// Sending half of a oneshot channel.
///
/// Consumes itself on `send()`. If dropped without sending, the slot
/// transitions to CLOSED and the receiver gets `Err(OneshotClosed)`.
pub struct OneshotSender<T: Copy> {
    slot: *mut OneshotSlot<T>,
    _marker: PhantomData<T>,
}

// SAFETY: Sender holds a raw pointer to global memory.
// Only one sender exists per slot (enforced by API).
unsafe impl<T: Copy + Send> Send for OneshotSender<T> {}

impl<T: Copy> OneshotSender<T> {
    /// Send a value through the channel.
    ///
    /// Writes the value to the slot and transitions state from EMPTY to SENT.
    /// The release store ensures the value is visible to the receiver's
    /// acquire load.
    ///
    /// # Safety
    /// - Slot must be in global memory
    /// - Must only be called once (enforced by consuming self)
    #[inline(always)]
    pub unsafe fn send(self, value: T) {
        let slot = &*self.slot;
        // Write value first (before state transition)
        core::ptr::write_volatile(slot.value.get() as *mut T, value);
        // Release store: value write is visible before SENT is observed
        sys_store_release_u32(slot.state.get(), ONESHOT_SENT);
        // Prevent drop from setting CLOSED
        core::mem::forget(self);
    }
}

impl<T: Copy> Drop for OneshotSender<T> {
    fn drop(&mut self) {
        // Sender dropped without sending → mark CLOSED
        unsafe {
            sys_store_release_u32((*self.slot).state.get(), ONESHOT_CLOSED);
        }
    }
}

/// Receiving half of a oneshot channel. Implements `Future`.
///
/// Polls the atomic state of the slot. Returns `Poll::Ready(Ok(value))`
/// when the sender has sent, `Poll::Ready(Err(OneshotClosed))` if the
/// sender was dropped, or `Poll::Pending` if no value yet.
pub struct OneshotReceiver<T: Copy> {
    slot: *const OneshotSlot<T>,
    _marker: PhantomData<T>,
}

// SAFETY: Receiver holds a raw pointer to global memory.
// Only one receiver exists per slot (enforced by API).
unsafe impl<T: Copy + Send> Send for OneshotReceiver<T> {}

impl<T: Copy> Future for OneshotReceiver<T> {
    type Output = Result<T, OneshotClosed>;

    #[inline(always)]
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        let slot = unsafe { &*self.slot };
        let state = unsafe { sys_load_acquire_u32(slot.state.get() as *const u32) };
        match state {
            ONESHOT_SENT => {
                // Acquire load above ensures we see the value written by sender
                let value = unsafe { core::ptr::read_volatile(slot.value.get() as *const T) };
                Poll::Ready(Ok(value))
            }
            ONESHOT_CLOSED => Poll::Ready(Err(OneshotClosed)),
            _ => Poll::Pending, // EMPTY — not ready yet
        }
    }
}

/// Create a oneshot channel from a pre-allocated slot.
///
/// Returns a (sender, receiver) pair. The sender can send exactly one value;
/// the receiver is a `Future` that resolves to that value.
///
/// # Safety
/// - `slot` must point to valid global memory (device or mapped)
/// - `slot` must not be concurrently used by another channel pair
/// - The slot must outlive both sender and receiver
///
/// # Example
///
/// ```rust,ignore
/// use gpu_runtime::channel::{OneshotSlot, oneshot};
///
/// let mut slot = OneshotSlot::<u32>::new();
/// let (tx, rx) = unsafe { oneshot(&mut slot) };
///
/// // In producer task:
/// unsafe { tx.send(42); }
///
/// // In consumer task (as Future):
/// // let value = rx.await; // Ok(42)
/// ```
#[inline(always)]
pub unsafe fn oneshot<T: Copy>(
    slot: &mut OneshotSlot<T>,
) -> (OneshotSender<T>, OneshotReceiver<T>) {
    // Reset slot state
    slot.reset();
    (
        OneshotSender {
            slot: slot as *mut OneshotSlot<T>,
            _marker: PhantomData,
        },
        OneshotReceiver {
            slot: slot as *const OneshotSlot<T>,
            _marker: PhantomData,
        },
    )
}

// ============================================================
// MPSC Channel — multi-producer, single-consumer
// ============================================================

/// Default capacity for MPSC channel (must be power of 2).
pub const MPSC_DEFAULT_CAPACITY: usize = 64;

/// Error returned when trying to send on a closed MPSC channel.
#[derive(Debug, Clone, Copy)]
pub struct MpscClosed;

/// Error returned when the channel is full.
#[derive(Debug, Clone, Copy)]
pub struct MpscFull;

/// A single slot in the MPSC ring buffer.
///
/// Uses a sequence number for publication ordering:
/// - When `sequence == slot_index`: slot is available for writing
/// - When `sequence == slot_index + 1`: slot contains a readable value
/// - After consumer reads: `sequence = slot_index + capacity` (recycles for next round)
#[repr(C)]
pub struct MpscSlot<T: Copy> {
    sequence: UnsafeCell<u32>,
    _pad: u32,
    value: UnsafeCell<MaybeUninit<T>>,
}

impl<T: Copy> MpscSlot<T> {
    /// Create an empty slot with initial sequence number.
    pub const fn new(seq: u32) -> Self {
        Self {
            sequence: UnsafeCell::new(seq),
            _pad: 0,
            value: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }
}

/// Pre-allocated MPSC channel storage.
///
/// Must reside in global memory (device or mapped). The ring buffer uses
/// CAS-based head advancement for producers and plain load/store for the
/// single consumer.
///
/// # Layout
/// - `head`: producer write cursor (CAS-contended by multiple senders)
/// - `tail`: consumer read cursor (only consumer writes)
/// - `closed`: shutdown flag
/// - `slots`: ring buffer of N slots with sequence numbers
///
/// # Capacity
/// N must be a power of 2 for efficient modulo via bitmask.
#[repr(C)]
pub struct MpscChannel<T: Copy, const N: usize = MPSC_DEFAULT_CAPACITY> {
    head: UnsafeCell<u32>,
    tail: UnsafeCell<u32>,
    closed: UnsafeCell<u32>,
    /// Stored waker for the consumer task. Set by consumer's poll(),
    /// called by producer's try_send() to wake the parked consumer.
    /// Layout: [data: *const (), vtable: &'static RawWakerVTable] (16 bytes).
    /// waker_set flag indicates if a waker is stored (0 = no, 1 = yes).
    waker_set: UnsafeCell<u32>,
    _waker_pad: u32,
    waker_bytes: UnsafeCell<[u8; 16]>,
    slots: [MpscSlot<T>; N],
}

// SAFETY: MpscChannel is accessed via system-scope atomics.
unsafe impl<T: Copy + Send, const N: usize> Send for MpscChannel<T, N> {}
unsafe impl<T: Copy + Send, const N: usize> Sync for MpscChannel<T, N> {}

impl<T: Copy, const N: usize> MpscChannel<T, N> {
    /// Initialize the channel. Must be called before use.
    ///
    /// Sets head=0, tail=0, closed=0, and each slot's sequence to its index.
    ///
    /// # Safety
    /// Must be called from a single thread before any senders/receivers access the channel.
    #[inline(always)]
    pub unsafe fn init(&self) {
        core::ptr::write_volatile(self.head.get(), 0);
        core::ptr::write_volatile(self.tail.get(), 0);
        core::ptr::write_volatile(self.closed.get(), 0);
        core::ptr::write_volatile(self.waker_set.get(), 0);
        let mut i = 0u32;
        while (i as usize) < N {
            core::ptr::write_volatile(self.slots[i as usize].sequence.get(), i);
            i += 1;
        }
    }

    /// Close the channel. After this, senders get `MpscClosed` and
    /// the receiver drains remaining items then gets `None`.
    #[inline(always)]
    pub unsafe fn close(&self) {
        sys_store_release_u32(self.closed.get(), 1);
    }

    /// Check if the channel is closed.
    #[inline(always)]
    pub unsafe fn is_closed(&self) -> bool {
        sys_load_acquire_u32(self.closed.get() as *const u32) != 0
    }

    /// Try to send a value directly on the channel (producer side).
    ///
    /// Returns `Ok(())` on success, `Err(MpscSendError)` if closed or full.
    ///
    /// # Safety
    /// Channel must be in global memory and initialized.
    #[inline(always)]
    pub unsafe fn try_send(&self, value: T) -> Result<(), MpscSendError<T>> {
        if self.is_closed() {
            return Err(MpscSendError::Closed(value));
        }

        let head = sys_load_acquire_u32(self.head.get() as *const u32);
        let tail = sys_load_acquire_u32(self.tail.get() as *const u32);

        if head.wrapping_sub(tail) >= N as u32 {
            return Err(MpscSendError::Full(value));
        }

        let old = sys_cas_u32(self.head.get(), head, head.wrapping_add(1));
        if old != head {
            return Err(MpscSendError::Full(value));
        }

        let idx = (head as usize) & (N - 1);
        let slot = &self.slots[idx];
        core::ptr::write_volatile(slot.value.get() as *mut T, value);
        sys_store_release_u32(slot.sequence.get(), head.wrapping_add(1));

        // Wake parked consumer if one is waiting
        self.wake_consumer();

        Ok(())
    }

    /// Store waker from the consumer's Context.
    ///
    /// Clones the waker and calls `wake_by_ref` semantics later from `try_send`.
    /// Instead of storing a full Waker (which requires vtable + drop), we just
    /// clone and immediately call `wake()` from `wake_consumer()`.
    ///
    /// The approach: store the waker as a cloned Waker in a raw u128 (data + vtable).
    /// But since our GPU waker has no-op drop and trivial clone, we can just store
    /// the raw data pointer (the packed work_queue|task_id).
    #[inline(always)]
    pub unsafe fn store_waker(&self, cx: &core::task::Context<'_>) {
        // Clone the waker and copy its bytes into channel storage.
        // GPU waker is 16 bytes (data + vtable), no-op drop, trivial clone.
        let waker = cx.waker().clone();
        let src = &waker as *const core::task::Waker as *const u8;
        let dst = (*self.waker_bytes.get()).as_mut_ptr();
        core::ptr::copy_nonoverlapping(src, dst, 16);
        core::mem::forget(waker); // GPU waker has no-op drop, but be explicit
                                  // Mark waker as set (release store ensures waker bytes visible)
        sys_store_release_u32(self.waker_set.get(), 1);
    }

    /// Wake the parked consumer task by calling the stored waker.
    #[inline(always)]
    unsafe fn wake_consumer(&self) {
        let is_set = sys_load_acquire_u32(self.waker_set.get() as *const u32);
        if is_set != 0 {
            // Clear flag (consume — consumer will re-store on next Pending)
            sys_store_release_u32(self.waker_set.get(), 0);
            // Reconstruct waker from stored bytes and call wake
            let src = (*self.waker_bytes.get()).as_ptr();
            let mut waker_buf = core::mem::MaybeUninit::<core::task::Waker>::uninit();
            core::ptr::copy_nonoverlapping(src, waker_buf.as_mut_ptr() as *mut u8, 16);
            let waker = waker_buf.assume_init();
            waker.wake();
        }
    }

    /// Try to receive a value directly from the channel (consumer side).
    ///
    /// Returns `Some(value)` if available, `None` if no value is ready.
    ///
    /// # Safety
    /// Channel must be in global memory and initialized.
    /// Only one consumer should call this at a time.
    #[inline(always)]
    pub unsafe fn try_recv(&self) -> Option<T> {
        let tail = sys_load_acquire_u32(self.tail.get() as *const u32);
        let idx = (tail as usize) & (N - 1);
        let slot = &self.slots[idx];

        let seq = sys_load_acquire_u32(slot.sequence.get() as *const u32);
        if seq != tail.wrapping_add(1) {
            return None;
        }

        let value = core::ptr::read_volatile(slot.value.get() as *const T);
        sys_store_release_u32(slot.sequence.get(), tail.wrapping_add(N as u32));
        sys_store_release_u32(self.tail.get(), tail.wrapping_add(1));

        Some(value)
    }
}

/// Sending half of an MPSC channel. Can be cloned for multiple producers.
///
/// Uses CAS on the shared head pointer to reserve a slot, then writes
/// the value and publishes via sequence number release-store.
pub struct MpscSender<T: Copy, const N: usize = MPSC_DEFAULT_CAPACITY> {
    channel: *const MpscChannel<T, N>,
    _marker: PhantomData<T>,
}

// SAFETY: Multiple senders can coexist; all access is via system-scope atomics.
unsafe impl<T: Copy + Send, const N: usize> Send for MpscSender<T, N> {}
unsafe impl<T: Copy + Send, const N: usize> Sync for MpscSender<T, N> {}

impl<T: Copy, const N: usize> Clone for MpscSender<T, N> {
    fn clone(&self) -> Self {
        Self {
            channel: self.channel,
            _marker: PhantomData,
        }
    }
}

impl<T: Copy, const N: usize> MpscSender<T, N> {
    /// Try to send a value. Returns `Ok(())` on success.
    ///
    /// - `Err(MpscClosed)` if channel is closed
    /// - `Err(MpscFull)` if the ring buffer is full (caller should retry later)
    ///
    /// # Safety
    /// Channel pointer must be valid and in global memory.
    #[inline(always)]
    pub unsafe fn try_send(&self, value: T) -> Result<(), MpscSendError<T>> {
        (*self.channel).try_send(value)
    }
}

/// Error type for MPSC send operations.
#[derive(Debug, Clone, Copy)]
pub enum MpscSendError<T> {
    /// Channel is closed. Contains the unsent value.
    Closed(T),
    /// Channel is full. Contains the unsent value. Caller should retry.
    Full(T),
}

/// Receiving half of an MPSC channel. Only one receiver should exist.
///
/// Polls the sequence number of the next slot. When the sequence indicates
/// the slot has been published, reads the value and advances the tail pointer.
pub struct MpscReceiver<T: Copy, const N: usize = MPSC_DEFAULT_CAPACITY> {
    channel: *const MpscChannel<T, N>,
    _marker: PhantomData<T>,
}

// SAFETY: Only one receiver exists per channel (enforced by API).
unsafe impl<T: Copy + Send, const N: usize> Send for MpscReceiver<T, N> {}

impl<T: Copy, const N: usize> MpscReceiver<T, N> {
    /// Try to receive a value. Returns `Some(value)` if available.
    ///
    /// Returns `None` if no value is ready (caller should retry later).
    ///
    /// # Safety
    /// Channel pointer must be valid and in global memory.
    /// Only one receiver should call this at a time.
    #[inline(always)]
    pub unsafe fn try_recv(&self) -> Option<T> {
        unsafe { (*self.channel).try_recv() }
    }

    /// Check if the channel is closed and empty.
    #[inline(always)]
    pub unsafe fn is_terminated(&self) -> bool {
        let ch = unsafe { &*self.channel };
        if !ch.is_closed() {
            return false;
        }
        let head = sys_load_acquire_u32(ch.head.get() as *const u32);
        let tail = sys_load_acquire_u32(ch.tail.get() as *const u32);
        head == tail
    }
}

/// Future adapter for MPSC receiver. Polls until a value is available
/// or the channel is closed and drained.
pub struct MpscRecvFuture<T: Copy, const N: usize = MPSC_DEFAULT_CAPACITY> {
    receiver: *const MpscReceiver<T, N>,
}

// SAFETY: Single receiver, same constraints as MpscReceiver.
unsafe impl<T: Copy + Send, const N: usize> Send for MpscRecvFuture<T, N> {}

impl<T: Copy, const N: usize> Future for MpscRecvFuture<T, N> {
    type Output = Option<T>;

    #[inline(always)]
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        unsafe {
            let rx = &*self.receiver;

            // Try to receive
            if let Some(value) = rx.try_recv() {
                return Poll::Ready(Some(value));
            }

            // No value — check if terminated (closed + empty)
            if rx.is_terminated() {
                return Poll::Ready(None);
            }

            // Store waker so producer can wake us on send
            let ch = &*rx.channel;
            ch.store_waker(cx);

            // Double-check: value may have been sent between try_recv and store_waker
            if let Some(value) = rx.try_recv() {
                return Poll::Ready(Some(value));
            }

            // Not ready yet — we're parked, waker will re-enqueue us
            Poll::Pending
        }
    }
}

impl<T: Copy, const N: usize> MpscReceiver<T, N> {
    /// Create a future that resolves to the next value, or `None` if closed.
    ///
    /// # Safety
    /// The returned future borrows the receiver; the receiver must outlive it.
    #[inline(always)]
    pub unsafe fn recv(&self) -> MpscRecvFuture<T, N> {
        MpscRecvFuture {
            receiver: self as *const MpscReceiver<T, N>,
        }
    }
}

/// Create an MPSC channel from a pre-allocated channel buffer.
///
/// Returns a (sender, receiver) pair. The sender can be cloned for
/// multiple producers. Only one receiver should exist.
///
/// # Safety
/// - `channel` must point to valid global memory (device or mapped)
/// - `channel` must not be concurrently used by another channel pair
/// - The channel must outlive all senders and the receiver
/// - N must be a power of 2
///
/// # Example
///
/// ```rust,ignore
/// use gpu_runtime::channel::{MpscChannel, mpsc};
///
/// let channel = MpscChannel::<u32, 64>::default_uninit();
/// let (tx, rx) = unsafe { mpsc(&channel) };
///
/// // Producer 1:
/// unsafe { tx.try_send(42).ok(); }
///
/// // Producer 2 (cloned sender):
/// let tx2 = tx.clone();
/// unsafe { tx2.try_send(99).ok(); }
///
/// // Consumer:
/// // let val = rx.recv().await; // Some(42) or Some(99)
/// ```
#[inline(always)]
pub unsafe fn mpsc<T: Copy, const N: usize>(
    channel: &MpscChannel<T, N>,
) -> (MpscSender<T, N>, MpscReceiver<T, N>) {
    channel.init();
    (
        MpscSender {
            channel: channel as *const MpscChannel<T, N>,
            _marker: PhantomData,
        },
        MpscReceiver {
            channel: channel as *const MpscChannel<T, N>,
            _marker: PhantomData,
        },
    )
}
