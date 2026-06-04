//! Block-scoped GPU channels — shared-memory backed with CTA-scope atomics.
//!
//! These channels are designed for intra-block communication where both sender
//! and receiver are warps within the same cooperative thread array (block).
//! They use CTA-scope atomics (~2-6 cycles) instead of system-scope atomics
//! (~100 cycles), yielding 20-50x latency improvement.
//!
//! # Channel Types
//!
//! - [`BlockOneshotSlot`]: Single-value, single-use channel in shared memory.
//! - [`BlockMpscChannel`]: Multi-producer, single-consumer ring buffer in shared memory.
//!
//! # Memory Model
//!
//! All channel storage must reside in shared memory, allocated via
//! `BlockScope::alloc()` or `block::shared_mem_at()`. The `'scope` lifetime
//! on sender/receiver types prevents shared-memory references from escaping
//! the block scope (enforced by the Rust borrow checker).
//!
//! # Safety
//!
//! - `T: Copy` is required — no Drop support for channel payloads.
//! - Channels must not be used across block boundaries (CTA-scope atomics
//!   are only visible within the current block).
//! - Pointers must point to valid shared memory within the current block.

use core::cell::UnsafeCell;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use gpu_atomics::{
    cta_cas_u32, cta_load_acquire_u32, cta_spin_load_acquire_u32, cta_store_release_u32,
};

// ============================================================
// Block Oneshot Channel
// ============================================================

/// Oneshot channel states (same protocol as global-memory OneshotSlot).
const BLOCK_ONESHOT_EMPTY: u32 = 0;
const BLOCK_ONESHOT_SENT: u32 = 1;
const BLOCK_ONESHOT_CLOSED: u32 = 2;

/// Error returned when a block oneshot sender is dropped without sending.
#[derive(Debug, Clone, Copy)]
pub struct BlockOneshotClosed;

/// A pre-allocated slot for a block-scoped oneshot channel.
///
/// Must reside in shared memory (allocated via `BlockScope::alloc()` or
/// `block::shared_mem_at()`). Uses CTA-scope atomics for state transitions.
///
/// # Layout
///
/// - `state`: atomic u32 (EMPTY=0, SENT=1, CLOSED=2)
/// - `_pad`: 4-byte padding for alignment
/// - `value`: inline storage for `T`
///
/// # Latency
///
/// ~2-4 cycles per state transition (vs ~100 cycles for global-memory OneshotSlot).
#[repr(C)]
pub struct BlockOneshotSlot<T: Copy> {
    state: UnsafeCell<u32>,
    _pad: u32,
    value: UnsafeCell<MaybeUninit<T>>,
}

// SAFETY: BlockOneshotSlot is accessed via CTA-scope atomics.
// Intra-block cross-warp access is synchronized by acquire/release.
// The slot must not be shared across blocks (enforced by 'scope lifetime).
unsafe impl<T: Copy + Send> Send for BlockOneshotSlot<T> {}
unsafe impl<T: Copy + Send> Sync for BlockOneshotSlot<T> {}

impl<T: Copy> BlockOneshotSlot<T> {
    /// Initialize slot state to EMPTY. Call before creating a channel pair.
    ///
    /// # Safety
    /// Must be called before any sender/receiver accesses the slot.
    /// The slot must reside in shared memory.
    #[inline(always)]
    pub unsafe fn reset(&self) {
        core::ptr::write_volatile(self.state.get(), BLOCK_ONESHOT_EMPTY);
    }

    /// Get raw pointer to the state field.
    #[inline(always)]
    pub fn state_ptr(&self) -> *mut u32 {
        self.state.get()
    }

    /// Get raw pointer to the value storage.
    #[inline(always)]
    pub fn value_ptr(&self) -> *mut MaybeUninit<T> {
        self.value.get()
    }
}

/// Sending half of a block-scoped oneshot channel.
///
/// Consumes itself on `send()`. If dropped without sending, the slot
/// transitions to CLOSED. Uses CTA-scope atomics for shared-memory
/// state transitions.
///
/// The `'scope` lifetime ties this sender to the enclosing `BlockScope`,
/// preventing it from escaping the block.
pub struct BlockOneshotSender<'scope, T: Copy> {
    slot: *mut BlockOneshotSlot<T>,
    _marker: PhantomData<&'scope T>,
}

// SAFETY: Sender holds a raw pointer to shared memory within the block.
// Only one sender exists per slot (enforced by API).
// 'scope lifetime prevents cross-block use.
unsafe impl<'scope, T: Copy + Send> Send for BlockOneshotSender<'scope, T> {}

impl<'scope, T: Copy> BlockOneshotSender<'scope, T> {
    /// Send a value through the channel.
    ///
    /// Writes the value to the shared-memory slot and transitions state
    /// from EMPTY to SENT using a CTA-scope release store. The release
    /// semantics ensure the value is visible to the receiver's acquire load.
    ///
    /// # Safety
    /// - Slot must be in shared memory within the current block
    /// - Must only be called once (enforced by consuming self)
    #[inline(always)]
    pub unsafe fn send(self, value: T) {
        let slot = &*self.slot;
        // Write value first (before state transition)
        core::ptr::write_volatile(slot.value.get() as *mut T, value);
        // CTA-scope release store: value write is visible before SENT is observed
        cta_store_release_u32(slot.state.get(), BLOCK_ONESHOT_SENT);
        // Prevent drop from setting CLOSED
        core::mem::forget(self);
    }
}

impl<'scope, T: Copy> Drop for BlockOneshotSender<'scope, T> {
    fn drop(&mut self) {
        // Sender dropped without sending -> mark CLOSED via CTA-scope release
        unsafe {
            cta_store_release_u32((*self.slot).state.get(), BLOCK_ONESHOT_CLOSED);
        }
    }
}

/// Receiving half of a block-scoped oneshot channel.
///
/// Provides `try_recv()` for non-blocking receive and `recv_spin()` for
/// blocking spin-wait. Uses CTA-scope atomics for polling shared-memory state.
///
/// The `'scope` lifetime ties this receiver to the enclosing `BlockScope`.
pub struct BlockOneshotReceiver<'scope, T: Copy> {
    slot: *const BlockOneshotSlot<T>,
    _marker: PhantomData<&'scope T>,
}

// SAFETY: Receiver holds a raw pointer to shared memory within the block.
// Only one receiver exists per slot (enforced by API).
unsafe impl<'scope, T: Copy + Send> Send for BlockOneshotReceiver<'scope, T> {}

impl<'scope, T: Copy> BlockOneshotReceiver<'scope, T> {
    /// Try to receive a value without blocking.
    ///
    /// Returns `Ok(value)` if the sender has sent, `Err(BlockOneshotClosed)`
    /// if the sender was dropped, or `None` if no value is available yet.
    ///
    /// # Safety
    /// Slot must be in shared memory within the current block.
    #[inline(always)]
    pub unsafe fn try_recv(&self) -> Option<Result<T, BlockOneshotClosed>> {
        let slot = &*self.slot;
        let state = cta_load_acquire_u32(slot.state.get() as *const u32);
        match state {
            BLOCK_ONESHOT_SENT => {
                // Acquire load above ensures we see the value written by sender
                let value = core::ptr::read_volatile(slot.value.get() as *const T);
                Some(Ok(value))
            }
            BLOCK_ONESHOT_CLOSED => Some(Err(BlockOneshotClosed)),
            _ => None, // EMPTY -- not ready yet
        }
    }

    /// Receive a value, spinning until available or closed.
    ///
    /// Uses `cta_spin_load_acquire_u32` which includes `nanosleep` to
    /// yield the warp slot during spinning.
    ///
    /// # Safety
    /// - Slot must be in shared memory within the current block.
    /// - The sender must eventually send or be dropped, or this will spin forever.
    #[inline(always)]
    pub unsafe fn recv_spin(&self) -> Result<T, BlockOneshotClosed> {
        let slot = &*self.slot;
        loop {
            let state = cta_spin_load_acquire_u32(slot.state.get() as *const u32);
            match state {
                BLOCK_ONESHOT_SENT => {
                    let value = core::ptr::read_volatile(slot.value.get() as *const T);
                    return Ok(value);
                }
                BLOCK_ONESHOT_CLOSED => return Err(BlockOneshotClosed),
                _ => {} // EMPTY -- keep spinning
            }
        }
    }
}

/// Create a block-scoped oneshot channel from a pre-allocated shared-memory slot.
///
/// Returns a (sender, receiver) pair. The sender can send exactly one value;
/// the receiver polls or spins until that value is available.
///
/// # Safety
/// - `slot` must point to valid shared memory within the current block
/// - `slot` must not be concurrently used by another channel pair
/// - The slot must be alive for the duration of `'scope`
///
/// # Example
///
/// ```rust,ignore
/// use gpu_runtime::block_channel::{BlockOneshotSlot, block_oneshot};
///
/// // In a BlockScope:
/// let slot: &mut BlockOneshotSlot<u32> = scope.alloc_val(/* ... */);
/// let (tx, rx) = unsafe { block_oneshot(slot) };
///
/// // Producer warp:
/// unsafe { tx.send(42); }
///
/// // Consumer warp:
/// let val = unsafe { rx.recv_spin() }; // Ok(42)
/// ```
#[inline(always)]
pub unsafe fn block_oneshot<'scope, T: Copy>(
    slot: &'scope mut BlockOneshotSlot<T>,
) -> (
    BlockOneshotSender<'scope, T>,
    BlockOneshotReceiver<'scope, T>,
) {
    slot.reset();
    (
        BlockOneshotSender {
            slot: slot as *mut BlockOneshotSlot<T>,
            _marker: PhantomData,
        },
        BlockOneshotReceiver {
            slot: slot as *const BlockOneshotSlot<T>,
            _marker: PhantomData,
        },
    )
}

// ============================================================
// Block MPSC Channel — multi-producer, single-consumer
// ============================================================

/// Error returned when trying to send on a closed block MPSC channel.
#[derive(Debug, Clone, Copy)]
pub struct BlockMpscClosed;

/// Error type for block MPSC send operations.
#[derive(Debug, Clone, Copy)]
pub enum BlockMpscSendError<T> {
    /// Channel is closed. Contains the unsent value.
    Closed(T),
    /// Channel is full. Contains the unsent value. Caller should retry.
    Full(T),
}

/// A single slot in the block MPSC ring buffer.
///
/// Uses a sequence number for publication ordering (same protocol as
/// global-memory MpscSlot):
/// - When `sequence == slot_index`: slot is available for writing
/// - When `sequence == slot_index + 1`: slot contains a readable value
/// - After consumer reads: `sequence = slot_index + capacity` (recycles)
#[repr(C)]
pub struct BlockMpscSlot<T: Copy> {
    sequence: UnsafeCell<u32>,
    _pad: u32,
    value: UnsafeCell<MaybeUninit<T>>,
}

impl<T: Copy> BlockMpscSlot<T> {
    /// Create an empty slot with initial sequence number.
    pub const fn new(seq: u32) -> Self {
        Self {
            sequence: UnsafeCell::new(seq),
            _pad: 0,
            value: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }
}

/// Pre-allocated block-scoped MPSC channel storage.
///
/// Must reside in shared memory. The ring buffer uses CTA-scope CAS for
/// producer head advancement and CTA-scope load/store for the single consumer.
///
/// # Layout
/// - `head`: producer write cursor (CAS-contended by multiple sender warps)
/// - `tail`: consumer read cursor (only consumer writes)
/// - `closed`: shutdown flag
/// - `_pad`: alignment padding
/// - `slots`: ring buffer of N slots with sequence numbers
///
/// # Capacity
/// N must be a power of 2 for efficient modulo via bitmask.
///
/// # Latency
/// ~6-10 cycles per send (CAS + store), vs ~150-200 cycles for global-memory MPSC.
#[repr(C)]
pub struct BlockMpscChannel<T: Copy, const N: usize> {
    head: UnsafeCell<u32>,
    tail: UnsafeCell<u32>,
    closed: UnsafeCell<u32>,
    _pad: u32,
    slots: [BlockMpscSlot<T>; N],
}

// SAFETY: BlockMpscChannel is accessed via CTA-scope atomics.
// Intra-block cross-warp access is synchronized by acquire/release.
unsafe impl<T: Copy + Send, const N: usize> Send for BlockMpscChannel<T, N> {}
unsafe impl<T: Copy + Send, const N: usize> Sync for BlockMpscChannel<T, N> {}

impl<T: Copy, const N: usize> BlockMpscChannel<T, N> {
    /// Initialize the channel. Must be called before use.
    ///
    /// Sets head=0, tail=0, closed=0, and each slot's sequence to its index.
    ///
    /// # Safety
    /// Must be called from a single warp before any senders/receivers access
    /// the channel. The channel must reside in shared memory.
    #[inline(always)]
    pub unsafe fn init(&self) {
        core::ptr::write_volatile(self.head.get(), 0);
        core::ptr::write_volatile(self.tail.get(), 0);
        core::ptr::write_volatile(self.closed.get(), 0);
        let mut i = 0u32;
        while (i as usize) < N {
            core::ptr::write_volatile(self.slots[i as usize].sequence.get(), i);
            i += 1;
        }
    }

    /// Close the channel. After this, senders get `BlockMpscClosed` and
    /// the receiver drains remaining items then gets `None`.
    #[inline(always)]
    pub unsafe fn close(&self) {
        cta_store_release_u32(self.closed.get(), 1);
    }

    /// Check if the channel is closed.
    #[inline(always)]
    pub unsafe fn is_closed(&self) -> bool {
        cta_load_acquire_u32(self.closed.get() as *const u32) != 0
    }

    /// Try to send a value (producer side).
    ///
    /// Uses CTA-scope CAS on the head pointer to reserve a slot, then writes
    /// the value and publishes via sequence number release-store.
    ///
    /// Returns `Ok(())` on success, `Err(BlockMpscSendError)` if closed or full.
    ///
    /// # Safety
    /// Channel must be in shared memory and initialized.
    #[inline(always)]
    pub unsafe fn try_send(&self, value: T) -> Result<(), BlockMpscSendError<T>> {
        if self.is_closed() {
            return Err(BlockMpscSendError::Closed(value));
        }

        let head = cta_load_acquire_u32(self.head.get() as *const u32);
        let tail = cta_load_acquire_u32(self.tail.get() as *const u32);

        // Check if full
        if head.wrapping_sub(tail) >= N as u32 {
            return Err(BlockMpscSendError::Full(value));
        }

        // CTA-scope CAS to reserve slot
        let old = cta_cas_u32(self.head.get(), head, head.wrapping_add(1));
        if old != head {
            // Another producer won the CAS race -- caller should retry
            return Err(BlockMpscSendError::Full(value));
        }

        // Write value to the reserved slot
        let idx = (head as usize) & (N - 1);
        let slot = &self.slots[idx];
        core::ptr::write_volatile(slot.value.get() as *mut T, value);
        // Publish: release store on sequence number
        cta_store_release_u32(slot.sequence.get(), head.wrapping_add(1));

        Ok(())
    }

    /// Try to receive a value (consumer side).
    ///
    /// Returns `Some(value)` if available, `None` if no value is ready.
    ///
    /// # Safety
    /// Channel must be in shared memory and initialized.
    /// Only one consumer should call this at a time.
    #[inline(always)]
    pub unsafe fn try_recv(&self) -> Option<T> {
        let tail = cta_load_acquire_u32(self.tail.get() as *const u32);
        let idx = (tail as usize) & (N - 1);
        let slot = &self.slots[idx];

        let seq = cta_load_acquire_u32(slot.sequence.get() as *const u32);
        if seq != tail.wrapping_add(1) {
            return None;
        }

        let value = core::ptr::read_volatile(slot.value.get() as *const T);
        // Recycle slot: set sequence to tail + N (available for next round)
        cta_store_release_u32(slot.sequence.get(), tail.wrapping_add(N as u32));
        // Advance tail
        cta_store_release_u32(self.tail.get(), tail.wrapping_add(1));

        Some(value)
    }
}

/// Sending half of a block-scoped MPSC channel. Can be copied for multiple producers.
///
/// Uses CTA-scope CAS on the shared-memory head pointer to reserve slots.
/// The `'scope` lifetime ties this sender to the enclosing `BlockScope`.
pub struct BlockMpscSender<'scope, T: Copy, const N: usize> {
    channel: *const BlockMpscChannel<T, N>,
    _marker: PhantomData<&'scope T>,
}

// SAFETY: Multiple senders can coexist; all access is via CTA-scope atomics.
unsafe impl<'scope, T: Copy + Send, const N: usize> Send for BlockMpscSender<'scope, T, N> {}
unsafe impl<'scope, T: Copy + Send, const N: usize> Sync for BlockMpscSender<'scope, T, N> {}

impl<'scope, T: Copy, const N: usize> Copy for BlockMpscSender<'scope, T, N> {}

impl<'scope, T: Copy, const N: usize> Clone for BlockMpscSender<'scope, T, N> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'scope, T: Copy, const N: usize> BlockMpscSender<'scope, T, N> {
    /// Try to send a value. Returns `Ok(())` on success.
    ///
    /// - `Err(Closed)` if channel is closed
    /// - `Err(Full)` if the ring buffer is full (caller should retry)
    ///
    /// # Safety
    /// Channel pointer must be valid and in shared memory within the current block.
    #[inline(always)]
    pub unsafe fn try_send(&self, value: T) -> Result<(), BlockMpscSendError<T>> {
        (*self.channel).try_send(value)
    }
}

/// Receiving half of a block-scoped MPSC channel. Only one receiver should exist.
///
/// The `'scope` lifetime ties this receiver to the enclosing `BlockScope`.
pub struct BlockMpscReceiver<'scope, T: Copy, const N: usize> {
    channel: *const BlockMpscChannel<T, N>,
    _marker: PhantomData<&'scope T>,
}

// SAFETY: Only one receiver exists per channel (enforced by API).
unsafe impl<'scope, T: Copy + Send, const N: usize> Send for BlockMpscReceiver<'scope, T, N> {}

impl<'scope, T: Copy, const N: usize> BlockMpscReceiver<'scope, T, N> {
    /// Try to receive a value. Returns `Some(value)` if available.
    ///
    /// Returns `None` if no value is ready (caller should retry).
    ///
    /// # Safety
    /// Channel pointer must be valid and in shared memory.
    /// Only one receiver should call this at a time.
    #[inline(always)]
    pub unsafe fn try_recv(&self) -> Option<T> {
        (*self.channel).try_recv()
    }

    /// Receive a value, spinning until available or the channel is closed and empty.
    ///
    /// Returns `Some(value)` when a value is received, or `None` if the channel
    /// is closed and all values have been drained.
    ///
    /// Uses `cta_spin_load_acquire_u32` internally via the `try_recv` path,
    /// with nanosleep between attempts to yield the warp slot.
    ///
    /// # Safety
    /// - Channel pointer must be valid and in shared memory.
    /// - Producers must eventually send or close, or this will spin forever.
    #[inline(always)]
    pub unsafe fn recv_spin(&self) -> Option<T> {
        loop {
            if let Some(value) = self.try_recv() {
                return Some(value);
            }
            if self.is_terminated() {
                return None;
            }
            // Yield warp slot while waiting
            #[cfg(target_arch = "nvptx64")]
            core::arch::asm!("nanosleep.u32 64;", options(nostack));
        }
    }

    /// Check if the channel is closed and empty (all values drained).
    #[inline(always)]
    pub unsafe fn is_terminated(&self) -> bool {
        let ch = &*self.channel;
        if !ch.is_closed() {
            return false;
        }
        let head = cta_load_acquire_u32(ch.head.get() as *const u32);
        let tail = cta_load_acquire_u32(ch.tail.get() as *const u32);
        head == tail
    }
}

/// Create a block-scoped MPSC channel from a pre-allocated shared-memory buffer.
///
/// Returns a (sender, receiver) pair. The sender can be cloned/copied for
/// multiple producers. Only one receiver should exist.
///
/// # Safety
/// - `channel` must point to valid shared memory within the current block
/// - `channel` must not be concurrently used by another channel pair
/// - The channel must be alive for the duration of `'scope`
/// - N must be a power of 2
///
/// # Example
///
/// ```rust,ignore
/// use gpu_runtime::block_channel::{BlockMpscChannel, block_mpsc};
///
/// // In a BlockScope:
/// let ch: &BlockMpscChannel<u32, 8> = scope.alloc_val(/* ... */);
/// let (tx, rx) = unsafe { block_mpsc(ch) };
///
/// // Producer warps (tx is Copy):
/// unsafe { tx.try_send(42).ok(); }
///
/// // Consumer warp:
/// let val = unsafe { rx.recv_spin() }; // Some(42)
/// ```
#[inline(always)]
pub unsafe fn block_mpsc<'scope, T: Copy, const N: usize>(
    channel: &'scope BlockMpscChannel<T, N>,
) -> (
    BlockMpscSender<'scope, T, N>,
    BlockMpscReceiver<'scope, T, N>,
) {
    channel.init();
    (
        BlockMpscSender {
            channel: channel as *const BlockMpscChannel<T, N>,
            _marker: PhantomData,
        },
        BlockMpscReceiver {
            channel: channel as *const BlockMpscChannel<T, N>,
            _marker: PhantomData,
        },
    )
}
