use core::cell::UnsafeCell;
use core::future::Future;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::pin::Pin;
use core::task::{Context, Poll};
use gpu_atomics::{sys_load_acquire_u32, sys_store_release_u32};

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
