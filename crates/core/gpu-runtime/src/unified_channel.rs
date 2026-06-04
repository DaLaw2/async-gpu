//! Unified GPU channel API — auto-selects transport based on scope.
//!
//! Users write a single API surface (`scope.oneshot()`, `gscope.oneshot()`)
//! and the transport is chosen automatically:
//!
//! - **Block scope** → shared memory with CTA-scope atomics (~2-6 cycles)
//! - **Grid scope** → global memory with system-scope atomics (~100 cycles)
//!
//! The unified enum types [`ScopedOneshotSender`] / [`ScopedOneshotReceiver`]
//! and [`ScopedMpscSender`] / [`ScopedMpscReceiver`] dispatch to the correct
//! transport at zero cost — the enum match compiles to a branch on a constant
//! known at construction time (no vtable, no indirection).
//!
//! # Design Rationale
//!
//! - **Enum, not trait object**: No vtable overhead on GPU. The discriminant is
//!   set at construction and never changes, so the branch predictor eliminates
//!   the overhead after the first iteration.
//! - **`'scope` lifetime**: Channels cannot escape their scope. For block scope,
//!   this is the shared memory lifetime. For grid scope, this is the global
//!   memory pool lifetime.
//! - **`T: Copy` required**: GPU channels do not support Drop types.
//! - **shfl.sync not included**: Per sc-channel.1 findings, shuffle is a
//!   broadcast primitive, not a channel.

use core::marker::PhantomData;

use crate::block_channel::{
    block_mpsc, block_oneshot, BlockMpscChannel, BlockMpscReceiver, BlockMpscSendError,
    BlockMpscSender, BlockOneshotClosed, BlockOneshotReceiver, BlockOneshotSender,
    BlockOneshotSlot,
};
use crate::channel::{
    mpsc, oneshot, MpscChannel, MpscReceiver, MpscSendError, MpscSender, OneshotSender, OneshotSlot,
};

// ============================================================
// Scoped Oneshot — unified enum
// ============================================================

/// Sending half of a scope-bound oneshot channel.
///
/// Dispatches to either block-scoped (shared memory, CTA atomics) or
/// grid-scoped (global memory, system atomics) transport depending on
/// how the channel was created.
pub enum ScopedOneshotSender<'scope, T: Copy> {
    /// Block-scoped: shared memory, CTA-scope atomics.
    Block(BlockOneshotSender<'scope, T>),
    /// Grid-scoped: global memory, system-scope atomics.
    /// Wraps the unscoped `OneshotSender` with a `'scope` lifetime.
    Grid(GridOneshotSender<'scope, T>),
}

impl<'scope, T: Copy> ScopedOneshotSender<'scope, T> {
    /// Send a value through the channel, consuming the sender.
    ///
    /// # Safety
    ///
    /// - For block variant: slot must be in shared memory within the current block.
    /// - For grid variant: slot must be in global memory.
    /// - Must only be called once (enforced by consuming self).
    #[inline(always)]
    pub unsafe fn send(self, value: T) {
        match self {
            ScopedOneshotSender::Block(tx) => tx.send(value),
            ScopedOneshotSender::Grid(tx) => tx.inner.send(value),
        }
    }
}

/// Receiving half of a scope-bound oneshot channel.
///
/// Dispatches to either block-scoped or grid-scoped transport.
pub enum ScopedOneshotReceiver<'scope, T: Copy> {
    /// Block-scoped: shared memory, CTA-scope atomics.
    Block(BlockOneshotReceiver<'scope, T>),
    /// Grid-scoped: global memory, system-scope atomics.
    Grid(GridOneshotReceiver<'scope, T>),
}

/// Error returned when a scoped oneshot sender is dropped without sending.
#[derive(Debug, Clone, Copy)]
pub struct ScopedOneshotClosed;

impl<'scope, T: Copy> ScopedOneshotReceiver<'scope, T> {
    /// Try to receive a value without blocking.
    ///
    /// Returns `Ok(value)` if the sender has sent, `Err(ScopedOneshotClosed)`
    /// if the sender was dropped, or `None` if no value is available yet.
    ///
    /// # Safety
    ///
    /// - For block variant: slot must be in shared memory within the current block.
    /// - For grid variant: slot must be in global memory.
    #[inline(always)]
    pub unsafe fn try_recv(&self) -> Option<Result<T, ScopedOneshotClosed>> {
        match self {
            ScopedOneshotReceiver::Block(rx) => rx.try_recv().map(|r| match r {
                Ok(v) => Ok(v),
                Err(BlockOneshotClosed) => Err(ScopedOneshotClosed),
            }),
            ScopedOneshotReceiver::Grid(rx) => {
                // Poll the OneshotSlot directly via its state pointer
                let slot = &*rx.slot;
                let state = gpu_atomics::sys_load_acquire_u32(slot.state_ptr() as *const u32);
                match state {
                    1 => {
                        // SENT — acquire above ensures value is visible
                        let value = core::ptr::read_volatile(slot.value_ptr() as *const T);
                        Some(Ok(value))
                    }
                    2 => Some(Err(ScopedOneshotClosed)), // CLOSED
                    _ => None,                           // EMPTY
                }
            }
        }
    }

    /// Receive a value, spinning until available or closed.
    ///
    /// Uses nanosleep to yield the warp slot during spinning.
    ///
    /// # Safety
    ///
    /// - Slot must be in the correct memory space for its variant.
    /// - The sender must eventually send or be dropped, or this will spin forever.
    #[inline(always)]
    pub unsafe fn recv_spin(&self) -> Result<T, ScopedOneshotClosed> {
        loop {
            if let Some(result) = self.try_recv() {
                return result;
            }
            #[cfg(target_arch = "nvptx64")]
            core::arch::asm!("nanosleep.u32 64;", options(nostack));
        }
    }
}

// ============================================================
// Grid-scoped oneshot wrappers — add 'scope lifetime
// ============================================================

/// Grid-scoped oneshot sender — wraps `OneshotSender<T>` with a `'scope` lifetime.
///
/// The `'scope` lifetime prevents this sender from escaping the `GridScope`.
pub struct GridOneshotSender<'scope, T: Copy> {
    inner: OneshotSender<T>,
    _marker: PhantomData<&'scope ()>,
}

/// Grid-scoped oneshot receiver — holds a raw pointer to the slot with
/// `'scope` lifetime, providing `try_recv` and `recv_spin` semantics
/// without requiring the `Future` trait.
pub struct GridOneshotReceiver<'scope, T: Copy> {
    slot: *const OneshotSlot<T>,
    _marker: PhantomData<&'scope ()>,
}

// SAFETY: Same constraints as OneshotSender/OneshotReceiver — single
// sender/receiver per slot, system-scope atomics for cross-block visibility.
unsafe impl<'scope, T: Copy + Send> Send for GridOneshotSender<'scope, T> {}
unsafe impl<'scope, T: Copy + Send> Send for GridOneshotReceiver<'scope, T> {}

// ============================================================
// Scoped MPSC — unified enum
// ============================================================

/// Sending half of a scope-bound MPSC channel.
///
/// Dispatches to block-scoped (shared memory) or grid-scoped (global memory).
/// Block-scoped senders are `Copy` (raw pointer). Grid-scoped senders are
/// `Clone` only (matches the underlying `MpscSender`).
pub enum ScopedMpscSender<'scope, T: Copy, const N: usize> {
    /// Block-scoped: shared memory, CTA-scope atomics.
    Block(BlockMpscSender<'scope, T, N>),
    /// Grid-scoped: global memory, system-scope atomics.
    Grid(GridMpscSender<'scope, T, N>),
}

/// Error type for scoped MPSC send operations.
#[derive(Debug, Clone, Copy)]
pub enum ScopedMpscSendError<T> {
    /// Channel is closed. Contains the unsent value.
    Closed(T),
    /// Channel is full. Contains the unsent value. Caller should retry.
    Full(T),
}

impl<'scope, T: Copy, const N: usize> Clone for ScopedMpscSender<'scope, T, N> {
    fn clone(&self) -> Self {
        match self {
            ScopedMpscSender::Block(tx) => ScopedMpscSender::Block(*tx),
            ScopedMpscSender::Grid(tx) => ScopedMpscSender::Grid(tx.clone()),
        }
    }
}

impl<'scope, T: Copy, const N: usize> ScopedMpscSender<'scope, T, N> {
    /// Try to send a value. Returns `Ok(())` on success.
    ///
    /// - `Err(Closed)` if channel is closed
    /// - `Err(Full)` if the ring buffer is full (caller should retry)
    ///
    /// # Safety
    ///
    /// Channel pointer must be valid and in the correct memory space.
    #[inline(always)]
    pub unsafe fn try_send(&self, value: T) -> Result<(), ScopedMpscSendError<T>> {
        match self {
            ScopedMpscSender::Block(tx) => tx.try_send(value).map_err(|e| match e {
                BlockMpscSendError::Closed(v) => ScopedMpscSendError::Closed(v),
                BlockMpscSendError::Full(v) => ScopedMpscSendError::Full(v),
            }),
            ScopedMpscSender::Grid(tx) => tx.inner.try_send(value).map_err(|e| match e {
                MpscSendError::Closed(v) => ScopedMpscSendError::Closed(v),
                MpscSendError::Full(v) => ScopedMpscSendError::Full(v),
            }),
        }
    }
}

/// Receiving half of a scope-bound MPSC channel.
pub enum ScopedMpscReceiver<'scope, T: Copy, const N: usize> {
    /// Block-scoped: shared memory, CTA-scope atomics.
    Block(BlockMpscReceiver<'scope, T, N>),
    /// Grid-scoped: global memory, system-scope atomics.
    Grid(GridMpscReceiver<'scope, T, N>),
}

impl<'scope, T: Copy, const N: usize> ScopedMpscReceiver<'scope, T, N> {
    /// Try to receive a value. Returns `Some(value)` if available.
    ///
    /// Returns `None` if no value is ready (caller should retry).
    ///
    /// # Safety
    ///
    /// Channel pointer must be valid and in the correct memory space.
    /// Only one receiver should call this at a time.
    #[inline(always)]
    pub unsafe fn try_recv(&self) -> Option<T> {
        match self {
            ScopedMpscReceiver::Block(rx) => rx.try_recv(),
            ScopedMpscReceiver::Grid(rx) => rx.inner.try_recv(),
        }
    }

    /// Receive a value, spinning until available or the channel is closed and empty.
    ///
    /// Returns `Some(value)` when a value is received, or `None` if the channel
    /// is closed and all values have been drained.
    ///
    /// # Safety
    ///
    /// - Channel pointer must be valid and in the correct memory space.
    /// - Producers must eventually send or close, or this will spin forever.
    #[inline(always)]
    pub unsafe fn recv_spin(&self) -> Option<T> {
        match self {
            ScopedMpscReceiver::Block(rx) => rx.recv_spin(),
            ScopedMpscReceiver::Grid(rx) => {
                // Spin-poll the global MPSC receiver
                loop {
                    if let Some(value) = rx.inner.try_recv() {
                        return Some(value);
                    }
                    if rx.inner.is_terminated() {
                        return None;
                    }
                    #[cfg(target_arch = "nvptx64")]
                    core::arch::asm!("nanosleep.u32 64;", options(nostack));
                }
            }
        }
    }
}

// ============================================================
// Grid-scoped MPSC wrappers — add 'scope lifetime
// ============================================================

/// Grid-scoped MPSC sender — wraps `MpscSender<T, N>` with a `'scope` lifetime.
pub struct GridMpscSender<'scope, T: Copy, const N: usize> {
    inner: MpscSender<T, N>,
    _marker: PhantomData<&'scope ()>,
}

// SAFETY: Same constraints as MpscSender.
unsafe impl<'scope, T: Copy + Send, const N: usize> Send for GridMpscSender<'scope, T, N> {}
unsafe impl<'scope, T: Copy + Send, const N: usize> Sync for GridMpscSender<'scope, T, N> {}

impl<'scope, T: Copy, const N: usize> Clone for GridMpscSender<'scope, T, N> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            _marker: PhantomData,
        }
    }
}

/// Grid-scoped MPSC receiver — wraps `MpscReceiver<T, N>` with a `'scope` lifetime.
pub struct GridMpscReceiver<'scope, T: Copy, const N: usize> {
    inner: MpscReceiver<T, N>,
    _marker: PhantomData<&'scope ()>,
}

// SAFETY: Same constraints as MpscReceiver.
unsafe impl<'scope, T: Copy + Send, const N: usize> Send for GridMpscReceiver<'scope, T, N> {}

// ============================================================
// BlockScope integration
// ============================================================

impl<'scope> crate::scope::BlockScope<'scope> {
    /// Create a block-scoped oneshot channel.
    ///
    /// Allocates the oneshot slot from shared memory via the scope's bump
    /// allocator. The returned sender and receiver are tied to `'scope` —
    /// they cannot outlive the block scope.
    ///
    /// Uses CTA-scope atomics (~2-6 cycles per state transition).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use gpu_runtime::scope::block_scope;
    ///
    /// block_scope(|scope| {
    ///     let (tx, rx) = scope.oneshot::<u32>();
    ///
    ///     scope.spawn(move || {
    ///         unsafe { tx.send(42); }
    ///     });
    ///
    ///     let value = unsafe { rx.recv_spin() }; // Ok(42)
    /// });
    /// ```
    pub fn oneshot<T: Copy>(
        &self,
    ) -> (
        ScopedOneshotSender<'scope, T>,
        ScopedOneshotReceiver<'scope, T>,
    ) {
        // BlockOneshotSlot contains UnsafeCell so it is not Copy — cannot use
        // the typed alloc()/alloc_uninit(). Use alloc_raw_bytes() to get
        // properly aligned memory, then cast. block_oneshot() calls reset().
        let slot: &'scope mut BlockOneshotSlot<T> = unsafe {
            let size = core::mem::size_of::<BlockOneshotSlot<T>>();
            let align = core::mem::align_of::<BlockOneshotSlot<T>>();
            let ptr = self.alloc_raw_bytes(size, align);
            &mut *(ptr as *mut BlockOneshotSlot<T>)
        };

        // Create the channel pair (initializes the slot)
        let (tx, rx) = unsafe { block_oneshot(slot) };

        (
            ScopedOneshotSender::Block(tx),
            ScopedOneshotReceiver::Block(rx),
        )
    }

    /// Create a block-scoped MPSC channel with capacity N.
    ///
    /// Allocates the ring buffer from shared memory. N must be a power of 2.
    /// The sender can be cloned for multiple producers.
    ///
    /// Uses CTA-scope atomics (~6-10 cycles per send).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use gpu_runtime::scope::block_scope;
    ///
    /// block_scope(|scope| {
    ///     let (tx, rx) = scope.mpsc::<u32, 8>();
    ///
    ///     scope.spawn(move || {
    ///         unsafe { tx.try_send(1).ok(); }
    ///         unsafe { tx.try_send(2).ok(); }
    ///     });
    ///
    ///     let v1 = unsafe { rx.recv_spin() }; // Some(1)
    ///     let v2 = unsafe { rx.recv_spin() }; // Some(2)
    /// });
    /// ```
    pub fn mpsc<T: Copy, const N: usize>(
        &self,
    ) -> (
        ScopedMpscSender<'scope, T, N>,
        ScopedMpscReceiver<'scope, T, N>,
    ) {
        // BlockMpscChannel contains UnsafeCell so it is not Copy — use
        // alloc_raw_bytes(). block_mpsc() calls init() to set up the ring buffer.
        let ch: &'scope BlockMpscChannel<T, N> = unsafe {
            let size = core::mem::size_of::<BlockMpscChannel<T, N>>();
            let align = core::mem::align_of::<BlockMpscChannel<T, N>>();
            let ptr = self.alloc_raw_bytes(size, align);
            &*(ptr as *const BlockMpscChannel<T, N>)
        };

        let (tx, rx) = unsafe { block_mpsc(ch) };

        (ScopedMpscSender::Block(tx), ScopedMpscReceiver::Block(rx))
    }
}

// ============================================================
// GridScope integration
// ============================================================

impl<'scope> crate::scope::GridScope<'scope> {
    /// Create a grid-scoped oneshot channel.
    ///
    /// Allocates the oneshot slot from the global memory pool. The returned
    /// sender and receiver are tied to `'scope` — they cannot outlive the
    /// grid scope.
    ///
    /// Uses system-scope atomics (~100 cycles per state transition) for
    /// cross-block visibility.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use gpu_runtime::scope::grid_scope;
    ///
    /// unsafe {
    ///     grid_scope(pool, pool_size, |gscope| {
    ///         let (tx, rx) = gscope.oneshot::<u32>();
    ///
    ///         // Hand tx to a worker block...
    ///         // tx.send(result) from the worker
    ///
    ///         let value = rx.recv_spin(); // Ok(result)
    ///     });
    /// }
    /// ```
    pub fn oneshot<T: Copy>(
        &self,
    ) -> (
        ScopedOneshotSender<'scope, T>,
        ScopedOneshotReceiver<'scope, T>,
    ) {
        // OneshotSlot contains UnsafeCell so it is not Copy — use
        // alloc_raw_bytes(). oneshot() calls reset() to initialize the slot.
        let slot: &'scope mut OneshotSlot<T> = unsafe {
            let size = core::mem::size_of::<OneshotSlot<T>>();
            let align = core::mem::align_of::<OneshotSlot<T>>();
            let ptr = self.alloc_raw_bytes(size, align);
            &mut *(ptr as *mut OneshotSlot<T>)
        };

        let slot_ptr = slot as *mut OneshotSlot<T>;

        // Create the sender/receiver pair (initializes the slot)
        let (tx, _rx) = unsafe { oneshot(slot) };

        (
            ScopedOneshotSender::Grid(GridOneshotSender {
                inner: tx,
                _marker: PhantomData,
            }),
            ScopedOneshotReceiver::Grid(GridOneshotReceiver {
                slot: slot_ptr as *const OneshotSlot<T>,
                _marker: PhantomData,
            }),
        )
    }

    /// Create a grid-scoped MPSC channel with capacity N.
    ///
    /// Allocates the ring buffer from the global memory pool. N must be a
    /// power of 2. The sender can be cloned for multiple producers.
    ///
    /// Uses system-scope atomics for cross-block visibility.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use gpu_runtime::scope::grid_scope;
    ///
    /// unsafe {
    ///     grid_scope(pool, pool_size, |gscope| {
    ///         let (tx, rx) = gscope.mpsc::<u32, 16>();
    ///
    ///         // Hand tx clones to worker blocks...
    ///
    ///         while let Some(v) = rx.recv_spin() {
    ///             // process v
    ///         }
    ///     });
    /// }
    /// ```
    pub fn mpsc<T: Copy, const N: usize>(
        &self,
    ) -> (
        ScopedMpscSender<'scope, T, N>,
        ScopedMpscReceiver<'scope, T, N>,
    ) {
        // MpscChannel contains UnsafeCell so it is not Copy — use
        // alloc_raw_bytes(). mpsc() calls init() to set up the ring buffer.
        let ch: &'scope MpscChannel<T, N> = unsafe {
            let size = core::mem::size_of::<MpscChannel<T, N>>();
            let align = core::mem::align_of::<MpscChannel<T, N>>();
            let ptr = self.alloc_raw_bytes(size, align);
            &*(ptr as *const MpscChannel<T, N>)
        };

        let (tx, rx) = unsafe { mpsc(ch) };

        (
            ScopedMpscSender::Grid(GridMpscSender {
                inner: tx,
                _marker: PhantomData,
            }),
            ScopedMpscReceiver::Grid(GridMpscReceiver {
                inner: rx,
                _marker: PhantomData,
            }),
        )
    }
}
