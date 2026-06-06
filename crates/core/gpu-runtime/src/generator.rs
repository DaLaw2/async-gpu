//! GPU Coroutine Generator — warp-cooperative generator/yield pattern.
//!
//! Provides [`GpuGenerator`], a warp-cooperative generator trait that mirrors
//! [`WarpFuture`](crate::warp_future::WarpFuture) but produces a stream of
//! values instead of a single result. Generators yield values to consumers
//! with zero buffering using warp-cooperative execution.
//!
//! # Key Types
//!
//! - [`WarpCoroutineState<Y, R>`] — result of resuming a generator (Yielded or Complete)
//! - [`GpuGenerator`] — trait for warp-cooperative generators
//! - [`WarpBroadcast`] — trait for broadcasting values from lane 0 to all lanes
//! - [`GeneratorTask`] — adapter wrapping a generator + consumer as a `Future`
//!
//! # Zero-Buffering Streaming Pipeline
//!
//! The [`for_each_yield`] combinator drives a generator inline — the producer
//! yields one value, the consumer processes it, then the producer yields the next.
//! At most ONE yielded value exists at any time.
//!
//! # Example
//!
//! ```rust,ignore
//! use gpu_runtime::generator::*;
//! use gpu_runtime::warp_future::WarpContext;
//!
//! // Drive a generator, processing each yielded value:
//! unsafe {
//!     let mut wcx = WarpContext::new();
//!     for_each_yield(&mut my_gen, |value, wcx| {
//!         // All 32 lanes see the same `value`
//!         // Process it with data-parallel SIMD
//!     }, &mut wcx);
//! }
//! ```

use crate::warp_future::WarpContext;
use gpu_atomics::{shfl_sync_idx_u32, syncwarp};

// ============================================================
// WarpCoroutineState — result of resuming a generator
// ============================================================

/// Result of resuming a warp-cooperative generator.
///
/// Mirror of `core::ops::CoroutineState` but warp-safe:
/// all 32 lanes observe the same variant after `resume_warp()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarpCoroutineState<Y, R> {
    /// Generator yielded a value. All lanes see the same `Y`.
    Yielded(Y),
    /// Generator completed with a return value. All lanes see the same `R`.
    Complete(R),
}

// ============================================================
// WarpBroadcast — broadcast a value from lane 0 to all lanes
// ============================================================

/// Broadcast a value from lane 0 to all lanes in a warp.
///
/// Implementations use the most efficient hardware path:
/// - Scalars (<=32 bits): single `shfl.sync.idx.b32`
/// - Multi-word (33-128 bits): multiple `shfl.sync.idx.b32` calls
///
/// # Safety
///
/// All active lanes must call `broadcast()` simultaneously with the
/// same `mask`. Only lane 0's `value` is used; other lanes' values
/// are overwritten with lane 0's.
pub unsafe trait WarpBroadcast: Copy {
    /// Broadcast this value from lane 0 to all lanes.
    /// Returns the broadcast value (same on all lanes).
    fn broadcast(value: Self, mask: u32) -> Self;
}

// --- Unit type: no broadcast needed ---

unsafe impl WarpBroadcast for () {
    #[inline(always)]
    fn broadcast(_value: (), _mask: u32) {}
}

// --- Single shfl.sync: 1 cycle ---

unsafe impl WarpBroadcast for u32 {
    #[inline(always)]
    fn broadcast(value: Self, mask: u32) -> Self {
        unsafe { shfl_sync_idx_u32(mask, value, 0) }
    }
}

unsafe impl WarpBroadcast for i32 {
    #[inline(always)]
    fn broadcast(value: Self, mask: u32) -> Self {
        let bits = value as u32;
        let result = unsafe { shfl_sync_idx_u32(mask, bits, 0) };
        result as i32
    }
}

unsafe impl WarpBroadcast for f32 {
    #[inline(always)]
    fn broadcast(value: Self, mask: u32) -> Self {
        let bits = value.to_bits();
        let result = unsafe { shfl_sync_idx_u32(mask, bits, 0) };
        f32::from_bits(result)
    }
}

unsafe impl WarpBroadcast for bool {
    #[inline(always)]
    fn broadcast(value: Self, mask: u32) -> Self {
        let bits = value as u32;
        let result = unsafe { shfl_sync_idx_u32(mask, bits, 0) };
        result != 0
    }
}

unsafe impl WarpBroadcast for u8 {
    #[inline(always)]
    fn broadcast(value: Self, mask: u32) -> Self {
        let wide = value as u32;
        let result = unsafe { shfl_sync_idx_u32(mask, wide, 0) };
        result as u8
    }
}

unsafe impl WarpBroadcast for u16 {
    #[inline(always)]
    fn broadcast(value: Self, mask: u32) -> Self {
        let wide = value as u32;
        let result = unsafe { shfl_sync_idx_u32(mask, wide, 0) };
        result as u16
    }
}

unsafe impl WarpBroadcast for i8 {
    #[inline(always)]
    fn broadcast(value: Self, mask: u32) -> Self {
        let wide = value as u32;
        let result = unsafe { shfl_sync_idx_u32(mask, wide, 0) };
        result as i8
    }
}

unsafe impl WarpBroadcast for i16 {
    #[inline(always)]
    fn broadcast(value: Self, mask: u32) -> Self {
        let wide = value as u32;
        let result = unsafe { shfl_sync_idx_u32(mask, wide, 0) };
        result as i16
    }
}

// --- Two shfl.sync calls: 2 cycles ---

unsafe impl WarpBroadcast for u64 {
    #[inline(always)]
    fn broadcast(value: Self, mask: u32) -> Self {
        let lo = value as u32;
        let hi = (value >> 32) as u32;
        let lo = unsafe { shfl_sync_idx_u32(mask, lo, 0) };
        let hi = unsafe { shfl_sync_idx_u32(mask, hi, 0) };
        (lo as u64) | ((hi as u64) << 32)
    }
}

unsafe impl WarpBroadcast for i64 {
    #[inline(always)]
    fn broadcast(value: Self, mask: u32) -> Self {
        u64::broadcast(value as u64, mask) as i64
    }
}

unsafe impl WarpBroadcast for f64 {
    #[inline(always)]
    fn broadcast(value: Self, mask: u32) -> Self {
        let bits = value.to_bits();
        let result = u64::broadcast(bits, mask);
        f64::from_bits(result)
    }
}

unsafe impl WarpBroadcast for (u32, u32) {
    #[inline(always)]
    fn broadcast(value: Self, mask: u32) -> Self {
        let a = unsafe { shfl_sync_idx_u32(mask, value.0, 0) };
        let b = unsafe { shfl_sync_idx_u32(mask, value.1, 0) };
        (a, b)
    }
}

// ============================================================
// GpuGenerator — warp-cooperative generator trait
// ============================================================

/// A generator representing an entire warp (32 lanes) in SIMT lockstep.
///
/// # Contract
///
/// - All active lanes must call `resume_warp()` simultaneously.
/// - The coroutine state discriminant is broadcast from lane 0 to all lanes
///   via `shfl.sync.idx.b32` (handled by the MIR pass automatically).
/// - Yielded values are broadcast from lane 0 to all lanes via
///   [`WarpBroadcast`].
/// - The resume argument `arg` is only consumed by lane 0; all lanes must
///   pass the same value (or a dummy — only lane 0's value is used).
///
/// # Safety
///
/// Implementing this trait requires maintaining warp convergence.
/// Breaking convergence causes deadlock or incorrect results.
pub unsafe trait GpuGenerator<R = ()> {
    /// Type of values yielded at each suspension point.
    type Yield: WarpBroadcast + Copy;

    /// Type of the final return value when the generator completes.
    type Return: WarpBroadcast + Copy;

    /// Resume the generator. Called by all active lanes simultaneously.
    ///
    /// Lane 0 calls the inner coroutine resume logic and obtains a
    /// `CoroutineState<Y, R>`. The discriminant (Yielded vs Complete)
    /// is broadcast via `shfl.sync`. The payload (Y or R) is broadcast
    /// via [`WarpBroadcast::broadcast()`]. All lanes return the same
    /// [`WarpCoroutineState<Y, R>`].
    fn resume_warp(
        &mut self,
        arg: R,
        wcx: &mut WarpContext,
    ) -> WarpCoroutineState<Self::Yield, Self::Return>;
}

// ============================================================
// for_each_yield — zero-buffered streaming pipeline combinator
// ============================================================

/// Drive a generator to completion, calling `consumer` on each yielded value.
///
/// All 32 lanes participate — the yielded value is broadcast to all lanes
/// before the consumer runs, enabling data-parallel consumption.
///
/// # Zero Buffering
///
/// There is no intermediate buffer. The producer yields one value,
/// the consumer processes it, then the producer yields the next.
/// At any point in time, at most ONE yielded value exists.
///
/// # Safety
///
/// All active lanes must call this simultaneously.
#[inline(always)]
pub unsafe fn for_each_yield<G, F>(
    generator: &mut G,
    mut consumer: F,
    wcx: &mut WarpContext,
) -> G::Return
where
    G: GpuGenerator,
    F: FnMut(G::Yield, &mut WarpContext),
{
    loop {
        match generator.resume_warp((), wcx) {
            WarpCoroutineState::Yielded(value) => {
                consumer(value, wcx);
            }
            WarpCoroutineState::Complete(ret) => {
                return ret;
            }
        }
        syncwarp(wcx.active_mask);
    }
}

// ============================================================
// GeneratorTask — adapter wrapping generators as Futures
// ============================================================

/// Wraps a [`GpuGenerator`] + consumer closure into a `Future<Output = ()>`
/// that the [`GpuExecutor`](crate::executor::GpuExecutor) can schedule.
///
/// Each poll drives one `resume_warp` + consumer call. When the generator
/// yields, the consumer processes the value and the task returns `Pending`
/// (to be re-polled). When the generator completes, the task returns `Ready`.
pub struct GeneratorTask<G, F> {
    generator: G,
    consumer: F,
    completed: bool,
}

impl<G, F> GeneratorTask<G, F>
where
    G: GpuGenerator,
    F: FnMut(G::Yield),
{
    /// Create a new generator task.
    pub fn new(generator: G, consumer: F) -> Self {
        Self {
            generator,
            consumer,
            completed: false,
        }
    }
}

impl<G, F> core::future::Future for GeneratorTask<G, F>
where
    G: GpuGenerator,
    F: FnMut(G::Yield),
{
    type Output = ();

    fn poll(
        self: core::pin::Pin<&mut Self>,
        _cx: &mut core::task::Context<'_>,
    ) -> core::task::Poll<()> {
        if self.completed {
            return core::task::Poll::Ready(());
        }
        let this = unsafe { self.get_unchecked_mut() };
        let mut wcx = unsafe { WarpContext::new() };

        match this.generator.resume_warp((), &mut wcx) {
            WarpCoroutineState::Yielded(value) => {
                (this.consumer)(value);
                core::task::Poll::Pending // come back for more
            }
            WarpCoroutineState::Complete(_) => {
                this.completed = true;
                core::task::Poll::Ready(())
            }
        }
    }
}

// ============================================================
// CounterGenerator — example/test generator
// ============================================================

/// A simple counter generator that yields values 0, 1, 2, ..., count-1
/// and returns the sum of all yielded values.
///
/// This serves as a reference implementation of [`GpuGenerator`] and
/// a test fixture for verifying generator compilation to PTX.
pub struct CounterGenerator {
    current: u32,
    count: u32,
    sum: u32,
}

impl CounterGenerator {
    /// Create a new counter generator that yields `count` values.
    pub fn new(count: u32) -> Self {
        Self {
            current: 0,
            count,
            sum: 0,
        }
    }
}

// SAFETY: CounterGenerator maintains warp convergence — all lanes
// observe the same state transitions via WarpBroadcast.
unsafe impl GpuGenerator for CounterGenerator {
    type Yield = u32;
    type Return = u32;

    #[inline(always)]
    fn resume_warp(&mut self, _arg: (), wcx: &mut WarpContext) -> WarpCoroutineState<u32, u32> {
        // Lane 0 computes the state transition
        let mut discrim: u32 = 0; // 0 = Yielded, 1 = Complete
        let mut payload: u32 = 0;

        if wcx.is_leader() {
            if self.current < self.count {
                let val = self.current;
                self.sum += val;
                self.current += 1;
                discrim = 0;
                payload = val;
            } else {
                discrim = 1;
                payload = self.sum;
            }
        }

        // Broadcast discriminant and payload to all lanes
        let discrim = unsafe { shfl_sync_idx_u32(wcx.active_mask, discrim, 0) };
        let payload = u32::broadcast(payload, wcx.active_mask);

        if discrim == 0 {
            WarpCoroutineState::Yielded(payload)
        } else {
            WarpCoroutineState::Complete(payload)
        }
    }
}

// ============================================================
// Shared memory broadcast helper (for types > 128 bits)
// ============================================================

/// Broadcast a value from lane 0 to all lanes via shared memory.
///
/// Use this for types larger than 128 bits that cannot be efficiently
/// broadcast via `shfl.sync`. The caller must provide a shared memory
/// slot of sufficient size and alignment.
///
/// # Safety
///
/// - All active lanes must call this simultaneously with the same `mask`.
/// - `smem_slot` must point to valid shared memory with proper alignment for `T`.
/// - Only lane 0's `value` is used.
#[inline(always)]
pub unsafe fn warp_broadcast_via_smem<T: Copy>(value: T, mask: u32, smem_slot: *mut T) -> T {
    let lid = gpu_atomics::lane_id();

    // Lane 0 writes value to shared memory
    if lid == 0 {
        core::ptr::write_volatile(smem_slot, value);
    }

    // Barrier to ensure lane 0's write is visible
    syncwarp(mask);

    // All lanes read from shared memory
    core::ptr::read_volatile(smem_slot)
}
