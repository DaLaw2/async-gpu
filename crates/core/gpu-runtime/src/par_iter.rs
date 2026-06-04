//! GPU parallel iterator — lazy, fused, data-parallel iterator chains.
//!
//! Provides a Rayon-like parallel iterator API for GPU kernels. Iterator
//! chains are built lazily via adapter methods (`map`, `enumerate`, `zip`)
//! and executed via terminal methods (`for_each`, `collect_into`, `fold`,
//! `sum`). All closures are fused at compile time via monomorphization —
//! no intermediate buffers, no heap allocation.
//!
//! # Design
//!
//! - **Lazy adapters**: `map`, `enumerate`, `zip` build a type-level chain.
//! - **Eager terminals**: `for_each`, `collect_into`, `fold`, `sum` dispatch
//!   via `BlockScope::spawn_all` for cooperative warp-parallel execution.
//! - **Fusion**: Rust monomorphization inlines the full chain into a single
//!   loop body per warp. `.map(f).map(g)` compiles to `g(f(x))` — zero
//!   intermediate buffers, register-to-register.
//! - **Copy-only**: All types are `Copy` (no heap, no Drop). Closures must
//!   be `Fn + Copy + Send + Sync`.
//!
//! # Example
//!
//! ```rust,ignore
//! use gpu_runtime::par_iter::*;
//!
//! let data = unsafe { GpuSlice::from_raw_parts(input_ptr, len) };
//! let out = unsafe { GpuSliceMut::from_raw_parts(output_ptr, len) };
//!
//! // Fused chain: compiles to a single loop per warp
//! data.par_iter()
//!     .map(|x| x * 2.0)
//!     .map(|x| x + 1.0)
//!     .collect_into(out);
//! ```

use crate::scope::block_scope;
use crate::thread::{NUM_WARPS, WARP_RESULT};
use core::sync::atomic::Ordering;

// ============================================================
// GpuSlice / GpuSliceMut — fat pointers to GPU global memory
// ============================================================

/// A reference to a contiguous region of `T` elements in GPU global memory.
///
/// This is the GPU equivalent of `&[T]`, but for device global memory.
/// It carries a raw pointer and length — no lifetime, because GPU global
/// memory outlives any kernel scope.
///
/// # Safety
///
/// The pointer must be valid for `len` elements of `T` in device global
/// memory for the duration of the kernel. The caller (host-side launch
/// code) is responsible for ensuring this.
#[derive(Clone, Copy)]
pub struct GpuSlice<T: Copy> {
    ptr: *const T,
    len: usize,
}

// SAFETY: On GPU, all global memory pointers are accessible from all warps.
// There is no concept of thread-local memory ownership.
unsafe impl<T: Copy> Send for GpuSlice<T> {}
unsafe impl<T: Copy> Sync for GpuSlice<T> {}

/// Mutable variant of [`GpuSlice`] for output buffers.
#[derive(Clone, Copy)]
pub struct GpuSliceMut<T: Copy> {
    ptr: *mut T,
    len: usize,
}

// SAFETY: Same reasoning as GpuSlice — GPU global memory is warp-accessible.
unsafe impl<T: Copy> Send for GpuSliceMut<T> {}
unsafe impl<T: Copy> Sync for GpuSliceMut<T> {}

impl<T: Copy> GpuSlice<T> {
    /// Create a `GpuSlice` from a raw pointer and length.
    ///
    /// # Safety
    ///
    /// `ptr` must point to `len` valid, aligned elements in GPU global memory.
    pub unsafe fn from_raw_parts(ptr: *const T, len: usize) -> Self {
        Self { ptr, len }
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the slice is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Raw pointer to the data.
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Create a parallel iterator over this slice.
    pub fn par_iter(&self) -> GpuParIter<T> {
        GpuParIter {
            ptr: self.ptr,
            len: self.len,
        }
    }
}

impl<T: Copy> GpuSliceMut<T> {
    /// Create a `GpuSliceMut` from a raw pointer and length.
    ///
    /// # Safety
    ///
    /// `ptr` must point to `len` valid, aligned, writable elements in GPU
    /// global memory.
    pub unsafe fn from_raw_parts(ptr: *mut T, len: usize) -> Self {
        Self { ptr, len }
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the slice is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Raw pointer to the data (const).
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Raw mutable pointer to the data.
    pub fn as_mut_ptr(&self) -> *mut T {
        self.ptr
    }
}

// ============================================================
// SendPtr / SendPtrMut — raw pointer wrappers for closure capture
// ============================================================

/// A raw pointer wrapper that implements `Send + Sync`.
///
/// On GPU, all global memory pointers are accessible from all warps —
/// there is no concept of thread-local memory ownership. This wrapper
/// makes it safe to capture device pointers in `spawn_all` closures.
///
/// # Safety
///
/// The wrapped pointer must point to GPU global memory that is valid
/// for the duration of the closure's execution.
#[derive(Clone, Copy)]
pub struct SendPtr<T>(*const T);

/// Mutable variant of [`SendPtr`].
#[derive(Clone, Copy)]
pub struct SendPtrMut<T>(*mut T);

unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
unsafe impl<T> Send for SendPtrMut<T> {}
unsafe impl<T> Sync for SendPtrMut<T> {}

impl<T> SendPtr<T> {
    /// Wrap a raw const pointer.
    pub fn new(ptr: *const T) -> Self {
        Self(ptr)
    }

    /// Unwrap to the raw const pointer.
    pub fn as_ptr(self) -> *const T {
        self.0
    }
}

impl<T> SendPtrMut<T> {
    /// Wrap a raw mutable pointer.
    pub fn new(ptr: *mut T) -> Self {
        Self(ptr)
    }

    /// Unwrap to a raw const pointer.
    pub fn as_ptr(self) -> *const T {
        self.0
    }

    /// Unwrap to the raw mutable pointer.
    pub fn as_mut_ptr(self) -> *mut T {
        self.0
    }
}

// ============================================================
// GpuZero / GpuOne / GpuMaxValue / GpuMinValue — identity traits
// ============================================================

/// Trait for types that have an additive identity (zero).
pub trait GpuZero: Copy {
    /// The additive identity element.
    fn zero() -> Self;
}

/// Trait for types that have a multiplicative identity (one).
pub trait GpuOne: Copy {
    /// The multiplicative identity element.
    fn one() -> Self;
}

/// Trait for types that have a maximum representable value.
pub trait GpuMaxValue: Copy {
    /// The maximum representable value.
    fn max_value() -> Self;
}

/// Trait for types that have a minimum representable value.
pub trait GpuMinValue: Copy {
    /// The minimum representable value.
    fn min_value() -> Self;
}

macro_rules! impl_gpu_zero_one {
    ($($t:ty, $zero:expr, $one:expr);* $(;)?) => {
        $(
            impl GpuZero for $t {
                #[inline(always)]
                fn zero() -> Self { $zero }
            }
            impl GpuOne for $t {
                #[inline(always)]
                fn one() -> Self { $one }
            }
        )*
    };
}

impl_gpu_zero_one! {
    f32, 0.0, 1.0;
    f64, 0.0, 1.0;
    u32, 0, 1;
    u64, 0, 1;
    i32, 0, 1;
    i64, 0, 1;
    usize, 0, 1;
}

macro_rules! impl_gpu_min_max {
    ($($t:ty, $min:expr, $max:expr);* $(;)?) => {
        $(
            impl GpuMinValue for $t {
                #[inline(always)]
                fn min_value() -> Self { $min }
            }
            impl GpuMaxValue for $t {
                #[inline(always)]
                fn max_value() -> Self { $max }
            }
        )*
    };
}

impl_gpu_min_max! {
    f32, f32::MIN, f32::MAX;
    f64, f64::MIN, f64::MAX;
    u32, u32::MIN, u32::MAX;
    u64, u64::MIN, u64::MAX;
    i32, i32::MIN, i32::MAX;
    i64, i64::MIN, i64::MAX;
    usize, usize::MIN, usize::MAX;
}

// ============================================================
// GpuParallelIterator trait
// ============================================================

/// Trait for GPU parallel iterators.
///
/// Lazy: adapter methods (`map`, `enumerate`, `zip`) build up a type-level
/// chain. Terminal methods (`for_each`, `fold`, `collect_into`, `sum`)
/// execute the chain via `BlockScope::spawn_all`.
///
/// All closures must be `Fn + Copy + Send + Sync`:
/// - `Fn` (not `FnMut`/`FnOnce`): multiple warps execute the same closure.
/// - `Copy`: no heap allocation, no Drop — GPU safety requirement.
/// - `Send + Sync`: closure data is copied to each warp's scratch buffer.
///
/// The `Copy` bound is the critical GPU safety gate. It excludes:
/// - `Box<T>`, `Vec<T>`, `String` (heap allocation)
/// - Closures capturing `&mut T` (FnMut, not Fn)
/// - Any type with `Drop` (GPU has no destructors)
///
/// Allowed captures: `f32`, `u32`, `i64`, `[f32; N]`, `SendPtr<T>`,
/// and any `Copy + Send + Sync` struct. Must fit in 256 bytes total
/// (warp scratch buffer limit).
pub trait GpuParallelIterator: Sized + Copy + Send + Sync + 'static {
    /// The element type produced by this iterator.
    type Item: Copy + Send + Sync + 'static;

    /// Number of elements this iterator will produce.
    fn len(&self) -> usize;

    /// Whether this iterator produces zero elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read element at logical index `i`.
    ///
    /// Called by each warp for its assigned indices. The full iterator
    /// chain is inlined into this call via monomorphization.
    ///
    /// # Safety
    ///
    /// `i` must be in range `0..self.len()`.
    unsafe fn get_unchecked(&self, i: usize) -> Self::Item;

    // === Adapter methods (lazy, build type-level chain) ===

    /// Apply a function to each element, producing a new element type.
    fn map<B, F>(self, f: F) -> GpuMap<Self, F>
    where
        B: Copy,
        F: Fn(Self::Item) -> B + Copy + Send + Sync,
    {
        GpuMap { inner: self, f }
    }

    /// Pair each element with its index: `(index, element)`.
    fn enumerate(self) -> GpuEnumerate<Self> {
        GpuEnumerate { inner: self }
    }

    /// Pair elements from two equal-length iterators.
    fn zip<Other: GpuParallelIterator>(self, other: Other) -> GpuZip<Self, Other> {
        GpuZip { a: self, b: other }
    }

    // === Terminal methods (eager, dispatch via spawn_all) ===

    /// Execute a side-effect for each element. No output buffer.
    fn for_each<F>(self, f: F)
    where
        F: Fn(Self::Item) + Copy + Send + Sync + 'static,
    {
        default_for_each(self, f);
    }

    /// Fold all elements to a single value using an associative operation.
    ///
    /// `identity` is the neutral element (e.g., 0 for sum, 1 for product).
    /// `fold_op` is an associative binary operation on `Item`.
    ///
    /// The fold_op must be associative:
    /// `fold_op(a, fold_op(b, c)) == fold_op(fold_op(a, b), c)`.
    /// This is required because GPU reduction partitions elements across
    /// warps and combines partial results — the order is not sequential.
    fn fold<F>(self, identity: Self::Item, fold_op: F) -> Self::Item
    where
        F: Fn(Self::Item, Self::Item) -> Self::Item + Copy + Send + Sync + 'static,
    {
        default_fold(self, identity, fold_op)
    }

    /// Materialize the iterator chain into an output buffer.
    ///
    /// The output buffer must be pre-allocated with the same length as
    /// the iterator. For 1:1 operations (map, enumerate, zip), writes
    /// `output[i] = chain(input[i])`.
    fn collect_into(self, output: GpuSliceMut<Self::Item>) {
        default_collect_into(self, output);
    }

    // === Convenience terminals (built on fold) ===

    /// Sum all elements.
    fn sum(self) -> Self::Item
    where
        Self::Item: core::ops::Add<Output = Self::Item> + GpuZero,
    {
        self.fold(GpuZero::zero(), |a, b| a + b)
    }

    /// Product of all elements.
    fn product(self) -> Self::Item
    where
        Self::Item: core::ops::Mul<Output = Self::Item> + GpuOne,
    {
        self.fold(GpuOne::one(), |a, b| a * b)
    }

    /// Minimum element.
    fn min(self) -> Self::Item
    where
        Self::Item: PartialOrd + GpuMaxValue,
    {
        self.fold(GpuMaxValue::max_value(), |a, b| if a < b { a } else { b })
    }

    /// Maximum element.
    fn max(self) -> Self::Item
    where
        Self::Item: PartialOrd + GpuMinValue,
    {
        self.fold(GpuMinValue::min_value(), |a, b| if a > b { a } else { b })
    }

    /// Count elements. For non-filter iterators, returns `len()`.
    fn count(self) -> usize {
        self.len()
    }
}

// ============================================================
// GpuParIter — base iterator over a GpuSlice
// ============================================================

/// The root parallel iterator over a [`GpuSlice`].
///
/// Created by [`GpuSlice::par_iter()`]. Each element is read via
/// `read_volatile` from GPU global memory.
#[derive(Clone, Copy)]
pub struct GpuParIter<T: Copy> {
    ptr: *const T,
    len: usize,
}

// SAFETY: GPU global memory is accessible from all warps.
unsafe impl<T: Copy> Send for GpuParIter<T> {}
unsafe impl<T: Copy> Sync for GpuParIter<T> {}

impl<T: Copy + Send + Sync + 'static> GpuParallelIterator for GpuParIter<T> {
    type Item = T;

    fn len(&self) -> usize {
        self.len
    }

    unsafe fn get_unchecked(&self, i: usize) -> T {
        core::ptr::read_volatile(self.ptr.add(i))
    }
}

// ============================================================
// GpuMap — map adapter
// ============================================================

/// Lazy map adapter. Fuses with the chain — no intermediate buffer.
///
/// Created by [`GpuParallelIterator::map()`].
#[derive(Clone, Copy)]
pub struct GpuMap<I, F> {
    inner: I,
    f: F,
}

impl<I, F, B> GpuParallelIterator for GpuMap<I, F>
where
    I: GpuParallelIterator,
    F: Fn(I::Item) -> B + Copy + Send + Sync + 'static,
    B: Copy + Send + Sync + 'static,
{
    type Item = B;

    fn len(&self) -> usize {
        self.inner.len()
    }

    unsafe fn get_unchecked(&self, i: usize) -> B {
        (self.f)(self.inner.get_unchecked(i))
    }
}

// ============================================================
// GpuEnumerate — enumerate adapter
// ============================================================

/// Pairs each element with its index: `(usize, Item)`.
///
/// Created by [`GpuParallelIterator::enumerate()`].
#[derive(Clone, Copy)]
pub struct GpuEnumerate<I> {
    inner: I,
}

impl<I: GpuParallelIterator> GpuParallelIterator for GpuEnumerate<I> {
    type Item = (usize, I::Item);

    fn len(&self) -> usize {
        self.inner.len()
    }

    unsafe fn get_unchecked(&self, i: usize) -> (usize, I::Item) {
        (i, self.inner.get_unchecked(i))
    }
}

// ============================================================
// GpuZip — zip adapter
// ============================================================

/// Pairs elements from two iterators: `(A::Item, B::Item)`.
///
/// Created by [`GpuParallelIterator::zip()`].
///
/// # Panics
///
/// Debug-asserts that both iterators have the same length.
#[derive(Clone, Copy)]
pub struct GpuZip<A, B> {
    a: A,
    b: B,
}

impl<A, B> GpuParallelIterator for GpuZip<A, B>
where
    A: GpuParallelIterator,
    B: GpuParallelIterator,
{
    type Item = (A::Item, B::Item);

    fn len(&self) -> usize {
        let la = self.a.len();
        let lb = self.b.len();
        debug_assert_eq!(la, lb, "GpuZip: iterators must have equal length");
        la
    }

    unsafe fn get_unchecked(&self, i: usize) -> (A::Item, B::Item) {
        (self.a.get_unchecked(i), self.b.get_unchecked(i))
    }
}

// ============================================================
// Terminal operation implementations
// ============================================================

/// Default `for_each` implementation for all iterator chains.
///
/// Dispatches via `block_scope` + `spawn_all`. Each warp processes
/// elements in round-robin: warp `wid` handles indices
/// `wid, wid + n_warps, wid + 2*n_warps, ...`.
fn default_for_each<I, F>(iter: I, f: F)
where
    I: GpuParallelIterator + 'static,
    F: Fn(I::Item) + Copy + Send + Sync + 'static,
{
    let len = iter.len();
    if len == 0 {
        return;
    }

    // `move` on both closures copies the Copy types (iter, f, len)
    // into the closure, satisfying the 'scope HRTB in block_scope.
    block_scope(move |scope| {
        scope.spawn_all(move |wid, n_warps| {
            let mut i = wid as usize;
            while i < len {
                let elem = unsafe { iter.get_unchecked(i) };
                f(elem);
                i += n_warps as usize;
            }
        });
    });
}

/// Default `fold` implementation.
///
/// Each warp reduces its partition to a partial result. Warp 0 then
/// combines all partial results sequentially (max 31 additions for
/// 32 warps).
///
/// Two-level hierarchy:
/// 1. Intra-warp: sequential fold over assigned elements (register-only).
/// 2. Cross-warp: warp 0 collects partial results from `WARP_RESULT` slots.
fn default_fold<I, F>(iter: I, identity: I::Item, fold_op: F) -> I::Item
where
    I: GpuParallelIterator + 'static,
    F: Fn(I::Item, I::Item) -> I::Item + Copy + Send + Sync + 'static,
{
    let len = iter.len();
    if len == 0 {
        return identity;
    }

    // The fold result type must fit in a u64 for cross-warp transfer
    // via WARP_RESULT slots.
    assert!(
        core::mem::size_of::<I::Item>() <= 8,
        "fold result type must fit in 8 bytes (u64)"
    );

    block_scope(move |scope| {
        scope.spawn_all(move |wid, n_warps| {
            let mut acc = identity;
            let mut i = wid as usize;
            while i < len {
                let elem = unsafe { iter.get_unchecked(i) };
                acc = fold_op(acc, elem);
                i += n_warps as usize;
            }
            // Write partial result to WARP_RESULT for warp 0 to collect.
            // Transmute the Item to u64 bits for storage.
            let bits = unsafe {
                let mut buf = 0u64;
                core::ptr::copy_nonoverlapping(
                    &acc as *const I::Item as *const u8,
                    &mut buf as *mut u64 as *mut u8,
                    core::mem::size_of::<I::Item>(),
                );
                buf
            };
            WARP_RESULT[wid as usize].store(bits, Ordering::Release);
        });
    });

    // Warp 0 combines partial results from all warps.
    let n_warps = NUM_WARPS.load(Ordering::Acquire) as usize;
    let total = if n_warps == 0 { 1 } else { n_warps };
    let mut combined = identity;
    #[allow(clippy::needless_range_loop)]
    for w in 0..total {
        let bits = WARP_RESULT[w].load(Ordering::Acquire);
        let partial = unsafe {
            let mut val = core::mem::MaybeUninit::<I::Item>::uninit();
            core::ptr::copy_nonoverlapping(
                &bits as *const u64 as *const u8,
                val.as_mut_ptr() as *mut u8,
                core::mem::size_of::<I::Item>(),
            );
            val.assume_init()
        };
        combined = fold_op(combined, partial);
    }
    combined
}

/// Default `collect_into` implementation.
///
/// Each warp writes its assigned elements directly to the output buffer.
/// For 1:1 chains (map, enumerate, zip), `output[i] = chain(input[i])`.
fn default_collect_into<I>(iter: I, output: GpuSliceMut<I::Item>)
where
    I: GpuParallelIterator + 'static,
{
    let len = iter.len();
    assert_eq!(
        len,
        output.len(),
        "collect_into: output buffer length mismatch"
    );
    if len == 0 {
        return;
    }

    let out_ptr = SendPtrMut::new(output.as_mut_ptr());

    block_scope(move |scope| {
        scope.spawn_all(move |wid, n_warps| {
            let mut i = wid as usize;
            while i < len {
                let elem = unsafe { iter.get_unchecked(i) };
                unsafe {
                    core::ptr::write_volatile(out_ptr.as_mut_ptr().add(i), elem);
                }
                i += n_warps as usize;
            }
        });
    });
}

// ============================================================
// Free-standing entry point
// ============================================================

/// Create a parallel iterator over a [`GpuSlice`].
///
/// Convenience function equivalent to `slice.par_iter()`.
pub fn par_iter<T: Copy>(slice: &GpuSlice<T>) -> GpuParIter<T> {
    slice.par_iter()
}
