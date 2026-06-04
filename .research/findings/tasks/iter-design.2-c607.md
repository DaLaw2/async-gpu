# iter-design.2 — GpuParallelIterator Trait API Surface + Closure Capture Rules

## Status: done

## Summary

Complete API surface design for the kernel-side `GpuParallelIterator` trait and supporting types. All types live in `gpu-runtime` (the kernel-side crate, `no_std`). The design builds exclusively on `BlockScope::spawn_all` for execution dispatch and uses Rust's monomorphization for chain fusion. An implementer can code directly from this spec.

---

## 1. Core Types

### 1.1 GpuSlice — GPU Data Reference

`GpuSlice<T>` is a fat pointer to data in GPU global memory. It is the kernel-side handle that iterator chains operate on. It does NOT own memory — ownership belongs to the host-side launch infrastructure (future `gpu-host` integration).

```rust
// crates/core/gpu-runtime/src/par_iter.rs

/// A reference to a contiguous region of `T` elements in GPU global memory.
///
/// This is the GPU equivalent of `&[T]` / `&mut [T]`, but for device global
/// memory. It carries a raw pointer and length — no lifetime, because GPU
/// global memory outlives any kernel scope.
///
/// # Safety
///
/// The pointer must be valid for `len` elements of `T` in device global memory
/// for the duration of the kernel. The caller (host-side launch code) is
/// responsible for ensuring this.
#[derive(Clone, Copy)]
pub struct GpuSlice<T: Copy> {
    ptr: *const T,
    len: usize,
}

/// Mutable variant for output buffers.
#[derive(Clone, Copy)]
pub struct GpuSliceMut<T: Copy> {
    ptr: *mut T,
    len: usize,
}

impl<T: Copy> GpuSlice<T> {
    /// Create a GpuSlice from a raw pointer and length.
    ///
    /// # Safety
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
        GpuParIter { ptr: self.ptr, len: self.len }
    }
}

impl<T: Copy> GpuSliceMut<T> {
    /// Create a GpuSliceMut from a raw pointer and length.
    ///
    /// # Safety
    /// `ptr` must point to `len` valid, aligned, writable elements in GPU global memory.
    pub unsafe fn from_raw_parts(ptr: *mut T, len: usize) -> Self {
        Self { ptr, len }
    }

    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
    pub fn as_ptr(&self) -> *const T { self.ptr }
    pub fn as_mut_ptr(&self) -> *mut T { self.ptr }
}
```

### 1.2 SendPtr — Safe Wrapper for Raw Pointers in Closures

Raw pointers are not `Send`/`Sync`. Closures that capture `GpuSlice` need this wrapper. (This is a common pattern in GPU Rust code.)

```rust
/// A raw pointer wrapper that implements Send + Sync.
///
/// On GPU, all global memory pointers are accessible from all warps —
/// there is no concept of thread-local memory ownership. This wrapper
/// makes it safe to capture device pointers in `spawn_all` closures.
///
/// # Safety
/// The wrapped pointer must point to GPU global memory that is valid
/// for the duration of the closure's execution.
#[derive(Clone, Copy)]
pub struct SendPtr<T>(*const T);

#[derive(Clone, Copy)]
pub struct SendPtrMut<T>(*mut T);

unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
unsafe impl<T> Send for SendPtrMut<T> {}
unsafe impl<T> Sync for SendPtrMut<T> {}

impl<T> SendPtr<T> {
    pub fn new(ptr: *const T) -> Self { Self(ptr) }
    pub fn as_ptr(self) -> *const T { self.0 }
}

impl<T> SendPtrMut<T> {
    pub fn new(ptr: *mut T) -> Self { Self(ptr) }
    pub fn as_ptr(self) -> *const T { self.0 }
    pub fn as_mut_ptr(self) -> *mut T { self.0 }
}
```

---

## 2. GpuParallelIterator Trait

### 2.1 Trait Definition

```rust
/// Trait for GPU parallel iterators.
///
/// Lazy: adapter methods (map, enumerate, zip) build up a type-level chain.
/// Terminal methods (for_each, fold, collect, sum) execute the chain via
/// `BlockScope::spawn_all`.
///
/// All closures must be `Fn + Copy + Send + Sync`:
/// - `Fn` (not `FnMut`/`FnOnce`): multiple warps execute the same closure
/// - `Copy`: no heap allocation, no Drop — GPU safety requirement
/// - `Send + Sync`: closure data is copied to each warp's scratch buffer
///
/// The `Copy` bound is the critical GPU safety gate. It excludes:
/// - `Box<T>`, `Vec<T>`, `String` (heap allocation)
/// - Closures capturing `&mut T` (FnMut, not Fn)
/// - Any type with `Drop` (GPU has no destructors)
///
/// Allowed captures: `f32`, `u32`, `i64`, `[f32; N]`, `SendPtr<T>`,
/// and any `Copy + Send + Sync` struct. Must fit in 256 bytes total
/// (warp scratch buffer limit).
pub trait GpuParallelIterator: Sized {
    /// The element type produced by this iterator.
    type Item: Copy;

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
    fn zip<Other>(self, other: Other) -> GpuZip<Self, Other>
    where
        Other: GpuParallelIterator,
    {
        GpuZip { a: self, b: other }
    }

    // === Terminal operations ===
    // These consume the chain and dispatch to spawn_all.

    /// Execute a side-effect for each element. No output buffer.
    fn for_each<F>(self, f: F)
    where
        F: Fn(Self::Item) + Copy + Send + Sync;

    /// Fold all elements to a single value using an associative operation.
    ///
    /// `identity` provides the neutral element (e.g., 0 for sum, 1 for product).
    /// `fold_op` is an associative binary operation.
    ///
    /// Execution: each warp reduces its partition, then warp 0 combines
    /// partial results.
    fn fold<B, ID, FOLD>(self, identity: ID, fold_op: FOLD) -> B
    where
        B: Copy + Send + Sync,
        ID: Fn() -> B + Copy + Send + Sync,
        FOLD: Fn(B, Self::Item) -> B + Copy + Send + Sync;

    /// Materialize the iterator chain into an output buffer.
    ///
    /// The caller provides the output buffer (pre-allocated in GPU global memory).
    /// For 1:1 operations (map, enumerate, zip), the output buffer must have
    /// the same length as the input.
    fn collect_into(self, output: GpuSliceMut<Self::Item>);

    // === Convenience terminals (built on fold) ===

    /// Sum all elements. Requires `Item: Add<Output=Item>` and a zero value.
    fn sum(self) -> Self::Item
    where
        Self::Item: core::ops::Add<Output = Self::Item> + GpuZero,
    {
        self.fold(
            || <Self::Item as GpuZero>::zero(),
            |acc, x| acc + x,
        )
    }

    /// Product of all elements.
    fn product(self) -> Self::Item
    where
        Self::Item: core::ops::Mul<Output = Self::Item> + GpuOne,
    {
        self.fold(
            || <Self::Item as GpuOne>::one(),
            |acc, x| acc * x,
        )
    }

    // --- Internal plumbing ---

    /// Number of elements this iterator will produce.
    /// Used by collect_into to verify output buffer size.
    fn len(&self) -> usize;

    /// Read element at logical index `i`.
    /// Called by each warp for its assigned indices.
    ///
    /// # Safety
    /// `i` must be in range `0..self.len()`.
    unsafe fn get_unchecked(&self, i: usize) -> Self::Item;
}
```

### 2.2 Why `collect_into` Instead of `collect() -> GpuVec`

The kernel side has no allocator. GPU global memory must be allocated by the host before kernel launch. Therefore:

- `collect_into(output: GpuSliceMut<T>)` writes to a pre-allocated output buffer
- The host-side API (future work in `gpu-host`) will provide `collect()` that allocates the output buffer and calls `collect_into` under the hood

This keeps the kernel-side API allocation-free, matching the `no_std` constraint.

### 2.3 GpuZero and GpuOne Traits

```rust
/// Trait for types that have an additive identity (zero).
pub trait GpuZero: Copy {
    fn zero() -> Self;
}

/// Trait for types that have a multiplicative identity (one).
pub trait GpuOne: Copy {
    fn one() -> Self;
}

// Implementations for primitive types
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
```

---

## 3. Adapter Types

### 3.1 GpuParIter — Base Iterator (from GpuSlice)

```rust
/// The root parallel iterator over a GpuSlice.
/// Created by `GpuSlice::par_iter()`.
#[derive(Clone, Copy)]
pub struct GpuParIter<T: Copy> {
    ptr: *const T,
    len: usize,
}

impl<T: Copy> GpuParallelIterator for GpuParIter<T> {
    type Item = T;

    fn len(&self) -> usize {
        self.len
    }

    unsafe fn get_unchecked(&self, i: usize) -> T {
        core::ptr::read_volatile(self.ptr.add(i))
    }

    fn for_each<F>(self, f: F)
    where
        F: Fn(T) + Copy + Send + Sync,
    {
        default_for_each(self, f);
    }

    fn fold<B, ID, FOLD>(self, identity: ID, fold_op: FOLD) -> B
    where
        B: Copy + Send + Sync,
        ID: Fn() -> B + Copy + Send + Sync,
        FOLD: Fn(B, T) -> B + Copy + Send + Sync,
    {
        default_fold(self, identity, fold_op)
    }

    fn collect_into(self, output: GpuSliceMut<T>) {
        default_collect_into(self, output);
    }
}
```

### 3.2 GpuMap — Map Adapter

```rust
/// Lazy map adapter. Fuses with the chain — no intermediate buffer.
#[derive(Clone, Copy)]
pub struct GpuMap<I, F> {
    inner: I,
    f: F,
}

impl<I, F, B> GpuParallelIterator for GpuMap<I, F>
where
    I: GpuParallelIterator,
    F: Fn(I::Item) -> B + Copy + Send + Sync,
    B: Copy,
{
    type Item = B;

    fn len(&self) -> usize {
        self.inner.len()
    }

    unsafe fn get_unchecked(&self, i: usize) -> B {
        (self.f)(self.inner.get_unchecked(i))
    }

    fn for_each<G>(self, g: G)
    where
        G: Fn(B) + Copy + Send + Sync,
    {
        default_for_each(self, g);
    }

    fn fold<C, ID, FOLD>(self, identity: ID, fold_op: FOLD) -> C
    where
        C: Copy + Send + Sync,
        ID: Fn() -> C + Copy + Send + Sync,
        FOLD: Fn(C, B) -> C + Copy + Send + Sync,
    {
        default_fold(self, identity, fold_op)
    }

    fn collect_into(self, output: GpuSliceMut<B>) {
        default_collect_into(self, output);
    }
}
```

**How fusion works**: When you write `.map(|x| x * 2.0).map(|x| x + 1.0)`, the type is:
```
GpuMap<GpuMap<GpuParIter<f32>, [closure1]>, [closure2]>
```
Calling `get_unchecked(i)` on the outer `GpuMap` calls `(closure2)((closure1)(inner.get_unchecked(i)))`. The Rust compiler inlines this into a single `x * 2.0 + 1.0` — register-to-register, zero intermediate buffers. This is automatic monomorphization; no MIR pass needed.

### 3.3 GpuEnumerate — Enumerate Adapter

```rust
/// Pairs each element with its index: `(usize, Item)`.
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

    fn for_each<F>(self, f: F)
    where
        F: Fn((usize, I::Item)) + Copy + Send + Sync,
    {
        default_for_each(self, f);
    }

    fn fold<B, ID, FOLD>(self, identity: ID, fold_op: FOLD) -> B
    where
        B: Copy + Send + Sync,
        ID: Fn() -> B + Copy + Send + Sync,
        FOLD: Fn(B, (usize, I::Item)) -> B + Copy + Send + Sync,
    {
        default_fold(self, identity, fold_op)
    }

    fn collect_into(self, output: GpuSliceMut<(usize, I::Item)>) {
        default_collect_into(self, output);
    }
}
```

### 3.4 GpuZip — Zip Adapter

```rust
/// Pairs elements from two iterators: `(A::Item, B::Item)`.
///
/// # Panics
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

    fn for_each<F>(self, f: F)
    where
        F: Fn((A::Item, B::Item)) + Copy + Send + Sync,
    {
        default_for_each(self, f);
    }

    fn fold<C, ID, FOLD>(self, identity: ID, fold_op: FOLD) -> C
    where
        C: Copy + Send + Sync,
        ID: Fn() -> C + Copy + Send + Sync,
        FOLD: Fn(C, (A::Item, B::Item)) -> C + Copy + Send + Sync,
    {
        default_fold(self, identity, fold_op)
    }

    fn collect_into(self, output: GpuSliceMut<(A::Item, B::Item)>) {
        default_collect_into(self, output);
    }
}
```

---

## 4. Terminal Operation Implementations

All terminal operations dispatch through `BlockScope::spawn_all`. The iterator chain's `get_unchecked(i)` compiles to a single inlined function per element.

### 4.1 for_each — Default Implementation

```rust
/// Default `for_each` implementation for all iterator chains.
///
/// Dispatches via `block_scope` + `spawn_all`. Each warp processes
/// elements in round-robin: warp `wid` handles indices `wid, wid + n_warps, ...`.
fn default_for_each<I, F>(iter: I, f: F)
where
    I: GpuParallelIterator,
    F: Fn(I::Item) + Copy + Send + Sync,
{
    let len = iter.len();
    if len == 0 {
        return;
    }

    // The closure captures `iter` (the full chain descriptor) and `f`.
    // Both are Copy — the entire chain is a value type with no heap pointers.
    // spawn_all copies this closure to each warp's scratch buffer.
    block_scope(|scope| {
        scope.spawn_all(|wid, n_warps| {
            let mut i = wid as usize;
            while i < len {
                // get_unchecked inlines the full chain: read → map → map → ...
                let elem = unsafe { iter.get_unchecked(i) };
                f(elem);
                i += n_warps as usize;
            }
        });
    });
}
```

### 4.2 fold — Default Implementation

```rust
/// Default `fold` implementation.
///
/// Each warp reduces its partition to a partial result. Warp 0 then
/// combines all partial results sequentially (max 31 additions for 32 warps).
///
/// For large reductions, this is a two-level hierarchy:
/// 1. Intra-warp: sequential fold over assigned elements (register-only)
/// 2. Cross-warp: warp 0 collects partial results from WARP_RESULT slots
///
/// The cross-warp reduction could use warp shuffle, but with only ~31 partial
/// results the sequential combine is negligible. Warp shuffle optimization
/// is a future enhancement.
fn default_fold<I, B, ID, FOLD>(iter: I, identity: ID, fold_op: FOLD) -> B
where
    I: GpuParallelIterator,
    B: Copy + Send + Sync,
    ID: Fn() -> B + Copy + Send + Sync,
    FOLD: Fn(B, I::Item) -> B + Copy + Send + Sync,
{
    use crate::thread::{WARP_RESULT, NUM_WARPS};
    use core::sync::atomic::Ordering;

    let len = iter.len();
    if len == 0 {
        return identity();
    }

    // Each warp writes its partial result to WARP_RESULT[wid] as a u64
    // (B must fit in 8 bytes — asserted below).
    assert!(
        core::mem::size_of::<B>() <= 8,
        "fold result type must fit in 8 bytes (u64)"
    );

    block_scope(|scope| {
        scope.spawn_all(|wid, n_warps| {
            let mut acc = identity();
            let mut i = wid as usize;
            while i < len {
                let elem = unsafe { iter.get_unchecked(i) };
                acc = fold_op(acc, elem);
                i += n_warps as usize;
            }
            // Write partial result to WARP_RESULT for warp 0 to collect
            let bits = unsafe {
                let mut buf = 0u64;
                core::ptr::copy_nonoverlapping(
                    &acc as *const B as *const u8,
                    &mut buf as *mut u64 as *mut u8,
                    core::mem::size_of::<B>(),
                );
                buf
            };
            WARP_RESULT[wid as usize].store(bits, Ordering::Release);
        });
    });

    // Warp 0 combines partial results
    let n_warps = NUM_WARPS.load(Ordering::Acquire) as usize;
    let total = if n_warps == 0 { 1 } else { n_warps };
    let mut combined = identity();
    for w in 0..total {
        let bits = WARP_RESULT[w].load(Ordering::Acquire);
        let partial = unsafe {
            let mut val = core::mem::MaybeUninit::<B>::uninit();
            core::ptr::copy_nonoverlapping(
                &bits as *const u64 as *const u8,
                val.as_mut_ptr() as *mut u8,
                core::mem::size_of::<B>(),
            );
            val.assume_init()
        };
        combined = fold_op(combined, unsafe {
            // Treat partial B as Item for the fold_op by transmuting.
            // This works because combined and partial are both B.
            // We need a separate combine path — see design note below.
            core::mem::transmute_copy(&partial)
        });
    }
    combined
}
```

**Design note on fold**: The `fold_op` signature is `Fn(B, I::Item) -> B`, which combines an accumulator of type `B` with an element of type `I::Item`. For cross-warp reduction we need to combine two `B` values. There are two approaches:

1. **Require `B == I::Item`** (simplest): This is what `sum()` and `product()` do — the accumulator and element are the same type.

2. **Add a separate `combine: Fn(B, B) -> B`** parameter: More general, but adds API complexity.

For MVP, approach (1) covers all practical cases. The `fold` signature should be revised to:

```rust
fn fold<F>(self, identity: Self::Item, fold_op: F) -> Self::Item
where
    F: Fn(Self::Item, Self::Item) -> Self::Item + Copy + Send + Sync;
```

This is cleaner and avoids the transmute. The cross-warp combine uses the same `fold_op`.

### 4.3 collect_into — Default Implementation

```rust
/// Default `collect_into` implementation.
///
/// Each warp writes its assigned elements directly to the output buffer.
/// For 1:1 chains (map, enumerate, zip), output[i] = chain(input[i]).
fn default_collect_into<I>(iter: I, output: GpuSliceMut<I::Item>)
where
    I: GpuParallelIterator,
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

    block_scope(|scope| {
        scope.spawn_all(|wid, n_warps| {
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
```

---

## 5. Revised fold Signature (MVP)

After the design note in section 4.2, the final MVP `fold` signature is:

```rust
pub trait GpuParallelIterator: Sized {
    type Item: Copy;

    // ... adapters unchanged ...

    /// Fold all elements to a single value.
    ///
    /// `identity`: neutral element (e.g., 0 for sum).
    /// `fold_op`: associative binary operation on `Item`.
    ///
    /// The fold_op must be associative: `fold_op(a, fold_op(b, c)) == fold_op(fold_op(a, b), c)`.
    /// This is required because GPU reduction partitions elements across warps
    /// and combines partial results — the order of operations is not sequential.
    fn fold<F>(self, identity: Self::Item, fold_op: F) -> Self::Item
    where
        F: Fn(Self::Item, Self::Item) -> Self::Item + Copy + Send + Sync;

    fn sum(self) -> Self::Item
    where
        Self::Item: core::ops::Add<Output = Self::Item> + GpuZero,
    {
        self.fold(GpuZero::zero(), |a, b| a + b)
    }

    fn product(self) -> Self::Item
    where
        Self::Item: core::ops::Mul<Output = Self::Item> + GpuOne,
    {
        self.fold(GpuOne::one(), |a, b| a * b)
    }

    fn min(self) -> Self::Item
    where
        Self::Item: PartialOrd + GpuMaxValue,
    {
        self.fold(GpuMaxValue::max_value(), |a, b| if a < b { a } else { b })
    }

    fn max(self) -> Self::Item
    where
        Self::Item: PartialOrd + GpuMinValue,
    {
        self.fold(GpuMinValue::min_value(), |a, b| if a > b { a } else { b })
    }

    fn count(self) -> usize {
        // This is a simplified count — for non-filter iterators, just return len.
        self.len()
    }
}
```

With helper traits:

```rust
pub trait GpuMaxValue: Copy {
    fn max_value() -> Self;
}
pub trait GpuMinValue: Copy {
    fn min_value() -> Self;
}

impl GpuMaxValue for f32 { fn max_value() -> Self { f32::MAX } }
impl GpuMinValue for f32 { fn min_value() -> Self { f32::MIN } }
impl GpuMaxValue for u32 { fn max_value() -> Self { u32::MAX } }
impl GpuMinValue for u32 { fn min_value() -> Self { u32::MIN } }
impl GpuMaxValue for i32 { fn max_value() -> Self { i32::MAX } }
impl GpuMinValue for i32 { fn min_value() -> Self { i32::MIN } }
// ... etc for u64, i64, f64
```

---

## 6. Closure Capture Rules

### 6.1 Type Bounds

Every closure passed to `map`, `for_each`, `fold`, etc. must satisfy:

```
F: Fn(...) -> ... + Copy + Send + Sync
```

**Why each bound**:

| Bound | Reason |
|-------|--------|
| `Fn` | Multiple warps call the same closure concurrently. `FnOnce` would be consumed on first call. `FnMut` requires `&mut self` which is unsound across warps. |
| `Copy` | The closure is `copy_nonoverlapping`'d to each warp's scratch buffer. `Copy` guarantees no heap allocation and no `Drop`. This is the key GPU safety gate — it excludes `Box`, `Vec`, `String`, `Arc`, and any type with a destructor. |
| `Send` | The closure crosses warp boundaries (copied to different warps). |
| `Sync` | Multiple warps may reference the closure data concurrently. |

### 6.2 What Users Can Capture

**Allowed** (all are `Copy + Send + Sync`):

```rust
// Scalars
let scale: f32 = 2.0;
data.par_iter().map(|x| x * scale).collect_into(output);

// Fixed-size arrays
let weights: [f32; 4] = [1.0, 0.5, 0.25, 0.125];
data.par_iter().map(|x| x * weights[0]).collect_into(output);

// Raw pointers (via SendPtr)
let lookup = SendPtr::new(lookup_table_ptr);
data.par_iter().map(|x| {
    let idx = x as usize;
    unsafe { *lookup.as_ptr().add(idx) }
}).collect_into(output);

// Other GpuSlices (GpuSlice is Copy)
let other = other_data;  // GpuSlice<f32>
data.par_iter().enumerate().map(|(i, x)| {
    x + unsafe { *other.as_ptr().add(i) }
}).collect_into(output);

// Copy structs
#[derive(Clone, Copy)]
struct Params { scale: f32, offset: f32 }
let p = Params { scale: 2.0, offset: 1.0 };
data.par_iter().map(|x| x * p.scale + p.offset).collect_into(output);
```

**Prohibited** (compiler errors, not runtime):

```rust
// Heap types — not Copy
let v = Vec::new();  // ERROR: Vec is not Copy
let s = String::new();  // ERROR: String is not Copy
let b = Box::new(42);  // ERROR: Box is not Copy

// Mutable references — closure becomes FnMut, not Fn
let mut counter = 0;
data.par_iter().for_each(|x| counter += 1);  // ERROR: FnMut, not Fn

// Trait objects — not Copy, not Sized
let f: &dyn Fn(f32) -> f32 = &|x| x;  // ERROR: not Copy
```

### 6.3 Size Limit

The closure (including all captured data) must fit in `SCRATCH_SIZE` (256 bytes). This is enforced by `BlockScope::spawn_all` at runtime via `assert!`.

For a typical iterator chain like `.map(|x| x * scale + offset)`, the closure captures `(scale: f32, offset: f32)` = 8 bytes. The entire chain type (which includes the base `GpuParIter` pointer + len + closure) is roughly:

- `GpuParIter<f32>`: 16 bytes (ptr + len)
- `GpuMap` wrapper: 0 extra (ZST for the Fn type) + inner
- Closure captures: 8 bytes (scale + offset)
- `GpuSliceMut` (output): 16 bytes
- Total in spawn_all closure: ~40 bytes — well within 256

Even complex chains with multiple captures stay under 256 bytes because each closure captures only scalar values and pointers (8 bytes each).

### 6.4 Closure vs Function Pointer

The existing `cooperative_map` requires function pointers (`fn(&CoopMapArgs)`). The iterator API uses closures via `spawn_all`, which supports captures. This is strictly more powerful:

- `spawn_all` copies the closure to each warp's scratch buffer (monomorphized, no trait object)
- Each warp reads its copy and calls the closure
- The closure body is inlined by the compiler (no indirect call overhead)

---

## 7. Composition with Existing APIs

### 7.1 Relationship to spawn_all

The iterator chain is syntactic sugar over `spawn_all`. This is the key composition:

```
data.par_iter()                      ─┐
    .map(|x| x * 2.0)                │  type-level chain
    .map(|x| x + 1.0)                │  (no execution yet)
    .collect_into(output)            ─┘  triggers spawn_all
```

The `collect_into` terminal compiles to approximately:

```rust
block_scope(|scope| {
    scope.spawn_all(|wid, n_warps| {
        let mut i = wid as usize;
        while i < len {
            let x = unsafe { core::ptr::read_volatile(input_ptr.add(i)) };
            let result = (x * 2.0) + 1.0;  // fused chain
            unsafe { core::ptr::write_volatile(output_ptr.add(i), result); }
            i += n_warps as usize;
        }
    });
});
```

### 7.2 Using par_iter Inside block_scope

Users can mix par_iter with manual scope operations:

```rust
block_scope(|scope| {
    // Manual allocation
    let scratch = scope.alloc::<f32>(256);

    // Use par_iter for bulk operations
    data.par_iter()
        .map(|x| x * 2.0)
        .collect_into(output);

    // Manual spawn for non-iterator work
    let h = scope.spawn(|| compute_something());
    let result = h.join();
});
```

**Important**: `par_iter` terminals create their own inner `block_scope` via the default implementations. This nests correctly because `BlockScope` supports up to 4 levels of nesting. However, if the user is already inside a `block_scope`, they should be aware that par_iter will create a nested scope.

### 7.3 Using par_iter Inside grid_scope

For cross-block parallelism, each block runs its own par_iter independently:

```rust
// Block 0 dispatches work
grid_scope(pool, pool_size, |gscope| {
    let data = gscope.alloc::<f32>(1024);
    // ... fill data ...
    gscope.set_expected_completions(num_blocks);
});

// Each worker block:
fn worker(data: GpuSlice<f32>, output: GpuSliceMut<f32>) {
    // Each block uses par_iter over its local partition
    data.par_iter()
        .map(|x| x * 2.0)
        .collect_into(output);
}
```

---

## 8. File Structure

All new code goes in a single module:

```
crates/core/gpu-runtime/src/par_iter.rs    # All types + implementations
crates/core/gpu-runtime/src/lib.rs         # Add: pub mod par_iter;
```

The module is self-contained. No changes to existing files except adding `pub mod par_iter;` to `lib.rs` and re-exporting key types from `prelude.rs`:

```rust
// In prelude.rs, add:
pub use crate::par_iter::{
    GpuParallelIterator, GpuParIter, GpuSlice, GpuSliceMut,
    SendPtr, SendPtrMut, GpuZero, GpuOne,
};
```

---

## 9. Complete User-Facing Examples

### 9.1 Element-wise Transform (map + collect)

```rust
use gpu_runtime::prelude::*;
use gpu_runtime::par_iter::*;

#[no_mangle]
pub unsafe extern "gpu-kernel" fn scale_add_kernel(
    input: *const f32,
    output: *mut f32,
    len: usize,
    scale: f32,
    offset: f32,
) {
    thread::gpu_main(|| {
        let data = GpuSlice::from_raw_parts(input, len);
        let out = GpuSliceMut::from_raw_parts(output, len);

        data.par_iter()
            .map(|x| x * scale + offset)
            .collect_into(out);
    });
}
```

### 9.2 Reduction (sum)

```rust
#[no_mangle]
pub unsafe extern "gpu-kernel" fn dot_product_kernel(
    a: *const f32,
    b: *const f32,
    result: *mut f32,
    len: usize,
) {
    thread::gpu_main(|| {
        let va = GpuSlice::from_raw_parts(a, len);
        let vb = GpuSlice::from_raw_parts(b, len);

        let dot = va.par_iter()
            .zip(vb.par_iter())
            .map(|(x, y)| x * y)
            .sum();

        core::ptr::write_volatile(result, dot);
    });
}
```

### 9.3 Enumerate + Map

```rust
#[no_mangle]
pub unsafe extern "gpu-kernel" fn iota_kernel(
    output: *mut f32,
    len: usize,
) {
    thread::gpu_main(|| {
        // Create a "dummy" input just to get indices
        let dummy = GpuSlice::from_raw_parts(output as *const f32, len);
        let out = GpuSliceMut::from_raw_parts(output, len);

        dummy.par_iter()
            .enumerate()
            .map(|(i, _)| i as f32)
            .collect_into(out);
    });
}
```

### 9.4 Side-Effect (for_each)

```rust
#[no_mangle]
pub unsafe extern "gpu-kernel" fn clamp_inplace_kernel(
    data: *mut f32,
    len: usize,
    min_val: f32,
    max_val: f32,
) {
    thread::gpu_main(|| {
        let slice = GpuSlice::from_raw_parts(data, len);
        let data_ptr = SendPtrMut::new(data);

        slice.par_iter()
            .enumerate()
            .for_each(|(i, x)| {
                let clamped = if x < min_val { min_val }
                              else if x > max_val { max_val }
                              else { x };
                core::ptr::write_volatile(data_ptr.as_mut_ptr().add(i), clamped);
            });
    });
}
```

### 9.5 Chained Maps (Fusion Demo)

```rust
#[no_mangle]
pub unsafe extern "gpu-kernel" fn fused_chain_kernel(
    input: *const f32,
    output: *mut f32,
    len: usize,
) {
    thread::gpu_main(|| {
        let data = GpuSlice::from_raw_parts(input, len);
        let out = GpuSliceMut::from_raw_parts(output, len);

        // All 4 maps fuse into ONE spawn_all call with ONE loop.
        // The compiler inlines get_unchecked through all 4 GpuMap layers.
        // Compiled code: output[i] = ((input[i] * 2.0 + 1.0) * 0.5).abs()
        data.par_iter()
            .map(|x| x * 2.0)
            .map(|x| x + 1.0)
            .map(|x| x * 0.5)
            .map(|x| if x < 0.0 { -x } else { x })  // abs
            .collect_into(out);
    });
}
```

### 9.6 Zip + Fold (Manual Reduce)

```rust
#[no_mangle]
pub unsafe extern "gpu-kernel" fn l2_distance_kernel(
    a: *const f32,
    b: *const f32,
    result: *mut f32,
    len: usize,
) {
    thread::gpu_main(|| {
        let va = GpuSlice::from_raw_parts(a, len);
        let vb = GpuSlice::from_raw_parts(b, len);

        // L2 distance: sqrt(sum((a[i] - b[i])^2))
        let sum_sq = va.par_iter()
            .zip(vb.par_iter())
            .map(|(x, y)| {
                let d = x - y;
                d * d
            })
            .fold(0.0f32, |acc, x| acc + x);

        // sqrt on warp 0
        let dist = gpu_runtime::math::sqrt_f32(sum_sq);
        core::ptr::write_volatile(result, dist);
    });
}
```

---

## 10. What Is NOT in This Design (Future Work)

| Feature | Why Deferred | Depends On |
|---------|-------------|------------|
| `filter(pred)` | Needs warp ballot compaction + atomic output counter | Warp ballot intrinsics work, but compaction kernel is complex |
| `flat_map(f)` | Two-pass kernel (count → allocate → write) | `filter` infrastructure + dynamic output sizing |
| `scan()` / prefix sum | Blelloch algorithm, multiple sync barriers | Shared memory tiling |
| `collect() -> GpuVec` | Needs kernel-side allocator or host-side output allocation | Host-side par_iter API in gpu-host |
| Host-side `gpu::par_iter(data)` | Kernel launch config auto-generation from chain type | gpu-host integration, kernel compilation pipeline |
| MIR pass for cross-boundary fusion | Optimizer, not required for correctness | MIR infrastructure |

---

## 11. Implementation Checklist

1. Create `crates/core/gpu-runtime/src/par_iter.rs`
2. Add types: `GpuSlice`, `GpuSliceMut`, `SendPtr`, `SendPtrMut`
3. Add traits: `GpuParallelIterator`, `GpuZero`, `GpuOne`, `GpuMaxValue`, `GpuMinValue`
4. Add adapters: `GpuParIter`, `GpuMap`, `GpuEnumerate`, `GpuZip`
5. Implement default terminal functions: `default_for_each`, `default_fold`, `default_collect_into`
6. Implement `GpuParallelIterator` for each adapter type
7. Add `pub mod par_iter;` to `lib.rs`
8. Add re-exports to `prelude.rs`
9. Write integration test: 1M f32 elements, `par_iter().map(|x| x * 2.0 + 1.0).collect_into(out)`, verify against CPU

## Files Changed: none (design document only)
