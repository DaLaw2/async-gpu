//! Type-level safety primitives for race-free GPU programming.
//!
//! Inspired by cuda-oxide's 3-tier safety model, adapted for async-gpu's
//! warp-cooperative execution model where each **warp** (not lane) is the
//! unit of parallelism.
//!
//! # Key Types
//!
//! - [`WarpIndex`] — opaque witness proving this code runs on a specific warp.
//!   Cannot be constructed, copied, sent, or stored. Obtained only from
//!   [`BlockScope::spawn_all_indexed`](crate::scope::BlockScope::spawn_all_indexed).
//!
//! - [`DisjointSlice`] — a slice where each warp gets exclusive access to its
//!   own partition. Access requires a `WarpIndex` witness, providing compile-time
//!   proof of ownership.
//!
//! - [`WarpHandle`] — witness that all 32 lanes are active and converged.
//!   Lifts warp-level operations (reduce, shuffle, ballot) from `unsafe` to safe.
//!
//! # Safety Model
//!
//! | Tier | Mechanism | unsafe? |
//! |------|-----------|---------|
//! | 1 | DisjointSlice + WarpIndex — per-warp exclusivity | No |
//! | 2 | WarpHandle — safe warp ops via witness | No |
//! | 3 | Raw PTX asm, manual pointer arithmetic | Yes |
//!
//! # Example
//!
//! ```rust,ignore
//! use gpu_runtime::scope::block_scope;
//!
//! block_scope(|scope| {
//!     let buf = scope.alloc::<f32>(256);
//!     let output = scope.disjoint_slice::<f32>(buf);
//!
//!     scope.spawn_all_indexed(|widx, warp| {
//!         // widx: WarpIndex — compile-time proof of warp identity
//!         // warp: WarpHandle — safe warp-level operations
//!         let my_partition = output.get_mut(&widx);
//!         for slot in my_partition.iter_mut() {
//!             *slot = 1.0;
//!         }
//!         let sum = warp.reduce_sum_f32(my_partition.len() as f32);
//!     });
//! });
//! ```

use core::marker::PhantomData;

// ============================================================
// WarpIndex — opaque witness of warp identity within a scope
// ============================================================

/// Opaque witness proving this code runs on a specific warp within a scope.
///
/// Cannot be constructed, copied, sent, or stored. Obtained only from
/// [`BlockScope::spawn_all_indexed`](crate::scope::BlockScope::spawn_all_indexed).
///
/// The `'scope` lifetime ties the witness to the enclosing scope, preventing
/// it from escaping to a different scope or being stored in shared memory.
///
/// # Properties
///
/// - `!Send` — cannot be sent to another warp (via `PhantomData<*const ()>`)
/// - `!Sync` — cannot be shared between warps
/// - `!Copy` — cannot be duplicated
/// - `!Clone` — cannot be cloned
/// - `'scope`-bounded — cannot outlive the enclosing scope
///
/// # Example
///
/// ```rust,ignore
/// scope.spawn_all_indexed(|widx, _warp| {
///     let id = widx.warp_id();     // which warp am I?
///     let total = widx.n_warps();  // how many warps total?
///     let my_slice = disjoint.get_mut(&widx); // my exclusive partition
/// });
/// ```
pub struct WarpIndex<'scope> {
    warp_id: u32,
    n_warps: u32,
    /// Makes this type `!Send + !Sync` (raw pointers are not Send/Sync).
    _not_send: PhantomData<*const ()>,
    /// Invariant lifetime — prevents covariance from allowing escape.
    _scope: PhantomData<&'scope mut &'scope ()>,
}

// Explicitly NOT implementing Send, Sync, Copy, Clone.
// PhantomData<*const ()> already makes it !Send + !Sync.
// Absence of derive(Copy, Clone) makes it !Copy + !Clone.

impl<'scope> WarpIndex<'scope> {
    /// Create a new WarpIndex. This is `pub(crate)` — only scope machinery
    /// can construct this type.
    #[inline(always)]
    pub(crate) fn new(warp_id: u32, n_warps: u32) -> Self {
        Self {
            warp_id,
            n_warps,
            _not_send: PhantomData,
            _scope: PhantomData,
        }
    }

    /// Returns this warp's ID (0-based index within the block).
    #[inline(always)]
    pub fn warp_id(&self) -> u32 {
        self.warp_id
    }

    /// Returns the total number of warps participating.
    #[inline(always)]
    pub fn n_warps(&self) -> u32 {
        self.n_warps
    }

    /// Convert a partition-local index to a global index using round-robin striding.
    ///
    /// Given `local_i` (the i-th element in this warp's partition), returns
    /// the corresponding index in the original array. This is the inverse of
    /// the round-robin partitioning used by `DisjointSlice`.
    ///
    /// `global_index = local_i * n_warps + warp_id`
    #[inline(always)]
    pub fn global_index(&self, local_i: usize) -> usize {
        local_i * self.n_warps as usize + self.warp_id as usize
    }
}

// ============================================================
// DisjointSlice — per-warp exclusive partitions of a slice
// ============================================================

/// A slice where each warp gets exclusive access to its own partition.
///
/// Access requires a [`WarpIndex`] witness with the same `'scope` lifetime,
/// providing compile-time proof that the caller is a specific warp within
/// the scope. Each warp sees only its own elements (round-robin striding),
/// so no two warps can access the same element.
///
/// # Partitioning
///
/// Elements are distributed round-robin across warps:
/// - Warp 0 gets indices: 0, n_warps, 2*n_warps, ...
/// - Warp 1 gets indices: 1, n_warps+1, 2*n_warps+1, ...
/// - Warp k gets indices: k, n_warps+k, 2*n_warps+k, ...
///
/// This matches async-gpu's existing partitioning convention in `par_iter`
/// and `spawn_all`.
///
/// # Safety Properties
///
/// - `!Send + !Sync` — cannot escape the scope
/// - Access requires `&WarpIndex<'scope>` — compile-time proof of warp identity
/// - Round-robin guarantees disjoint partitions — no aliasing possible
/// - Zero runtime overhead: partition bounds computed from warp_id/n_warps
///
/// # Example
///
/// ```rust,ignore
/// block_scope(|scope| {
///     let buf = scope.alloc::<f32>(1024);
///     let disjoint = scope.disjoint_slice::<f32>(buf);
///
///     scope.spawn_all_indexed(|widx, _warp| {
///         let my_partition = disjoint.get_mut(&widx);
///         for (i, slot) in my_partition.iter_mut().enumerate() {
///             let global_i = widx.global_index(i);
///             *slot = global_i as f32;
///         }
///     });
/// });
/// ```
pub struct DisjointSlice<'scope, T: Copy> {
    ptr: *mut T,
    len: usize,
    /// Phantom mutable borrow — prevents aliasing at the type level.
    _scope: PhantomData<&'scope mut [T]>,
}

// DisjointSlice is !Send + !Sync because it contains a raw pointer (*mut T)
// and we do NOT implement Send/Sync for it. The PhantomData<&'scope mut [T]>
// also prevents it from being covariant over 'scope.

impl<'scope, T: Copy> DisjointSlice<'scope, T> {
    /// Create a new DisjointSlice. This is `pub(crate)` — only scope machinery
    /// can construct this type.
    ///
    /// # Safety
    ///
    /// - `ptr` must point to `len` valid, initialized elements of `T`.
    /// - The memory must remain valid for `'scope`.
    /// - No other code may access the same memory while this DisjointSlice exists,
    ///   except through this DisjointSlice's accessor methods.
    #[inline(always)]
    pub(crate) unsafe fn new(ptr: *mut T, len: usize) -> Self {
        Self {
            ptr,
            len,
            _scope: PhantomData,
        }
    }

    /// Returns the total number of elements in the underlying slice.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the underlying slice is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns this warp's exclusive mutable partition of the slice.
    ///
    /// Elements are distributed contiguously across warps: warp `k` gets a
    /// contiguous sub-slice `[start..start+count)` where the total is divided
    /// as evenly as possible, with the first `remainder` warps getting one
    /// extra element.
    ///
    /// # Safety Argument
    ///
    /// This method takes `&self` but returns `&mut [T]`. This is sound because:
    /// 1. The `WarpIndex<'scope>` witness guarantees the caller is a specific warp.
    /// 2. Each warp produces a different `WarpIndex` (constructed by `spawn_all_indexed`).
    /// 3. The contiguous partitioning is deterministic and non-overlapping.
    /// 4. Therefore, no two callers can receive overlapping `&mut [T]` slices.
    ///
    /// This is the same pattern as `Cell::get` / `UnsafeCell::get` — interior
    /// mutability where safety is enforced by external invariants (here, the
    /// warp identity witness).
    ///
    /// # Returns
    ///
    /// A mutable slice of this warp's contiguous partition. The slice may be
    /// empty if `len < n_warps` and this warp has no elements.
    #[inline(always)]
    #[allow(clippy::mut_from_ref)]
    pub fn get_mut(&self, idx: &WarpIndex<'scope>) -> &mut [T] {
        if self.len == 0 {
            return &mut [];
        }

        let wid = idx.warp_id() as usize;
        let nw = idx.n_warps() as usize;

        // Contiguous partitioning: distribute remainder across first `remainder` warps
        let chunk = self.len / nw;
        let remainder = self.len % nw;
        let start = wid * chunk + if wid < remainder { wid } else { remainder };
        let count = chunk + if wid < remainder { 1 } else { 0 };

        if count == 0 {
            return &mut [];
        }

        // SAFETY: Each warp gets a disjoint contiguous sub-slice.
        // The WarpIndex witness guarantees the caller is the correct warp.
        // No two warps can produce the same WarpIndex (they are constructed
        // by spawn_all_indexed with the warp's actual ID).
        unsafe { core::slice::from_raw_parts_mut(self.ptr.add(start), count) }
    }

    /// Bounds-checked immutable read of a single element by global index.
    ///
    /// Returns `None` if `global_index >= len`.
    ///
    /// # Note
    ///
    /// This does NOT enforce disjoint access — any warp can read any element.
    /// For mutable access with disjoint guarantees, use [`get_mut`](Self::get_mut).
    /// Immutable reads are always safe (no data race on reads).
    #[inline(always)]
    pub fn get(&self, global_index: usize) -> Option<&T> {
        if global_index >= self.len {
            return None;
        }
        // SAFETY: bounds check above guarantees global_index < len.
        unsafe { Some(&*self.ptr.add(global_index)) }
    }

    /// Returns the raw pointer and length of the underlying buffer.
    ///
    /// This is an escape hatch for advanced use cases (e.g., passing to
    /// warp-cooperative operations that need the full buffer). The caller
    /// must ensure proper synchronization.
    ///
    /// # Safety
    ///
    /// The caller must not create aliasing mutable references to partitions
    /// that are currently borrowed via `get_mut`.
    #[inline(always)]
    pub unsafe fn raw_parts(&self) -> (*mut T, usize) {
        (self.ptr, self.len)
    }
}

// ============================================================
// WarpHandle — witness that all lanes are active and converged
// ============================================================

/// Witness that all 32 lanes of this warp are active and converged.
///
/// Constructed only by trusted entry points
/// ([`BlockScope::spawn_all_indexed`](crate::scope::BlockScope::spawn_all_indexed)),
/// this type lifts warp-level operations from `unsafe` to safe.
///
/// The safety contract: when you hold a `WarpHandle`, all 32 lanes are
/// executing the same code path. This is exactly the precondition that
/// warp shuffle/vote/reduce operations require.
///
/// # Properties
///
/// - `!Send + !Sync` — cannot escape the warp
/// - `!Copy + !Clone` — cannot be duplicated
/// - `'scope`-bounded — cannot outlive the enclosing scope
///
/// # Example
///
/// ```rust,ignore
/// scope.spawn_all_indexed(|widx, warp| {
///     let sum = warp.reduce_sum_f32(my_val);  // safe! WarpHandle proves convergence
///     let max = warp.reduce_max_f32(my_val);
/// });
/// ```
pub struct WarpHandle<'scope> {
    /// Active lane mask — always 0xFFFF_FFFF for full-warp entry points.
    mask: u32,
    /// Makes this type `!Send + !Sync`.
    _not_send: PhantomData<*const ()>,
    /// Invariant lifetime.
    _scope: PhantomData<&'scope ()>,
}

// Explicitly NOT implementing Send, Sync, Copy, Clone.

impl<'scope> WarpHandle<'scope> {
    /// Create a new WarpHandle. This is `pub(crate)` — only scope machinery
    /// can construct this type.
    #[inline(always)]
    pub(crate) fn new() -> Self {
        Self {
            mask: 0xFFFF_FFFF,
            _not_send: PhantomData,
            _scope: PhantomData,
        }
    }

    /// Returns the active lane mask (always `0xFFFF_FFFF` for full-warp handles).
    #[inline(always)]
    pub fn active_mask(&self) -> u32 {
        self.mask
    }

    /// Butterfly reduction: sum of `val` across all 32 lanes.
    ///
    /// Safe because `WarpHandle` proves all lanes are active.
    #[inline(always)]
    pub fn reduce_sum_f32(&self, val: f32) -> f32 {
        // SAFETY: WarpHandle guarantees all lanes are active and converged.
        unsafe { crate::warp::reduce_sum_f32(val) }
    }

    /// Butterfly reduction: sum of `val` (u32) across all 32 lanes.
    ///
    /// Safe because `WarpHandle` proves all lanes are active.
    #[inline(always)]
    pub fn reduce_sum_u32(&self, val: u32) -> u32 {
        unsafe { crate::warp::reduce_sum_u32(val) }
    }

    /// Butterfly reduction: maximum of `val` across all 32 lanes.
    ///
    /// Safe because `WarpHandle` proves all lanes are active.
    #[inline(always)]
    pub fn reduce_max_f32(&self, val: f32) -> f32 {
        unsafe { crate::warp::reduce_max_f32(val) }
    }

    /// Butterfly reduction: minimum of `val` across all 32 lanes.
    ///
    /// Safe because `WarpHandle` proves all lanes are active.
    #[inline(always)]
    pub fn reduce_min_f32(&self, val: f32) -> f32 {
        unsafe { crate::warp::reduce_min_f32(val) }
    }

    /// Butterfly shuffle: exchange `val` with lane at `(lane_id ^ offset)`.
    ///
    /// Safe because `WarpHandle` proves all lanes are active.
    #[inline(always)]
    pub fn shfl_bfly_u32(&self, val: u32, offset: u32) -> u32 {
        unsafe { crate::warp::shfl_bfly_u32(self.mask, val, offset) }
    }

    /// Shuffle down: read `val` from `(lane_id + delta)`.
    ///
    /// Safe because `WarpHandle` proves all lanes are active.
    #[inline(always)]
    pub fn shfl_down_u32(&self, val: u32, delta: u32) -> u32 {
        unsafe { crate::warp::shfl_down_u32(self.mask, val, delta) }
    }

    /// Shuffle up: read `val` from `(lane_id - delta)`.
    ///
    /// Safe because `WarpHandle` proves all lanes are active.
    #[inline(always)]
    pub fn shfl_up_u32(&self, val: u32, delta: u32) -> u32 {
        unsafe { crate::warp::shfl_up_u32(self.mask, val, delta) }
    }

    /// Warp vote: ballot — returns bitmask of lanes where `predicate` is true.
    ///
    /// Safe because `WarpHandle` proves all lanes are active.
    #[inline(always)]
    pub fn ballot(&self, predicate: bool) -> u32 {
        unsafe { crate::warp::ballot(self.mask, predicate) }
    }

    /// Warp vote: true if `predicate` is true for ALL lanes.
    ///
    /// Safe because `WarpHandle` proves all lanes are active.
    #[inline(always)]
    pub fn all(&self, predicate: bool) -> bool {
        unsafe { crate::warp::all(self.mask, predicate) }
    }

    /// Warp vote: true if `predicate` is true for ANY lane.
    ///
    /// Safe because `WarpHandle` proves all lanes are active.
    #[inline(always)]
    pub fn any(&self, predicate: bool) -> bool {
        unsafe { crate::warp::any(self.mask, predicate) }
    }
}
