//! Structured concurrency scopes with lifetime-bounded memory allocation.
//!
//! Provides two scope primitives:
//!
//! - [`BlockScope`] — block-level scope over shared memory, coordinates warps
//!   within a single block.
//! - [`GridScope`] — grid-level scope over a pre-allocated global memory pool,
//!   coordinates work across multiple blocks via system-scope atomics.
//!
//! Both use the `for<'scope>` higher-ranked trait bound pattern (like Rayon's
//! `scope()`) to prevent allocated references from escaping their scope.
//!
//! # Key types
//!
//! - [`SharedMemAllocator`] — watermark/bump allocator over shared memory
//! - [`BlockScope`] — structured concurrency scope for a single block
//! - [`ScopeJoinHandle`] — handle to a task spawned within a BlockScope
//! - [`block_scope()`] — entry function that creates a `BlockScope`
//! - [`GridScope`] — structured concurrency scope across GPU blocks
//! - [`grid_scope()`] — entry function that creates a `GridScope`
//!
//! # Examples
//!
//! ## BlockScope (shared memory, intra-block)
//!
//! ```rust,ignore
//! use gpu_runtime::scope::block_scope;
//!
//! block_scope(|scope| {
//!     let buf = scope.alloc::<f32>(256);
//!     scope.spawn_all(|wid, n_warps| {
//!         let mut i = wid as usize;
//!         while i < 256 {
//!             buf[i] = i as f32;
//!             i += n_warps as usize;
//!         }
//!     });
//! });
//! ```
//!
//! ## GridScope (global memory, cross-block)
//!
//! ```rust,ignore
//! use gpu_runtime::scope::grid_scope;
//!
//! // pool is a pre-allocated global memory region (at least 8 bytes for header)
//! unsafe {
//!     grid_scope(pool, pool_size, |gscope| {
//!         let data = gscope.alloc::<f32>(1024);
//!         // ... distribute work across blocks ...
//!         // Blocks signal completion via gscope.completion_counter_ptr()
//!     });
//! }
//! ```
//!
//! # Composing with cooperative APIs
//!
//! Scope-allocated shared memory composes cleanly with the cooperative APIs
//! in [`crate::thread`] (`cooperative_map`, `cooperative_reduce`,
//! `cooperative_map_with_params`). The key rules:
//!
//! 1. **Scope-allocated buffers as cooperative src/dst**: `scope.alloc::<T>(n)`
//!    returns `&'scope mut [T]` backed by shared memory. Call `.as_ptr()` /
//!    `.as_mut_ptr()` and cast to `*const u8` / `*mut u8` to pass to
//!    `cooperative_map()`. The pointers remain valid for the scope's lifetime.
//!
//! 2. **cooperative_map inside a scope**: Works correctly as long as no
//!    `scope.spawn()` tasks are in-flight (all must be joined first).
//!    `cooperative_map` wakes all warps via `STATUS_COOPERATIVE`, which is
//!    the same mechanism as `scope.spawn_all()`. After it returns, all
//!    warps are back to `STATUS_IDLE`.
//!
//! 3. **Prefer `spawn_all` over `cooperative_map` inside scopes**:
//!    `scope.spawn_all()` is the preferred way to do cooperative work within
//!    a scope. It can capture scope-allocated references in its closure
//!    (closures vs function pointers), provides error detection via
//!    `error_mask()`, and integrates with the scope's warp tracking.
//!    `cooperative_map` uses function pointers and global statics for
//!    argument passing, which is more restrictive.
//!
//! 4. **Do NOT interleave spawn and cooperative_map**: If you call
//!    `scope.spawn()` and a warp is still running, calling `cooperative_map()`
//!    will corrupt that warp's status. Always `join_all()` before using any
//!    cooperative API. `spawn_all()` already enforces this with an assertion.
//!
//! # Safety model
//!
//! - Only warp 0 (block 0 for GridScope) may call scope entry and `alloc()`.
//! - The `for<'scope>` HRTB on entry functions prevents scope-allocated
//!   references from escaping the closure.
//! - `PhantomData<&'scope mut &'scope ()>` enforces lifetime invariance.
//! - `T: Copy` is required for all allocations (no Drop, no destructors needed).
//! - GridScope uses system-scope atomics for cross-block visibility.

use core::cell::UnsafeCell;
use core::marker::PhantomData;
use core::sync::atomic::Ordering;

use crate::thread::{
    lane_id, nanosleep_short, warp_id, NUM_WARPS, SCRATCH, SCRATCH_SIZE, STATUS_ASSIGNED,
    STATUS_COOPERATIVE, STATUS_DONE, STATUS_IDLE, STATUS_TRAPPED, WARP_DATA, WARP_FN, WARP_RESULT,
    WARP_STATUS,
};

// ============================================================
// SharedMemAllocator — watermark/bump allocator over shared memory
// ============================================================

/// Maximum nesting depth for scopes.
const MAX_SCOPE_DEPTH: usize = 4;

/// Watermark (bump) allocator over the block's shared memory.
///
/// Maintains a stack of watermarks: each nested [`block_scope`] pushes
/// a new mark; scope exit pops back, logically freeing all allocations
/// made within that scope.
///
/// Lives in global memory (static), tracks offsets into shared memory.
/// Only warp 0 may access this allocator.
pub struct SharedMemAllocator {
    /// Current allocation offset (bytes from shared_mem_ptr base).
    watermark: u32,
    /// Stack of saved watermarks for nested scopes.
    stack: [u32; MAX_SCOPE_DEPTH],
    /// Current nesting depth.
    depth: u32,
    /// Total shared memory available (set at init time).
    capacity: u32,
}

impl SharedMemAllocator {
    /// Create an uninitialized allocator. Must call [`init`](Self::init)
    /// before use.
    const fn uninit() -> Self {
        Self {
            watermark: 0,
            stack: [0; MAX_SCOPE_DEPTH],
            depth: 0,
            capacity: 0,
        }
    }

    /// Initialize the allocator with the total shared memory capacity in bytes.
    ///
    /// Must be called once from warp 0 before any `block_scope` calls.
    pub fn init(&mut self, capacity: u32) {
        self.watermark = 0;
        self.depth = 0;
        self.capacity = capacity;
    }

    /// Push the current watermark onto the stack, returning the saved mark.
    /// Called at scope entry.
    ///
    /// # Panics
    ///
    /// Panics if nesting depth exceeds [`MAX_SCOPE_DEPTH`].
    fn push(&mut self) -> u32 {
        assert!(
            (self.depth as usize) < MAX_SCOPE_DEPTH,
            "SharedMemAllocator: nesting depth exceeded (max 4)"
        );
        let saved = self.watermark;
        self.stack[self.depth as usize] = saved;
        self.depth += 1;
        saved
    }

    /// Pop back to the previously saved watermark.
    /// Called at scope exit — logically frees all allocations since the
    /// corresponding [`push`](Self::push).
    ///
    /// # Panics
    ///
    /// Panics if there is no corresponding push.
    fn pop(&mut self) {
        assert!(self.depth > 0, "SharedMemAllocator: pop without push");
        self.depth -= 1;
        self.watermark = self.stack[self.depth as usize];
    }

    /// Bump-allocate `size` bytes, aligned to `align`.
    ///
    /// Returns the byte offset from the shared memory base, or `None` if
    /// the allocation would exceed capacity.
    fn alloc_raw(&mut self, size: usize, align: usize) -> Option<u32> {
        // Round up watermark to required alignment
        let aligned = (self.watermark as usize + align - 1) & !(align - 1);
        let new_watermark = aligned + size;
        if new_watermark > self.capacity as usize {
            return None;
        }
        self.watermark = new_watermark as u32;
        Some(aligned as u32)
    }

    /// Returns the number of bytes remaining (approximate — does not account
    /// for alignment padding of the next allocation).
    fn available(&self) -> usize {
        (self.capacity as usize).saturating_sub(self.watermark as usize)
    }
}

// SAFETY: Only warp 0 accesses the allocator. The single-writer invariant
// is enforced by debug_assert in all public allocation methods.
unsafe impl Sync for AllocatorCell {}

/// Wrapper to hold the allocator in a static with interior mutability.
struct AllocatorCell(UnsafeCell<SharedMemAllocator>);

impl AllocatorCell {
    const fn new() -> Self {
        Self(UnsafeCell::new(SharedMemAllocator::uninit()))
    }

    /// Get a raw mutable pointer to the allocator.
    ///
    /// # Safety
    ///
    /// Caller must be warp 0, lane 0. No other warp may access the allocator
    /// concurrently. The caller must ensure proper aliasing rules.
    #[inline(always)]
    fn as_ptr(&self) -> *mut SharedMemAllocator {
        self.0.get()
    }
}

/// Global allocator instance. Lives in global memory, tracks offsets into
/// shared memory. Only warp 0 accesses this.
static ALLOCATOR: AllocatorCell = AllocatorCell::new();

/// Initialize the shared memory allocator with the given capacity.
///
/// Must be called once from warp 0 before any `block_scope` calls.
/// Typically called at kernel entry with the launch config's
/// `shared_mem_bytes` value.
///
/// # Safety
///
/// Must be called from warp 0, lane 0.
pub unsafe fn init_shared_mem_allocator(capacity: u32) {
    debug_assert_eq!(
        warp_id(),
        0,
        "init_shared_mem_allocator: must be called from warp 0"
    );
    (&mut *ALLOCATOR.as_ptr()).init(capacity);
}

// ============================================================
// BlockScope — structured concurrency scope for a single block
// ============================================================

/// A structured concurrency scope bound to a single GPU block.
///
/// Allocations come from shared memory via the watermark allocator.
/// Spawned closures are `'scope`-bounded: they can borrow data that
/// lives at least as long as the scope, but cannot outlive it.
///
/// Created by [`block_scope()`]. All spawned tasks are joined before
/// the scope closure returns.
pub struct BlockScope<'scope> {
    /// Bitmask of warp IDs spawned within this scope (max 32 warps).
    spawned_warps: u32,
    /// Number of warps spawned (for iteration during join).
    spawn_count: u32,
    /// Byte offset of the cancellation flag in shared memory.
    cancel_flag_offset: u32,
    /// Bitmask of warps that trapped (panicked) during execution.
    /// Each set bit corresponds to a warp that entered STATUS_TRAPPED.
    error_mask: u32,
    /// Invariant lifetime — prevents covariance from allowing escape.
    _marker: PhantomData<&'scope mut &'scope ()>,
}

impl<'scope> BlockScope<'scope> {
    /// Allocate a zero-initialized mutable slice of `count` elements from
    /// shared memory.
    ///
    /// Returns `&'scope mut [T]` — the slice is valid for the lifetime of
    /// this scope. When the scope exits, the watermark is popped and this
    /// memory is logically reclaimed.
    ///
    /// # Panics
    ///
    /// Panics if the allocation would exceed shared memory capacity.
    ///
    /// # Alignment
    ///
    /// Automatically aligns to `core::mem::align_of::<T>()`.
    pub fn alloc<T: Copy>(&self, count: usize) -> &'scope mut [T] {
        debug_assert_eq!(warp_id(), 0, "scope.alloc() can only be called from warp 0");
        let size = core::mem::size_of::<T>() * count;
        let align = core::mem::align_of::<T>();

        let offset = unsafe { &mut *ALLOCATOR.as_ptr() }
            .alloc_raw(size, align)
            .expect("BlockScope::alloc: shared memory exhausted");

        unsafe {
            let ptr = crate::block::shared_mem_at::<T>(offset as usize);
            // Zero-initialize the region
            core::ptr::write_bytes(ptr, 0, count);
            core::slice::from_raw_parts_mut(ptr, count)
        }
    }

    /// Allocate a mutable slice WITHOUT zero-initialization.
    ///
    /// # Safety
    ///
    /// The caller must initialize all elements before reading them.
    pub unsafe fn alloc_uninit<T: Copy>(&self, count: usize) -> &'scope mut [T] {
        debug_assert_eq!(
            warp_id(),
            0,
            "scope.alloc_uninit() can only be called from warp 0"
        );
        let size = core::mem::size_of::<T>() * count;
        let align = core::mem::align_of::<T>();

        let offset = (&mut *ALLOCATOR.as_ptr())
            .alloc_raw(size, align)
            .expect("BlockScope::alloc_uninit: shared memory exhausted");

        let ptr = crate::block::shared_mem_at::<T>(offset as usize);
        core::slice::from_raw_parts_mut(ptr, count)
    }

    /// Allocate a single value in shared memory, initialized to `val`.
    pub fn alloc_val<T: Copy>(&self, val: T) -> &'scope mut T {
        let slot = self.alloc::<T>(1);
        slot[0] = val;
        &mut slot[0]
    }

    /// Returns the number of bytes remaining in the shared memory pool.
    pub fn available_bytes(&self) -> usize {
        unsafe { &mut *ALLOCATOR.as_ptr() }.available()
    }

    /// Spawn a closure on an idle warp within this block.
    ///
    /// The closure is bounded by `'scope` — it can borrow anything that
    /// lives at least as long as the scope (including scope-allocated
    /// shared memory). The closure runs on a single warp (lane 0 executes
    /// the closure body; all 32 lanes participate in any warp-level ops
    /// the closure may use internally).
    ///
    /// Returns a [`ScopeJoinHandle`] that can be joined explicitly, or
    /// will be joined implicitly when the scope exits.
    ///
    /// # Panics
    ///
    /// Panics if no idle warps are available or if the closure + result
    /// exceeds the per-warp scratch buffer size (256 bytes).
    pub fn spawn<F, T>(&mut self, f: F) -> ScopeJoinHandle<'scope, T>
    where
        F: FnOnce() -> T + Send + 'scope,
        T: Send + 'scope,
    {
        debug_assert_eq!(warp_id(), 0, "scope.spawn() can only be called from warp 0");

        // Type-erased trampoline: called by worker warp's lane 0
        fn trampoline<F, T>(raw: *mut u8)
        where
            F: FnOnce() -> T,
        {
            let lid = crate::index::thread_idx_x() % 32;
            if lid == 0 {
                let f = unsafe { core::ptr::read(raw as *const F) };
                let result = f();

                // Write result after the closure data in the scratch buffer
                let result_slot = unsafe {
                    let slot = raw.add(core::mem::size_of::<F>()) as *mut T;
                    core::ptr::write(slot, result);
                    slot
                };

                let wid = crate::index::thread_idx_x() / 32;
                WARP_RESULT[wid as usize].store(result_slot as u64, Ordering::Release);
            }
        }

        let n_warps = NUM_WARPS.load(Ordering::Acquire) as usize;

        // Find an idle warp (linear scan from warp 1)
        let target_warp = loop {
            let found =
                (1..n_warps).find(|&i| WARP_STATUS[i].load(Ordering::Acquire) == STATUS_IDLE);
            if let Some(w) = found {
                break w;
            }
            nanosleep_short();
        };

        // Copy closure data into the warp's scratch buffer
        let scratch_ptr = SCRATCH[target_warp].as_ptr() as *mut u8;
        let closure_size = core::mem::size_of::<F>();
        assert!(
            closure_size + core::mem::size_of::<T>() <= SCRATCH_SIZE,
            "scope.spawn: closure + result too large for scratch buffer"
        );
        unsafe {
            core::ptr::write(scratch_ptr as *mut F, f);
        }

        // Set up warp slot
        let trampoline_fn = trampoline::<F, T> as fn(*mut u8);
        WARP_FN[target_warp].store(trampoline_fn as usize as u64, Ordering::Relaxed);
        WARP_DATA[target_warp].store(scratch_ptr as u64, Ordering::Relaxed);
        WARP_RESULT[target_warp].store(0, Ordering::Relaxed);

        // Wake the worker warp
        WARP_STATUS[target_warp].store(STATUS_ASSIGNED, Ordering::Release);

        // Record in spawned bitmask
        self.spawned_warps |= 1 << target_warp;
        self.spawn_count += 1;

        ScopeJoinHandle {
            warp_id: target_warp,
            _marker: PhantomData,
        }
    }

    /// Spawn a closure cooperatively across all warps, data-parallel style.
    ///
    /// All idle warps and warp 0 execute the closure. The closure receives
    /// `(warp_id, n_warps)` so each warp can determine its data partition.
    /// All warps are joined before this function returns (synchronous).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// block_scope(|scope| {
    ///     let data = scope.alloc::<f32>(1024);
    ///     scope.spawn_all(|wid, n_warps| {
    ///         let mut i = wid as usize;
    ///         while i < 1024 {
    ///             data[i] = i as f32;
    ///             i += n_warps as usize;
    ///         }
    ///     });
    /// });
    /// ```
    pub fn spawn_all<F>(&mut self, f: F)
    where
        F: Fn(u32, u32) + Send + Sync + 'scope,
    {
        debug_assert_eq!(
            warp_id(),
            0,
            "scope.spawn_all() can only be called from warp 0"
        );

        // Safety: spawn_all wakes ALL worker warps with STATUS_COOPERATIVE.
        // If any warps are still in-flight from a prior spawn(), their status
        // would be corrupted. Assert that all spawned warps have been joined.
        assert!(
            self.spawned_warps == 0,
            "scope.spawn_all: all spawned tasks must be joined before calling spawn_all"
        );

        let n_warps = NUM_WARPS.load(Ordering::Acquire) as usize;
        let total = if n_warps == 0 { 1 } else { n_warps };

        if total <= 1 {
            // Single warp: just call directly on warp 0
            if lane_id() == 0 {
                f(0, 1);
            }
            return;
        }

        // Trampoline for worker warps: reads the Fn from warp 0's scratch,
        // calls it with (warp_id, n_warps).
        fn spawn_all_trampoline<F: Fn(u32, u32)>(raw: *mut u8) {
            let lid = crate::index::thread_idx_x() % 32;
            if lid == 0 {
                let f = unsafe { &*(raw as *const F) };
                let wid = crate::index::thread_idx_x() / 32;
                let nw = NUM_WARPS.load(Ordering::Acquire);
                f(wid, nw);
            }
        }

        let closure_size = core::mem::size_of::<F>();
        assert!(
            closure_size <= SCRATCH_SIZE,
            "scope.spawn_all: closure too large for scratch buffer"
        );

        let trampoline_fn = spawn_all_trampoline::<F> as fn(*mut u8);

        // Copy closure to each worker's scratch buffer and wake them
        if lane_id() == 0 {
            for i in 1..total {
                let scratch_ptr = SCRATCH[i].as_ptr() as *mut u8;
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        &f as *const F as *const u8,
                        scratch_ptr,
                        closure_size,
                    );
                }
                WARP_FN[i].store(trampoline_fn as usize as u64, Ordering::Relaxed);
                WARP_DATA[i].store(scratch_ptr as u64, Ordering::Relaxed);
                WARP_STATUS[i].store(STATUS_COOPERATIVE, Ordering::Release);
            }
        }

        // Warp 0 also participates
        if lane_id() == 0 {
            f(0, total as u32);
        }

        // Join all workers (with trap detection)
        #[allow(clippy::needless_range_loop)]
        for i in 1..total {
            loop {
                let s = WARP_STATUS[i].load(Ordering::Acquire);
                if s == STATUS_DONE {
                    WARP_STATUS[i].store(STATUS_IDLE, Ordering::Release);
                    break;
                }
                if s == STATUS_TRAPPED {
                    // Warp is dead — record in error mask, do not reset to IDLE.
                    self.error_mask |= 1 << i;
                    break;
                }
                nanosleep_short();
            }
        }
    }

    /// Request cooperative cancellation of all tasks in this scope.
    ///
    /// Sets a flag in shared memory. Spawned tasks should check
    /// [`is_cancelled()`](Self::is_cancelled) at appropriate points and
    /// return early. This is cooperative — tasks are not forcibly stopped.
    pub fn cancel(&self) {
        unsafe {
            let flag = crate::block::shared_mem_at::<u32>(self.cancel_flag_offset as usize);
            core::ptr::write_volatile(flag, 1);
        }
    }

    /// Check if this scope has been cancelled.
    pub fn is_cancelled(&self) -> bool {
        unsafe {
            let flag = crate::block::shared_mem_at::<u32>(self.cancel_flag_offset as usize);
            core::ptr::read_volatile(flag) != 0
        }
    }

    /// Join all spawned warps that have not yet been joined, and reset
    /// the bitmask so the warps can be reused.
    ///
    /// Detects `STATUS_TRAPPED` warps (panicked) and records them in the
    /// error mask instead of spinning forever. Trapped warps are NOT reset
    /// to `STATUS_IDLE` because they are dead and cannot re-enter the
    /// worker loop.
    ///
    /// Can be called explicitly mid-scope to synchronize, or is called
    /// automatically at scope exit (via `Drop`).
    pub fn join_all(&mut self) {
        let mut mask = self.spawned_warps;
        while mask != 0 {
            let wid = mask.trailing_zeros() as usize;
            // Spin-wait until the warp finishes or traps
            loop {
                let status = WARP_STATUS[wid].load(Ordering::Acquire);
                if status == STATUS_DONE {
                    WARP_STATUS[wid].store(STATUS_IDLE, Ordering::Release);
                    break;
                }
                if status == STATUS_TRAPPED {
                    // Warp is dead — record in error mask.
                    // Do NOT reset to IDLE; the warp cannot re-enter
                    // the worker loop after a trap.
                    self.error_mask |= 1 << wid;
                    break;
                }
                nanosleep_short();
            }
            mask &= !(1 << wid);
        }
        self.spawned_warps = 0;
        self.spawn_count = 0;
    }

    /// Returns the error bitmask. Each set bit corresponds to a warp
    /// that trapped (panicked) during execution within this scope.
    pub fn error_mask(&self) -> u32 {
        self.error_mask
    }

    /// Returns `true` if any spawned warp trapped during this scope.
    pub fn has_errors(&self) -> bool {
        self.error_mask != 0
    }
}

impl<'scope> Drop for BlockScope<'scope> {
    fn drop(&mut self) {
        // Safety net: join any un-joined spawned warps
        if self.spawned_warps != 0 {
            self.join_all();
        }
        // Pop the allocator watermark
        unsafe {
            (&mut *ALLOCATOR.as_ptr()).pop();
        }
    }
}

// ============================================================
// ScopeJoinHandle — handle to a task spawned within a BlockScope
// ============================================================

/// Handle to a task spawned within a [`BlockScope`].
///
/// The `'scope` lifetime ties this handle to the enclosing scope.
/// The result can be retrieved via [`join()`](Self::join), or the
/// scope will implicitly join all handles at exit.
pub struct ScopeJoinHandle<'scope, T> {
    warp_id: usize,
    _marker: PhantomData<&'scope T>,
}

impl<'scope, T> ScopeJoinHandle<'scope, T> {
    /// Block (spin-wait) until the spawned warp completes and return its result.
    ///
    /// Resets the warp to idle so it can be reused.
    ///
    /// # Panics
    ///
    /// Panics if the spawned warp trapped (GPU panic). The warp is dead
    /// and cannot produce a result.
    pub fn join(self) -> T {
        // Spin until the warp is done or trapped
        loop {
            let status = WARP_STATUS[self.warp_id].load(Ordering::Acquire);
            if status == STATUS_DONE {
                break;
            }
            if status == STATUS_TRAPPED {
                panic!(
                    "ScopeJoinHandle::join: warp {} trapped (GPU panic)",
                    self.warp_id
                );
            }
            nanosleep_short();
        }

        // Read the result
        let result_ptr = WARP_RESULT[self.warp_id].load(Ordering::Acquire) as *mut T;
        let result = unsafe { core::ptr::read(result_ptr) };

        // Reset the slot to IDLE for reuse
        WARP_STATUS[self.warp_id].store(STATUS_IDLE, Ordering::Release);

        result
    }
}

// ============================================================
// block_scope() — entry function
// ============================================================

/// Enter a block-level structured concurrency scope.
///
/// The closure receives a `&mut BlockScope<'scope>` handle for allocating
/// shared memory and spawning warps. All spawned work is joined before
/// this function returns.
///
/// # Lifetime guarantee
///
/// The `'scope` lifetime is shorter than any reference the closure borrows
/// from the enclosing function. This means `scope.alloc()` returns
/// `&'scope mut [T]` that can be passed to spawned closures — but cannot
/// escape the `block_scope` call.
///
/// # Composing with cooperative APIs
///
/// Scope-allocated buffers can be used with [`crate::thread::cooperative_map`]
/// and friends by converting to raw pointers:
///
/// ```rust,ignore
/// use gpu_runtime::scope::block_scope;
/// use gpu_runtime::thread;
///
/// block_scope(|scope| {
///     let src = scope.alloc::<f32>(256);
///     let dst = scope.alloc::<f32>(256);
///
///     // Initialize with spawn_all (preferred — closures, error tracking)
///     scope.spawn_all(|wid, nw| {
///         let mut i = wid as usize;
///         while i < 256 { src[i] = i as f32; i += nw as usize; }
///     });
///
///     // cooperative_map also works (function pointers, no scope tracking)
///     thread::cooperative_map(
///         src.as_ptr() as *const u8,
///         dst.as_mut_ptr() as *mut u8,
///         256,
///         |args| {
///             let s = args.src as *const f32;
///             let d = args.dst as *mut f32;
///             let mut i = args.warp_id as usize;
///             while i < args.len {
///                 unsafe {
///                     let v = core::ptr::read_volatile(s.add(i));
///                     core::ptr::write_volatile(d.add(i), v * 2.0);
///                 }
///                 i += args.n_warps as usize;
///             }
///         },
///     );
/// });
/// ```
///
/// **Important**: Do not call `cooperative_map` while `scope.spawn()` tasks
/// are still in-flight. Join all spawned tasks first.
///
/// # Safety
///
/// - Must be called from warp 0 (the main thread) within `gpu_main`.
/// - Shared memory must be large enough for the scope's allocations
///   plus internal bookkeeping (cancel flag: 4 bytes).
/// - The shared memory allocator must have been initialized via
///   [`init_shared_mem_allocator()`].
///
/// # Example
///
/// ```rust,ignore
/// use gpu_runtime::scope::block_scope;
///
/// let result: f32 = block_scope(|scope| {
///     let buf = scope.alloc::<f32>(256);
///     buf[0] = 42.0;
///     buf[0]
/// });
/// ```
pub fn block_scope<F, R>(f: F) -> R
where
    F: for<'scope> FnOnce(&mut BlockScope<'scope>) -> R,
{
    debug_assert_eq!(warp_id(), 0, "block_scope() must be called from warp 0");

    // 1. Push allocator watermark
    unsafe {
        (&mut *ALLOCATOR.as_ptr()).push();
    }

    // 2. Allocate cancel flag (4 bytes, u32) from shared memory
    let cancel_flag_offset = unsafe { &mut *ALLOCATOR.as_ptr() }
        .alloc_raw(core::mem::size_of::<u32>(), core::mem::align_of::<u32>())
        .expect("block_scope: not enough shared memory for cancel flag");

    // Zero-initialize the cancel flag
    unsafe {
        let flag = crate::block::shared_mem_at::<u32>(cancel_flag_offset as usize);
        core::ptr::write_volatile(flag, 0);
    }

    // 3. Construct BlockScope
    let mut scope = BlockScope {
        spawned_warps: 0,
        spawn_count: 0,
        cancel_flag_offset,
        error_mask: 0,
        _marker: PhantomData,
    };

    // 4. Call the closure
    let result = f(&mut scope);

    // 5. Join all spawned warps (explicit path — Drop is the safety net)
    scope.join_all();

    // 6. Pop allocator watermark (via Drop)
    //    The Drop impl will pop the watermark. We let it run naturally
    //    when `scope` goes out of scope here.

    result
}

// ============================================================
// GridScope — structured concurrency scope across GPU blocks
// ============================================================

/// Size of the GridScope header in the global memory pool.
///
/// Layout: `[completion_counter: u32, cancel_flag: u32]` = 8 bytes.
/// The pool offset starts after this header, so user allocations
/// begin at byte 8.
const GRID_SCOPE_HEADER_SIZE: u32 = 8;

/// A structured concurrency scope that coordinates across GPU blocks.
///
/// Allocations come from a pre-allocated global memory pool. Block
/// completion is tracked via an atomic counter in global memory using
/// system-scope atomics for cross-block visibility.
///
/// Created by [`grid_scope()`]. The `'scope` lifetime prevents global
/// memory references from escaping the scope closure.
///
/// # Memory layout
///
/// The first 8 bytes of the pool are reserved for internal bookkeeping:
/// - Bytes 0..4: `completion_counter` (u32, system-scope atomic)
/// - Bytes 4..8: `cancel_flag` (u32, system-scope atomic)
///
/// User allocations via [`alloc()`](Self::alloc) begin at byte 8 (or later,
/// after alignment).
///
/// # GridScope does NOT nest
///
/// Unlike `BlockScope` (which supports up to 4 levels of nesting),
/// `GridScope` has no nesting support. The bump allocator is a simple
/// offset with no watermark stack.
pub struct GridScope<'scope> {
    /// Global memory pool base pointer (caller-owned).
    pool_base: *mut u8,
    /// Current allocation offset within the pool (bytes from pool_base).
    /// Wrapped in `UnsafeCell` because `alloc()` takes `&self` but must
    /// bump this offset. Only the coordinator block/warp mutates this.
    pool_offset: UnsafeCell<u32>,
    /// Pool capacity in bytes.
    pool_capacity: u32,
    /// Atomic completion counter (in global memory, at pool_base + 0).
    /// Incremented by each block when it finishes its scope work.
    completion_counter: *mut u32,
    /// Total number of completions expected before the scope can exit.
    /// Wrapped in `UnsafeCell` because `set_expected_completions()` takes
    /// `&self`. Only the coordinator block/warp mutates this.
    expected_completions: UnsafeCell<u32>,
    /// Cancellation flag in global memory (at pool_base + 4).
    cancel_flag: *mut u32,
    /// Invariant lifetime marker — prevents covariance from allowing escape.
    _marker: PhantomData<&'scope mut &'scope ()>,
}

impl<'scope> GridScope<'scope> {
    /// Allocate a zero-initialized mutable slice of `count` elements from
    /// the global memory pool.
    ///
    /// Returns `&'scope mut [T]` — the slice is valid for the lifetime of
    /// this scope. When the scope exits, the pool offset is reset and this
    /// memory is logically reclaimed (the pool itself is not freed — it is
    /// caller-owned).
    ///
    /// # Panics
    ///
    /// Panics if the allocation would exceed the pool capacity.
    ///
    /// # Alignment
    ///
    /// Automatically aligns to `core::mem::align_of::<T>()`.
    pub fn alloc<T: Copy>(&self, count: usize) -> &'scope mut [T] {
        let size = core::mem::size_of::<T>() * count;
        let align = core::mem::align_of::<T>();

        // SAFETY: Only the coordinator block/warp calls alloc() — single writer.
        let offset = unsafe { *self.pool_offset.get() };

        // Bump allocator: round up offset to alignment, then advance
        let aligned = (offset as usize + align - 1) & !(align - 1);
        let new_offset = aligned + size;
        assert!(
            new_offset <= self.pool_capacity as usize,
            "GridScope::alloc: global memory pool exhausted"
        );

        unsafe { *self.pool_offset.get() = new_offset as u32 };

        unsafe {
            let ptr = self.pool_base.add(aligned) as *mut T;
            // Zero-initialize the region
            core::ptr::write_bytes(ptr, 0, count);
            core::slice::from_raw_parts_mut(ptr, count)
        }
    }

    /// Allocate a mutable slice WITHOUT zero-initialization.
    ///
    /// # Safety
    ///
    /// The caller must initialize all elements before reading them.
    pub unsafe fn alloc_uninit<T: Copy>(&self, count: usize) -> &'scope mut [T] {
        let size = core::mem::size_of::<T>() * count;
        let align = core::mem::align_of::<T>();

        let offset = *self.pool_offset.get();
        let aligned = (offset as usize + align - 1) & !(align - 1);
        let new_offset = aligned + size;
        assert!(
            new_offset <= self.pool_capacity as usize,
            "GridScope::alloc_uninit: global memory pool exhausted"
        );

        *self.pool_offset.get() = new_offset as u32;

        let ptr = self.pool_base.add(aligned) as *mut T;
        core::slice::from_raw_parts_mut(ptr, count)
    }

    /// Allocate a single value in global memory, initialized to `val`.
    pub fn alloc_val<T: Copy>(&self, val: T) -> &'scope mut T {
        let slot = self.alloc::<T>(1);
        slot[0] = val;
        &mut slot[0]
    }

    /// Returns the number of bytes remaining in the global memory pool.
    ///
    /// This is approximate — does not account for alignment padding of
    /// the next allocation.
    pub fn available_bytes(&self) -> usize {
        let offset = unsafe { *self.pool_offset.get() };
        (self.pool_capacity as usize).saturating_sub(offset as usize)
    }

    /// Request cooperative cancellation of all blocks in this scope.
    ///
    /// Sets a flag in global memory using a system-scope release store.
    /// Blocks should check [`is_cancelled()`](Self::is_cancelled) at
    /// checkpoints and return early. This is cooperative — blocks are
    /// not forcibly stopped.
    pub fn cancel(&self) {
        unsafe {
            gpu_atomics::sys_store_release_u32(self.cancel_flag, 1);
        }
    }

    /// Check if this scope has been cancelled.
    ///
    /// Uses a system-scope acquire load for cross-block visibility.
    pub fn is_cancelled(&self) -> bool {
        unsafe { gpu_atomics::sys_load_acquire_u32(self.cancel_flag as *const u32) != 0 }
    }

    /// Spin-wait until the completion counter reaches `expected`.
    ///
    /// Blocks (from block 0, warp 0) call this to wait for all dispatched
    /// work to finish. Each worker block increments the completion counter
    /// (via [`completion_counter_ptr()`](Self::completion_counter_ptr))
    /// when it finishes its portion of work.
    ///
    /// Uses system-scope spin-load (with nanosleep) to avoid LLVM
    /// hoisting the load out of the loop.
    pub fn wait_for_completions(&self, expected: u32) {
        loop {
            let done = unsafe {
                gpu_atomics::sys_spin_load_acquire_u32(self.completion_counter as *const u32)
            };
            if done >= expected {
                break;
            }
        }
    }

    /// Returns the raw pointer to the completion counter in global memory.
    ///
    /// Worker blocks should atomically increment this counter (using
    /// `gpu_atomics::sys_fetch_add_u32(ptr, 1)`) when they finish their
    /// portion of scope work. The coordinator block uses
    /// [`wait_for_completions()`](Self::wait_for_completions) to poll
    /// this counter.
    pub fn completion_counter_ptr(&self) -> *mut u32 {
        self.completion_counter
    }

    /// Returns the raw pointer to the cancellation flag in global memory.
    ///
    /// Worker blocks can poll this flag (using
    /// `gpu_atomics::sys_load_acquire_u32(ptr)`) to check for cooperative
    /// cancellation. Non-zero means cancelled.
    pub fn cancel_flag_ptr(&self) -> *const u32 {
        self.cancel_flag as *const u32
    }

    /// Returns the number of completions expected.
    pub fn expected_completions(&self) -> u32 {
        unsafe { *self.expected_completions.get() }
    }

    /// Allocate and initialize `count` work slots from the global memory pool.
    ///
    /// Returns a mutable slice of [`crate::grid_work::BlockWorkSlot`] ready
    /// for use with [`dispatch_work_to_slot`](Self::dispatch_work_to_slot)
    /// and [`crate::grid_work::grid_worker_loop`].
    ///
    /// All slots are zero-initialized with status set to
    /// [`crate::grid_work::SLOT_IDLE`] via system-scope release stores.
    ///
    /// # Panics
    ///
    /// Panics if the allocation would exceed the pool capacity.
    pub fn alloc_work_slots(&self, count: usize) -> &'scope mut [crate::grid_work::BlockWorkSlot] {
        let slots = self.alloc::<crate::grid_work::BlockWorkSlot>(count);
        // alloc() already zeroes memory. Re-initialize with proper atomic
        // release stores so worker blocks on other SMs see consistent state.
        unsafe {
            crate::grid_work::init_work_slots(slots);
        }
        slots
    }

    /// Dispatch work to a specific slot allocated by [`alloc_work_slots`](Self::alloc_work_slots).
    ///
    /// Writes the function pointer and arguments to the slot, then
    /// transitions it to `WORK_AVAILABLE` via system-scope release store.
    ///
    /// The work function signature must be `fn(args: &[u64; 4]) -> u64`.
    ///
    /// # Safety
    ///
    /// - `slot` must be a slot returned by `alloc_work_slots` on this scope.
    /// - `work_fn` must be a valid function pointer with the expected signature.
    /// - The slot must be in IDLE or COMPLETED state.
    pub unsafe fn dispatch_work_to_slot(
        &self,
        slot: &mut crate::grid_work::BlockWorkSlot,
        work_fn: u64,
        args: [u64; 4],
    ) {
        crate::grid_work::dispatch_work(slot, work_fn, args);
    }

    /// Set the number of completions to wait for at scope exit.
    ///
    /// This is set by the user after dispatching work to blocks.
    /// The scope's `Drop` (and `grid_scope()` exit) will spin-wait
    /// until the completion counter reaches this value.
    pub fn set_expected_completions(&self, n: u32) {
        // SAFETY: Only the coordinator block/warp mutates this.
        unsafe { *self.expected_completions.get() = n };
    }
}

impl<'scope> Drop for GridScope<'scope> {
    fn drop(&mut self) {
        // Wait for all expected completions before exiting.
        // This ensures all blocks have finished their work.
        let expected = *self.expected_completions.get_mut();
        if expected > 0 {
            self.wait_for_completions(expected);
        }

        // Reset pool offset to 0 — logically frees all allocations.
        // The pool memory itself is caller-owned and not freed here.
        *self.pool_offset.get_mut() = 0;
    }
}

// ============================================================
// grid_scope() — entry function
// ============================================================

/// Enter a grid-level structured concurrency scope.
///
/// The closure receives a `&GridScope<'scope>` handle for allocating
/// global memory and coordinating block-level work. All dispatched
/// blocks should be joined (via the completion counter) before this
/// function returns.
///
/// `pool` is a pre-allocated global memory region for scope allocations.
/// `pool_size` is its size in bytes. The first 8 bytes are reserved for
/// internal bookkeeping (completion counter + cancel flag).
///
/// # Lifetime guarantee
///
/// The `'scope` lifetime is shorter than any reference the closure borrows
/// from the enclosing function. This means `scope.alloc()` returns
/// `&'scope mut [T]` that cannot escape the `grid_scope` call.
///
/// # Safety
///
/// - `pool` must point to valid global device memory of at least `pool_size` bytes.
/// - `pool_size` must be at least 8 (for the header). Larger pools allow user
///   allocations.
/// - Must be called from the coordinator block (typically block 0, warp 0).
/// - The pool must be exclusively owned by this scope for its duration (no
///   concurrent access from other code paths).
///
/// # Example
///
/// ```rust,ignore
/// use gpu_runtime::scope::grid_scope;
///
/// unsafe {
///     grid_scope(pool, pool_size, |gscope| {
///         let buf = gscope.alloc::<f32>(256);
///         // ... fill buf, dispatch to blocks ...
///         gscope.set_expected_completions(num_blocks);
///     });
///     // Scope exited: all blocks completed, pool is logically freed.
/// }
/// ```
pub unsafe fn grid_scope<F, R>(pool: *mut u8, pool_size: u32, f: F) -> R
where
    F: for<'scope> FnOnce(&GridScope<'scope>) -> R,
{
    assert!(
        pool_size >= GRID_SCOPE_HEADER_SIZE,
        "grid_scope: pool_size must be at least {} bytes for header",
        GRID_SCOPE_HEADER_SIZE
    );

    // 1. Initialize completion counter and cancel flag in the pool header.
    //    Layout: [completion_counter: u32 @ offset 0, cancel_flag: u32 @ offset 4]
    let completion_counter = pool as *mut u32;
    let cancel_flag = pool.add(4) as *mut u32;

    // Use system-scope release stores for cross-block visibility.
    gpu_atomics::sys_store_release_u32(completion_counter, 0);
    gpu_atomics::sys_store_release_u32(cancel_flag, 0);

    // 2. Construct GridScope with pool_base, offset past the header.
    let scope = GridScope {
        pool_base: pool,
        pool_offset: UnsafeCell::new(GRID_SCOPE_HEADER_SIZE),
        pool_capacity: pool_size,
        completion_counter,
        expected_completions: UnsafeCell::new(0),
        cancel_flag,
        _marker: PhantomData,
    };

    // 3. Call the closure
    let result = f(&scope);

    // 4. Wait for completions and reset pool (via Drop).
    //    Drop will spin-wait on the completion counter if expected_completions > 0,
    //    then reset pool_offset to 0.
    drop(scope);

    result
}
