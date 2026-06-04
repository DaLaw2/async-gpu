# sc-design.2 — BlockScope/GridScope API Design

## Status: done
## Summary: Concrete Rust API design for `BlockScope<'scope>` and `GridScope<'scope>`, modeled after Rayon's `scope()`. BlockScope owns a shared-memory watermark allocator and spawns warps with `'scope`-bounded closures; GridScope coordinates across blocks via global memory with atomic completion counters. Both integrate cleanly with the existing `thread::spawn`, `cooperative_map`, and `block::sync` infrastructure. The `'scope` lifetime prevents shared memory references from escaping their hardware scope, enforced entirely by the Rust borrow checker at compile time.

## 1. Type Definitions

### 1.1 Shared Memory Watermark Allocator

```rust
/// Watermark (bump) allocator over the block's shared memory.
///
/// Maintains a stack of watermarks: each nested `block_scope` pushes
/// a new mark; scope exit pops back, logically freeing all allocations
/// made within that scope.
///
/// Lives in global memory (static), tracks offsets into shared memory.
pub struct SharedMemAllocator {
    /// Current allocation offset (bytes from shared_mem_ptr base).
    watermark: u32,
    /// Stack of saved watermarks for nested scopes (max depth 4).
    stack: [u32; 4],
    /// Current nesting depth.
    depth: u32,
    /// Total shared memory available (set at kernel launch).
    capacity: u32,
}

impl SharedMemAllocator {
    /// Initialize with the total shared memory size declared at launch.
    pub fn init(capacity: u32) -> Self;

    /// Push the current watermark and return the saved mark.
    /// Called at scope entry.
    fn push(&mut self) -> u32;

    /// Pop back to a saved watermark.
    /// Called at scope exit — logically frees all allocations since push.
    fn pop(&mut self);

    /// Bump-allocate `count * size_of::<T>()` bytes, aligned to `align_of::<T>()`.
    /// Returns the byte offset from shared memory base, or None if OOM.
    fn alloc_raw(&mut self, size: usize, align: usize) -> Option<u32>;
}
```

### 1.2 BlockScope

```rust
/// A structured concurrency scope bound to a single GPU block.
///
/// Allocations come from shared memory via watermark allocator.
/// Spawned closures are `'scope`-bounded: they can borrow data that
/// lives at least as long as the scope, but cannot outlive it.
///
/// Created by [`block_scope()`]. All spawned tasks are joined before
/// the scope closure returns.
pub struct BlockScope<'scope> {
    /// Saved watermark for this scope's shared memory region.
    saved_watermark: u32,
    /// Warp IDs spawned within this scope (bitmask, max 32 warps).
    spawned_warps: u32,
    /// Number of warps spawned (for join counting).
    spawn_count: u32,
    /// Cancellation flag in shared memory (offset from base).
    cancel_flag_offset: u32,
    /// Invariant lifetime — prevents covariance from allowing escape.
    _marker: PhantomData<&'scope mut &'scope ()>,
}
```

### 1.3 ScopeJoinHandle

```rust
/// Handle to a task spawned within a `BlockScope`.
///
/// The `'scope` lifetime ties this handle to the enclosing scope.
/// The result can be retrieved via `join()`, or the scope will
/// implicitly join all handles at exit.
pub struct ScopeJoinHandle<'scope, T> {
    warp_id: usize,
    _marker: PhantomData<&'scope T>,
}

impl<'scope, T> ScopeJoinHandle<'scope, T> {
    /// Block (spin-wait) until the spawned warp completes and return its result.
    pub fn join(self) -> T;
}
```

### 1.4 GridScope

```rust
/// A structured concurrency scope that coordinates across GPU blocks.
///
/// Allocations come from a pre-allocated global memory pool.
/// Block completion is tracked via atomic counters in global memory.
///
/// Created by [`grid_scope()`]. All spawned block-level work is joined
/// before the scope closure returns.
pub struct GridScope<'scope> {
    /// Global memory pool base pointer.
    pool_base: *mut u8,
    /// Current allocation offset within the pool.
    pool_offset: u32,
    /// Pool capacity in bytes.
    pool_capacity: u32,
    /// Atomic completion counter (in global memory).
    /// Incremented by each block when it finishes its scope work.
    completion_counter: *mut u32,
    /// Total number of blocks spawned in this scope.
    expected_completions: u32,
    /// Cancellation flag in global memory.
    cancel_flag: *mut u32,
    /// Invariant lifetime marker.
    _marker: PhantomData<&'scope mut &'scope ()>,
}
```

### 1.5 GridScopeJoinHandle

```rust
/// Handle to a block-scope task spawned within a `GridScope`.
///
/// Since blocks are independently scheduled, join polls an atomic
/// counter in global memory rather than spinning on a warp status flag.
pub struct GridScopeJoinHandle<'scope, T> {
    /// Slot in global memory where the block writes its result.
    result_slot: *mut T,
    /// Per-block completion flag (in global memory).
    done_flag: *mut u32,
    _marker: PhantomData<&'scope T>,
}

impl<'scope, T: Copy> GridScopeJoinHandle<'scope, T> {
    /// Spin-wait until the block completes and return its result.
    pub fn join(self) -> T;

    /// Check if the block has completed without blocking.
    pub fn is_done(&self) -> bool;
}
```

## 2. Scope Creation API

### 2.1 BlockScope entry — closure-based (Rayon model)

```rust
/// Enter a block-level structured concurrency scope.
///
/// The closure receives a `&BlockScope<'scope>` handle for allocating
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
/// # Example
///
/// ```rust,ignore
/// use gpu_runtime::scope::block_scope;
///
/// let input: &[f32] = /* ... */;
/// let result: f32 = block_scope(|scope| {
///     let buf = scope.alloc::<f32>(256);
///     // ... use buf within this scope ...
///     42.0
/// });
/// ```
///
/// # Safety
///
/// - Must be called from warp 0 (the main thread) within `gpu_main`.
/// - Shared memory must be large enough for the scope's allocations
///   plus internal bookkeeping (cancel flag: 4 bytes).
pub fn block_scope<'env, F, R>(f: F) -> R
where
    F: for<'scope> FnOnce(&'scope BlockScope<'scope>) -> R,
    // No Send bound — all warps are in the same block.
{
    // 1. Push allocator watermark
    // 2. Allocate cancel_flag (4 bytes) from shared memory
    // 3. Construct BlockScope
    // 4. Call f(&scope)
    // 5. Join all spawned warps (spin-wait on WARP_STATUS)
    // 6. Pop allocator watermark (logically free all shared memory)
    // 7. Return result
    ...
}
```

The `for<'scope>` higher-ranked trait bound is the key mechanism (identical to Rayon):
- The caller cannot name `'scope` — it is universally quantified.
- Any `&'scope T` obtained from the scope cannot be stored in a variable with a longer lifetime.
- The compiler rejects code that tries to smuggle a shared memory reference out.

### 2.2 GridScope entry

```rust
/// Enter a grid-level structured concurrency scope.
///
/// The closure receives a `&GridScope<'scope>` handle for allocating
/// global memory and dispatching block-level work. All dispatched
/// blocks are joined before this function returns.
///
/// `pool` is a pre-allocated global memory region for scope allocations.
/// `pool_size` is its size in bytes.
///
/// # Safety
///
/// - `pool` must point to valid global device memory of at least `pool_size` bytes.
/// - Must be called from the "coordinator" block (typically block 0, warp 0).
/// - Other blocks must be in a state where they can receive dispatched work
///   (e.g., polling a work queue or command buffer).
pub unsafe fn grid_scope<'env, F, R>(
    pool: *mut u8,
    pool_size: u32,
    f: F,
) -> R
where
    F: for<'scope> FnOnce(&'scope GridScope<'scope>) -> R,
{
    // 1. Initialize completion counter and cancel flag in pool
    // 2. Construct GridScope with pool_base, offsets
    // 3. Call f(&scope)
    // 4. Spin-wait: poll completion_counter until == expected_completions
    // 5. Return result
    ...
}
```

## 3. Resource Allocation

### 3.1 BlockScope allocation (shared memory)

```rust
impl<'scope> BlockScope<'scope> {
    /// Allocate a mutable slice of `count` elements from shared memory.
    ///
    /// Returns `&'scope mut [T]` — the slice is valid for the lifetime
    /// of this scope. When the scope exits, the watermark is popped and
    /// this memory is logically reclaimed.
    ///
    /// # Panics
    ///
    /// Panics if the allocation would exceed shared memory capacity (48KB on SM75).
    ///
    /// # Alignment
    ///
    /// Automatically aligns to `core::mem::align_of::<T>()`.
    pub fn alloc<T: Copy>(&self, count: usize) -> &'scope mut [T] {
        // 1. Call ALLOCATOR.alloc_raw(count * size_of::<T>(), align_of::<T>())
        // 2. Convert offset to pointer: shared_mem_ptr().add(offset) as *mut T
        // 3. Zero-initialize the region
        // 4. Transmute to &'scope mut [T] via core::slice::from_raw_parts_mut
        ...
    }

    /// Allocate a mutable slice WITHOUT zero-initialization.
    ///
    /// # Safety
    ///
    /// The caller must initialize all elements before reading them.
    pub unsafe fn alloc_uninit<T: Copy>(&self, count: usize) -> &'scope mut [T];

    /// Allocate a single value in shared memory, initialized to `val`.
    pub fn alloc_val<T: Copy>(&self, val: T) -> &'scope mut T {
        let slot = self.alloc::<T>(1);
        slot[0] = val;
        &mut slot[0]
    }

    /// Returns the number of bytes remaining in the shared memory pool.
    pub fn available_bytes(&self) -> usize;
}
```

### 3.2 GridScope allocation (global memory)

```rust
impl<'scope> GridScope<'scope> {
    /// Allocate a mutable slice of `count` elements from the global memory pool.
    ///
    /// Returns `&'scope mut [T]` — the slice is valid for the lifetime
    /// of this scope. When the scope exits, the pool is logically reclaimed
    /// (the whole pool is freed at once, no per-element deallocation).
    ///
    /// # Panics
    ///
    /// Panics if the allocation would exceed the pool capacity.
    pub fn alloc<T: Copy>(&self, count: usize) -> &'scope mut [T];

    /// Allocate without zero-initialization.
    ///
    /// # Safety
    ///
    /// The caller must initialize all elements before reading them.
    pub unsafe fn alloc_uninit<T: Copy>(&self, count: usize) -> &'scope mut [T];

    /// Allocate a single value in global memory.
    pub fn alloc_val<T: Copy>(&self, val: T) -> &'scope mut T;

    /// Returns the number of bytes remaining in the global memory pool.
    pub fn available_bytes(&self) -> usize;
}
```

## 4. Task Spawning

### 4.1 BlockScope::spawn — warp-level task spawning

```rust
impl<'scope> BlockScope<'scope> {
    /// Spawn a closure on an idle warp within this block.
    ///
    /// The closure is bounded by `'scope` — it can borrow anything that
    /// lives at least as long as the scope (including scope-allocated
    /// shared memory). The closure runs on a single warp (lane 0 executes
    /// the closure body; all 32 lanes participate in any warp-level ops).
    ///
    /// Returns a `ScopeJoinHandle` that can be joined explicitly, or
    /// will be joined implicitly when the scope exits.
    ///
    /// # Panics
    ///
    /// Panics if no idle warps are available.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// block_scope(|scope| {
    ///     let shared_buf = scope.alloc::<f32>(128);
    ///     let h = scope.spawn(|| {
    ///         // Can read/write shared_buf — it lives for 'scope
    ///         shared_buf[0] = 1.0;
    ///         42u32
    ///     });
    ///     let result = h.join(); // 42
    /// });
    /// ```
    pub fn spawn<F, T>(&self, f: F) -> ScopeJoinHandle<'scope, T>
    where
        F: FnOnce() -> T + Send + 'scope,
        T: Send + 'scope,
    {
        // 1. Find idle warp (same mechanism as thread::spawn)
        // 2. Write closure to warp's SCRATCH buffer
        // 3. Set WARP_FN, WARP_DATA, WARP_STATUS = ASSIGNED
        // 4. Record warp ID in self.spawned_warps bitmask
        // 5. Return ScopeJoinHandle { warp_id, _marker }
        ...
    }

    /// Spawn multiple closures, one per available warp, data-parallel style.
    ///
    /// Equivalent to calling `scope.spawn()` once per available warp, but
    /// more efficient: all warps are woken simultaneously rather than one
    /// at a time. The closure receives the warp's ID and total warp count.
    ///
    /// All warps are joined before this function returns (synchronous).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// block_scope(|scope| {
    ///     let data = scope.alloc::<f32>(1024);
    ///     // ... fill data ...
    ///
    ///     scope.spawn_all(|warp_id, n_warps| {
    ///         let mut i = warp_id as usize;
    ///         while i < 1024 {
    ///             data[i] *= 2.0;
    ///             i += n_warps as usize;
    ///         }
    ///     });
    ///     // All warps have completed; data is doubled.
    /// });
    /// ```
    pub fn spawn_all<F>(&self, f: F)
    where
        F: Fn(u32, u32) + Send + Sync + 'scope,
    {
        // 1. Wake all idle warps with STATUS_COOPERATIVE
        // 2. Warp 0 also participates
        // 3. Join all warps before returning
        // Implemented via existing cooperative() machinery
        ...
    }
}
```

Key difference from `thread::spawn`: the `'scope` bound replaces `'static`. This is what allows borrowing scope-allocated shared memory in spawned closures.

### 4.2 GridScope::spawn_block — block-level task dispatching

```rust
impl<'scope> GridScope<'scope> {
    /// Dispatch a unit of work to a block.
    ///
    /// The closure describes work for an entire block. It runs within
    /// a `block_scope` on the target block, so it has access to that
    /// block's shared memory.
    ///
    /// The `args` parameter passes data to the block via global memory.
    /// It must be `Copy` because it is written to a global memory slot
    /// that the target block reads.
    ///
    /// Returns a `GridScopeJoinHandle` for waiting on the result.
    ///
    /// # How it works (SM75 without cooperative launch)
    ///
    /// The target block must already be running a "worker loop" that
    /// polls a command slot in global memory. `spawn_block` writes
    /// the work descriptor to that slot and publishes it atomically.
    /// The target block picks it up, executes within its own
    /// `block_scope`, writes the result, and signals completion.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// grid_scope(pool, pool_size, |scope| {
    ///     let global_buf = scope.alloc::<f32>(4096);
    ///
    ///     let h1 = scope.spawn_block(BlockWorkArgs {
    ///         data: global_buf.as_mut_ptr(),
    ///         offset: 0,
    ///         len: 2048,
    ///     }, |args| {
    ///         // This runs on the target block inside its own block_scope.
    ///         // args.data points to global memory — safe across blocks.
    ///         for i in args.offset..args.offset + args.len {
    ///             unsafe { *args.data.add(i) = i as f32; }
    ///         }
    ///         2048u32 // return count processed
    ///     });
    ///
    ///     let count = h1.join();
    /// });
    /// ```
    pub unsafe fn spawn_block<A, F, T>(
        &self,
        args: A,
        f: F,
    ) -> GridScopeJoinHandle<'scope, T>
    where
        A: Copy + Send + 'scope,
        F: FnOnce(A) -> T + Send + 'scope,
        T: Copy + Send + 'scope,
    {
        // 1. Allocate result slot + done flag from the global pool
        // 2. Serialize work descriptor (fn pointer + args) to a command slot
        // 3. Publish command via atomic store (release)
        // 4. Increment expected_completions
        // 5. Return GridScopeJoinHandle { result_slot, done_flag }
        ...
    }
}
```

## 5. Scope Join and Cleanup

### 5.1 BlockScope exit sequence

When the `block_scope` closure returns, the following happens automatically:

```
1. Join all spawned warps:
   for each warp_id in self.spawned_warps bitmask {
       loop {
           if WARP_STATUS[warp_id].load(Acquire) == STATUS_DONE {
               WARP_STATUS[warp_id].store(STATUS_IDLE, Release);
               break;
           }
           nanosleep_short();
       }
   }

2. Pop the shared memory watermark:
   ALLOCATOR.pop()
   // This logically frees ALL shared memory allocated during the scope.
   // No per-element destructor — T: Copy, so no Drop.

3. Return the closure's return value to the caller.
```

No `bar.sync` at scope exit — the polling join is sufficient and avoids the deadlock risk of `bar.sync` when not all warps participate. The user can call `block::sync()` explicitly inside the scope if they need a full-block barrier.

### 5.2 GridScope exit sequence

```
1. Spin-wait on completion counter:
   loop {
       let done = sys_load_acquire_u32(self.completion_counter);
       if done >= self.expected_completions {
           break;
       }
       nanosleep_short();
   }

2. The global memory pool is NOT freed here — the caller owns the pool
   and is responsible for its lifetime. The scope merely resets pool_offset
   so the pool can be reused.

3. Return the closure's return value.
```

### 5.3 Cancellation

```rust
impl<'scope> BlockScope<'scope> {
    /// Request cancellation of all tasks in this scope.
    ///
    /// Sets a flag in shared memory. Spawned tasks should check
    /// `scope.is_cancelled()` at appropriate points and return early.
    /// This is cooperative — tasks are not forcibly stopped.
    pub fn cancel(&self) {
        unsafe {
            let flag = block::shared_mem_at::<u32>(self.cancel_flag_offset as usize);
            core::ptr::write_volatile(flag, 1);
        }
    }

    /// Check if this scope has been cancelled.
    pub fn is_cancelled(&self) -> bool {
        unsafe {
            let flag = block::shared_mem_at::<u32>(self.cancel_flag_offset as usize);
            core::ptr::read_volatile(flag) != 0
        }
    }
}

impl<'scope> GridScope<'scope> {
    /// Request cancellation of all blocks in this scope.
    ///
    /// Sets a flag in global memory. Blocks should check
    /// `scope.is_cancelled()` at checkpoints.
    pub fn cancel(&self) {
        unsafe { sys_store_release_u32(self.cancel_flag, 1); }
    }

    /// Check if this scope has been cancelled.
    pub fn is_cancelled(&self) -> bool {
        unsafe { sys_load_acquire_u32(self.cancel_flag as *const u32) != 0 }
    }
}
```

## 6. Integration with Existing APIs

### 6.1 Composing with thread::spawn

`thread::spawn` requires `F: 'static`. Code inside a `block_scope` can still call `thread::spawn` if the closure captures only `'static` data. However, the idiomatic path within a scope is `scope.spawn()`, which allows borrowing `'scope` data.

```rust
block_scope(|scope| {
    let shared = scope.alloc::<f32>(64);

    // OK: scope.spawn borrows shared (which is 'scope)
    let h1 = scope.spawn(|| {
        shared[0] = 1.0;
    });

    // COMPILE ERROR: thread::spawn requires 'static, shared is 'scope
    // let h2 = thread::spawn(|| { shared[1] = 2.0; });

    // OK: thread::spawn with only 'static data
    let h3 = thread::spawn(|| 42u32);

    h1.join();
    h3.join();
});
```

### 6.2 Composing with cooperative_map

`scope.spawn_all()` subsumes `cooperative_map` for scope-aware code. However, `cooperative_map` remains available for cases where you want the function-pointer API without a scope:

```rust
block_scope(|scope| {
    let input = scope.alloc::<f32>(1024);
    let output = scope.alloc::<f32>(1024);
    // ... fill input ...

    // Option A: scope.spawn_all (can borrow scope data directly)
    scope.spawn_all(|warp_id, n_warps| {
        let mut i = warp_id as usize;
        while i < 1024 {
            output[i] = input[i] * 2.0;
            i += n_warps as usize;
        }
    });

    // Option B: cooperative_map (function pointer, no closures)
    // Still works inside a scope — doesn't conflict.
    thread::cooperative_map(
        input.as_ptr() as *const u8,
        output.as_mut_ptr() as *mut u8,
        1024,
        |args| { /* ... */ },
    );
});
```

### 6.3 Composing with block::sync()

`block::sync()` can be called inside a scope, but the user must ensure all active warps participate:

```rust
block_scope(|scope| {
    // No warps spawned yet — only warp 0 is active.
    // DO NOT call block::sync() here — other warps are in the worker loop,
    // not at a matching sync point.

    // spawn_all wakes all warps to execute the closure synchronously:
    scope.spawn_all(|warp_id, n_warps| {
        // All warps are executing here — block::sync() is safe.
        let smem = scope.alloc::<f32>(32); // allocated once by warp 0 before spawn_all
        unsafe { block::sync(); }
        // ... use shared memory with barrier guarantees ...
    });
});
```

**Design note:** `scope.alloc()` must be called from warp 0 (the scope owner) BEFORE `spawn_all`. Inside `spawn_all`, all warps can read/write the allocated region. We do NOT allow `scope.alloc()` from within a spawned closure because the watermark allocator is not thread-safe (it's a simple bump pointer managed by warp 0).

### 6.4 Composing with channels

Existing channels (oneshot, MPSC) work unchanged — they use global memory. Within a BlockScope, users can create shared-memory channels (future work: BlockChannel). For now, the existing global-memory channels compose naturally:

```rust
block_scope(|scope| {
    let mut slot = OneshotSlot::<u32>::new();
    let (tx, rx) = unsafe { oneshot(&mut slot) };

    scope.spawn(move || {
        unsafe { tx.send(42); }
    });

    // Spin-poll the receiver (or use executor if available)
    loop {
        let poll = unsafe {
            Pin::new_unchecked(&mut rx).poll(&mut cx)
        };
        if let Poll::Ready(Ok(val)) = poll {
            break; // val == 42
        }
    }
});
```

## 7. Lifetime Mechanics

### 7.1 How `'scope` prevents escape

The core mechanism is the higher-ranked trait bound on `block_scope`:

```rust
pub fn block_scope<'env, F, R>(f: F) -> R
where
    F: for<'scope> FnOnce(&'scope BlockScope<'scope>) -> R,
```

The `for<'scope>` means the closure must work for ANY choice of `'scope`. Since the caller cannot name a specific `'scope`, they cannot assign scope-derived references to variables that outlive the call:

```rust
let mut escaped: &mut [f32] = &mut [];

block_scope(|scope| {
    let buf = scope.alloc::<f32>(64);
    // COMPILE ERROR: cannot assign 'scope reference to 'escaped
    // because 'scope might be shorter than the lifetime of 'escaped.
    escaped = buf;
});
```

### 7.2 PhantomData invariance

The `BlockScope` uses `PhantomData<&'scope mut &'scope ()>` to make the `'scope` parameter **invariant**. This prevents the compiler from shortening or lengthening `'scope`:

- Without invariance, covariance could allow `&'scope [T]` to be weakened to `&'short [T]` where `'short` is a named lifetime, potentially allowing escape.
- Invariance forces `'scope` to be exactly the compiler-chosen lifetime, not a subtype of it.

### 7.3 The Send bound on spawned closures

`scope.spawn` requires `F: Send + 'scope`. The `Send` bound is needed because the closure is executed on a different warp (different "thread" in GPU terms). The `'scope` bound (instead of `'static`) is what makes the whole thing useful — it allows borrowing scope-allocated data.

```rust
pub fn spawn<F, T>(&self, f: F) -> ScopeJoinHandle<'scope, T>
where
    F: FnOnce() -> T + Send + 'scope,
    T: Send + 'scope,
```

### 7.4 Interior mutability pattern for alloc

Since `block_scope` passes `&BlockScope` (shared reference), and `alloc` needs to bump the watermark, the allocator uses interior mutability. The allocator state lives in global memory (a `static` with `UnsafeCell`), and only warp 0 mutates it — no synchronization needed because scope.alloc() is only callable from warp 0.

## 8. User-Facing Examples

### 8.1 BlockScope: parallel vector add with shared memory

```rust
use gpu_runtime::scope::block_scope;
use gpu_runtime::thread;

/// Compute output[i] = a[i] + b[i] using shared memory tiling.
/// a, b, output are in global memory. n is the element count.
fn vec_add_tiled(a: *const f32, b: *const f32, output: *mut f32, n: usize) {
    thread::gpu_main(|| {
        const TILE: usize = 256;

        for tile_start in (0..n).step_by(TILE) {
            let tile_len = core::cmp::min(TILE, n - tile_start);

            block_scope(|scope| {
                // Allocate two tiles in shared memory
                let sa = scope.alloc::<f32>(tile_len);
                let sb = scope.alloc::<f32>(tile_len);

                // Load from global → shared (all warps cooperate)
                scope.spawn_all(|wid, nw| {
                    let mut i = wid as usize;
                    while i < tile_len {
                        unsafe {
                            sa[i] = *a.add(tile_start + i);
                            sb[i] = *b.add(tile_start + i);
                        }
                        i += nw as usize;
                    }
                });

                // Compute and store (all warps cooperate)
                scope.spawn_all(|wid, nw| {
                    let mut i = wid as usize;
                    while i < tile_len {
                        unsafe {
                            *output.add(tile_start + i) = sa[i] + sb[i];
                        }
                        i += nw as usize;
                    }
                });
            });
            // Scope exited: shared memory for sa, sb is reclaimed.
            // Next tile reuses the same shared memory region.
        }
    });
}
```

### 8.2 GridScope: multi-block reduce

```rust
use gpu_runtime::scope::{block_scope, grid_scope};
use gpu_runtime::thread;

/// Sum all elements of `data[0..n]` across multiple blocks.
/// `pool` is a pre-allocated global memory region for GridScope bookkeeping.
/// `partial_sums` is a global memory array with one slot per block.
unsafe fn multi_block_reduce(
    data: *const f32,
    n: usize,
    pool: *mut u8,
    pool_size: u32,
    partial_sums: *mut f32,
    num_blocks: u32,
) -> f32 {
    // Phase 1: Each block computes a partial sum.
    // (In a real implementation, this would be dispatched to worker blocks
    //  via the command buffer. Here we show the logical structure.)
    grid_scope(pool, pool_size, |gscope| {
        let elements_per_block = (n + num_blocks as usize - 1) / num_blocks as usize;

        for block_id in 0..num_blocks {
            let offset = block_id as usize * elements_per_block;
            let count = core::cmp::min(elements_per_block, n - offset);

            gscope.spawn_block(
                (data, offset, count, partial_sums, block_id),
                |(data, offset, count, partial_sums, block_id)| {
                    // Each block uses its own block_scope for local computation
                    block_scope(|bscope| {
                        let local = bscope.alloc::<f32>(1);
                        local[0] = 0.0;

                        bscope.spawn_all(|wid, nw| {
                            let mut sum = 0.0f32;
                            let mut i = wid as usize;
                            while i < count {
                                sum += *data.add(offset + i);
                                i += nw as usize;
                            }
                            // Warp-level reduce, then atomic add to local[0]
                            let warp_sum = gpu_runtime::warp::reduce_sum_f32(sum);
                            if gpu_runtime::index::thread_idx_x() % 32 == 0 {
                                // Atomic add to shared memory accumulator
                                // (in practice, use a shared-memory reduce tree)
                                gpu_atomics::atomic_add_f32_shared(
                                    local.as_mut_ptr(), warp_sum
                                );
                            }
                        });

                        // Write block's partial sum to global memory
                        *partial_sums.add(block_id as usize) = local[0];
                    });
                },
            );
        }

        // grid_scope exit: all blocks have completed.
        // Phase 2: Sum the partial sums (on block 0).
        let mut total = 0.0f32;
        for i in 0..num_blocks as usize {
            total += *partial_sums.add(i);
        }
        total
    })
}
```

### 8.3 Nested scopes: BlockScope containing BlockScopes

```rust
use gpu_runtime::scope::block_scope;
use gpu_runtime::thread;

/// Nested scopes demonstrate watermark allocator stacking.
fn nested_scope_example() {
    thread::gpu_main(|| {
        block_scope(|outer| {
            // Outer scope: allocate 4KB
            let big_buf = outer.alloc::<f32>(1024); // 4096 bytes

            // Fill big_buf cooperatively
            outer.spawn_all(|wid, nw| {
                let mut i = wid as usize;
                while i < 1024 {
                    big_buf[i] = i as f32;
                    i += nw as usize;
                }
            });

            // Inner scope: allocate temporary scratch space
            let partial_sum = block_scope(|inner| {
                // This allocates ABOVE the outer scope's watermark.
                // inner has access to big_buf (it lives for 'outer which
                // is longer than 'inner).
                let scratch = inner.alloc::<f32>(32); // 128 bytes

                inner.spawn_all(|wid, nw| {
                    // Each warp sums its partition of big_buf
                    let mut sum = 0.0f32;
                    let mut i = wid as usize;
                    while i < 1024 {
                        sum += big_buf[i];
                        i += nw as usize;
                    }
                    scratch[wid as usize] = sum;
                });

                // Reduce scratch[0..n_warps] on warp 0
                let n_warps = thread::available_parallelism() + 1;
                let mut total = 0.0f32;
                for i in 0..n_warps {
                    total += scratch[i];
                }
                total
            });
            // Inner scope exited: scratch is reclaimed.
            // big_buf is still valid (outer scope is still alive).

            // Store result
            big_buf[0] = partial_sum;
        });
        // Outer scope exited: big_buf is reclaimed.
    });
}
```

## 9. Migration Path

### 9.1 thread::spawn → scope.spawn

Before (current API — requires `'static`):
```rust
thread::gpu_main(|| {
    let h1 = thread::spawn(|| compute_a());
    let h2 = thread::spawn(|| compute_b());
    let r1 = h1.join();
    let r2 = h2.join();
});
```

After (scope API — allows borrowing):
```rust
thread::gpu_main(|| {
    block_scope(|scope| {
        let shared_data = scope.alloc::<f32>(256);
        let h1 = scope.spawn(|| {
            shared_data[0] = compute_a(); // Can borrow shared_data!
        });
        let h2 = scope.spawn(|| {
            shared_data[1] = compute_b();
        });
        h1.join();
        h2.join();
    });
});
```

**Migration effort:** Wrap the spawn+join pattern in `block_scope`. Existing code that uses `thread::spawn` with `'static` closures continues to work unchanged.

### 9.2 cooperative / cooperative_map → scope.spawn_all

Before:
```rust
thread::gpu_main(|| {
    unsafe {
        thread::cooperative(&|| {
            let wid = thread::current_id() as usize;
            let n = thread::available_parallelism() + 1;
            // ...
        });
    }
});
```

After:
```rust
thread::gpu_main(|| {
    block_scope(|scope| {
        scope.spawn_all(|wid, n_warps| {
            // Same body, but now can borrow scope-allocated data.
        });
    });
});
```

**Migration effort:** Replace `cooperative(|| { ... })` with `scope.spawn_all(|wid, nw| { ... })`. The new API is safe (no `unsafe` block needed) and can access scope-allocated shared memory.

### 9.3 Raw shared_mem_ptr → scope.alloc

Before:
```rust
unsafe {
    let smem = block::shared_mem_ptr();
    let a = smem as *mut f32;                    // offset 0
    let b = smem.add(256) as *mut f32;           // offset 256
    // Manual offset tracking, no lifetime safety
}
```

After:
```rust
block_scope(|scope| {
    let a = scope.alloc::<f32>(64);   // 256 bytes, automatically offset
    let b = scope.alloc::<f32>(64);   // next 256 bytes
    // Lifetime-safe, no manual offset math
});
```

**Migration effort:** Replace manual offset arithmetic with `scope.alloc()` calls. The compiler enforces lifetime safety that was previously the programmer's responsibility.

### 9.4 Backward compatibility

All existing APIs (`thread::spawn`, `cooperative_map`, `block::sync`, etc.) remain unchanged and fully functional. The scope API is additive — it provides a safer, more ergonomic way to do the same things, plus enables patterns (borrowing shared memory across warps) that were previously impossible with safe code.

## 10. Open Questions

### 10.1 spawn_all vs block::sync interaction

**Question:** Should `spawn_all` internally use `bar.sync` instead of polling?

**Tradeoff:** `bar.sync` is ~4 cycles but requires ALL threads in the block to participate. The current thread pool has warps in a polling loop — they are not at a `bar.sync` point. To use `bar.sync`, we'd need to restructure the worker loop to park at barriers rather than polling status flags.

**Current decision:** Use polling (consistent with existing thread pool). Revisit after measuring performance — if the polling overhead is significant for `spawn_all` workloads, consider a separate `block_barrier_scope` that restructures the worker loop around `bar.sync`.

### 10.2 GridScope block dispatch mechanism

**Question:** How exactly do blocks receive work from `GridScope::spawn_block`?

**Options:**
1. **Command buffer (existing `cmd.rs`):** Block worker loops poll a shared command ring buffer. GridScope writes work descriptors to the ring. This is closest to the existing infrastructure.
2. **Global work queue (existing `executor.rs`):** Reuse the executor's `WorkQueue` at grid level. Each block's main loop dequeues tasks.
3. **Pre-assigned slots:** Each block has a dedicated slot in global memory. The coordinator writes directly to that block's slot.

**Current decision:** Start with option 3 (pre-assigned slots) — simplest to implement and avoids contention. Each block polls its own dedicated `BlockWorkSlot` in global memory. The coordinator writes to slot[block_id].

### 10.3 Error propagation across scopes

**Question:** What happens when a spawned warp panics (traps)?

**Current status:** GPU panics call `trap;` which terminates the warp. The scope's join loop would spin forever waiting for `STATUS_DONE` that never comes.

**Proposed solution:** Add a `STATUS_TRAPPED: u32 = 5` status. The panic handler sets this before trapping. The join loop checks for TRAPPED and propagates the error. But this requires changes to the panic handler to NOT immediately trap — instead set the status and THEN trap.

**Current decision:** Defer. For now, panics in spawned warps will deadlock the scope (same as current `thread::spawn` + `join`). This matches the existing behavior and is consistent with GPU programming expectations. Address when adding the error-bitmask infrastructure.

### 10.4 alloc thread-safety within spawn_all

**Question:** Can `scope.alloc()` be called from inside `spawn_all`?

**Answer:** No. The watermark allocator is not thread-safe — it's a simple bump pointer managed by warp 0. All allocations must happen before `spawn_all` (or `scope.spawn`). This is enforced by the API: `scope.alloc()` takes `&self` (shared ref) and internally uses `UnsafeCell`, but it asserts that `warp_id() == 0`.

If per-warp allocation is needed, allocate a large block and partition it manually:

```rust
block_scope(|scope| {
    let n_warps = thread::available_parallelism() + 1;
    let pool = scope.alloc::<f32>(n_warps * 64); // 64 f32s per warp

    scope.spawn_all(|wid, nw| {
        let my_slice = &mut pool[wid as usize * 64..(wid as usize + 1) * 64];
        // Each warp writes only to its own partition — no races.
    });
});
```

### 10.5 Scope nesting limits

**Question:** How deep can scopes nest?

**Answer:** The watermark stack is fixed at depth 4. This supports patterns like `grid_scope > block_scope > block_scope > block_scope`, which is more than any reasonable use case. Exceeding depth 4 panics at runtime. The shared memory constraint (48KB) is the practical limit anyway — each nesting level reduces available space.

## Files Changed: none (design only)
