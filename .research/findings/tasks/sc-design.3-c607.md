# sc-design.3 — Cancellation Propagation Semantics

## Status: done
## Summary: Defines cancellation propagation for nested BlockScope and GridScope hierarchies. The model is "chain-walk with short-circuit": each scope has its own cancel flag at the appropriate memory tier, and `is_cancelled()` checks the local flag OR the parent flag (if any). Parent→child propagation is automatic (child inherits a pointer to parent's flag); child→parent escalation is opt-in via `ScopePolicy::Supervised` vs `ScopePolicy::Nursery`. GridScope→BlockScope cancellation uses a global memory flag readable by all blocks. Cleanup is cooperative throughout — no preemption, no unwinding.

## 1. Propagation Model

### 1.1 Design choice: chain-walk with short-circuit (Option B)

Each scope maintains its own cancel flag at the appropriate memory tier (shared memory for BlockScope, global memory for GridScope). A scope also holds an optional pointer to its parent's cancel flag. `is_cancelled()` checks the local flag first (cheap read), and if not set, walks up to the parent flag.

**Why not Option A (independent flags, explicit propagation)?**
- Requires the parent to enumerate and set every child flag. In a dynamic spawn pattern, the parent may not know how many children exist. Adds O(N) writes at cancellation time.

**Why not Option C (single flag per tree)?**
- A single flag conflates parent and child cancellation. If child scope C1 wants to cancel its own subtree without affecting sibling C2, a shared flag cannot express this. Also, a BlockScope flag lives in shared memory while a GridScope flag lives in global memory — a single flag can't span both tiers.

**Why Option B (chain-walk)?**
- Each scope can cancel independently (set its own flag). Parent cancellation automatically propagates because children check the parent's flag. O(1) to cancel (one write), O(depth) to check (but depth ≤ 4, so constant in practice). Memory tier boundaries are handled naturally: a BlockScope child points to its parent's shared-memory flag, or to a GridScope's global-memory flag if the parent is grid-level.

### 1.2 Flag layout

```
GridScope (global memory)
├── cancel_flag: *mut u32  (global memory, system-scope atomics)
│
├── Block 0 → BlockScope (shared memory)
│   ├── cancel_flag_offset: u32  (shared memory, volatile read)
│   ├── parent_cancel: *const u32  → GridScope.cancel_flag (global memory)
│   │
│   └── nested BlockScope (shared memory)
│       ├── cancel_flag_offset: u32  (shared memory, volatile read)
│       └── parent_cancel: *const u32  → parent BlockScope.cancel_flag (shared memory)
│
├── Block 1 → BlockScope (shared memory)
│   ├── cancel_flag_offset: u32
│   └── parent_cancel: *const u32  → GridScope.cancel_flag
...
```

Each `is_cancelled()` call reads at most 2 pointers (local + parent). For a deeply nested BlockScope (depth 3), the walk goes: inner.flag → middle.flag → outer.flag → grid.flag. This is at most 4 reads, all within shared memory except the grid-level flag (~100 cycles for the global read). In practice, once any flag in the chain is set to 1, the walk short-circuits immediately.

## 2. Parent → Child Propagation

### 2.1 Mechanism

Child scopes inherit a `parent_cancel` pointer at construction time. The pointer is set by the `block_scope` / `grid_scope` entry function based on the enclosing scope.

```rust
pub struct BlockScope<'scope> {
    /// Bitmask of warp IDs spawned within this scope.
    spawned_warps: u32,
    /// Number of warps spawned.
    spawn_count: u32,
    /// Byte offset of this scope's cancellation flag in shared memory.
    cancel_flag_offset: u32,
    /// Optional pointer to the parent scope's cancel flag.
    /// - For nested BlockScope: points to shared memory (parent's cancel_flag).
    /// - For BlockScope inside GridScope: points to global memory (GridScope's cancel_flag).
    /// - For top-level BlockScope: null.
    parent_cancel: *const u32,
    /// Invariant lifetime marker.
    _marker: PhantomData<&'scope mut &'scope ()>,
}
```

### 2.2 Updated `is_cancelled()` — chain-walk

```rust
impl<'scope> BlockScope<'scope> {
    /// Check if this scope or any ancestor scope has been cancelled.
    ///
    /// Reads the local cancel flag first (shared memory, ~2 cycles).
    /// If not set, checks the parent cancel flag (shared or global memory).
    /// Short-circuits on the first set flag.
    ///
    /// Cost: 1 volatile read (best case) to 2 reads (worst case).
    /// The parent pointer is never more than 1 level up because each
    /// scope only stores its immediate parent — but since parent
    /// `is_cancelled()` also walks up, the chain check happens at
    /// each scope's own check time, not recursively here.
    ///
    /// Design note: we do NOT walk the full chain recursively. Each scope
    /// checks local + parent. The parent scope's own `is_cancelled()` check
    /// (at its yield points) handles grandparent propagation. This gives
    /// O(1) per check instead of O(depth).
    pub fn is_cancelled(&self) -> bool {
        // Check local flag (shared memory, ~2 cycles)
        let local = unsafe {
            let flag = crate::block::shared_mem_at::<u32>(self.cancel_flag_offset as usize);
            core::ptr::read_volatile(flag) != 0
        };
        if local {
            return true;
        }

        // Check parent flag if present
        if !self.parent_cancel.is_null() {
            // Parent may be in shared or global memory — volatile read works for both.
            // For global memory (GridScope parent), this costs ~100 cycles.
            // For shared memory (BlockScope parent), ~2 cycles.
            unsafe { core::ptr::read_volatile(self.parent_cancel) != 0 }
        } else {
            false
        }
    }
}
```

**Why only check one level up, not the full chain?**

Consider nested scopes: outer → middle → inner. When outer cancels:
1. `outer.cancel_flag` is set to 1.
2. Middle's spawned warps check `middle.is_cancelled()` → reads `middle.flag` (0) then `outer.flag` (1) → returns true. Middle can then call `middle.cancel()` to set its own flag.
3. Inner's spawned warps check `inner.is_cancelled()` → reads `inner.flag` (0) then `middle.flag`.

If middle has not yet set its own flag, inner won't see the cancellation until middle does. This is by design: cancellation propagation is gated by each scope's yield points. Middle must acknowledge the cancellation (by checking `is_cancelled()` and calling `cancel()` on itself) before inner sees it. This prevents a warp deep in a nested scope from seeing cancellation before its parent scope has had a chance to do cleanup.

However, for the common case where immediate propagation is desired, we provide a `propagate_cancel` helper:

```rust
impl<'scope> BlockScope<'scope> {
    /// If the parent scope is cancelled, set this scope's cancel flag too.
    ///
    /// Call this at the top of a scope body to enable automatic propagation:
    /// ```rust,ignore
    /// block_scope(|outer| {
    ///     outer.spawn(|| {
    ///         block_scope(|inner| {
    ///             // At the start of inner scope, propagate parent cancel
    ///             inner.propagate_cancel();
    ///             // ... work ...
    ///         });
    ///     });
    /// });
    /// ```
    pub fn propagate_cancel(&self) {
        if !self.parent_cancel.is_null() {
            let parent_cancelled = unsafe {
                core::ptr::read_volatile(self.parent_cancel) != 0
            };
            if parent_cancelled {
                self.cancel();
            }
        }
    }
}
```

### 2.3 Constructing the parent chain

The `block_scope` entry function is updated with an overload that accepts a parent cancel pointer:

```rust
/// Enter a block-level scope with an explicit parent cancel pointer.
///
/// Used internally when nesting scopes. The parent pointer enables
/// `is_cancelled()` to check the parent scope's flag.
///
/// For top-level scopes, use `block_scope()` (parent = null).
pub fn block_scope_with_parent<F, R>(parent_cancel: *const u32, f: F) -> R
where
    F: for<'scope> FnOnce(&mut BlockScope<'scope>) -> R,
{
    debug_assert_eq!(warp_id(), 0, "block_scope() must be called from warp 0");

    unsafe { (&mut *ALLOCATOR.as_ptr()).push(); }

    let cancel_flag_offset = unsafe { &mut *ALLOCATOR.as_ptr() }
        .alloc_raw(core::mem::size_of::<u32>(), core::mem::align_of::<u32>())
        .expect("block_scope: not enough shared memory for cancel flag");

    unsafe {
        let flag = crate::block::shared_mem_at::<u32>(cancel_flag_offset as usize);
        core::ptr::write_volatile(flag, 0);
    }

    let mut scope = BlockScope {
        spawned_warps: 0,
        spawn_count: 0,
        cancel_flag_offset,
        parent_cancel,
        _marker: PhantomData,
    };

    let result = f(&mut scope);
    scope.join_all();
    result
}

/// Enter a top-level block scope (no parent).
pub fn block_scope<F, R>(f: F) -> R
where
    F: for<'scope> FnOnce(&mut BlockScope<'scope>) -> R,
{
    block_scope_with_parent(core::ptr::null(), f)
}
```

For nested scopes inside a spawned closure, the user obtains the parent cancel pointer via a new accessor:

```rust
impl<'scope> BlockScope<'scope> {
    /// Returns a raw pointer to this scope's cancel flag in shared memory.
    ///
    /// Pass this to `block_scope_with_parent()` when creating nested scopes
    /// inside spawned closures, to enable parent→child cancellation propagation.
    pub fn cancel_ptr(&self) -> *const u32 {
        unsafe {
            crate::block::shared_mem_at::<u32>(self.cancel_flag_offset as usize) as *const u32
        }
    }
}
```

### 2.4 Nested scope example with propagation

```rust
block_scope(|outer| {
    let data = outer.alloc::<f32>(256);
    let parent_ptr = outer.cancel_ptr();

    outer.spawn(|| {
        // Nested scope: parent_cancel points to outer's flag
        block_scope_with_parent(parent_ptr, |inner| {
            let scratch = inner.alloc::<f32>(32);

            inner.spawn_all(|wid, nw| {
                let mut i = wid as usize;
                while i < 32 {
                    // Check cancellation at each iteration
                    if inner.is_cancelled() {
                        return; // Early exit — parent or self cancelled
                    }
                    scratch[i] = data[i] * 2.0;
                    i += nw as usize;
                }
            });
        });
    });

    // If outer decides to cancel:
    outer.cancel(); // inner.is_cancelled() will return true on next check
});
```

## 3. Child → Parent Escalation

### 3.1 Two policies: Nursery vs Supervised

Inspired by Trio nurseries (child failure cancels siblings) and Kotlin supervisorScope (child failure isolated), we support two escalation policies:

```rust
/// Policy for how child scope failures affect the parent scope.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ScopePolicy {
    /// Nursery mode (default): if any spawned task signals failure,
    /// the scope's cancel flag is set, cancelling all siblings.
    /// Matches Trio nursery semantics.
    Nursery,

    /// Supervised mode: child failures are recorded but do not
    /// cancel siblings or the parent. Each child runs to completion
    /// independently. Matches Kotlin supervisorScope semantics.
    Supervised,
}
```

### 3.2 Error bitmask and first-error slot

Each scope carries an error bitmask (one bit per warp) and a slot for the first error's warp ID:

```rust
pub struct BlockScope<'scope> {
    spawned_warps: u32,
    spawn_count: u32,
    cancel_flag_offset: u32,
    parent_cancel: *const u32,

    /// Byte offset of the error bitmask in shared memory (u32, one bit per warp).
    error_mask_offset: u32,
    /// Byte offset of the first error's warp ID in shared memory (u32).
    first_error_warp_offset: u32,
    /// Escalation policy.
    policy: ScopePolicy,

    _marker: PhantomData<&'scope mut &'scope ()>,
}
```

### 3.3 Task failure reporting

A spawned task reports failure by calling `scope.report_error()`. In Nursery mode, this also cancels the scope.

```rust
impl<'scope> BlockScope<'scope> {
    /// Report that the calling warp has encountered an error.
    ///
    /// Sets the warp's bit in the error bitmask. In `Nursery` mode,
    /// also sets the cancel flag to cancel all siblings.
    ///
    /// This is cooperative: the warp should return after calling this.
    /// It is NOT automatically called by the panic handler (panics
    /// call `trap;` which kills the warp with no cleanup).
    pub fn report_error(&self) {
        let wid = warp_id() as u32;

        // Set error bit for this warp
        unsafe {
            let mask = crate::block::shared_mem_at::<u32>(self.error_mask_offset as usize);
            let old = core::ptr::read_volatile(mask);
            core::ptr::write_volatile(mask, old | (1 << wid));

            // Record first error (CAS-like: only write if slot is still 0xFFFFFFFF)
            let first = crate::block::shared_mem_at::<u32>(self.first_error_warp_offset as usize);
            if core::ptr::read_volatile(first) == 0xFFFF_FFFF {
                core::ptr::write_volatile(first, wid);
            }
        }

        // Nursery mode: cancel all siblings
        if self.policy == ScopePolicy::Nursery {
            self.cancel();
        }
    }

    /// Returns the error bitmask after scope exit.
    /// Each set bit corresponds to a warp that called `report_error()`.
    pub fn error_mask(&self) -> u32 {
        unsafe {
            let mask = crate::block::shared_mem_at::<u32>(self.error_mask_offset as usize);
            core::ptr::read_volatile(mask)
        }
    }

    /// Returns the warp ID of the first error, or `None` if no errors.
    pub fn first_error_warp(&self) -> Option<u32> {
        unsafe {
            let first = crate::block::shared_mem_at::<u32>(self.first_error_warp_offset as usize);
            let val = core::ptr::read_volatile(first);
            if val == 0xFFFF_FFFF { None } else { Some(val) }
        }
    }

    /// Returns true if any spawned task reported an error.
    pub fn has_errors(&self) -> bool {
        self.error_mask() != 0
    }
}
```

### 3.4 GPU trap reality

When a warp executes `trap;`, it dies immediately — no cleanup, no `report_error()` call. The scope's `join_all()` loop would spin forever waiting for `STATUS_DONE` that never comes. This is addressed by a new warp status:

```rust
/// Warp trapped (panic handler set this before calling trap;).
pub(crate) const STATUS_TRAPPED: u32 = 6;
```

The panic handler is updated to set `STATUS_TRAPPED` before trapping:

```rust
// In panic handler (simplified):
fn panic_handler(info: &PanicInfo) {
    let wid = warp_id();
    // ... send panic message via hostcall (best-effort) ...
    // ... write to GpuKernelResult ...

    // Signal the scope that this warp trapped
    WARP_STATUS[wid].store(STATUS_TRAPPED, Ordering::Release);

    // Now trap — the warp is dead after this
    unsafe { core::arch::asm!("trap;"); }
}
```

The scope's `join_all()` is updated to detect trapped warps:

```rust
impl<'scope> BlockScope<'scope> {
    fn join_all(&mut self) {
        let mut mask = self.spawned_warps;
        while mask != 0 {
            let wid = mask.trailing_zeros() as usize;
            loop {
                let status = WARP_STATUS[wid].load(Ordering::Acquire);
                match status {
                    STATUS_DONE => {
                        WARP_STATUS[wid].store(STATUS_IDLE, Ordering::Release);
                        break;
                    }
                    STATUS_TRAPPED => {
                        // Warp is dead. Record in error mask.
                        unsafe {
                            let emask = crate::block::shared_mem_at::<u32>(
                                self.error_mask_offset as usize,
                            );
                            let old = core::ptr::read_volatile(emask);
                            core::ptr::write_volatile(emask, old | (1 << wid));
                        }
                        // Do NOT reset to IDLE — the warp is dead and cannot
                        // re-enter the worker loop. It stays TRAPPED permanently.
                        break;
                    }
                    _ => nanosleep_short(),
                }
            }
            mask &= !(1 << wid);
        }
        self.spawned_warps = 0;
        self.spawn_count = 0;
    }
}
```

### 3.5 Child→parent escalation in Nursery mode

When a child scope completes with errors and its parent uses Nursery policy, the child scope's exit sequence can escalate to the parent:

```rust
// In block_scope_with_parent, after the closure returns:
let result = f(&mut scope);
scope.join_all();

// Escalation: if child has errors and parent exists, report to parent
if scope.has_errors() && !parent_cancel.is_null() {
    // In Nursery mode, the child already cancelled itself (which the parent
    // sees via is_cancelled() chain-walk). Additionally, we can set the
    // parent's cancel flag directly for immediate propagation:
    unsafe { core::ptr::write_volatile(parent_cancel as *mut u32, 1); }
}
```

This gives Trio-like semantics: child failure → cancel siblings → propagate to parent.

In Supervised mode, the child's errors are isolated. The parent can query them after the child scope exits, but no automatic cancellation happens.

## 4. GridScope Cancellation

### 4.1 GridScope cancel flag location

GridScope's cancel flag lives in global memory, accessible by all blocks via system-scope atomics:

```rust
pub struct GridScope<'scope> {
    pool_base: *mut u8,
    pool_offset: u32,
    pool_capacity: u32,
    /// Atomic completion counter (global memory).
    completion_counter: *mut u32,
    expected_completions: u32,
    /// Cancellation flag (global memory). System-scope atomics.
    cancel_flag: *mut u32,
    /// Error bitmask for blocks (global memory). One bit per block.
    /// (u32 supports up to 32 blocks; extend to u64 if needed.)
    block_error_mask: *mut u32,
    /// First error block ID (global memory). 0xFFFFFFFF = no error.
    first_error_block: *mut u32,
    /// Escalation policy.
    policy: ScopePolicy,

    _marker: PhantomData<&'scope mut &'scope ()>,
}
```

### 4.2 GridScope → BlockScope propagation

When a GridScope dispatches work to a block, the block's `block_scope` receives the GridScope's cancel flag as its parent:

```rust
impl<'scope> GridScope<'scope> {
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
        // ... allocate result slot, done flag ...
        // The work descriptor includes the GridScope's cancel_flag pointer
        // so the target block can pass it to block_scope_with_parent.
        let work = BlockWorkDescriptor {
            args,
            f,
            result_slot,
            done_flag,
            parent_cancel: self.cancel_flag as *const u32,  // <-- key
        };
        // ... publish to target block's command slot ...
    }
}
```

On the target block, the worker loop creates a BlockScope with the GridScope's cancel flag as parent:

```rust
// Target block worker loop (simplified):
fn block_worker_loop(cmd_slot: *const BlockWorkDescriptor<A, F, T>) {
    let desc = unsafe { core::ptr::read_volatile(cmd_slot) };

    block_scope_with_parent(desc.parent_cancel, |scope| {
        // scope.is_cancelled() checks:
        //   1. This block's local cancel flag (shared memory, ~2 cycles)
        //   2. GridScope's cancel flag (global memory, ~100 cycles)
        let result = (desc.f)(desc.args);
        unsafe { core::ptr::write(desc.result_slot, result); }
    });

    // Signal completion
    unsafe { gpu_atomics::sys_store_release_u32(desc.done_flag, 1); }
}
```

### 4.3 Block failure → other blocks (GridScope)

When a block fails, the GridScope handles it according to its policy:

**Nursery policy (default):**
A block failure (trap or `report_error`) sets the GridScope's `cancel_flag` in global memory. All other blocks see this on their next `is_cancelled()` check. This is the "one block fails, all blocks stop" semantic.

**Supervised policy:**
A block failure sets the per-block error bit in `block_error_mask` but does not touch the `cancel_flag`. Other blocks continue executing. The coordinator collects the error mask at scope exit.

```rust
impl<'scope> GridScope<'scope> {
    /// Request cancellation of all blocks in this scope.
    pub fn cancel(&self) {
        unsafe { gpu_atomics::sys_store_release_u32(self.cancel_flag, 1); }
    }

    /// Check if this scope has been cancelled.
    pub fn is_cancelled(&self) -> bool {
        unsafe { gpu_atomics::sys_load_acquire_u32(self.cancel_flag as *const u32) != 0 }
    }

    /// Report that a block has failed. Called by the block's scope exit code.
    ///
    /// `block_id` identifies the failing block.
    pub fn report_block_error(&self, block_id: u32) {
        unsafe {
            // Atomic OR to set the block's error bit.
            // (Use CAS loop since there's no atomic OR on all architectures.)
            loop {
                let old = gpu_atomics::sys_load_acquire_u32(
                    self.block_error_mask as *const u32,
                );
                let new = old | (1 << block_id);
                if gpu_atomics::sys_cas_u32(self.block_error_mask, old, new) == old {
                    break;
                }
            }

            // Record first error block
            gpu_atomics::sys_cas_u32(self.first_error_block, 0xFFFF_FFFF, block_id);
        }

        // Nursery mode: cancel all blocks
        if self.policy == ScopePolicy::Nursery {
            self.cancel();
        }
    }

    /// Returns the block error bitmask after scope exit.
    pub fn block_error_mask(&self) -> u32 {
        unsafe { gpu_atomics::sys_load_acquire_u32(self.block_error_mask as *const u32) }
    }

    /// Returns true if any block reported an error.
    pub fn has_errors(&self) -> bool {
        self.block_error_mask() != 0
    }
}
```

### 4.4 Cancellation cost analysis

| Check | Memory tier | Latency | When |
|-------|------------|---------|------|
| BlockScope local flag | Shared memory | ~2 cycles | Every yield point |
| BlockScope parent (another BlockScope) | Shared memory | ~2 cycles | Every yield point, if local not set |
| BlockScope parent (GridScope) | Global memory | ~100 cycles | Every yield point, if local not set |
| GridScope cancel flag | Global memory | ~100 cycles | Block-level checkpoints |

For the common non-cancelled path, the total cost is ~4 cycles per check (two shared memory reads). The 100-cycle global read only happens when the local flag is clear and the parent is a GridScope — amortized over many iterations of a compute loop, this is negligible.

## 5. Cleanup Semantics

### 5.1 In-flight spawned warps

When a scope is cancelled, spawned warps are NOT forcibly stopped. Cooperative cancellation means:

1. Parent calls `scope.cancel()`.
2. Each spawned warp checks `scope.is_cancelled()` at its next yield point.
3. The warp returns early (the task closure returns).
4. The scope's `join_all()` waits until all warps reach `STATUS_DONE` or `STATUS_TRAPPED`.

**There is no "instant kill" of in-flight warps.** This matches GPU reality: warps can only be stopped by `trap;` (which kills the warp with no cleanup) or by cooperative return. The scope guarantees that `join_all()` blocks until all warps have terminated, regardless of whether they acknowledged the cancellation.

Guideline for task implementations:

```rust
// Good: check cancellation in the inner loop
scope.spawn(|| {
    for i in 0..1_000_000 {
        if scope.is_cancelled() {
            return; // Early exit
        }
        heavy_computation(i);
    }
});

// Bad: never checks cancellation — will run to completion even if cancelled
scope.spawn(|| {
    for i in 0..1_000_000 {
        heavy_computation(i);
    }
});
```

### 5.2 Shared memory allocations

Shared memory cleanup is handled by the watermark allocator, not by cancellation:

- When a BlockScope exits (whether by normal completion or after cancellation), `Drop` pops the watermark. This logically frees all shared memory allocated within the scope.
- `T: Copy` is required for all allocations, so no destructors need to run.
- The watermark pop happens regardless of whether tasks completed successfully or were cancelled. The scope's `join_all()` guarantees all warps have stopped before the pop.

**Cancellation does NOT change the cleanup sequence.** The scope still waits for all warps to finish, then pops the watermark. The only difference is that cancelled warps should return early, making the wait shorter.

### 5.3 Channel operations

Blocked channel operations (e.g., `recv()` spinning for a value) should check cancellation to avoid deadlock:

```rust
impl<'scope> BlockScope<'scope> {
    /// Spin-receive with cancellation awareness.
    ///
    /// Returns `Some(value)` if the channel delivers, or `None` if the
    /// scope was cancelled before a value arrived.
    pub fn recv_or_cancel<T: Copy>(
        &self,
        slot: &BlockOneshotSlot<T>,
    ) -> Option<T> {
        loop {
            // Try to receive
            if let Some(val) = slot.try_recv() {
                return Some(val);
            }
            // Check cancellation
            if self.is_cancelled() {
                return None;
            }
            nanosleep_short();
        }
    }
}
```

This is a convenience — the user can also write their own cancel-aware recv loop. The key principle is that cancellation does not forcibly interrupt a channel operation; the receiver must poll for both the value and the cancel flag.

### 5.4 Cleanup ordering

The full scope exit sequence with cancellation support:

```
1. Closure returns (either naturally or early due to cancellation check).

2. join_all():
   For each spawned warp:
     - Spin until STATUS_DONE or STATUS_TRAPPED.
     - If TRAPPED: record in error bitmask.
     - If DONE: reset to STATUS_IDLE.

3. Error escalation (if parent_cancel is set and policy is Nursery):
   - If error_mask != 0, set parent's cancel flag.

4. Drop (watermark pop):
   - Pop the shared memory allocator watermark.
   - All shared memory from this scope is logically freed.

5. Return to caller. Caller can inspect error state via the scope's
   return value or by querying error_mask()/first_error_warp() on the
   scope before it drops (requires careful structuring — see API section).
```

## 6. API Design

### 6.1 Scope creation with policy

```rust
/// Enter a block scope with explicit parent and policy.
///
/// This is the full-featured entry point. Convenience wrappers
/// (`block_scope`, `block_scope_with_parent`) call this with defaults.
pub fn block_scope_full<F, R>(
    parent_cancel: *const u32,
    policy: ScopePolicy,
    f: F,
) -> ScopeResult<R>
where
    F: for<'scope> FnOnce(&mut BlockScope<'scope>) -> R,
{
    // ... setup allocator, flags, error mask ...
    let mut scope = BlockScope { /* ... */ };
    let result = f(&mut scope);
    scope.join_all();

    let error_mask = scope.error_mask();
    let first_error = scope.first_error_warp();

    // Escalate if needed
    if error_mask != 0 && !parent_cancel.is_null() && policy == ScopePolicy::Nursery {
        unsafe { core::ptr::write_volatile(parent_cancel as *mut u32, 1); }
    }

    // scope drops here → watermark pop

    if error_mask != 0 {
        ScopeResult::Err(ScopeError {
            value: result,
            error_mask,
            first_error_warp: first_error,
        })
    } else {
        ScopeResult::Ok(result)
    }
}

/// Result type for scope execution.
#[derive(Debug)]
pub enum ScopeResult<T> {
    /// All tasks completed successfully.
    Ok(T),
    /// Some tasks failed. The closure's return value is still available
    /// (it ran to completion), but spawned tasks had errors.
    Err(ScopeError<T>),
}

/// Error information from a scope with failed tasks.
#[derive(Debug)]
pub struct ScopeError<T> {
    /// The closure's return value (the closure itself completed).
    pub value: T,
    /// Bitmask of warps that reported errors or trapped.
    pub error_mask: u32,
    /// Warp ID of the first error, if any.
    pub first_error_warp: Option<u32>,
}

impl<T> ScopeResult<T> {
    /// Extract the value, ignoring errors.
    pub fn into_value(self) -> T {
        match self {
            ScopeResult::Ok(v) => v,
            ScopeResult::Err(e) => e.value,
        }
    }

    /// Returns true if all tasks completed without errors.
    pub fn is_ok(&self) -> bool {
        matches!(self, ScopeResult::Ok(_))
    }
}
```

### 6.2 Convenience wrappers

```rust
/// Top-level block scope (no parent, nursery policy).
/// This is the most common entry point.
pub fn block_scope<F, R>(f: F) -> R
where
    F: for<'scope> FnOnce(&mut BlockScope<'scope>) -> R,
{
    block_scope_full(core::ptr::null(), ScopePolicy::Nursery, f).into_value()
}

/// Block scope with parent cancel pointer (nursery policy).
/// Used for nesting.
pub fn block_scope_with_parent<F, R>(parent_cancel: *const u32, f: F) -> R
where
    F: for<'scope> FnOnce(&mut BlockScope<'scope>) -> R,
{
    block_scope_full(parent_cancel, ScopePolicy::Nursery, f).into_value()
}

/// Block scope with explicit policy and error reporting.
/// Use when you need to inspect errors after scope exit.
pub fn block_scope_supervised<F, R>(parent_cancel: *const u32, f: F) -> ScopeResult<R>
where
    F: for<'scope> FnOnce(&mut BlockScope<'scope>) -> R,
{
    block_scope_full(parent_cancel, ScopePolicy::Supervised, f)
}
```

### 6.3 GridScope creation with cancellation

```rust
/// Enter a grid-level scope with cancellation support.
pub unsafe fn grid_scope<F, R>(
    pool: *mut u8,
    pool_size: u32,
    f: F,
) -> ScopeResult<R>
where
    F: for<'scope> FnOnce(&'scope GridScope<'scope>) -> R,
{
    // Allocate cancel_flag, block_error_mask, first_error_block from pool
    // ...
    let scope = GridScope {
        cancel_flag,
        block_error_mask,
        first_error_block,
        policy: ScopePolicy::Nursery,
        // ...
    };

    let result = f(&scope);

    // Wait for all blocks to complete
    // ...

    let error_mask = scope.block_error_mask();
    if error_mask != 0 {
        ScopeResult::Err(ScopeError {
            value: result,
            error_mask,
            first_error_warp: scope.first_error_block().map(|b| b),
        })
    } else {
        ScopeResult::Ok(result)
    }
}
```

### 6.4 Full API surface summary

**New types:**
- `ScopePolicy` — `Nursery` | `Supervised`
- `ScopeResult<T>` — `Ok(T)` | `Err(ScopeError<T>)`
- `ScopeError<T>` — `{ value: T, error_mask: u32, first_error_warp: Option<u32> }`

**New on BlockScope:**
- `cancel_ptr(&self) -> *const u32` — for constructing parent chains
- `report_error(&self)` — cooperative error reporting
- `propagate_cancel(&self)` — manually pull parent cancellation
- `error_mask(&self) -> u32` — query error bitmask
- `first_error_warp(&self) -> Option<u32>` — query first error
- `has_errors(&self) -> bool` — any errors?
- `recv_or_cancel(...)` — cancel-aware channel receive

**Updated on BlockScope:**
- `is_cancelled(&self) -> bool` — now checks local + parent flag
- Internal: `parent_cancel` field, `error_mask_offset`, `first_error_warp_offset`, `policy`

**New on GridScope:**
- `report_block_error(&self, block_id: u32)` — block-level error reporting
- `block_error_mask(&self) -> u32` — query block error bitmask
- `has_errors(&self) -> bool`

**New free functions:**
- `block_scope_with_parent(parent_cancel, f) -> R`
- `block_scope_full(parent_cancel, policy, f) -> ScopeResult<R>`
- `block_scope_supervised(parent_cancel, f) -> ScopeResult<R>`

**New constant:**
- `STATUS_TRAPPED: u32 = 6`

**Shared memory overhead per BlockScope:**
- Cancel flag: 4 bytes
- Error bitmask: 4 bytes
- First error warp: 4 bytes
- **Total: 12 bytes per scope** (was 4 bytes for cancel flag alone)

## 7. Examples

### 7.1 Nested scopes with automatic parent→child cancellation

```rust
use gpu_runtime::scope::{block_scope, block_scope_with_parent};

fn nested_cancel_example() {
    thread::gpu_main(|| {
        block_scope(|outer| {
            let data = outer.alloc::<f32>(1024);
            let parent_ptr = outer.cancel_ptr();

            // Spawn a warp that creates a nested scope
            let h = outer.spawn(|| {
                block_scope_with_parent(parent_ptr, |inner| {
                    let scratch = inner.alloc::<f32>(64);

                    inner.spawn_all(|wid, nw| {
                        let mut i = wid as usize;
                        while i < 64 {
                            // This check sees both inner and outer cancellation
                            if inner.is_cancelled() {
                                return;
                            }
                            scratch[i] = heavy_compute(data[i * 16]);
                            i += nw as usize;
                        }
                    });
                });
            });

            // Meanwhile on warp 0: decide to cancel everything
            if some_condition() {
                outer.cancel();
                // All warps in inner scope will see cancellation on next check
            }

            h.join();
        });
    });
}
```

### 7.2 Supervised scope — isolate failures

```rust
use gpu_runtime::scope::{block_scope_supervised, ScopeResult, ScopePolicy};

fn supervised_example() {
    thread::gpu_main(|| {
        let result = block_scope_supervised(core::ptr::null(), |scope| {
            let outputs = scope.alloc::<f32>(4);

            // Spawn 4 independent tasks
            let h0 = scope.spawn(|| { outputs[0] = safe_compute(0); });
            let h1 = scope.spawn(|| { outputs[1] = safe_compute(1); });
            let h2 = scope.spawn(|| {
                // This task encounters an error
                if bad_input() {
                    scope.report_error();
                    return; // Don't cancel siblings (supervised mode)
                }
                outputs[2] = safe_compute(2);
            });
            let h3 = scope.spawn(|| { outputs[3] = safe_compute(3); });

            h0.join(); h1.join(); h2.join(); h3.join();

            // outputs[0], [1], [3] are valid. outputs[2] may be 0.0 (zero-init).
            (outputs[0], outputs[1], outputs[2], outputs[3])
        });

        match result {
            ScopeResult::Ok(vals) => { /* all good */ }
            ScopeResult::Err(err) => {
                // err.value still has the tuple
                // err.error_mask tells us which warps failed
                let vals = err.value;
                let mask = err.error_mask;
                // Use the valid outputs, skip the failed ones
            }
        }
    });
}
```

### 7.3 GridScope → BlockScope cancellation

```rust
use gpu_runtime::scope::{grid_scope, block_scope_with_parent};

unsafe fn grid_cancel_example(
    pool: *mut u8,
    pool_size: u32,
    data: *const f32,
    n: usize,
) {
    let result = grid_scope(pool, pool_size, |gscope| {
        // Dispatch work to 4 blocks
        let h0 = gscope.spawn_block((data, 0, n / 4), |(d, off, len)| {
            // This runs on block 0 inside its own block_scope.
            // The block_scope's parent_cancel points to gscope.cancel_flag.
            block_scope_with_parent(gscope.cancel_ptr(), |bscope| {
                bscope.spawn_all(|wid, nw| {
                    let mut i = wid as usize;
                    while i < len {
                        if bscope.is_cancelled() {
                            // GridScope was cancelled — stop early.
                            // This check reads:
                            //   1. bscope.cancel_flag (shared mem, ~2 cyc)
                            //   2. gscope.cancel_flag (global mem, ~100 cyc)
                            return;
                        }
                        process(*d.add(off + i));
                        i += nw as usize;
                    }
                });
            });
        });

        // ... spawn h1, h2, h3 similarly ...

        // If coordinator detects a problem, cancel all blocks:
        if global_error_detected() {
            gscope.cancel();
            // All blocks' is_cancelled() → true on next check
        }

        gscope.join_all_handles(); // Wait for all blocks
    });
}
```

### 7.4 Nursery mode — child failure cancels siblings

```rust
use gpu_runtime::scope::block_scope;

fn nursery_cancel_example() {
    thread::gpu_main(|| {
        // Default block_scope uses Nursery policy
        block_scope(|scope| {
            let results = scope.alloc::<f32>(4);

            let h0 = scope.spawn(|| {
                // Long computation — checks cancellation
                for i in 0..10000 {
                    if scope.is_cancelled() { return; }
                    results[0] += slow_step(i);
                }
            });

            let h1 = scope.spawn(|| {
                // This task finds bad data and reports error
                if validate_input().is_err() {
                    scope.report_error(); // Sets cancel flag (Nursery mode)
                    return;
                }
                results[1] = compute();
            });

            // h0 will see cancellation on next is_cancelled() check
            // because h1's report_error() set the scope's cancel flag.

            h0.join();
            h1.join();

            // scope.has_errors() == true
        });
    });
}
```

### 7.5 Cancel-aware channel receive

```rust
use gpu_runtime::scope::block_scope;
use gpu_runtime::block_channel::{BlockOneshotSlot, block_oneshot};

fn channel_cancel_example() {
    thread::gpu_main(|| {
        block_scope(|scope| {
            let slot = scope.alloc_val(BlockOneshotSlot::<u32>::new());
            let (tx, rx) = unsafe { block_oneshot(slot) };

            // Producer warp
            scope.spawn(move || {
                if scope.is_cancelled() { return; }
                let result = expensive_computation();
                unsafe { tx.send(result); }
            });

            // Consumer (warp 0): wait for result or cancellation
            match scope.recv_or_cancel(&rx) {
                Some(value) => {
                    // Got the value — use it
                    use_result(value);
                }
                None => {
                    // Scope was cancelled before value arrived
                    // Clean up and return
                }
            }
        });
    });
}
```

## Files Changed: none (design only)
