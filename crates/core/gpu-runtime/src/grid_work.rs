//! Cross-block work coordination for GridScope.
//!
//! Provides work-dispatch primitives that let a coordinator block (block 0)
//! distribute work to independently-scheduled worker blocks via global memory.
//! Designed for SM75 WITHOUT cooperative launch — blocks are independently
//! scheduled and may not all be resident simultaneously.
//!
//! # Architecture
//!
//! ```text
//!   Coordinator (block 0)              Worker blocks (1..N)
//!   ┌─────────────────────┐            ┌─────────────────────┐
//!   │ alloc_work_slots(N) │            │ grid_worker_loop()  │
//!   │                     │            │   poll slot.status   │
//!   │ dispatch_work(slot, │──[global]──│   if WORK_AVAILABLE │
//!   │   fn_ptr, args)     │   memory   │     execute work    │
//!   │                     │            │     write result     │
//!   │ poll results        │◄─[global]──│     set COMPLETED   │
//!   └─────────────────────┘   memory   └─────────────────────┘
//! ```
//!
//! # Status state machine
//!
//! ```text
//!   IDLE ──[coordinator writes work]──► WORK_AVAILABLE
//!     ▲                                      │
//!     │                              [worker picks up]
//!     │                                      ▼
//!     └────[coordinator reads result]── COMPLETED
//!
//!   Any state ──[coordinator cancels]──► CANCELLED
//! ```
//!
//! # Memory model
//!
//! All status transitions use system-scope atomics (`sys_store_release_u32` /
//! `sys_spin_load_acquire_u32`) for cross-block visibility on SM75+. The
//! coordinator writes work descriptors before releasing `WORK_AVAILABLE`;
//! the worker acquires the status and is guaranteed to see the written data.
//!
//! # Example
//!
//! ```rust,ignore
//! use gpu_runtime::scope::grid_scope;
//! use gpu_runtime::grid_work::{self, BlockWorkSlot};
//!
//! // Coordinator block (block 0):
//! unsafe {
//!     grid_scope(pool, pool_size, |gscope| {
//!         let slots = gscope.alloc::<BlockWorkSlot>(num_workers);
//!         grid_work::init_work_slots(slots);
//!
//!         // Dispatch work to block 1
//!         grid_work::dispatch_work(
//!             &mut slots[0],
//!             my_work_fn as u64,
//!             [input_ptr as u64, count as u64, 0, 0],
//!         );
//!
//!         // Wait for completion
//!         grid_work::wait_slot_completed(&slots[0]);
//!         let result = grid_work::read_result(&slots[0]);
//!     });
//! }
//!
//! // Worker block (block N):
//! unsafe {
//!     // slot_ptr provided by host or discovered via device global
//!     grid_work::grid_worker_loop(slot_ptr);
//! }
//! ```

/// Work slot status: idle, no work assigned.
pub const SLOT_IDLE: u32 = 0;

/// Work slot status: coordinator has written work, worker should execute.
pub const SLOT_WORK_AVAILABLE: u32 = 1;

/// Work slot status: worker has finished, result is available.
pub const SLOT_COMPLETED: u32 = 2;

/// Work slot status: work is cancelled, worker should exit.
pub const SLOT_CANCELLED: u32 = 3;

/// A work descriptor slot in global memory for one block.
///
/// The coordinator block writes `work_fn`, `args`, then transitions
/// `status` to [`SLOT_WORK_AVAILABLE`]. The worker block polls `status`,
/// executes the work, writes `result`, then transitions to [`SLOT_COMPLETED`].
///
/// All fields are `u64`-aligned for system-scope atomic access.
/// The `status` field is the synchronization point — it must be accessed
/// via system-scope atomics only.
///
/// # Layout
///
/// ```text
/// offset 0:  status   (u32, atomic — state machine)
/// offset 4:  _pad     (u32, alignment padding)
/// offset 8:  work_fn  (u64, function pointer cast)
/// offset 16: args     ([u64; 4], work arguments)
/// offset 48: result   (u64, written by worker)
/// ```
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockWorkSlot {
    /// Status field — accessed via system-scope atomics.
    /// Values: [`SLOT_IDLE`], [`SLOT_WORK_AVAILABLE`], [`SLOT_COMPLETED`], [`SLOT_CANCELLED`].
    pub status: u32,
    /// Padding for u64 alignment of subsequent fields.
    pub _pad: u32,
    /// Function pointer for the work (as u64).
    ///
    /// The function signature is `fn(args: &[u64; 4]) -> u64`.
    /// Cast with `my_fn as fn(&[u64; 4]) -> u64 as usize as u64`.
    pub work_fn: u64,
    /// Work arguments (as raw u64 values). Interpretation is up to the
    /// work function — typically pointers and sizes.
    pub args: [u64; 4],
    /// Result value (written by worker, read by coordinator after COMPLETED).
    pub result: u64,
}

/// Initialize an array of work slots to [`SLOT_IDLE`].
///
/// Zeroes all fields and sets each slot's status to `SLOT_IDLE` using
/// a system-scope release store for cross-block visibility.
///
/// # Safety
///
/// - `slots` must point to valid global memory with at least `count` elements.
/// - Must be called before any worker begins polling.
pub unsafe fn init_work_slots(slots: &mut [BlockWorkSlot]) {
    for slot in slots.iter_mut() {
        slot._pad = 0;
        slot.work_fn = 0;
        slot.args = [0; 4];
        slot.result = 0;
        // Status must be set last with release semantics so workers see zeroed fields.
        let status_ptr = &mut slot.status as *mut u32;
        gpu_atomics::sys_store_release_u32(status_ptr, SLOT_IDLE);
    }
}

/// Dispatch work to a specific block's work slot.
///
/// Writes the function pointer and arguments, then transitions the slot
/// status to [`SLOT_WORK_AVAILABLE`] with a system-scope release store.
/// The release guarantees that the worker block will see `work_fn` and
/// `args` after it acquires the `WORK_AVAILABLE` status.
///
/// # Safety
///
/// - `slot` must point to a valid [`BlockWorkSlot`] in global memory.
/// - The slot must be in [`SLOT_IDLE`] or [`SLOT_COMPLETED`] state.
/// - `work_fn` must be a valid function pointer cast to u64 with signature
///   `fn(args: &[u64; 4]) -> u64`.
/// - The coordinator must not write to the slot again until the worker
///   transitions it to [`SLOT_COMPLETED`].
pub unsafe fn dispatch_work(slot: &mut BlockWorkSlot, work_fn: u64, args: [u64; 4]) {
    // Write work descriptor fields (plain stores — no ordering needed yet).
    slot.work_fn = work_fn;
    slot.args = args;
    slot.result = 0;

    // Release-store the status — makes work_fn and args visible to the worker.
    let status_ptr = &mut slot.status as *mut u32;
    gpu_atomics::sys_store_release_u32(status_ptr, SLOT_WORK_AVAILABLE);
}

/// Cancel a work slot.
///
/// Transitions the slot to [`SLOT_CANCELLED`] via system-scope release store.
/// If a worker is polling, it will see the cancellation and exit. If the
/// worker has already started executing, the cancellation takes effect after
/// the current work item completes (cooperative cancellation).
///
/// # Safety
///
/// - `slot` must point to a valid [`BlockWorkSlot`] in global memory.
pub unsafe fn cancel_slot(slot: &mut BlockWorkSlot) {
    let status_ptr = &mut slot.status as *mut u32;
    gpu_atomics::sys_store_release_u32(status_ptr, SLOT_CANCELLED);
}

/// Read the status of a work slot using system-scope acquire load.
///
/// Returns the current status value. Use this for non-spinning one-shot
/// checks. For polling loops, use [`poll_slot_status`] instead.
///
/// # Safety
///
/// - `slot` must point to a valid [`BlockWorkSlot`] in global memory.
pub unsafe fn load_slot_status(slot: &BlockWorkSlot) -> u32 {
    let status_ptr = &slot.status as *const u32;
    gpu_atomics::sys_load_acquire_u32(status_ptr)
}

/// Poll the status of a work slot using spin-loop-safe system-scope acquire load.
///
/// Returns the current status value. Includes `nanosleep` to yield the
/// warp slot during spinning. Use inside polling loops.
///
/// # Safety
///
/// - `slot` must point to a valid [`BlockWorkSlot`] in global memory.
pub unsafe fn poll_slot_status(slot: &BlockWorkSlot) -> u32 {
    let status_ptr = &slot.status as *const u32;
    gpu_atomics::sys_spin_load_acquire_u32(status_ptr)
}

/// Spin-wait until a work slot reaches [`SLOT_COMPLETED`] or [`SLOT_CANCELLED`].
///
/// Returns the terminal status. Uses spin-loop-safe system-scope loads
/// with nanosleep yield.
///
/// # Safety
///
/// - `slot` must point to a valid [`BlockWorkSlot`] in global memory.
/// - The slot must have been dispatched (status = [`SLOT_WORK_AVAILABLE`])
///   or this will spin forever.
pub unsafe fn wait_slot_done(slot: &BlockWorkSlot) -> u32 {
    loop {
        let s = poll_slot_status(slot);
        if s == SLOT_COMPLETED || s == SLOT_CANCELLED {
            return s;
        }
    }
}

/// Read the result from a completed work slot.
///
/// # Safety
///
/// - `slot` must point to a valid [`BlockWorkSlot`] in global memory.
/// - The slot must be in [`SLOT_COMPLETED`] state (call after [`wait_slot_done`]).
/// - The result value was written by the worker with a release store before
///   transitioning to COMPLETED, so the acquire on status guarantees visibility.
pub unsafe fn read_result(slot: &BlockWorkSlot) -> u64 {
    slot.result
}

/// Reset a completed or cancelled slot back to [`SLOT_IDLE`] for reuse.
///
/// # Safety
///
/// - `slot` must point to a valid [`BlockWorkSlot`] in global memory.
/// - The slot must be in [`SLOT_COMPLETED`] or [`SLOT_CANCELLED`] state.
/// - No worker should be polling this slot when it is reset.
pub unsafe fn reset_slot(slot: &mut BlockWorkSlot) {
    slot.work_fn = 0;
    slot.args = [0; 4];
    slot.result = 0;
    let status_ptr = &mut slot.status as *mut u32;
    gpu_atomics::sys_store_release_u32(status_ptr, SLOT_IDLE);
}

/// Worker block main loop. Polls the work slot assigned to this block.
///
/// Executes work when status transitions to [`SLOT_WORK_AVAILABLE`].
/// The work function is called with a reference to the `args` array
/// and its return value is written to `result`. After execution, the
/// slot transitions to [`SLOT_COMPLETED`].
///
/// The loop exits when:
/// - The slot transitions to [`SLOT_CANCELLED`] (coordinator requested exit)
/// - A work item completes (single-shot mode). The worker writes COMPLETED
///   and returns, allowing the coordinator to dispatch more work via
///   another call to `grid_worker_loop` or `grid_worker_loop_continuous`.
///
/// # Safety
///
/// - `slot` must point to a valid [`BlockWorkSlot`] in global memory.
/// - The function pointer stored in `work_fn` must be valid and have
///   signature `fn(args: &[u64; 4]) -> u64`.
/// - Must be called from the worker block's warp 0.
pub unsafe fn grid_worker_loop(slot: *mut BlockWorkSlot) {
    loop {
        let status_ptr = &(*slot).status as *const u32 as *mut u32;
        let s = gpu_atomics::sys_spin_load_acquire_u32(status_ptr as *const u32);

        if s == SLOT_CANCELLED {
            // Coordinator requested exit — stop polling.
            return;
        }

        if s == SLOT_WORK_AVAILABLE {
            // Read work descriptor (acquire on status guarantees visibility).
            let work_fn_raw = (*slot).work_fn;
            let args = &(*slot).args;

            // Cast and call the work function.
            let work_fn: fn(&[u64; 4]) -> u64 = core::mem::transmute(work_fn_raw as usize);
            let result = work_fn(args);

            // Write result, then release-store COMPLETED.
            (*slot).result = result;
            gpu_atomics::sys_store_release_u32(status_ptr, SLOT_COMPLETED);

            // Single-shot: return after one work item.
            // The coordinator can dispatch more work and the worker can
            // call grid_worker_loop again, or use grid_worker_loop_continuous.
            return;
        }

        // SLOT_IDLE or SLOT_COMPLETED — keep polling (nanosleep is in
        // sys_spin_load_acquire_u32).
    }
}

/// Continuous worker block loop. Like [`grid_worker_loop`] but keeps
/// polling for new work after each completion instead of returning.
///
/// The loop exits only when the slot transitions to [`SLOT_CANCELLED`].
///
/// After completing a work item, the worker sets the slot to [`SLOT_COMPLETED`].
/// The coordinator must reset the slot to [`SLOT_IDLE`] (or dispatch new work
/// directly) before the worker will pick up the next item.
///
/// # Safety
///
/// Same requirements as [`grid_worker_loop`].
pub unsafe fn grid_worker_loop_continuous(slot: *mut BlockWorkSlot) {
    loop {
        let status_ptr = &(*slot).status as *const u32 as *mut u32;
        let s = gpu_atomics::sys_spin_load_acquire_u32(status_ptr as *const u32);

        if s == SLOT_CANCELLED {
            return;
        }

        if s == SLOT_WORK_AVAILABLE {
            let work_fn_raw = (*slot).work_fn;
            let args = &(*slot).args;

            let work_fn: fn(&[u64; 4]) -> u64 = core::mem::transmute(work_fn_raw as usize);
            let result = work_fn(args);

            (*slot).result = result;
            gpu_atomics::sys_store_release_u32(status_ptr, SLOT_COMPLETED);

            // Continue polling — coordinator will dispatch more work or cancel.
        }

        // SLOT_IDLE or SLOT_COMPLETED — keep polling.
    }
}
