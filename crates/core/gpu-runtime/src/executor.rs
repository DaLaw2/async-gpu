use core::cell::UnsafeCell;
use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};
use gpu_atomics::{
    lane_id, shfl_sync_idx_u32, syncwarp, sys_cas_u64, sys_load_acquire_u64,
    sys_spin_load_acquire_u32, sys_store_release_u32,
};

/// Maximum number of tasks the executor can hold.
pub const MAX_TASKS: usize = 256;

/// Maximum size of a spawned future in bytes.
pub const TASK_FUTURE_MAX_SIZE: usize = 512;

/// Sentinel value for empty queue/slot entries.
pub const EMPTY_SENTINEL: u32 = 0xFFFF_FFFF;

/// Maximum polls before a task is considered stuck.
/// Kept conservative to avoid GPU TDR timeouts on complex futures.
const MAX_POLLS_PER_TASK: u32 = 1_000;

// Task slot states
const SLOT_FREE: u32 = 0;
const SLOT_QUEUED: u32 = 1;
const SLOT_RUNNING: u32 = 2;

/// Error type for executor operations.
#[derive(Debug, Clone, Copy)]
pub enum ExecutorError {
    /// Work queue is full.
    QueueFull,
    /// No free task slots available.
    NoFreeSlots,
    /// Future exceeds `TASK_FUTURE_MAX_SIZE` bytes.
    FutureTooLarge,
}

/// Handle to a spawned task.
#[derive(Clone, Copy, Debug)]
pub struct TaskId(pub u32);

/// Statistics returned when a warp exits the executor loop.
#[derive(Clone, Copy, Debug)]
pub struct ExecutorStats {
    /// Number of tasks this warp executed to completion.
    pub tasks_executed: u32,
    /// Total number of poll calls this warp made.
    pub polls_total: u32,
}

// ================================================================
// Tagged pointer helpers (same pattern as hostcall protocol)
// ================================================================

#[inline(always)]
const fn tagged_value(tag: u32, index: u32) -> u64 {
    ((tag as u64) << 32) | (index as u64)
}

#[inline(always)]
fn tagged_tag(v: u64) -> u32 {
    (v >> 32) as u32
}

#[inline(always)]
fn tagged_index(v: u64) -> u32 {
    v as u32
}

// ================================================================
// WorkQueue — bounded lock-free MPMC FIFO
// ================================================================

/// Bounded lock-free MPMC FIFO queue.
///
/// Uses tagged CAS on head and tail to prevent ABA. The buffer is a
/// circular array of task slot indices.
#[repr(C)]
pub struct WorkQueue {
    /// Consumer index (dequeue here). Tagged u64 for ABA prevention.
    pub head: UnsafeCell<u64>,
    /// Producer index (enqueue here). Tagged u64 for ABA prevention.
    pub tail: UnsafeCell<u64>,
    /// Circular buffer of task slot indices. EMPTY_SENTINEL = unoccupied.
    pub buffer: [UnsafeCell<u32>; MAX_TASKS],
}

#[allow(clippy::new_without_default)]
impl WorkQueue {
    /// Create a new empty work queue.
    pub const fn new() -> Self {
        #[allow(clippy::declare_interior_mutable_const)]
        const EMPTY: UnsafeCell<u32> = UnsafeCell::new(EMPTY_SENTINEL);
        Self {
            head: UnsafeCell::new(tagged_value(0, 0)),
            tail: UnsafeCell::new(tagged_value(0, 0)),
            buffer: [EMPTY; MAX_TASKS],
        }
    }

    /// Enqueue a task index. Returns Err if the queue is full.
    ///
    /// # Safety
    /// Must be called from a single lane (typically lane 0).
    #[inline(always)]
    pub unsafe fn enqueue(&self, task_id: u32) -> Result<(), ExecutorError> {
        let head_ptr = self.head.get();
        let tail_ptr = self.tail.get();

        loop {
            let old_tail = sys_load_acquire_u64(tail_ptr as *const _);
            let old_head = sys_load_acquire_u64(head_ptr as *const _);

            let tail_idx = tagged_index(old_tail);
            let head_idx = tagged_index(old_head);

            // Check if full (allow wraparound comparison)
            if tail_idx.wrapping_sub(head_idx) >= MAX_TASKS as u32 {
                return Err(ExecutorError::QueueFull);
            }

            let slot = tail_idx & (MAX_TASKS as u32 - 1);
            let new_tag = tagged_tag(old_tail).wrapping_add(1);
            let new_tail = tagged_value(new_tag, tail_idx.wrapping_add(1));

            if sys_cas_u64(tail_ptr, old_tail, new_tail) == old_tail {
                // Write the task ID into the buffer slot
                sys_store_release_u32(self.buffer[slot as usize].get(), task_id);
                return Ok(());
            }
            // CAS failed — retry
        }
    }

    /// Dequeue a task index. Returns EMPTY_SENTINEL if the queue is empty.
    ///
    /// # Safety
    /// Must be called from a single lane (typically lane 0).
    #[inline(always)]
    pub unsafe fn dequeue(&self) -> u32 {
        let head_ptr = self.head.get();
        let tail_ptr = self.tail.get();

        loop {
            let old_head = sys_load_acquire_u64(head_ptr as *const _);
            let old_tail = sys_load_acquire_u64(tail_ptr as *const _);

            let head_idx = tagged_index(old_head);
            let tail_idx = tagged_index(old_tail);

            // Empty check
            if head_idx == tail_idx {
                return EMPTY_SENTINEL;
            }

            let slot = head_idx & (MAX_TASKS as u32 - 1);

            // Read the task ID
            let task_id = sys_spin_load_acquire_u32(self.buffer[slot as usize].get() as *const _);
            if task_id == EMPTY_SENTINEL {
                // Producer hasn't written yet — retry
                continue;
            }

            let new_tag = tagged_tag(old_head).wrapping_add(1);
            let new_head = tagged_value(new_tag, head_idx.wrapping_add(1));

            if sys_cas_u64(head_ptr, old_head, new_head) == old_head {
                // Clear the buffer slot
                sys_store_release_u32(self.buffer[slot as usize].get(), EMPTY_SENTINEL);
                return task_id;
            }
            // CAS failed — retry
        }
    }
}

// ================================================================
// TaskSlot — type-erased future storage
// ================================================================

/// Type alias for a type-erased poll function pointer.
pub type PollFn = unsafe fn(*mut u8, &mut Context<'_>) -> Poll<()>;

/// A fixed-size slot for storing a type-erased future.
///
/// The future bytes are stored inline. The `poll_fn` pointer provides
/// type-erased access to `Future::poll()`.
#[repr(C)]
pub struct TaskSlot {
    /// Slot state: FREE / QUEUED / RUNNING
    pub state: UnsafeCell<u32>,
    /// Type-erased poll function.
    pub poll_fn: UnsafeCell<Option<PollFn>>,
    /// Size of the stored future in bytes (for debugging).
    pub future_size: UnsafeCell<u32>,
    /// Padding to align `future_bytes` to 8 bytes.
    /// Without this, future_bytes starts at offset 20 (after state=4 + pad=4 +
    /// poll_fn=8 + future_size=4), which is not 8-byte aligned. Futures containing
    /// pointers (8 bytes on nvptx64) need 8-byte aligned storage.
    pub _pad: u32,
    /// Inline storage for the future (now at offset 24 = 8-byte aligned).
    pub future_bytes: UnsafeCell<[u8; TASK_FUTURE_MAX_SIZE]>,
}

#[allow(clippy::new_without_default)]
impl TaskSlot {
    /// Create a new free task slot.
    pub const fn new() -> Self {
        Self {
            state: UnsafeCell::new(SLOT_FREE),
            poll_fn: UnsafeCell::new(None),
            future_size: UnsafeCell::new(0),
            _pad: 0,
            future_bytes: UnsafeCell::new([0u8; TASK_FUTURE_MAX_SIZE]),
        }
    }
}

/// Type-erased poll trampoline. Casts the raw bytes back to `F` and polls.
///
/// # Safety
/// `ptr` must point to a valid `F` that was previously copied into the slot.
#[inline(always)]
unsafe fn erased_poll<F: Future<Output = ()>>(ptr: *mut u8, cx: &mut Context<'_>) -> Poll<()> {
    let future = &mut *(ptr as *mut F);
    Pin::new_unchecked(future).poll(cx)
}

// ================================================================
// Free slot stack (tagged CAS, same as hostcall free packets)
// ================================================================

/// Tagged free-slot stack head.
/// Bits 63-48: epoch tag, Bits 15-0: slot index (0xFFFF = empty)
#[repr(C)]
pub struct FreeSlotStack {
    head: UnsafeCell<u64>,
}

/// Encode a free-stack tagged pointer.
#[inline(always)]
const fn free_tagged(tag: u16, index: u16) -> u64 {
    ((tag as u64) << 48) | (index as u64)
}

#[inline(always)]
fn free_tag(v: u64) -> u16 {
    (v >> 48) as u16
}

#[inline(always)]
fn free_index(v: u64) -> u16 {
    v as u16
}

const FREE_NULL: u16 = 0xFFFF;

impl FreeSlotStack {
    /// Create a stack pre-populated with all slot indices [0..count).
    ///
    /// The `next` links are stored in the task slots' `future_size` field
    /// (reused as a next pointer when the slot is FREE).
    pub const fn empty() -> Self {
        Self {
            head: UnsafeCell::new(free_tagged(0, FREE_NULL)),
        }
    }

    /// Initialize the stack with slots [0..count). Must be called once
    /// before any pop/push operations.
    ///
    /// # Safety
    /// `slots` must point to a valid TaskSlot array of at least `count` elements.
    pub unsafe fn init(&self, slots: *mut TaskSlot, count: usize) {
        // Build a linked list: slot[0] -> slot[1] -> ... -> slot[count-1] -> NULL
        // We store the "next" pointer in the slot's `future_size` field (reused).
        for i in 0..count {
            let slot = &*slots.add(i);
            let next = if i + 1 < count {
                (i + 1) as u32
            } else {
                FREE_NULL as u32
            };
            core::ptr::write_volatile(slot.future_size.get(), next);
            core::ptr::write_volatile(slot.state.get(), SLOT_FREE);
        }
        // Head points to slot 0
        core::ptr::write_volatile(self.head.get(), free_tagged(0, 0));
    }

    /// Pop a free slot index. Returns FREE_NULL if none available.
    ///
    /// # Safety
    /// `slots` must point to the TaskSlot array.
    #[inline(always)]
    pub unsafe fn pop(&self, slots: *const TaskSlot) -> u16 {
        let head_ptr = self.head.get();
        loop {
            let old_head = sys_load_acquire_u64(head_ptr as *const _);
            let idx = free_index(old_head);
            if idx == FREE_NULL {
                return FREE_NULL;
            }
            // Read next pointer from the slot's future_size field
            let slot = &*slots.add(idx as usize);
            let next_idx = core::ptr::read_volatile(slot.future_size.get()) as u16;
            let new_tag = free_tag(old_head).wrapping_add(1);
            let new_head = free_tagged(new_tag, next_idx);
            if sys_cas_u64(head_ptr, old_head, new_head) == old_head {
                return idx;
            }
        }
    }

    /// Push a slot back onto the free stack.
    ///
    /// # Safety
    /// `slots` must point to the TaskSlot array. The slot must not be in use.
    #[inline(always)]
    pub unsafe fn push(&self, slots: *mut TaskSlot, slot_idx: u16) {
        let head_ptr = self.head.get();
        let slot = &*slots.add(slot_idx as usize);
        loop {
            let old_head = sys_load_acquire_u64(head_ptr as *const _);
            // Store current head as our "next"
            core::ptr::write_volatile(slot.future_size.get(), free_index(old_head) as u32);
            let new_tag = free_tag(old_head).wrapping_add(1);
            let new_head = free_tagged(new_tag, slot_idx);
            if sys_cas_u64(head_ptr, old_head, new_head) == old_head {
                return;
            }
        }
    }
}

// ================================================================
// GpuExecutor — the main executor struct
// ================================================================

/// GPU-side async task executor with work-stealing.
///
/// Allocated in global memory. Host initializes, kernel warps call `run()`.
#[repr(C)]
pub struct GpuExecutor {
    /// Lock-free MPMC work queue.
    pub work_queue: WorkQueue,
    /// Free slot recycling stack.
    free_slots: FreeSlotStack,
    /// Number of active tasks (for shutdown detection).
    tasks_active: UnsafeCell<u32>,
    /// Shutdown flag (0 = running, 1 = shutting down).
    shutdown: UnsafeCell<u32>,
    /// Total tasks spawned (diagnostic counter).
    tasks_spawned: UnsafeCell<u32>,
    /// Total tasks completed (diagnostic counter).
    tasks_completed: UnsafeCell<u32>,
    /// Task slot arena.
    pub slots: [TaskSlot; MAX_TASKS],
}

// SAFETY: GpuExecutor is designed for concurrent access across warps/blocks.
// All mutable state is protected by atomic CAS operations.
unsafe impl Send for GpuExecutor {}
unsafe impl Sync for GpuExecutor {}
unsafe impl Send for WorkQueue {}
unsafe impl Sync for WorkQueue {}
unsafe impl Send for TaskSlot {}
unsafe impl Sync for TaskSlot {}
unsafe impl Send for FreeSlotStack {}
unsafe impl Sync for FreeSlotStack {}

/// No-op waker vtable for GPU (no real wake mechanism).
const NOOP_VTABLE: core::task::RawWakerVTable = core::task::RawWakerVTable::new(
    |_| core::task::RawWaker::new(core::ptr::null(), &NOOP_VTABLE),
    |_| {},
    |_| {},
    |_| {},
);

#[inline(always)]
fn noop_waker() -> core::task::Waker {
    unsafe {
        core::task::Waker::from_raw(core::task::RawWaker::new(core::ptr::null(), &NOOP_VTABLE))
    }
}

#[allow(clippy::new_without_default)]
impl GpuExecutor {
    /// Create a new executor with all slots free.
    ///
    /// After construction, call `init()` to set up the free slot linked list.
    pub const fn new() -> Self {
        #[allow(clippy::declare_interior_mutable_const)]
        const SLOT: TaskSlot = TaskSlot::new();
        Self {
            work_queue: WorkQueue::new(),
            free_slots: FreeSlotStack::empty(),
            tasks_active: UnsafeCell::new(0),
            shutdown: UnsafeCell::new(0),
            tasks_spawned: UnsafeCell::new(0),
            tasks_completed: UnsafeCell::new(0),
            slots: [SLOT; MAX_TASKS],
        }
    }

    /// Initialize the executor. Must be called once before `spawn()` or `run()`.
    ///
    /// Sets up the free slot linked list.
    ///
    /// # Safety
    /// Must be called by exactly one thread (e.g., lane 0 of the first warp).
    pub unsafe fn init(&self) {
        self.free_slots
            .init(self.slots.as_ptr() as *mut TaskSlot, MAX_TASKS);
        core::ptr::write_volatile(self.tasks_active.get(), 0);
        core::ptr::write_volatile(self.shutdown.get(), 0);
        core::ptr::write_volatile(self.tasks_spawned.get(), 0);
        core::ptr::write_volatile(self.tasks_completed.get(), 0);
    }

    /// Spawn a new async task onto the executor.
    ///
    /// The future is copied into a task slot and enqueued for execution.
    /// Any warp currently in `run()` may pick it up.
    ///
    /// # Safety
    /// - `self` must point to valid executor memory in global/mapped space
    /// - The future must be safe to poll from any warp
    /// - Should be called from lane 0 only (single-lane operation)
    #[inline(always)]
    pub unsafe fn spawn<F: Future<Output = ()>>(&self, future: F) -> Result<TaskId, ExecutorError> {
        let size = core::mem::size_of::<F>();
        if size > TASK_FUTURE_MAX_SIZE {
            return Err(ExecutorError::FutureTooLarge);
        }

        // Allocate a free slot
        let slot_idx = self.free_slots.pop(self.slots.as_ptr());
        if slot_idx == FREE_NULL {
            return Err(ExecutorError::NoFreeSlots);
        }

        let slot = &self.slots[slot_idx as usize];

        // Copy future bytes into the slot
        core::ptr::copy_nonoverlapping(
            &future as *const F as *const u8,
            (*slot.future_bytes.get()).as_mut_ptr(),
            size,
        );
        core::mem::forget(future); // ownership transferred to slot

        // Set the type-erased poll function
        core::ptr::write(slot.poll_fn.get(), Some(erased_poll::<F> as _));
        core::ptr::write_volatile(slot.future_size.get(), size as u32);

        // Mark as queued (release store ensures all prior writes visible)
        sys_store_release_u32(slot.state.get(), SLOT_QUEUED);

        // Enqueue into work queue
        self.work_queue.enqueue(slot_idx as u32)?;

        // Increment spawn counter
        let old = core::ptr::read_volatile(self.tasks_spawned.get());
        core::ptr::write_volatile(self.tasks_spawned.get(), old.wrapping_add(1));

        Ok(TaskId(slot_idx as u32))
    }

    /// Enter the executor loop (ExitOnEmpty policy).
    ///
    /// Dequeues and polls tasks until the queue is empty. All 32 lanes of
    /// the calling warp must call this simultaneously.
    ///
    /// `mask` must be the warp's active lane mask (from `activemask()`).
    /// Taking it as a parameter avoids GPU hangs caused by LLVM nvptx
    /// codegen issues with `activemask()` called inside inlined methods.
    ///
    /// # Safety
    /// - Must be called by all active lanes of a warp simultaneously
    /// - `self` must point to valid executor memory in global/mapped space
    #[inline(always)]
    pub unsafe fn run(&self, mask: u32) -> ExecutorStats {
        let lid = lane_id();
        let mut tasks_executed: u32 = 0;
        let mut polls_total: u32 = 0;

        // Build waker + context. Use ManuallyDrop to avoid drop glue.
        let waker = core::mem::ManuallyDrop::new(noop_waker());
        let mut cx = Context::from_waker(&waker);

        let mut outer_count: u32 = 0;
        loop {
            // Safety valve: prevent infinite outer loops
            outer_count += 1;
            if outer_count > MAX_TASKS as u32 + 2 {
                break;
            }

            // Lane 0 dequeues, broadcasts to all lanes
            let mut task_id: u32 = EMPTY_SENTINEL;
            if lid == 0 {
                // Inline dequeue: read head/tail, check empty, CAS
                let head_ptr = self.work_queue.head.get();
                let tail_ptr = self.work_queue.tail.get();
                let old_head = sys_load_acquire_u64(head_ptr as *const _);
                let old_tail = sys_load_acquire_u64(tail_ptr as *const _);
                let head_idx = old_head as u32;
                let tail_idx = old_tail as u32;
                if head_idx != tail_idx {
                    let slot = head_idx & (MAX_TASKS as u32 - 1);
                    let buf_ptr = self.work_queue.buffer[slot as usize].get();
                    let tid = sys_spin_load_acquire_u32(buf_ptr as *const _);
                    if tid != EMPTY_SENTINEL {
                        let new_tag = (old_head >> 32).wrapping_add(1);
                        let new_head = (new_tag << 32) | (head_idx.wrapping_add(1) as u64);
                        if sys_cas_u64(head_ptr, old_head, new_head) == old_head {
                            sys_store_release_u32(buf_ptr, EMPTY_SENTINEL);
                            task_id = tid;
                        }
                    }
                }
            }
            let task_id = shfl_sync_idx_u32(mask, task_id, 0);
            syncwarp(mask);

            if task_id == EMPTY_SENTINEL {
                break;
            }

            if task_id as usize >= MAX_TASKS {
                break;
            }

            let slot = &self.slots[task_id as usize];
            if lid == 0 {
                sys_store_release_u32(slot.state.get(), SLOT_RUNNING);
            }
            syncwarp(mask);

            let poll_fn = core::ptr::read_volatile(slot.poll_fn.get());
            if poll_fn.is_none() {
                if lid == 0 {
                    core::ptr::write_volatile(slot.state.get(), SLOT_FREE);
                }
                syncwarp(mask);
                continue;
            }
            let poll_fn = poll_fn.unwrap();
            let future_ptr = (*slot.future_bytes.get()).as_mut_ptr();

            // Inner poll loop — only lane 0 polls
            let mut polls: u32 = 0;
            loop {
                let mut is_ready: u32 = 0;
                if lid == 0 {
                    let result = poll_fn(future_ptr, &mut cx);
                    is_ready = match result {
                        Poll::Ready(()) => 1,
                        Poll::Pending => 0,
                    };
                }
                let is_ready = shfl_sync_idx_u32(mask, is_ready, 0);
                syncwarp(mask);
                polls_total += 1;
                polls += 1;

                if is_ready != 0 {
                    if lid == 0 {
                        core::ptr::write_volatile(slot.state.get(), SLOT_FREE);
                        let old = core::ptr::read_volatile(self.tasks_completed.get());
                        core::ptr::write_volatile(self.tasks_completed.get(), old.wrapping_add(1));
                    }
                    syncwarp(mask);
                    tasks_executed += 1;
                    break;
                }
                if polls >= MAX_POLLS_PER_TASK {
                    if lid == 0 {
                        core::ptr::write_volatile(slot.state.get(), SLOT_FREE);
                    }
                    syncwarp(mask);
                    break;
                }
                #[cfg(target_arch = "nvptx64")]
                core::arch::asm!("nanosleep.u32 1000;", options(nostack));
            }
        }

        ExecutorStats {
            tasks_executed,
            polls_total,
        }
    }

    /// Signal shutdown. Warps in `run()` will exit after draining the queue.
    ///
    /// # Safety
    /// Must be called by lane 0 of exactly one warp.
    #[inline(always)]
    pub unsafe fn shutdown(&self) {
        sys_store_release_u32(self.shutdown.get(), 1);
    }

    /// Get the number of tasks spawned (diagnostic).
    pub unsafe fn spawned_count(&self) -> u32 {
        core::ptr::read_volatile(self.tasks_spawned.get() as *const u32)
    }

    /// Get the number of tasks completed (diagnostic).
    pub unsafe fn completed_count(&self) -> u32 {
        core::ptr::read_volatile(self.tasks_completed.get() as *const u32)
    }
}
