# Design: GPU-Side Async Task Spawning Executor

## 1. Problem Statement

The current async_gpu executor is strictly **one-future-per-warp**: `block_on()` and `SpinExecutor::run()` each spin-poll a single `Future` until completion, with no ability to launch additional work dynamically. This is sufficient for linear pipelines (open -> read -> process -> write) but inadequate for three important classes of GPU workloads:

**Data-dependent parallelism.** A kernel reads an index file, discovers N items, and wants to process each concurrently. Today, N must be known at kernel launch time (grid dimensions are fixed). With `spawn()`, the kernel can create tasks dynamically as data arrives.

**GPU-driven servers.** A GPU kernel running `tcp_accept()` in a loop wants to spawn a handler task for each incoming connection. Without dynamic task spawning, the kernel must serialize connection handling or pre-allocate a fixed number of handler warps.

**Recursive/graph algorithms.** BFS, tree traversal, and divide-and-conquer algorithms naturally produce work items during execution. Dynamic spawning maps directly to this pattern.

### Why not just launch more warps at host level?

Host-driven approaches require a GPU-host round-trip for every scheduling decision (~10-50us via hostcall). GPU-side spawning keeps scheduling on-device with sub-microsecond CAS latency, enabling fine-grained task parallelism that would be impractical with host involvement.

## 2. Architecture

### High-Level Design

```
                    GPU Global Memory
    ┌──────────────────────────────────────────────┐
    │  GpuExecutor (fixed at kernel launch)        │
    │  ┌────────────────────────────────────────┐  │
    │  │ Work Queue (lock-free MPMC)            │  │
    │  │   head: AtomicU64 (tagged)             │  │
    │  │   tail: AtomicU64 (tagged)             │  │
    │  │   slots: [TaskSlot; MAX_TASKS]         │  │
    │  ├────────────────────────────────────────┤  │
    │  │ Task Storage (bump-allocated)          │  │
    │  │   [TaskHeader + Future bytes] ...      │  │
    │  ├────────────────────────────────────────┤  │
    │  │ Executor State                         │  │
    │  │   active_warps: AtomicU32              │  │
    │  │   shutdown: AtomicU32                  │  │
    │  │   tasks_spawned: AtomicU64             │  │
    │  │   tasks_completed: AtomicU64           │  │
    │  └────────────────────────────────────────┘  │
    └──────────────────────────────────────────────┘
            ▲               ▲               ▲
            │               │               │
       ┌────┴────┐    ┌────┴────┐    ┌────┴────┐
       │ Warp 0  │    │ Warp 1  │    │ Warp 2  │
       │ (poll)  │    │ (steal) │    │ (idle)  │
       └─────────┘    └─────────┘    └─────────┘
```

### Executor State Machine (per warp)

```
         ┌─────────┐
         │  IDLE    │◄──────────────────────┐
         └────┬────┘                        │
              │ try_dequeue()               │
              ▼                             │
         ┌─────────┐    Poll::Ready    ┌────┴────┐
         │ RUNNING ├──────────────────►│COMPLETE │
         └────┬────┘                   └─────────┘
              │ Poll::Pending
              ▼
         ┌─────────┐
         │ YIELDED │── nanosleep ──► RUNNING (re-poll)
         └─────────┘
```

Each warp runs this loop:
1. Dequeue a task from the work queue (CAS on head)
2. Pin the task's future in place
3. Spin-poll the future (same pattern as current `block_on`)
4. On completion, mark the task slot as free, increment `tasks_completed`
5. Go back to step 1

### Memory Layout

The executor occupies a single contiguous allocation in CUDA mapped memory (or device global memory), set up by the host before kernel launch:

```
Offset  Size       Content
0       64         Executor header (queue head/tail, counters, flags)
64      N*16       TaskSlot array (N = MAX_TASKS, e.g. 256)
64+N*16 M          Task storage arena (bump-allocated future bodies)
```

## 3. Work Queue Design

### Lock-Free MPMC Queue

The queue uses the same tagged-CAS pattern proven in the hostcall protocol's free/ready stacks. However, instead of a stack (LIFO), we use a bounded MPMC queue (FIFO) for fairness.

**Why not reuse the existing stack pattern directly?** Stacks are LIFO, which can starve early-spawned tasks. A FIFO queue ensures tasks are processed roughly in spawn order, which matters for server workloads (first connection should be handled first).

### Queue Structure

```rust
#[repr(C, align(64))]
pub struct WorkQueue {
    /// Head index (consumers dequeue here). Tagged to prevent ABA.
    /// Bits 63-32: monotonic tag, Bits 31-0: slot index
    head: AtomicU64,

    _pad0: [u8; 56],  // cache-line separation

    /// Tail index (producers enqueue here). Tagged to prevent ABA.
    tail: AtomicU64,

    _pad1: [u8; 56],

    /// Bounded circular buffer of task slot indices.
    /// Each entry is a u32 index into the TaskSlot array.
    /// EMPTY_SENTINEL (0xFFFFFFFF) means the slot is not occupied.
    buffer: [AtomicU32; MAX_TASKS],
}
```

### Enqueue (spawn side)

```
1. old_tail = load_acquire(tail)
2. old_head = load_acquire(head)
3. if (old_tail - old_head) >= MAX_TASKS: return Err(QueueFull)
4. slot_idx = old_tail & (MAX_TASKS - 1)    // power-of-2 wrap
5. CAS(tail, old_tail, old_tail + 1)        // with tag increment
6. if CAS failed: goto 1 (retry)
7. store_release(buffer[slot_idx], task_id)
```

### Dequeue (warp executor side)

```
1. old_head = load_acquire(head)
2. old_tail = load_acquire(tail)
3. if old_head == old_tail: return None      // queue empty
4. slot_idx = old_head & (MAX_TASKS - 1)
5. task_id = load_acquire(buffer[slot_idx])
6. if task_id == EMPTY_SENTINEL: nanosleep + goto 1  // producer hasn't written yet
7. CAS(head, old_head, old_head + 1)
8. if CAS failed: goto 1 (retry)
9. store_release(buffer[slot_idx], EMPTY_SENTINEL)   // clear slot
10. return Some(task_id)
```

### ABA Prevention

Both `head` and `tail` use the upper 32 bits as a monotonically increasing tag, identical to the hostcall protocol's tagged pointers. Since the tag increments on every CAS, and the buffer is bounded, ABA cannot occur within 2^32 operations per index position.

### Contention Mitigation

With many warps competing on `head`, CAS contention could spike. Mitigations:

1. **Exponential backoff**: After a failed CAS, `nanosleep` with doubling delay (64ns, 128ns, ..., capped at 1000ns).
2. **Per-block local queues** (future optimization): Each block maintains a small local queue. Warps dequeue locally first, falling back to the global queue. This mirrors the hostcall protocol's per-block sharding.

## 4. Scheduling Policy

### Task Claiming

Only **lane 0** of each warp performs the dequeue CAS. The resulting task ID is broadcast to all lanes via `shfl.sync.idx.b32`. This ensures:
- Minimal CAS contention (one CAS per warp, not per thread)
- All lanes converge on the same task (warp-cooperative invariant)

```rust
// Pseudocode — lane 0 dequeues, broadcasts to all
let mask = activemask();
let lid = lane_id();

let mut task_id: u32 = EMPTY_SENTINEL;
if lid == 0 {
    task_id = work_queue.try_dequeue();
}
let task_id = shfl_sync_idx_u32(mask, task_id, 0);
syncwarp(mask);
```

### Empty Queue Behavior

When the queue is empty, a warp has three options (configurable via policy):

1. **Spin-wait** (default): `nanosleep` loop checking the queue. Good when new tasks are expected soon (server workloads).
2. **Exit**: The warp decrements `active_warps` and returns. The last warp to exit triggers kernel completion. Good for batch workloads.
3. **Cooperative yield**: The warp polls the queue with long nanosleep intervals (1000ns+), minimizing power consumption while remaining available. Good for mixed workloads.

Policy is set at executor creation:

```rust
pub enum IdlePolicy {
    /// Spin-wait with nanosleep until new work arrives or shutdown
    SpinWait,
    /// Exit immediately when queue is empty
    ExitOnEmpty,
    /// Poll with long sleep intervals
    CooperativeYield { sleep_ns: u32 },
}
```

### Shutdown Protocol

1. Producer sets `shutdown` flag via `store_release`
2. Consumer warps see shutdown, drain remaining queue entries, then exit
3. Last warp (detected by `fetch_sub` on `active_warps` returning 1) performs final cleanup

## 5. Lifetime & Memory

### The Core Challenge

Spawned futures must be stored somewhere in GPU memory. Unlike CPU runtimes (tokio, async-std) which use `Box<dyn Future>` with heap allocation, GPU constraints are severe:

- **No `dyn` dispatch on nvptx64**: Trait objects require vtables, which work in theory but LLVM's nvptx backend has historically been unreliable with indirect calls.
- **No `free()`**: The existing bump allocator never frees. Task storage that is never reclaimed limits total spawnable tasks.
- **No thread-local storage**: `#[thread_local]` is not supported on nvptx64.

### Approach: Type-Erased Task Slots with Arena Allocation

Each task is stored in a fixed-size arena slot. The future is type-erased via a manually constructed vtable (no `dyn` — just a function pointer stored alongside the future bytes).

```rust
#[repr(C, align(64))]
pub struct TaskSlot {
    /// Task state: FREE / QUEUED / RUNNING / COMPLETED
    state: AtomicU32,
    /// Pointer to the poll function (type-erased).
    /// Signature: unsafe fn(future_ptr: *mut u8, cx: &mut Context) -> Poll<()>
    poll_fn: unsafe fn(*mut u8, &mut Context<'_>) -> Poll<()>,
    /// Size of the future in bytes (for debugging/validation)
    future_size: u32,
    /// Padding to cache-line boundary
    _pad: [u8; 44],
    /// Inline storage for the future (fixed max size)
    future_bytes: [u8; TASK_FUTURE_MAX_SIZE],
}
```

`TASK_FUTURE_MAX_SIZE` is a compile-time constant (e.g., 512 bytes). Futures larger than this cannot be spawned. This is acceptable because:
- Most I/O futures in async_gpu are small (GpuOpenFuture is ~48 bytes)
- Composed futures (chains of .await) are larger but typically under 256 bytes
- A 512-byte limit with 256 slots = 128KB total, reasonable for GPU global memory

### spawn() Implementation

```rust
pub unsafe fn spawn<F: Future<Output = ()> + Send>(
    executor: *mut GpuExecutor,
    future: F,
) -> Result<TaskId, SpawnError> {
    // 1. Validate size
    if core::mem::size_of::<F>() > TASK_FUTURE_MAX_SIZE {
        return Err(SpawnError::FutureTooLarge);
    }

    // 2. Allocate a task slot (CAS on free list or bump pointer)
    let slot_id = alloc_task_slot(executor)?;

    // 3. Copy future bytes into the slot
    let slot = &mut (*executor).task_slots[slot_id];
    core::ptr::copy_nonoverlapping(
        &future as *const F as *const u8,
        slot.future_bytes.as_mut_ptr(),
        core::mem::size_of::<F>(),
    );
    core::mem::forget(future); // ownership transferred to slot

    // 4. Set the type-erased poll function
    slot.poll_fn = erased_poll::<F>;

    // 5. Mark slot as QUEUED
    slot.state.store(TASK_QUEUED, Ordering::Release);

    // 6. Enqueue task_id into the work queue
    (*executor).work_queue.enqueue(slot_id)?;

    Ok(TaskId(slot_id))
}

/// Type-erased poll trampoline
unsafe fn erased_poll<F: Future<Output = ()>>(
    ptr: *mut u8,
    cx: &mut Context<'_>,
) -> Poll<()> {
    let future = &mut *(ptr as *mut F);
    Pin::new_unchecked(future).poll(cx)
}
```

### Task Slot Recycling

When a task completes, its slot transitions to `FREE` state and is pushed onto a free-slot stack (another tagged CAS stack, same pattern as hostcall free packets). This allows slot reuse without a general-purpose allocator.

```
spawn() ──► QUEUED ──► RUNNING ──► COMPLETED ──► FREE
                 ▲                                  │
                 └──────────────────────────────────┘
                         (slot recycled)
```

### Pinning Guarantees

Once a future is copied into a `TaskSlot`, it is never moved — the slot has a fixed address in global memory. This satisfies `Pin`'s contract. The `erased_poll` function constructs `Pin::new_unchecked` each time, which is safe because the future's address is stable for its entire lifetime.

## 6. Integration with MIR Pass

### How `#[warp_cooperative]` Applies to Spawned Tasks

The `#[warp_cooperative]` MIR pass transforms `async fn` at compile time by inserting `bar.warp.sync` at yield points and broadcasting the state discriminant from lane 0. This transformation is independent of the executor — it modifies the generated `Future::poll()` implementation itself.

For spawned tasks, this means:
- The future stored in a `TaskSlot` already contains the warp synchronization instructions (they were inserted by the MIR pass at compile time)
- The executor does not need to add any synchronization — it just calls `poll()` as normal
- All 32 lanes of the warp call `erased_poll()` together; the embedded `bar.warp.sync` keeps them converged

### Constraint: All Lanes Must Poll the Same Task

This is the most critical invariant. When a warp dequeues a task, **all 32 lanes must participate in polling that task**. The dequeue protocol (section 4) ensures this by having lane 0 dequeue and broadcast. But the executor loop must also maintain this invariant:

```rust
// Executor main loop (all 32 lanes execute this)
loop {
    let task_id = warp_dequeue(&work_queue);  // lane 0 dequeues, broadcasts

    if task_id == EMPTY_SENTINEL {
        match idle_policy {
            IdlePolicy::ExitOnEmpty => break,
            _ => { nanosleep(); continue; }
        }
    }

    // All 32 lanes poll the same task
    let slot = &mut task_slots[task_id as usize];
    let poll_fn = slot.poll_fn;
    let future_ptr = slot.future_bytes.as_mut_ptr();

    let waker = noop_waker();
    let mut cx = Context::from_waker(&waker);

    loop {
        let result = poll_fn(future_ptr, &mut cx);  // all lanes call this
        match result {
            Poll::Ready(()) => {
                // Task complete — recycle slot (lane 0 only)
                if lane_id() == 0 {
                    recycle_slot(executor, task_id);
                }
                syncwarp(activemask());
                break;
            }
            Poll::Pending => {
                nanosleep();
            }
        }
    }
}
```

### Limitation: Heterogeneous Lane Behavior

Standard `core::future::Future` is per-thread — each lane could in principle be polling a different future. But warp-cooperative execution requires all lanes to execute the same code. This means:

- **One task per warp**: A warp cannot split into sub-groups working on different tasks.
- **Lane-divergent futures are not supported**: If a future branches differently per lane (e.g., `if lane_id() == 0 { ... }`), the MIR pass handles convergence, but the task must still be uniform across the warp.

This is a fundamental constraint of SIMT architecture, not a limitation of the executor design.

## 7. API Surface

### Core Types

```rust
/// Error type for executor operations.
#[derive(Debug)]
pub enum ExecutorError {
    /// Work queue is full (MAX_TASKS tasks already enqueued).
    QueueFull,
    /// No free task slots available.
    NoFreeSlots,
    /// Future exceeds TASK_FUTURE_MAX_SIZE bytes.
    FutureTooLarge,
    /// Executor has been shut down.
    Shutdown,
}

/// Handle to a spawned task.
#[derive(Clone, Copy, Debug)]
pub struct TaskId(u32);

/// Idle policy when no tasks are available.
pub enum IdlePolicy {
    SpinWait,
    ExitOnEmpty,
    CooperativeYield { sleep_ns: u32 },
}
```

### GpuExecutor

```rust
/// GPU-side async task executor with work-stealing.
///
/// Allocated in global memory by the host. Warps call `run()` to enter
/// the executor loop, and `spawn()` to enqueue new tasks.
#[repr(C, align(64))]
pub struct GpuExecutor {
    work_queue: WorkQueue,
    task_slots: [TaskSlot; MAX_TASKS],
    free_slot_stack: AtomicU64,  // tagged, same pattern as hostcall
    active_warps: AtomicU32,
    shutdown: AtomicU32,
    tasks_spawned: AtomicU64,
    tasks_completed: AtomicU64,
    idle_policy: IdlePolicy,
}

impl GpuExecutor {
    /// Spawn a new async task onto the executor.
    ///
    /// The future is copied into a task slot and enqueued for execution.
    /// Any warp currently in `run()` may pick it up.
    ///
    /// # Safety
    /// - `self` must point to valid executor memory in global/mapped space
    /// - The future must be safe to poll from any warp
    /// - The future must not exceed TASK_FUTURE_MAX_SIZE bytes
    pub unsafe fn spawn<F: Future<Output = ()>>(
        &self,
        future: F,
    ) -> Result<TaskId, ExecutorError>;

    /// Enter the executor loop.
    ///
    /// The calling warp dequeues and executes tasks until shutdown or the
    /// idle policy triggers exit. All 32 lanes of the warp must call this.
    ///
    /// # Safety
    /// - Must be called by all active lanes of a warp simultaneously
    /// - `self` must point to valid executor memory
    pub unsafe fn run(&self) -> ExecutorStats;

    /// Run a single future to completion, then enter the executor loop
    /// to help process spawned tasks.
    ///
    /// Equivalent to: spawn(future); run();
    ///
    /// # Safety
    /// Same as `run()`.
    pub unsafe fn block_on<F: Future<Output = ()>>(
        &self,
        future: F,
    ) -> ExecutorStats;

    /// Signal shutdown. Warps in `run()` will drain remaining tasks and exit.
    ///
    /// # Safety
    /// Must be called by lane 0 of exactly one warp.
    pub unsafe fn shutdown(&self);
}

/// Statistics returned when a warp exits the executor loop.
pub struct ExecutorStats {
    pub tasks_executed: u32,
    pub polls_total: u64,
}
```

### GpuChannel

For inter-task communication, a simple single-producer single-consumer (SPSC) channel:

```rust
/// Single-value channel for communicating between GPU tasks.
///
/// One task sends a value, another task awaits it. Built on the same
/// atomic CAS pattern as the hostcall protocol.
pub struct GpuChannel<T: Copy> {
    value: core::cell::UnsafeCell<core::mem::MaybeUninit<T>>,
    state: AtomicU32,  // 0 = empty, 1 = filled, 2 = closed
}

impl<T: Copy> GpuChannel<T> {
    /// Send a value. Returns Err if the channel is already filled or closed.
    pub unsafe fn send(&self, value: T) -> Result<(), T>;

    /// Returns a future that resolves when a value is available.
    pub fn recv(&self) -> GpuRecvFuture<'_, T>;
}

/// Future returned by GpuChannel::recv().
///
/// Polls the channel's state atomically. When the state transitions
/// to FILLED, reads the value and returns Poll::Ready.
pub struct GpuRecvFuture<'a, T: Copy> {
    channel: &'a GpuChannel<T>,
}

impl<T: Copy> Future for GpuRecvFuture<'_, T> {
    type Output = Option<T>;  // None if channel closed without sending
    // ...
}
```

### Usage Example

```rust
#[warp_cooperative]
async fn server_loop(executor: &GpuExecutor, buf: *mut u8, listener_fd: i32) {
    loop {
        // Accept a connection (warp-cooperative await)
        let client_fd = GpuTcpAcceptFuture::new(buf, listener_fd).await;
        let client_fd = match client_fd {
            Ok(fd) => fd,
            Err(_) => break,
        };

        // Spawn a handler task for this connection
        let _ = executor.spawn(handle_client(buf, client_fd));
    }
}

#[warp_cooperative]
async fn handle_client(buf: *mut u8, fd: i32) {
    let mut data = [0u8; 256];
    let n = GpuTcpReadFuture::new(buf, fd, &mut data).await.unwrap_or(0);
    if n > 0 {
        GpuTcpWriteFuture::new(buf, fd, &data[..n]).await.ok();
    }
    GpuTcpCloseFuture::new(buf, fd).await.ok();
}
```

## 8. Open Problems

### 8.1 Indirect Function Calls on nvptx64

The type-erased `poll_fn` approach requires indirect function calls (`call.uni` in PTX). LLVM's nvptx backend supports this, but:
- Indirect calls prevent inlining, which is how current futures achieve zero-cost abstractions
- The compiler may not be able to prove convergence properties through an indirect call
- **Untested assumption**: `#[warp_cooperative]` MIR transformations in the callee should still work through indirect dispatch, but this needs hardware validation

**Mitigation**: If indirect calls prove problematic, an alternative is an enum-based dispatcher where all spawnable future types are known at compile time:
```rust
enum AnyTask {
    HandleClient(HandleClientFuture),
    ProcessItem(ProcessItemFuture),
    // ... all types enumerated
}
```
This loses generality but avoids indirect calls entirely.

### 8.2 Future Size Limits

The fixed `TASK_FUTURE_MAX_SIZE` is a hard limit. Deeply nested async state machines can exceed it. We need to:
- Profile typical future sizes in real workloads
- Consider a tiered system: small (128B), medium (512B), large (2KB) slots
- Determine if the compiler can report future sizes to give better error messages

### 8.3 Memory Reclamation

The free-slot-stack approach recycles `TaskSlot` entries, but the future bytes within a slot are simply overwritten. If a future's `Drop` implementation has side effects (e.g., closing a file descriptor), we need to call `drop_in_place` before recycling. This requires storing a drop function pointer alongside `poll_fn`:
```rust
drop_fn: Option<unsafe fn(*mut u8)>,  // calls drop_in_place::<F>
```

**Open question**: Do any current async_gpu futures have meaningful `Drop` impls? The existing I/O futures (GpuOpenFuture, etc.) appear to be trivially droppable (raw pointers + enums).

### 8.4 Priority and Fairness

The current FIFO queue provides basic fairness but no priority. Some tasks (e.g., connection accept) might need higher priority than data processing. Options:
- Multiple queues (high/low priority) with weighted dequeuing
- Priority field in `TaskSlot` with a sorted insertion (expensive on GPU)
- Accept FIFO as sufficient for v1

### 8.5 Stack Overflow Risk

Each spawned task runs within the calling warp's stack context. Deeply recursive spawn chains (task A spawns B, B spawns C, ...) don't consume extra stack because `spawn()` just enqueues — the tasks run independently. But a single task with deep call stacks could overflow the GPU thread's limited stack space (~1KB default).

This is an existing problem (not specific to the executor), but dynamic spawning makes it more likely that users will hit it.

### 8.6 Warp Utilization

The one-task-per-warp model means 32 threads are dedicated to a single task, even if that task only needs lane 0 (e.g., a sequential I/O operation). Lanes 1-31 spin in `bar.warp.sync` barriers doing nothing useful.

**Future direction**: Allow lane-divergent execution where each lane runs an independent task (without `#[warp_cooperative]`). This would require a fundamentally different executor model and is out of scope for v1.

### 8.7 Deadlock Potential

If all warp executor slots are occupied by tasks that have spawned children and are awaiting their results (via `GpuChannel`), but no warps are free to execute the children, the system deadlocks. This is the classic thread-pool exhaustion problem.

**Mitigations**:
- Document the risk clearly
- Provide `executor.capacity()` so tasks can check before spawning
- Consider a "inline execution" mode where `spawn()` falls back to synchronous execution when the queue is full

### 8.8 Hostcall Buffer Contention

More concurrent tasks means more concurrent hostcall requests. The existing hostcall protocol uses per-block sharding to reduce contention, but dynamic task spawning could overwhelm the packet pool. The packet count is fixed at kernel launch.

**Mitigation**: The executor's `MAX_TASKS` should be sized relative to the hostcall packet count. A good rule of thumb: `MAX_TASKS <= packet_count * 2` (since tasks alternate between computing and waiting for hostcall responses).

## 9. Comparison with CPU-Side Runtimes

| Aspect | tokio | async-std | Embassy | GpuExecutor (this design) |
|--------|-------|-----------|---------|---------------------------|
| **Target** | Multi-core CPU | Multi-core CPU | Embedded (single-core) | GPU (SIMT warps) |
| **Task storage** | `Box<dyn Future>` on heap | `Box<dyn Future>` on heap | Static slots (compile-time) | Fixed-size arena slots |
| **Scheduling** | Work-stealing deques | Work-stealing deques | Cooperative run queue | Lock-free MPMC queue |
| **Waker** | Epoll/kqueue notification | Epoll/kqueue notification | Interrupt-driven | No-op (spin-poll) |
| **Dynamic dispatch** | Yes (trait objects) | Yes (trait objects) | No (monomorphized) | Type-erased fn ptr |
| **Concurrency unit** | OS thread | OS thread | Single core | Warp (32 lanes) |
| **Preemption** | Cooperative (yield points) | Cooperative | Cooperative | Cooperative (await points) |
| **Memory reclaim** | Full (Drop + dealloc) | Full (Drop + dealloc) | Static (no reclaim) | Slot recycling (no dealloc) |

### Key Differences

**No Waker mechanism.** CPU runtimes use Wakers to avoid busy-waiting: a task registers interest in an I/O event, and the OS notifies the runtime when data is ready. On GPU, there is no OS — the hostcall protocol is the only I/O path, and it uses polling (check the `CONTROL_READY` flag). Adding a Waker mechanism would require the host to write directly to GPU memory to wake specific tasks, which could be explored in the future but adds complexity for v1.

**Warp as scheduling unit.** CPU runtimes schedule individual tasks onto individual threads. The GpuExecutor schedules individual tasks onto entire warps (32 threads). This is forced by SIMT architecture: all 32 lanes must execute the same instruction. There is no way to have lane 0 run task A while lane 1 runs task B (they share a program counter).

**Closest analog: Embassy.** Embassy's static-slot, no-alloc, cooperative model is the closest match. Both use fixed task storage, spin-polling, and cooperative scheduling. The main difference is that Embassy targets single-core embedded, while GpuExecutor targets massively parallel SIMT. Embassy's `#[embassy_executor::task]` attribute is analogous to our requirement that spawnable tasks must fit in `TASK_FUTURE_MAX_SIZE` bytes.

## Appendix A: Sizing Guidelines

| Configuration | MAX_TASKS | TASK_FUTURE_MAX_SIZE | Total Memory | Use Case |
|---------------|-----------|----------------------|--------------|----------|
| Minimal | 32 | 256 B | ~12 KB | Simple pipelines |
| Default | 256 | 512 B | ~136 KB | Server workloads |
| Large | 1024 | 512 B | ~528 KB | Data-parallel batch |
| Max | 4096 | 1024 B | ~4.2 MB | Heavy graph algorithms |

These sizes are well within GPU global memory budgets (typically 4-48 GB).

## Appendix B: Implementation Phases

**Phase 1** (MVP): Single global work queue, fixed-size slots, `ExitOnEmpty` policy only. Prove that indirect `poll_fn` calls work on nvptx64 with `#[warp_cooperative]`.

**Phase 2** (Channels): Add `GpuChannel` for inter-task communication. Add `SpinWait` idle policy.

**Phase 3** (Optimization): Per-block local queues (sharding). Task slot size tiers. Priority support.

**Phase 4** (Advanced): Waker-like mechanism where host completion of a hostcall directly enqueues the waiting task. Would eliminate spin-polling for I/O-bound tasks.
