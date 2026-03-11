# async-runtime.2: GPU Executor Architecture Design
**Cycle**: 16 | **Theme**: async-runtime | **Kind**: design | **Status**: done

## Summary

This document defines the GPU executor architecture for running Rust async/await on NVIDIA GPUs. The design adopts a **per-thread (per-lane) executor** model following VectorWare's proven approach, with Embassy's `arch-spin` as the base template. Each GPU thread owns a private executor instance with task storage in local/register memory, a waker based on Embassy's existing atomic-flag mechanism, and an async hostcall integration that allows GPU threads to yield to other tasks while waiting for host responses. The design minimizes register pressure by limiting the default task count per executor and deferring inter-thread communication to the hostcall layer.

## Architecture Overview

```
GPU Grid
├── Block 0
│   ├── Warp 0 (32 lanes in SIMT lockstep)
│   │   ├── Lane 0: [Executor] ─ polls ─> [Task A, Task B]
│   │   ├── Lane 1: [Executor] ─ polls ─> [Task A, Task B]
│   │   ├── ...
│   │   └── Lane 31: [Executor] ─ polls ─> [Task A, Task B]
│   ├── Warp 1
│   │   └── (same: per-lane executor)
│   └── ...
├── Block 1
│   └── (same structure)
└── ...

Per-Lane Executor (private to each GPU thread):
┌──────────────────────────────────────┐
│  GpuExecutor                         │
│  ├─ run_queue: AtomicU32 (bitmask)   │  ← Which tasks are ready to poll
│  ├─ task_count: u8                   │
│  └─ tasks: [TaskSlot; N]            │  ← Inline array, N ≤ 8
│       ├─ state: u8                   │
│       ├─ poll_fn: fn(*)              │
│       └─ future: F (erased)         │
└──────────────────────────────────────┘
         ▲                     │
         │ wake(task_id)       │ hostcall(...)
         │ sets bit in         │ returns Poll::Pending
         │ run_queue            ▼
┌──────────────────────────────────────┐
│  Hostcall Mapped Memory             │
│  (ADR-3 two-stack protocol)          │
│  GPU writes request → spins on       │
│  control.READY → waker fires         │
└──────────────────────────────────────┘
```

**Key insight**: Because each lane has its own private executor, there is **zero contention** on executor data structures. No atomic CAS needed for the run queue. No critical section needed. The only cross-thread synchronization is in the hostcall layer (already designed in ADR-3) and in any explicit inter-task channels (deferred to a later task).

## Design Decisions

### D1: Executor Granularity — Per-Thread (Per-Lane)

**Decision**: Each GPU thread (lane) runs its own independent executor instance, following VectorWare's demonstrated approach.

**Rationale**:
1. **Proven feasibility**: VectorWare explicitly confirms "each GPU thread runs its own executor instance" and demonstrates async/await working on GPU with this model.
2. **Zero synchronization overhead**: A private executor needs no atomics, no locks, no critical sections for its internal run queue. State transitions are plain register operations.
3. **Simplicity**: The execution model maps directly to Embassy's `arch-spin` — a tight spin-poll loop on a single "thread." Each GPU lane *is* that single thread.
4. **SIMT compatibility**: All 32 lanes in a warp execute the same executor poll loop. Divergence occurs only when different lanes' tasks are in different states — this is inherent to the async model and unavoidable regardless of granularity.
5. **Register pressure is manageable**: VectorWare's blog acknowledges increased register pressure but states the approach works. The executor state itself is small (~16-32 bytes); the dominant cost is the Future state machine, which exists regardless of executor granularity.

**Alternatives considered**:
- **Per-warp executor** (1 executor shared by 32 lanes): Would reduce register pressure by 32x for executor state but requires complex warp-level synchronization. Lane 0 would run the executor while other lanes idle or assist via warp intrinsics. This trades parallelism for efficiency — unsuitable because the primary goal is capability (running async code), not throughput.
- **Per-block executor** (1 executor per thread block): Even more sharing, but requires shared-memory synchronization (`__syncthreads`, spinlocks). Introduces serialization bottlenecks. Rejected.
- **Hybrid**: Per-warp executor with per-lane task pools. Too complex for the initial implementation. Could be revisited as an optimization if register pressure proves catastrophic.

### D2: Memory Layout

**Decision**: Executor state and task storage reside in **local memory (registers/stack)**, with hostcall buffers in **global mapped memory**.

**Layout**:

| Component | Memory Space | Rationale |
|-----------|-------------|-----------|
| `GpuExecutor` struct | Registers/local | Private per-thread, hot path, must be fast |
| `TaskSlot` array | Registers/local | Private per-thread, polled every iteration |
| Future state machines | Registers/local (may spill) | Compiler-generated, per-task |
| Run queue bitmask | Register | Single u32, fits in one register |
| Hostcall buffer | Global (mapped) | Shared GPU↔CPU, already in global per ADR-3 |
| Waker data (task ID) | Register | Just an index into the task array |

**Task storage strategy**: Instead of Embassy's intrusive linked-list `TaskStorage<F>` (which uses `AtomicPtr` for the run queue), use a **fixed-size inline array** of task slots. For a per-thread executor with no contention:
- No linked list needed — a simple bitmask tracks which tasks are ready.
- Array index serves as the task ID (0..N-1).
- Maximum N = 8 tasks per executor (configurable). This keeps the bitmask in a single register (`u8` or `u32`).

**Register budget estimate**:
- Executor overhead: ~4-8 registers (bitmask, task count, loop variables)
- Per task: ~2 registers for metadata (state, poll_fn pointer) + Future state machine
- A simple hostcall future: ~4-8 registers (buffer pointer, packet index, state enum)
- Total for 2 tasks: ~20-30 registers out of 255 max per thread
- At 32 registers/thread: 65536/32 = 2048 threads = 64 warps per SM (full occupancy on most GPUs)
- At 64 registers/thread: 1024 threads = 32 warps per SM (~50% occupancy, still acceptable)

**Future state machine placement**: The Rust compiler generates state machines for `async fn`. These are stack-allocated by default and will be placed in local memory. Small futures fit in registers; larger ones spill to local memory (which is backed by L1 cache on modern GPUs, ~26-cycle latency). This is acceptable — the alternative (shared memory) wastes a scarce per-block resource on per-thread data.

### D3: Waker Implementation

**Decision**: Use a **lightweight bitmask waker** that sets a bit in the executor's run-queue register. No atomic operations needed since the waker and executor run on the same thread.

**Design**:
```rust
// Waker data encodes: executor pointer + task index
// Since executor is private to the thread, we only need the task index.
//
// For intra-thread waking (the common case):
//   waker.wake() sets bit `task_id` in executor.run_queue
//
// RawWaker data layout (1 word = 64 bits on nvptx64):
//   Bits 63..8:  pointer to GpuExecutor (or unused if single executor per thread)
//   Bits  7..0:  task index (0..255, but realistically 0..7)

static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, wake, wake, drop);

unsafe fn wake(data: *const ()) {
    let task_id = data as usize & 0xFF;
    // In practice, the executor is accessed via a thread-local-like
    // mechanism (a register variable or function parameter).
    // Set the ready bit for this task.
    CURRENT_EXECUTOR.run_queue |= 1 << task_id;
}

unsafe fn clone(data: *const ()) -> RawWaker {
    RawWaker::new(data, &VTABLE)
}

unsafe fn drop(_: *const ()) {
    // No-op: task IDs are indices, no ownership
}
```

**Key properties**:
- **No atomics**: Since the waker only fires within the same thread that owns the executor, a plain `|=` suffices. No `AtomicU32` needed.
- **No heap**: Task ID is encoded directly in the waker data pointer.
- **1-word storage**: Compatible with Rust's `RawWaker` API.
- **Embassy-compatible**: This follows Embassy's pattern (1-word data, static vtable, no-op drop) but simplifies the wake path by removing the atomic enqueue.

**Cross-thread waking (for future inter-task channels)**:
If a task on Lane 0 needs to wake a task on Lane 5 (same warp), this waker design does not support it — the `|=` would write to the wrong thread's register. Cross-thread waking requires:
1. Shared-memory mailbox: each thread checks a `__shared__ u32 wake_flags[32]` array.
2. Warp shuffle: `__shfl_sync` to communicate wake signals.
3. Global-memory flag: for cross-warp waking.

**Decision**: Defer cross-thread waking to the inter-task communication design (future task). The per-thread executor + per-thread waker handles the primary use case: async hostcall where the waker fires from within the same thread's poll loop after detecting the host response.

### D4: Hostcall Integration (Async)

**Decision**: Implement hostcall as an async operation that returns `Poll::Pending` while waiting for the host response, allowing the executor to poll other tasks.

**Async hostcall flow**:
```
GPU Thread (Lane 0) Timeline:
─────────────────────────────────────────────────────
1. Task A calls `hostcall_async(PRINT, args).await`
2. HostcallFuture::poll():
   - First poll: pop free packet, fill, push to ready, ring doorbell
   - Return Poll::Pending
   - (waker stored internally — will be invoked by self on next poll)
3. Executor polls Task B (if any other task is ready)
4. Executor re-polls Task A (HostcallFuture::poll()):
   - Check control.READY flag via sys_load_acquire_u32
   - If not ready: return Poll::Pending (re-enqueue self via waker)
   - If ready: read response, push packet to free stack, return Poll::Ready
5. Task A continues after .await
─────────────────────────────────────────────────────
```

**HostcallFuture state machine**:
```rust
enum HostcallState {
    /// Initial state: need to acquire a packet and submit request
    Init { service: u32, args: [u64; 7] },
    /// Waiting for host response
    Waiting { packet_idx: u16 },
    /// Completed (terminal)
    Done,
}

struct HostcallFuture {
    buffer: *const HostcallBuffer,
    state: HostcallState,
}

impl Future for HostcallFuture {
    type Output = Result<[u64; 7], HostcallError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = unsafe { self.get_unchecked_mut() };
        match this.state {
            HostcallState::Init { service, ref args } => {
                // Pop free packet (may spin briefly if pool exhausted)
                let pkt_idx = hc_pop_free(this.buffer)?;
                // Fill packet
                hc_fill_packet(this.buffer, pkt_idx, service, args);
                // Push to ready stack + ring doorbell
                hc_push_ready(this.buffer, pkt_idx);
                hc_ring_doorbell(this.buffer);
                // Transition to waiting
                this.state = HostcallState::Waiting { packet_idx: pkt_idx };
                // Wake self so executor re-polls us
                cx.waker().wake_by_ref();
                Poll::Pending
            }
            HostcallState::Waiting { packet_idx } => {
                let control = sys_load_acquire_u32(
                    hc_control_ptr(this.buffer, packet_idx)
                );
                if control & READY_BIT != 0 {
                    // Host responded
                    let response = hc_read_response(this.buffer, packet_idx);
                    hc_push_free(this.buffer, packet_idx);
                    this.state = HostcallState::Done;
                    Poll::Ready(Ok(response))
                } else {
                    // Not ready yet — wake self for next poll cycle
                    cx.waker().wake_by_ref();
                    Poll::Pending
                }
            }
            HostcallState::Done => panic!("polled after completion"),
        }
    }
}
```

**Key design points**:
1. **Self-waking pattern**: The future wakes itself (`cx.waker().wake_by_ref()`) to ensure the executor re-polls it. This is the standard pattern for polling-based futures without external notification.
2. **Non-blocking**: The future never spins internally — it checks once and returns `Pending` if not ready. The spin happens implicitly via the executor's poll loop.
3. **Overlap opportunity**: While Task A waits for a hostcall response (~microseconds for host processing), the executor can poll Task B. This is the KEY value proposition of async on GPU — multiple hostcalls can be in flight from the same thread.
4. **Packet lifetime**: The packet is held from `Init` → `Waiting` → response read → freed. This matches the synchronous protocol but stretches the hold time across multiple poll cycles.
5. **Pool pressure**: Long-held packets reduce the free pool. With N=64 packets and potentially thousands of threads, packet exhaustion is possible. Mitigation: size the pool for the expected concurrency, or back-pressure via `Poll::Pending` when the free stack is empty (retry on next poll).

**Sync fallback**: For cases where async overhead is undesirable (single-task kernels), retain the synchronous `hostcall()` function that spins internally. The async version is opt-in.

### D5: Critical Section Strategy

**Decision**: **No critical section needed** for the per-thread executor. Provide a no-op implementation to satisfy Embassy's linker requirement.

**Analysis by scope**:

| Scope | Critical Section Need | Strategy |
|-------|----------------------|----------|
| Per-thread executor internals | None — single owner | Plain register operations |
| Embassy's `critical-section` crate dependency | Linker satisfaction only | No-op impl: `acquire()` → no-op, `release()` → no-op |
| Intra-warp communication (future) | Warp-level | `__ballot_sync` / `__shfl_sync` intrinsics |
| Inter-warp communication (future) | Block-level | Shared-memory spinlock via `gpu-atomics` |
| Inter-block communication | Grid-level | Global-memory spinlock via `gpu-atomics` sys-scope |

**No-op critical section implementation**:
```rust
// In gpu-executor crate (or gpu-kernel)
use critical_section::RawRestoreState;

struct GpuCriticalSection;

critical_section::set_impl!(GpuCriticalSection);

unsafe impl critical_section::Impl for GpuCriticalSection {
    unsafe fn acquire() -> RawRestoreState {
        // No-op: per-thread executor has no shared mutable state
        // GPU threads cannot be preempted (no interrupts to disable)
        ()
    }
    unsafe fn release(_: RawRestoreState) {
        // No-op
    }
}
```

**Rationale**: On embedded CPUs, critical sections disable interrupts to prevent preemption. GPU threads are never preempted mid-instruction — the warp scheduler only switches at warp granularity, and all 32 lanes within a warp execute the same instruction. A per-thread executor has no shared mutable state that could be corrupted by "concurrent" access. The no-op is both correct and optimal.

**When this breaks**: If Embassy's `Mutex<CriticalSection, T>` or `Channel` is used for inter-task communication across different GPU threads, the no-op critical section is WRONG — it provides no actual mutual exclusion. This will be addressed in a separate task (async-runtime.1.3) with a proper spinlock implementation. For now, the scope is limited to per-thread-private executor state.

## Executor Poll Loop

The main execution entry point for each GPU thread:

```rust
#[no_mangle]
pub unsafe extern "ptx-kernel" fn async_kernel(hostcall_buf: *mut u8) {
    // Each thread gets its own executor on the stack (→ registers/local memory)
    let mut executor = GpuExecutor::new();

    // Spawn initial tasks
    executor.spawn(my_async_task(hostcall_buf));
    executor.spawn(another_task(hostcall_buf));

    // Spin-poll loop (matches Embassy arch-spin model)
    loop {
        let did_work = executor.poll();
        if executor.all_tasks_completed() {
            break;  // Kernel exit
        }
        if !did_work {
            // All tasks are Pending, none ready to poll
            // Use nanosleep to reduce power and yield to warp scheduler
            unsafe { asm!("nanosleep.u32 100;"); }
        }
    }
}
```

**Exit condition**: Unlike Embassy on embedded (runs forever), GPU kernels must terminate. The executor tracks completed tasks and exits when all are done. This is a key divergence from Embassy's `arch-spin` which loops indefinitely.

**Nanosleep optimization**: When no tasks are ready (all waiting for hostcall responses), the thread issues a PTX `nanosleep` instruction. This:
1. Reduces power consumption
2. Hints the warp scheduler to schedule other warps
3. Avoids burning cycles on futile polls
4. Available on SM 7.0+ (our minimum target)

## Implementation Plan

### Phase 1: Minimal GpuExecutor (async-runtime.3)
1. Implement `GpuExecutor` with fixed-size task array (N=4)
2. Implement bitmask run queue (plain `u32`, no atomics)
3. Implement GPU `RawWaker` with task-index encoding
4. Implement `poll()` loop with task completion tracking
5. Test: compile a trivial `async fn` that returns `Poll::Ready` immediately
6. Verify PTX output: register usage, no unexpected extern calls

### Phase 2: Async Hostcall Future (depends on Phase 1 + hostcall.4)
1. Implement `HostcallFuture` with the 3-state machine above
2. Wire into the existing hostcall protocol (gpu-protocol crate)
3. Test: `async fn` that prints via async hostcall, executor polls it to completion
4. Test: two tasks issuing hostcalls concurrently from the same thread

### Phase 3: Embassy Integration Path (depends on async-runtime.1.2, 1.3)
1. If LTO resolves Embassy cross-crate calls → use Embassy directly with no-op critical section
2. If not → use the custom GpuExecutor from Phase 1 as the primary executor
3. Embassy compatibility: ensure `GpuExecutor` can accept Embassy-style tasks (same waker vtable)
4. Evaluate register pressure difference between custom executor and Embassy

### Phase 4: Optimization (future)
1. Measure register pressure with `ptxas -v` and `cuobjdump`
2. If occupancy < 25%: reduce task count, simplify future state machines
3. Implement `nanosleep` backoff with adaptive delay
4. Profile async vs sync hostcall latency
5. Consider warp-cooperative optimizations (elected-lane executor) if needed

## Risk Assessment

### R1: Register Pressure Kills Occupancy
**Risk**: High
**Impact**: If the executor + future state machines consume >64 registers per thread, occupancy drops below 25%, and the GPU is severely underutilized.
**Mitigation**:
- Default to N=2 tasks per executor (minimal overhead)
- Use `#[inline(always)]` aggressively to avoid function-call register spills
- Measure early with `ptxas -v --register-usage`
- Fallback: reduce to N=1 (essentially `block_on` — still useful for async hostcall)
- Nuclear option: per-warp executor where only lane 0 carries executor state

### R2: Warp Divergence Degrades Throughput
**Risk**: Medium
**Impact**: Different lanes at different await points cause serialized execution within the warp.
**Mitigation**:
- This is inherent to the async model and acknowledged by VectorWare
- For uniform workloads (all threads doing the same async operations in lockstep), divergence is minimal
- For heterogeneous workloads, divergence is the cost of flexibility
- Mitigation: structure kernels so that all lanes execute the same async fn with the same control flow

### R3: Packet Pool Exhaustion Under Async
**Risk**: Medium
**Impact**: Synchronous hostcall holds a packet for one spin-wait. Async hostcall holds a packet across multiple poll cycles (potentially much longer). With thousands of threads, the pool may be exhausted.
**Mitigation**:
- Size pool proportionally to expected concurrent async hostcalls
- Back-pressure: `HostcallFuture::poll()` returns `Pending` when free pool is empty, retries next poll cycle
- Limit concurrent hostcalls per thread (e.g., max 1 outstanding at a time)
- Monitor and warn if packet exhaustion is frequent

### R4: Self-Waking Overhead
**Risk**: Low
**Impact**: The `cx.waker().wake_by_ref()` pattern on every `Pending` return is a function pointer call through the vtable. On GPU, indirect calls may be expensive if they prevent inlining.
**Mitigation**:
- Mark waker functions `#[inline(always)]`
- For the bitmask waker, the wake operation is a single `OR` instruction
- If vtable dispatch is problematic, bypass `RawWaker` entirely and set the bit directly

### R5: Future State Machine Size Unknown
**Risk**: Medium
**Impact**: Complex `async fn` bodies generate large state machines that spill from registers to local memory (global memory backed by L1 cache), increasing latency.
**Mitigation**:
- Keep GPU async functions small and shallow (few await points)
- Avoid deep async call chains
- Measure with `sizeof::<impl Future>()` equivalent in PTX
- Split large futures into smaller helper functions to reduce captured state

## Open Questions

1. **LTO for Embassy**: Does `-C lto=fat` resolve `Executor::spawn` / `Executor::poll` cross-crate extern calls? This determines whether we use Embassy directly or the custom `GpuExecutor`. (Tracked by async-runtime.1.2)

2. **`TaskPool` vs inline array**: Embassy uses `static TaskPool<F, N>` which lives in global memory. Our inline array lives on the stack (registers/local). Is there a correctness issue with stack-allocated futures being passed to wakers that store raw pointers? The pointer must remain valid for the executor's lifetime — guaranteed since the executor owns the array and lives on the same stack frame.

3. **Warp-cooperative hostcall**: The current hostcall protocol (ADR-3) allocates one packet per warp, but only the calling lane fills its slots. In the async model, different lanes may issue hostcalls at different times. Should each lane get its own packet (wastes 31/32 of payload), or should lanes cooperate to share a packet (complex coordination)?

4. **Nanosleep duration tuning**: What is the optimal nanosleep duration for the idle-poll case? Too short: wasted energy. Too long: increased latency when host responds. Needs empirical measurement.

5. **Type erasure for task array**: Embassy uses `TaskStorage<F>` with type-erased futures (stores `unsafe fn(TaskRef)` poll function). For our inline array, each slot needs to hold a different future type. Options: (a) all tasks must be the same type (limiting), (b) enum of known future types, (c) type-erased trait object with `dyn Future` (requires vtable in global memory), (d) separate typed arrays per future type (Embassy's approach). Decision deferred to implementation.

6. **Kernel exit synchronization**: When one lane's tasks complete but others are still running, the completed lane must wait (GPU threads in a block must exit together). Use `nanosleep` loop or `bar.sync` to wait for all lanes to finish.

## Impact on Downstream Tasks

| Task | Impact |
|------|--------|
| **async-runtime.3** (minimal async on GPU) | Direct implementation target: build `GpuExecutor` per this design |
| **async-runtime.1.2** (LTO test) | Determines Embassy vs custom executor path |
| **async-runtime.1.3** (critical section) | No-op CS is sufficient for per-thread executor; spinlock needed only for inter-thread sync |
| **gpu-std.2** (libc shim) | Libc calls become async hostcalls driven by this executor |
| **integration.1** (end-to-end) | This executor is the runtime foundation for async std on GPU |
| **hostcall.4** (existing sync impl) | Async `HostcallFuture` wraps the same protocol, reusing gpu-protocol crate |

## ADR

This design introduces **ADR-4** (see decisions.md).
