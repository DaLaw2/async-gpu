# async-runtime.2.1: GPU Executor Architecture Design (Rework)
**Cycle**: 19 | **Theme**: async-runtime | **Kind**: design | **Status**: done

## Summary

This document reworks the GPU executor design (ADR-4) based on rv3 review findings. The core architecture remains sound — per-thread executor, bitmask run queue, no-op critical section — but seven issues are resolved:

1. **Embassy-first phasing**: Embassy with fat LTO is Phase 1 (confirmed working by async-runtime.1.2). Custom GpuExecutor is Phase 2, built only if register pressure demands it.
2. **Per-lane packets for async hostcall**: Each lane independently acquires its own packet. Pool must be sized at 32x compared to synchronous warp-cooperative mode. Warp-cooperative async is deferred as a Phase 4 optimization.
3. **Poll-all-tasks model**: Self-waking is eliminated. The executor unconditionally polls all non-completed tasks every cycle, matching Embassy's `arch-spin` behavior. Nanosleep triggers only when all tasks are completed (pre-exit) or if an explicit "all waiting" heuristic is added later.
4. **Pool exhaustion via Pending**: When the free packet pool is empty, `HostcallFuture` returns `Poll::Pending` instead of propagating a hard error. The next poll cycle retries the allocation.
5. **Stack-local executor reference**: No `CURRENT_EXECUTOR` global. The executor pointer is passed as a function argument through the call stack. The waker encodes only the task index; the executor is always the caller.
6. **CS misuse warning**: No-op critical section includes a doc-level warning and `#[cfg]` guard recommendation against inter-thread use.
7. **Early register pressure validation**: Measurement via `ptxas -v` happens immediately after Phase 1 (Embassy integration), not deferred to Phase 4.

## Architecture Overview

```
GPU Grid
├── Block 0
│   ├── Warp 0 (32 lanes in SIMT lockstep)
│   │   ├── Lane 0: [Embassy Executor] ─ polls ─> [Task A, Task B]
│   │   ├── Lane 1: [Embassy Executor] ─ polls ─> [Task A, Task B]
│   │   ├── ...
│   │   └── Lane 31: [Embassy Executor] ─ polls ─> [Task A, Task B]
│   ├── Warp 1
│   │   └── (same: per-lane executor)
│   └── ...
└── ...

Per-Lane Executor (private to each GPU thread):
┌──────────────────────────────────────────────┐
│  Embassy Executor (via arch-spin + fat LTO)  │
│  ├─ run_queue: AtomicPtr (task linked list)  │  ← Embassy internals
│  ├─ TaskPool<F, N>                           │  ← Typed task storage
│  └─ __pender: no-op                         │  ← GPU: nothing to wake
└──────────────────────────────────────────────┘
         ▲                         │
         │ wake(task_id)           │ hostcall_async(...)
         │ (embassy internal)      │ returns Poll::Pending
         │                         ▼
┌──────────────────────────────────────────────┐
│  Hostcall Mapped Memory (per-lane packets)   │
│  GPU writes request → polls control.READY    │
│  Pool sized for per-lane concurrency         │
└──────────────────────────────────────────────┘
```

**Key change from v1**: Embassy is the primary executor, not a fallback. Fat LTO resolves all cross-crate calls (async-runtime.1.2). The custom GpuExecutor from the original design is retained only as a measured fallback if Embassy's register pressure proves unacceptable.

## Design Decisions

### D1: Executor Granularity — Per-Thread (Per-Lane) [Unchanged]

**Decision**: Each GPU thread (lane) runs its own independent executor instance.

**Rationale**: Unchanged from async-runtime.2. VectorWare confirms this model. Zero synchronization overhead. Maps directly to Embassy's `arch-spin`.

### D2: Embassy as Primary Executor [NEW — addresses rv3 Issue 5]

**Decision**: Use Embassy's executor directly via `arch-spin` feature + `lto = "fat"`. The custom `GpuExecutor` is the fallback, built only if register pressure exceeds acceptable thresholds.

**Rationale**:
1. **LTO confirmed**: async-runtime.1.2 proved that fat LTO resolves ALL Embassy cross-crate calls — `Executor::spawn`, `Executor::poll`, `panic_fmt`, `ARENA`, `critical_section_acquire/release`. Zero unresolved externs after providing `__pender`.
2. **No fork needed**: Embassy works unmodified. No vendoring, no patching.
3. **Type erasure solved**: Embassy's `TaskPool<F, N>` handles heterogeneous future types via separate typed pools, eliminating Open Question 5 from the original design.
4. **Battle-tested**: Embassy's run-queue, waker vtable, and task state machine are production-proven. Re-implementing them adds risk with no benefit unless register pressure forces it.

**Embassy integration requirements**:
```rust
// 1. Cargo.toml
[dependencies]
embassy-executor = { version = "0.7", features = ["arch-spin"] }

[profile.release]
lto = "fat"

// 2. Provide __pender (no-op — GPU executor is self-polling)
#[no_mangle]
unsafe extern "C" fn __pender(_: *mut ()) {
    // No-op: the spin-loop executor does not need external wake notification
}

// 3. Provide no-op critical section (see D5)
```

**Fallback trigger**: If `ptxas -v` shows >64 registers per thread with Embassy (measured in Phase 1), switch to the stripped-down custom GpuExecutor from the original design (bitmask run queue, no linked list, no AtomicPtr).

### D3: Memory Layout [Unchanged]

Same as async-runtime.2. Executor state in registers/local memory. Hostcall buffers in global mapped memory.

### D4: Poll-All-Tasks Model [CHANGED — addresses rv3 Issue 7]

**Decision**: Remove explicit self-waking from `HostcallFuture`. The executor unconditionally polls all non-completed tasks every cycle.

**Original problem**: Every pending `HostcallFuture` called `cx.waker().wake_by_ref()`, keeping itself in the run queue permanently. This made the run queue always non-empty while any hostcall was in flight, rendering the nanosleep idle path dead code. The async waker machinery (vtable dispatch, bitmask operations) added overhead over a simple loop.

**New model**:
```rust
// Embassy arch-spin already does this:
// - Polls all ready tasks in a tight loop
// - When no tasks are ready, loops back and checks again
// - The "spin" in arch-spin IS the poll-all model

// HostcallFuture no longer self-wakes:
impl Future for HostcallFuture {
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = unsafe { self.get_unchecked_mut() };
        match this.state {
            HostcallState::Init { service, ref args } => {
                match hc_try_pop_free(this.buffer) {
                    Some(pkt_idx) => {
                        hc_fill_packet(this.buffer, pkt_idx, service, args);
                        hc_push_ready(this.buffer, pkt_idx);
                        hc_ring_doorbell(this.buffer);
                        this.state = HostcallState::Waiting { packet_idx: pkt_idx };
                        // No self-wake — executor will re-poll us next cycle
                        Poll::Pending
                    }
                    None => {
                        // Pool exhausted — stay in Init, retry next poll cycle
                        // No self-wake needed — poll-all model guarantees re-poll
                        Poll::Pending
                    }
                }
            }
            HostcallState::Waiting { packet_idx } => {
                let control = sys_load_acquire_u32(
                    hc_control_ptr(this.buffer, packet_idx)
                );
                if control & READY_BIT != 0 {
                    let response = hc_read_response(this.buffer, packet_idx);
                    hc_push_free(this.buffer, packet_idx);
                    this.state = HostcallState::Done;
                    Poll::Ready(Ok(response))
                } else {
                    // Not ready — no self-wake, executor re-polls next cycle
                    Poll::Pending
                }
            }
            HostcallState::Done => panic!("polled after completion"),
        }
    }
}
```

**Nanosleep**: With the poll-all model, the executor always has tasks to poll (until all complete). Nanosleep is useful only in a degenerate case (all tasks pending, nothing productive to do). For now, the executor spin-polls without nanosleep — this matches Embassy's `arch-spin` behavior exactly. Nanosleep can be added as a Phase 4 optimization with a heuristic like "if no task transitioned state this cycle, nanosleep before next cycle."

**Why this is better**:
1. No waker vtable dispatch overhead on every Pending return
2. Simpler code — no `cx.waker().wake_by_ref()` calls
3. Matches Embassy's actual behavior (arch-spin polls all tasks unconditionally)
4. Nanosleep was dead code anyway with self-waking; now the model is honest about being a spin-poll

### D5: Per-Lane Packets for Async Hostcall [NEW — addresses rv3 Issue 1]

**Decision**: Async hostcall uses per-lane packet acquisition. Each lane independently pops a packet from the free pool, fills only its own slot, and waits for the response independently. The warp-cooperative model (ADR-3) applies only to synchronous hostcall.

**Rationale**:
The synchronous hostcall in ADR-3 uses `__activemask()` to determine which lanes participate, then elected-lane pops a single packet, all active lanes fill their respective slots, and elected-lane pushes to ready. This requires all participating lanes to reach the hostcall at the same time — a natural property of synchronous code in SIMT lockstep.

In async mode, different lanes may reach `hostcall_async().await` at different times (due to different task states, different `await` points, or different control flow). Warp-cooperative filling is not feasible without an explicit barrier, which would defeat the purpose of async.

**Design**:
```
Synchronous hostcall (ADR-3, unchanged):
  - 1 packet per warp, 32 lanes fill slots cooperatively
  - Pool size: num_warps × max_concurrent_hostcalls

Async hostcall (this design):
  - 1 packet per lane per outstanding hostcall
  - Only the calling lane's slot is used (slots[lane_id][0..7])
  - active_mask = 1 << lane_id (single lane)
  - Pool size: num_lanes × max_concurrent_async_hostcalls_per_lane
```

**Pool sizing implications**:
- Worst case (all lanes async): 32x more packets than synchronous mode
- Typical case (1 async hostcall per lane at a time): num_threads packets
- Example: 2048 threads, 1 concurrent hostcall each = 2048 packets × 128 bytes = 256 KB
- This is feasible — mapped memory is allocated from system RAM, not GPU VRAM

**Warp-cooperative async (Phase 4 optimization)**:
A future optimization could batch per-lane async hostcalls within a warp:
- When multiple lanes have pending `Init` state hostcalls for the same service, use `__ballot_sync` to detect them and cooperatively fill a single packet
- This reduces pool pressure back to ~1x but adds significant complexity
- Deferred until pool pressure is empirically demonstrated to be a problem

### D6: Pool Exhaustion via Pending [CHANGED — addresses rv3 Issue 2]

**Decision**: When the free packet pool is empty, `HostcallFuture::poll()` returns `Poll::Pending` and stays in the `Init` state. The next poll cycle retries the allocation.

**Original problem**: The `?` operator on `hc_pop_free()` propagated `PoolExhausted` as a hard error, making the future resolve to `Err(PoolExhausted)`. The caller would need to handle this error and potentially retry, but the error was terminal — the future was consumed.

**New behavior**:
```rust
HostcallState::Init { service, ref args } => {
    match hc_try_pop_free(this.buffer) {
        Some(pkt_idx) => {
            // Success: fill and submit packet, transition to Waiting
            ...
            Poll::Pending
        }
        None => {
            // Pool exhausted: stay in Init state, retry next cycle
            // Back-pressure: naturally limits outstanding hostcalls
            Poll::Pending
        }
    }
}
```

**Properties**:
1. **Non-destructive**: The future stays in `Init` and can retry on the next poll
2. **Natural back-pressure**: Lanes that can't acquire packets simply wait, reducing contention
3. **No starvation**: As other lanes' hostcalls complete and packets are freed, waiting lanes will eventually acquire packets
4. **Diagnostic hook**: Could add a counter for pool-exhaustion events to detect undersized pools

**`hc_try_pop_free` vs `hc_pop_free`**: The existing `hc_pop_free` spins until a packet is available (or returns `PoolExhausted` after timeout). For async, we need a non-blocking `hc_try_pop_free` that attempts one CAS and returns `None` if the free stack is empty. This is a minor addition to the gpu-protocol crate.

### D7: Stack-Local Executor Reference [CHANGED — addresses rv3 Issue 4]

**Decision**: The executor reference is passed as a function argument through the call stack. No global `CURRENT_EXECUTOR` variable.

**Original problem**: The waker's `wake()` function accessed `CURRENT_EXECUTOR` as a global, but GPU has no thread-local storage. The mechanism was left unspecified.

**Resolution**: With Embassy as the primary executor, this issue is largely moot — Embassy's waker stores the task pointer directly in `RawWaker::data` and uses `AtomicU32` state flags within the task struct itself. The waker does not need to find "the executor"; it marks the task as ready by setting an atomic flag in the task's own state.

For the custom GpuExecutor fallback, the resolution is:
```rust
// The executor lives on the kernel function's stack frame.
// It is passed as &mut to any function that needs to spawn or wake tasks.
// The waker does NOT need the executor — it only needs the task's ready-bit location.

unsafe fn wake(data: *const ()) {
    // data encodes a pointer to the task's ready-flag (a u32 on the stack)
    let ready_flag = data as *mut u32;
    // Safe because waker only fires within the same thread's poll loop
    *ready_flag = 1;
}
```

**Why no global is needed**: The waker fires only within the same thread that owns the executor (intra-thread waking). The poll loop calls `Future::poll()`, which may call `cx.waker().wake_by_ref()` — but this is always on the same call stack. The executor's `poll()` method checks the ready flags directly. Cross-thread waking is deferred to a separate design (inter-task channels).

### D8: Critical Section Misuse Warning [CHANGED — addresses rv3 Issue 3]

**Decision**: The no-op critical section implementation includes explicit documentation warning against inter-thread use, plus a recommendation for compile-time guards.

**Implementation**:
```rust
/// GPU no-op critical section implementation.
///
/// # SAFETY WARNING: Single-Thread Scope Only
///
/// This critical section provides NO mutual exclusion between GPU threads.
/// It is correct ONLY for per-thread-private data (e.g., per-lane Embassy executor).
///
/// DO NOT use `embassy_sync::Mutex<CriticalSectionRawMutex, T>`,
/// `embassy_sync::Channel`, or any other Embassy sync primitive that relies
/// on `critical_section` for inter-thread synchronization on GPU. These will
/// silently produce data races.
///
/// For inter-thread synchronization on GPU, use:
/// - Intra-warp: `__ballot_sync` / `__shfl_sync` intrinsics (gpu-atomics crate)
/// - Inter-warp (same block): shared-memory spinlock
/// - Inter-block: global-memory spinlock with system-scope atomics
///
/// A proper GPU-aware `CriticalSectionRawMutex` replacement using spinlocks
/// is planned (async-runtime.1.3).
struct GpuCriticalSection;

critical_section::set_impl!(GpuCriticalSection);

unsafe impl critical_section::Impl for GpuCriticalSection {
    unsafe fn acquire() -> RawRestoreState {
        // No-op: per-thread executor has no shared mutable state.
        // GPU threads cannot be preempted (no interrupts).
        ()
    }
    unsafe fn release(_: RawRestoreState) {
        // No-op
    }
}
```

**Compile-time guard recommendation**: When implementing inter-thread sync primitives, provide a `GpuMutex<T>` wrapper that uses spinlocks internally, rather than exposing `embassy_sync::Mutex` with the no-op CS. This prevents accidental misuse at the API level.

## HostcallFuture State Machine (Complete)

```rust
enum HostcallState {
    /// Need to acquire a packet and submit the request.
    /// Retries on pool exhaustion (returns Pending, stays in Init).
    Init { service: u32, args: [u64; 7] },
    /// Packet submitted, waiting for host response.
    /// Polls control flag each cycle.
    Waiting { packet_idx: u16 },
    /// Completed (terminal state).
    Done,
}

struct HostcallFuture {
    buffer: *const HostcallBuffer,
    lane_id: u32,  // This lane's index within the warp (0..31)
    state: HostcallState,
}

impl Future for HostcallFuture {
    type Output = [u64; 7];  // No Result — pool exhaustion is Pending, not Err

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = unsafe { self.get_unchecked_mut() };
        match this.state {
            HostcallState::Init { service, ref args } => {
                // Try to pop a free packet (non-blocking)
                let Some(pkt_idx) = hc_try_pop_free(this.buffer) else {
                    // Pool exhausted — stay in Init, will be re-polled next cycle
                    return Poll::Pending;
                };
                // Fill packet: per-lane mode
                // active_mask = single lane only
                let active_mask = 1u32 << this.lane_id;
                hc_fill_packet_header(this.buffer, pkt_idx, service, active_mask);
                hc_fill_lane_slots(this.buffer, pkt_idx, this.lane_id, args);
                // Submit: push to ready stack + ring doorbell
                hc_push_ready(this.buffer, pkt_idx);
                hc_ring_doorbell(this.buffer);
                // Transition to waiting
                this.state = HostcallState::Waiting { packet_idx: pkt_idx };
                Poll::Pending
            }
            HostcallState::Waiting { packet_idx } => {
                let control = sys_load_acquire_u32(
                    hc_control_ptr(this.buffer, packet_idx)
                );
                if control & READY_BIT != 0 {
                    // Host responded — read response from this lane's slots
                    let response = hc_read_lane_response(
                        this.buffer, packet_idx, this.lane_id
                    );
                    // Return packet to free pool
                    hc_push_free(this.buffer, packet_idx);
                    this.state = HostcallState::Done;
                    Poll::Ready(response)
                } else {
                    // Not ready yet — will be re-polled next cycle
                    Poll::Pending
                }
            }
            HostcallState::Done => panic!("polled after completion"),
        }
    }
}
```

**Changes from original**:
1. `Output` is `[u64; 7]` not `Result<[u64; 7], HostcallError>` — pool exhaustion is handled internally via Pending
2. No `cx.waker().wake_by_ref()` calls — poll-all model handles re-polling
3. `hc_try_pop_free` (non-blocking) replaces `hc_pop_free` (spinning/error)
4. `lane_id` field + `active_mask = 1 << lane_id` for per-lane packet mode
5. `hc_fill_lane_slots` fills only this lane's slot row, not the full packet

## Executor Poll Loop

### Primary Path: Embassy

```rust
use embassy_executor::{Executor, Spawner};

#[no_mangle]
unsafe extern "C" fn __pender(_: *mut ()) {
    // No-op: arch-spin executor polls continuously, no external wake needed
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn async_kernel(hostcall_buf: *mut u8) {
    // Each thread gets its own Embassy executor on the stack
    let executor = Executor::new(core::ptr::null_mut());

    // Embassy's arch-spin executor runs a spin-poll loop internally:
    //   loop {
    //       poll_all_ready_tasks();
    //       // arch-spin: immediately loop back (no WFE/sleep)
    //   }
    //
    // We need a modified version that exits when all tasks complete.
    // Options:
    //   (a) Use executor.run() with a top-level future that joins all tasks
    //   (b) Use executor.poll() in our own loop with exit detection
    //
    // Option (b) gives us control over exit:
    executor.run(|spawner| {
        spawner.spawn(my_async_task(hostcall_buf)).unwrap();
        spawner.spawn(another_task(hostcall_buf)).unwrap();
    });
    // Note: executor.run() never returns in Embassy's arch-spin.
    // For GPU, we need to modify the exit condition — see Implementation Plan.
}
```

**Exit condition**: Embassy's `arch-spin` loops forever (designed for embedded MCUs that never halt). For GPU kernels, we need the executor to exit when all tasks complete. Options:
1. **Spawn a "supervisor" task** that `join!`s all other tasks, then sets a flag checked by a modified poll loop
2. **Patch Embassy's arch-spin** to add an `all_tasks_done()` check (minimal fork — 1 line change)
3. **Use `Executor::poll()` directly** (if exposed) in a custom loop with exit detection

This is a Phase 1 implementation detail to resolve during async-runtime.3.

### Fallback Path: Custom GpuExecutor

Only used if Phase 1 register pressure measurement shows Embassy exceeds 64 registers per thread.

```rust
pub unsafe extern "ptx-kernel" fn async_kernel(hostcall_buf: *mut u8) {
    let mut executor = GpuExecutor::new();

    executor.spawn(my_async_task(hostcall_buf));
    executor.spawn(another_task(hostcall_buf));

    // Poll-all loop (no waker-driven scheduling)
    loop {
        let any_incomplete = executor.poll_all();
        if !any_incomplete {
            break;  // All tasks completed, kernel can exit
        }
        // Spin immediately — no nanosleep in baseline
        // (All tasks are polled every cycle regardless of ready state)
    }
}

impl GpuExecutor {
    /// Poll all non-completed tasks. Returns true if any task is still incomplete.
    fn poll_all(&mut self) -> bool {
        let mut any_incomplete = false;
        for i in 0..self.task_count {
            if self.tasks[i].state != TaskState::Completed {
                any_incomplete = true;
                // Create a no-op waker (poll-all model doesn't use wakers for scheduling)
                let waker = noop_waker();
                let mut cx = Context::from_waker(&waker);
                let poll_result = unsafe {
                    Pin::new_unchecked(&mut self.tasks[i].future).poll(&mut cx)
                };
                if let Poll::Ready(result) = poll_result {
                    self.tasks[i].state = TaskState::Completed;
                    self.tasks[i].result = Some(result);
                }
            }
        }
        any_incomplete
    }
}
```

**Simplification from original**: With poll-all semantics, the bitmask run queue and custom waker vtable are unnecessary. The executor simply iterates over all tasks each cycle. This reduces the custom executor to ~30 lines of code and minimal register usage.

## Implementation Plan (Reordered)

### Phase 1: Embassy Integration + Register Measurement [addresses rv3 Issue 5, 8]
1. Add `embassy-executor` dependency with `arch-spin` feature to gpu-kernel
2. Set `lto = "fat"` in `[profile.release]`
3. Provide `__pender` (no-op) and no-op critical section
4. Compile a trivial `async fn` that returns `Poll::Ready` immediately
5. **Measure register pressure** with `ptxas -v --register-usage`
6. If ≤64 registers: proceed with Embassy (Phase 2)
7. If >64 registers: switch to custom GpuExecutor fallback
8. Resolve exit condition (supervisor task vs patched poll loop)

**Decision gate**: Embassy register pressure determines Phase 2 path.

### Phase 2: Async Hostcall Future (on top of Phase 1 executor)
1. Implement non-blocking `hc_try_pop_free` in gpu-protocol crate
2. Implement `HostcallFuture` with per-lane packet semantics
3. Handle pool exhaustion via `Poll::Pending` (stay in Init state)
4. Test: single task issuing async hostcall, executor polls to completion
5. Test: two tasks issuing concurrent async hostcalls from same thread
6. Measure register pressure again with hostcall futures

### Phase 3: Pool Sizing Validation
1. Calculate required pool size for target concurrency: `num_threads × max_async_hostcalls_per_thread`
2. Test with intentionally small pool to verify back-pressure behavior
3. Document pool sizing guidelines for users

### Phase 4: Optimization (future)
1. Nanosleep heuristic: if no task changed state this cycle, `nanosleep` before next cycle
2. Warp-cooperative async hostcall: batch per-lane Inits via `__ballot_sync`
3. Adaptive pool sizing based on runtime exhaustion counters
4. Profile async vs sync hostcall latency

## Risk Assessment

### R1: Register Pressure Kills Occupancy [Unchanged — but mitigated by early measurement]
**Risk**: High
**Mitigation**: Measured in Phase 1, not deferred to Phase 4. Fallback path (custom executor) is fully designed and ready to implement.

### R2: Per-Lane Packets Exhaust Pool [NEW]
**Risk**: Medium
**Impact**: 32x more packets needed than synchronous mode. With 2048 threads and 1 concurrent hostcall each, need 2048 packets.
**Mitigation**:
- Pool sizing is user-configurable (host allocates the buffer)
- 2048 × 128 bytes = 256 KB — trivial for host memory
- Back-pressure via Pending prevents crashes on exhaustion
- Phase 4 warp-cooperative optimization reduces pressure to ~1x

### R3: Embassy Exit Condition [NEW]
**Risk**: Low
**Impact**: Embassy's `arch-spin` loops forever. GPU kernels must exit.
**Mitigation**: Multiple solutions available (supervisor task, patched poll loop, direct `poll()` usage). Low-risk because the change is small and localized.

### R4: Warp Divergence [Unchanged]
**Risk**: Medium
**Mitigation**: Inherent to async. Structure kernels for uniform control flow across lanes.

### R5: Poll-All Overhead with Many Tasks [NEW]
**Risk**: Low
**Impact**: Polling completed-but-not-yet-reaped tasks wastes cycles. With N=4-8 tasks max, overhead is negligible.
**Mitigation**: Skip completed tasks in the poll loop (both Embassy and custom executor do this).

## Changes from async-runtime.2

| Aspect | Original (async-runtime.2) | Reworked (async-runtime.2.1) |
|--------|---------------------------|------------------------------|
| Primary executor | Custom GpuExecutor | Embassy (arch-spin + fat LTO) |
| Fallback executor | Embassy (Phase 3) | Custom GpuExecutor (if reg pressure) |
| Hostcall packets | Ambiguous (per-lane code, per-warp protocol) | Explicitly per-lane for async |
| Pool exhaustion | `?` → hard error | `Poll::Pending` → retry next cycle |
| Scheduling model | Self-waking (run queue bitmask) | Poll-all (no explicit waker scheduling) |
| Nanosleep | On idle (dead code due to self-wake) | Deferred to Phase 4 heuristic |
| CURRENT_EXECUTOR | Global (unspecified mechanism) | Not needed (stack-local / Embassy internal) |
| CS documentation | Correctness noted but no warning | Explicit warning + guard recommendation |
| Register measurement | Phase 4 | Phase 1 (immediately after Embassy integration) |
| HostcallFuture Output | `Result<[u64; 7], HostcallError>` | `[u64; 7]` (errors handled internally) |

## Impact on Downstream Tasks

| Task | Impact |
|------|--------|
| **async-runtime.3** (minimal async on GPU) | Direct implementation target: Embassy-first per this design |
| **async-runtime.1.3** (critical section) | No-op CS confirmed sufficient for per-thread executor; document warnings |
| **gpu-std.2** (libc shim) | Libc calls become async hostcalls with per-lane packets |
| **integration.1** (end-to-end) | Embassy executor is the runtime foundation |
| **gpu-protocol** | Need `hc_try_pop_free` (non-blocking pop) addition |

## ADR

This rework updates **ADR-4** (see decisions.md).
