# Finding: GPU-Side Async Task Spawning Executor Design

- **Task**: executor-design.1
- **Cycle**: 310
- **Date**: 2026-03-15
- **Status**: Complete (design doc written)

## Summary

Produced a comprehensive design document (`DESIGN-executor.md`) for a GPU-side async task spawning executor. The design enables `spawn()` from within running GPU kernels, with warp-level work stealing via a lock-free MPMC queue.

## Key Design Decisions

### 1. Lock-Free MPMC Queue (FIFO)
- Uses the same tagged-CAS pattern proven in the hostcall protocol
- FIFO instead of LIFO (hostcall uses stacks) for fairness in server workloads
- Head and tail on separate cache lines to reduce contention
- Lane 0 performs CAS, broadcasts result to all 32 lanes via `shfl.sync.idx.b32`

### 2. Type-Erased Task Slots
- Fixed-size arena slots (default 512 bytes per future)
- `poll_fn: unsafe fn(*mut u8, &mut Context) -> Poll<()>` for type erasure
- Avoids `dyn Future` trait objects (unreliable on nvptx64 LLVM backend)
- Slot recycling via a tagged CAS free-list (same pattern as hostcall free packets)

### 3. Warp-as-Scheduling-Unit
- Each warp dequeues and executes one task at a time
- All 32 lanes participate in every poll (required by SIMT architecture)
- `#[warp_cooperative]` MIR transformations in the future's `poll()` remain valid through indirect dispatch

### 4. Three Idle Policies
- `SpinWait`: nanosleep loop (server workloads)
- `ExitOnEmpty`: warp exits immediately (batch workloads)
- `CooperativeYield`: long sleep intervals (mixed workloads)

## Critical Open Problems

1. **Indirect function calls on nvptx64**: The type-erased `poll_fn` requires `call.uni` in PTX. Whether `#[warp_cooperative]` transformations survive indirect dispatch is untested.
2. **Deadlock potential**: If all warps block on spawned child tasks with no free warps to execute them, classic thread-pool exhaustion occurs.
3. **Warp utilization**: 32 lanes dedicated to one task wastes compute when the task only needs lane 0 for sequential I/O.
4. **Future size limits**: Fixed `TASK_FUTURE_MAX_SIZE` may be too small for deeply nested async state machines.

## Closest CPU Analog

Embassy (embedded Rust async runtime) — both use:
- Static/fixed task storage
- No heap allocation
- Cooperative spin-polling
- No Waker mechanism (Embassy uses interrupts, GPU uses no-op wakers)

## Files

- Design document: `DESIGN-executor.md` (project root)
- Reference: `crates/core/gpu-runtime/src/lib.rs` (current `block_on`, `SpinExecutor`, `WarpExecutor`)
- Reference: `ARCHITECTURE.md` (hostcall protocol, tagged CAS pattern)

## Implementation Phases

1. MVP: Single global queue, fixed slots, prove indirect poll_fn works on nvptx64
2. Channels: `GpuChannel<T>` for inter-task communication
3. Optimization: Per-block sharding, slot size tiers, priority
4. Advanced: Host-driven waker mechanism to eliminate spin-polling
