# executor-impl.2: WorkQueue + TaskSlot arena
**Cycle**: 318 | **Theme**: executor-impl | **Kind**: experiment | **Status**: done

## Summary
Implemented the GPU-side async executor core in `gpu-runtime::executor` module:
lock-free MPMC work queue, type-erased task slots with arena allocation, free slot
recycling via tagged CAS stack, and the `GpuExecutor` with `spawn()` + `run()` API.

## Changes Made

### 1. New `executor` module in `crates/core/gpu-runtime/src/lib.rs`

**WorkQueue** — bounded lock-free MPMC FIFO:
- Uses tagged CAS on head/tail (u64) for ABA prevention
- Circular buffer of u32 task indices (MAX_TASKS = 256)
- `enqueue()` and `dequeue()` with CAS retry loops
- Same pattern as hostcall protocol's Treiber stacks, but FIFO for fairness

**TaskSlot** — type-erased future storage:
- `#[repr(C)]` with state, poll_fn, future_size, future_bytes
- `PollFn` type alias: `unsafe fn(*mut u8, &mut Context) -> Poll<()>`
- `TASK_FUTURE_MAX_SIZE = 512` bytes inline storage
- `erased_poll::<F>()` trampoline for type erasure

**FreeSlotStack** — tagged CAS free list:
- Reuses TaskSlot's `future_size` field as next pointer when FREE
- Tagged u64 head for ABA prevention (same pattern as hostcall)
- `init()`, `pop()`, `push()` operations

**GpuExecutor** — main executor:
- `new()` / `init()` for setup
- `spawn<F: Future<Output = ()>>()` — allocate slot, copy future, enqueue
- `run()` — executor loop with ExitOnEmpty policy
  - Lane 0 dequeues, broadcasts task_id via shfl.sync
  - All 32 lanes poll the same task (warp-cooperative)
  - Spin-poll with nanosleep yield, MAX_POLLS_PER_TASK timeout
  - Slot recycling on completion
- `shutdown()` — signal all warps to exit
- Diagnostic: `spawned_count()`, `completed_count()`

### 2. Prelude update
- Added `GpuExecutor`, `ExecutorError`, `ExecutorStats`, `TaskId` to prelude

## Design Decisions
- **FIFO queue** (not stack): fairness for server workloads
- **ExitOnEmpty policy only** for MVP: simplest, sufficient for batch workloads
- **No Drop support**: spawned futures must be trivially droppable (matches existing I/O futures)
- **Lane 0 CAS + shfl.sync broadcast**: minimizes contention (1 CAS per warp, not per thread)
- **MAX_TASKS = 256**: 256 * (64 header + 512 data) ≈ 144KB — fits easily in GPU global memory

## Verification
- Compiles on x86_64 (stub path): `cargo check --target x86_64-pc-windows-msvc`
- Clippy clean with `-D warnings`
- GPU-side (nvptx64) verification deferred to executor-impl.4 (demo kernel)

## Impact on Downstream Tasks
- **executor-impl.3 (executor loop + spawn API)**: Effectively done — `run()` and `spawn()` are implemented here. executor-impl.3 can focus on higher-level wrappers or be marked done.
- **executor-impl.4 (demo kernel)**: Can now write a kernel that creates a GpuExecutor, spawns tasks, and runs them.
- **gpu-executor epic**: Criteria 1 (lock-free MPMC queue) and 2 (TaskSlot arena) met.
