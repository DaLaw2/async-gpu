# executor-impl.4: Demo kernel with dynamic task spawning
**Cycle**: 323 | **Theme**: executor-impl | **Kind**: experiment | **Status**: done

## Summary
GPU-side async executor demo kernel — spawns 8 type-erased futures (4 WriteValueFuture + 4 CounterFuture) into a GpuExecutor in mapped memory. A single warp (32 lanes) runs the executor loop; lane 0 dequeues tasks, polls futures via indirect function pointers, and broadcasts results to all lanes via `shfl.sync`.

## Findings

### Q: Can type-erased futures with indirect fn pointers run on GPU?
A: **Yes.** The LLVM nvptx backend correctly generates `.callprototype` + indirect `call` instructions for `erased_poll::<F>`. Both `WriteValueFuture` (completes in 1 poll) and `CounterFuture` (yields once, completes in 2 polls) work correctly via function pointer indirection.
**Confidence**: high (verified end-to-end on GPU)

### Q: Does the full executor pipeline work?
A: **Yes.** spawn() → enqueue() → dequeue() → poll() → complete cycle works. Results:
- spawned=8, completed=8, tasks_executed=8, polls_total=12
- All WriteValueFuture values correct: [42, 100, 255, 1337]
- CounterFuture counter = 4 (all 4 incremented)
- Success flag = 1

## Bugs Found and Fixed

### 1. CUDA_ERROR_MISALIGNED_ADDRESS (TaskSlot alignment)
- **Root cause**: `future_bytes` in `TaskSlot` was at offset 20 (not 8-byte aligned). Futures containing pointers (8 bytes on nvptx64) caused misaligned access.
- **Fix**: Added `_pad: u32` field after `future_size` to push `future_bytes` to offset 24 (8-byte aligned).

### 2. All 32 lanes calling poll_fn simultaneously
- **Root cause**: Original `run()` called `poll_fn(future_ptr, &mut cx)` from ALL 32 lanes. Futures aren't designed for concurrent access by 32 threads writing to the same memory.
- **Fix**: Only lane 0 calls poll_fn, result is broadcast via `shfl_sync_idx_u32(mask, is_ready, 0)`.

### 3. Executor `run()` hangs on GPU despite correct logic
- **Root cause**: The LLVM nvptx backend generates problematic PTX when `activemask()` is called inside an inlined method, combined with complex loop unrolling (5x unrolled inner poll loop) and the Waker drop glue generating indirect function calls at scope exit. The exact failure mode is unclear but appears to be a warp convergence issue in the generated control flow.
- **Fix**: Created a simplified `run(mask)` that takes the warp mask as a parameter (computed externally), inlines the dequeue logic, uses `ManuallyDrop` for the Waker, and keeps simpler control flow. This generates cleaner PTX and works reliably.

## Key Design Decisions
1. **Lane 0 polls, shfl broadcasts**: The warp-cooperative pattern where only lane 0 executes the future and broadcasts the result via `shfl.sync` to all lanes.
2. **No free-stack recycling during run()**: Completed slots are marked FREE but not pushed back to the FreeSlotStack. This avoids CAS operations in the hot loop and prevents a subtle hang. Slots can be reclaimed via a separate pass if needed.
3. **Mask passed externally**: `run(mask)` takes the warp mask as a parameter instead of calling `activemask()` internally. This produces reliable PTX codegen.
4. **Safety valve**: Outer loop bounded by `MAX_TASKS + 2` iterations to prevent infinite loops.

## Open Questions
- Why does the original `run()` with `activemask()` inside hang while the same logic with mask as parameter works? Likely an LLVM nvptx codegen issue with inlined `asm!("activemask.b32 ...")` inside complex control flow.

## Impact on Downstream Tasks
- **gpu-executor epic**: All 4 criteria met — MPMC queue compiles to PTX, TaskSlot arena works, executor loop polls to completion, demo kernel spawns 8 tasks.
- **Next**: gpu-executor epic can be closed. Future work: multi-warp execution, dynamic spawn during run, integration with hostcall futures.
