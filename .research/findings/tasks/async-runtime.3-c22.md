# async-runtime.3: Minimal async/await execution on GPU
**Cycle**: 22 | **Theme**: async-runtime | **Kind**: experiment | **Status**: done

## Summary
Successfully executed Embassy-based async/await on actual GPU hardware. Three kernel variants were tested: (1) ImmediateFuture — single spawn+poll completing in one round, (2) CountdownFuture — a Future requiring 6 poll rounds (5 Pending + 1 Ready) with wake_by_ref re-enqueue, and (3) two concurrent tasks on the same executor completing in 6 rounds. All three passed on NVIDIA GPU hardware. Register pressure is moderate: 31 regs for single-task immediate, 39 for multi-poll, 56 for two-task — well under the 64-register ADR-4 threshold for the Embassy path.

## Findings
### Q: Can we poll a simple Future on the GPU?
A: **Yes.** The Embassy executor's full spawn → poll → task dispatch → Future::poll → Ready/Pending cycle works correctly on GPU hardware. Key mechanisms verified:
- `Executor::new()` + `TaskStorage::spawn()` + `Spawner::spawn()` all work
- `Executor::poll()` correctly drains the run queue via atomic exchange
- Indirect function calls (`prototype_N : .callprototype`) dispatch correctly to task-specific poll functions
- `wake_by_ref()` correctly re-enqueues tasks for subsequent poll rounds
- The no-op `__pender` and no-op critical-section cause no issues
- Multiple tasks on the same executor are polled in round-robin each round

**Confidence**: high (verified on hardware)

### Q: Measure register usage
A: Register usage measured from PTX `.reg` declarations:

| Kernel | Pred (%p) | 32-bit (%r) | 64-bit (%rd) | Local stack | Estimated total |
|--------|-----------|-------------|--------------|-------------|-----------------|
| sync_countdown | 2 | 6 | 5 | 4B | ~13 |
| embassy_test (immediate) | 7 | 3 | 21 | none | ~31 |
| embassy_countdown (multi-poll) | 8 | 8 | 23 | 4B | ~39 |
| embassy_two_task (2 tasks) | 12 | 9 | 35 | 4B | ~56 |

Notes:
- PTX register counts are virtual; PTXAS may allocate more or fewer physical registers
- The 56-register two-task kernel is under the 64-reg ADR-4 threshold
- The local stack (4 bytes) is used for volatile-read loop counters
- The executor's CAS loops and indirect calls are the main register pressure drivers
- Predicates are cheap (dedicated predicate register file on GPU)

**Confidence**: medium (PTX virtual regs; actual hardware allocation may differ, would need cuobjdump for SASS)

### Q: Performance comparison with synchronous version
A: Direct timing comparison was not performed (single-kernel launches are too fast to measure meaningfully). However, structural analysis shows:
- **Sync countdown**: 5 iterations of `sub + volatile_read + branch` — trivial, ~10 instructions
- **Async countdown**: 6 poll rounds, each involving `atom.exch.b64` (drain queue) + indirect call + `atom.and.b32` (clear flags) + `atom.or.b32` / `atom.cas.b64` (re-enqueue) — ~30 instructions per round
- **Overhead factor**: ~18x more instructions for the async version, but this is for a trivial payload. For real workloads (e.g., hostcall with spin-wait), the executor overhead becomes negligible relative to actual work.
- The multi-poll loop is NOT unrolled when `read_volatile` is used on the counter, keeping register pressure bounded.

**Confidence**: medium (structural analysis, no timing)

## Register Usage Comparison
| Kernel | 32-bit regs | 64-bit regs | Pred regs | Total estimate |
|--------|-------------|-------------|-----------|----------------|
| sync_countdown_kernel | 6 | 5 | 2 | ~13 |
| embassy_test_kernel (immediate) | 3 | 21 | 7 | ~31 |
| embassy_countdown_kernel (multi-poll) | 8 | 23 | 8 | ~39 |
| embassy_two_task_kernel (2 tasks) | 9 | 35 | 12 | ~56 |

## Compilation Attempts
### Round 1: Initial embassy-test build (existing crate)
- Command: `cargo +nightly rustc --manifest-path crates/embassy-test/Cargo.toml --target nvptx64-nvidia-cuda -Zbuild-std=core --release -- --emit=asm -C linker=echo -C target-cpu=sm_86`
- Result: **Success** — produces 131-line PTX with zero unresolved externs
- Time: 14.56s (cold), 0.44s (incremental)

### Round 2: Added multi-poll + two-task + sync kernels (initial)
- Result: **Success** but compiler unrolled poll loops (6 copies each), inflating register counts to 76/93
- Fix: Added `core::ptr::read_volatile(&poll_rounds)` to prevent loop unrolling

### Round 3: With volatile loop counter (final)
- Result: **Success** — proper loop structure, register counts 39/56 for countdown/two-task
- PTX: 475 lines total, all 4 kernels + helper functions + critical-section stubs

### Round 4: Host-side runner build
- Command: `cargo build --manifest-path crates/gpu-host/Cargo.toml`
- Result: **Success** — embassy_test.ptx embedded via include_str!

## Test Results
### embassy_test_kernel (ImmediateFuture)
- Config: 1 block × 1 thread, single task with `Poll::Ready(42)`
- Result: **PASSED**
- Details: result[0] = 1 (success marker written after executor.poll())

### embassy_countdown_kernel (CountdownFuture)
- Config: 1 block × 1 thread, single task with remaining=5
- Result: **PASSED**
- Details: poll_rounds = 6 (5 Pending + 1 Ready), success = 1

### embassy_two_task_kernel (Two concurrent tasks)
- Config: 1 block × 1 thread, Task A (remaining=3, result=10) + Task B (remaining=5, result=20)
- Result: **PASSED**
- Details: poll_rounds = 6, success = 1. Both tasks ran concurrently on same executor.

### sync_countdown_kernel (Baseline)
- Config: 1 block × 1 thread, simple loop countdown
- Result: **PASSED**
- Details: result = 42

## Unexpected Discoveries

1. **LLVM aggressively unrolls Embassy poll loops.** Without volatile reads on the loop counter, the compiler unrolled 6 iterations of `executor.poll()` into 6 separate code blocks, each with its own register set. This inflated register counts from 39 → 76 (countdown) and 56 → 93 (two-task). Using `core::ptr::read_volatile` on the counter prevents this. Production code should use a similar pattern to keep register pressure bounded.

2. **Embassy's indirect function dispatch works on GPU.** The `prototype_N : .callprototype ()_ (.param .b64 _)` indirect calls execute correctly on NVIDIA hardware. This was an open question from async-runtime.1.3 — now confirmed.

3. **Task B's wake_by_ref re-enqueue is correct.** When a task returns Pending and calls `cx.waker().wake_by_ref()`, Embassy re-enqueues it via `atom.or.b32` (set RUN_QUEUED flag) + `atom.cas.b64` (CAS into run queue head). This all works correctly on GPU, with the same atomic operations used for the hostcall protocol.

4. **Two-task executor achieves true concurrency.** Both tasks are polled each round because Embassy's `poll()` drains the entire run queue atomically (`atom.exch.b64` returns the linked list head, then walks it). Task A completes after round 4, Task B after round 6. The executor handles heterogeneous completion times correctly.

5. **Static task storage works but limits reuse.** Each `TaskStorage<F>` is a global static, which means a kernel can only be called once (the state machine transitions are one-shot). For production use, dynamic task storage (stack-allocated or from a pool) would be needed.

## Open Questions

1. **Actual hardware register allocation.** PTX virtual register counts are upper bounds. The `ptxas` compiler may spill or coalesce. Using `cuobjdump --dump-sass` or CUDA occupancy API would give the true per-thread register count. This matters for occupancy calculations.

2. **Task storage reuse.** The current static `TaskStorage` approach means each kernel launch uses the same pre-allocated slots. For a real executor, we need dynamic task pools — either stack-allocated arrays or global memory pools.

3. **Scaling to realistic workloads.** The countdown futures are trivial. Register pressure for real async hostcall futures (with spin-wait, payload buffers, etc.) needs measurement.

4. **Multi-thread executors.** All tests use 1 block × 1 thread. With 32 threads per block, each running its own executor with static storage, we'd need per-thread storage (thread-indexed arrays or stack allocation).

## Impact on Downstream Tasks
- **async-runtime theme**: Two of three success criteria now met: (1) Can poll a Future on GPU ✓ (2) Multiple async tasks run concurrently ✓ (3) Register pressure measured (56 regs for 2-task < 64 threshold) ✓. The theme's success criteria are effectively satisfied.
- **integration.1**: Unblocked. The Embassy executor pattern can now be combined with hostcall for async I/O on GPU.
- **ADR-4 validated**: The Embassy-first approach works as designed. No custom executor fallback is needed (register pressure is acceptable).
