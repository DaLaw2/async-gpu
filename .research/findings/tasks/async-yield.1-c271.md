# async-yield.1: Design async hostcall mechanism — how does a sync PAL fn yield?
**Cycle**: 271 | **Theme**: async-yield | **Kind**: investigation | **Status**: done

## Summary
Investigated how to bridge the sync PAL layer (std::io::Write returning io::Result) with async hostcall Futures. Found that there are two viable approaches: (A) MIR-level transform that converts sync I/O calls into .await points, or (B) runtime-level yield where the spin loop itself yields efficiently. Approach B is already partially implemented via `nanosleep.u32 64` in `sys_spin_load_acquire_u32`. The real win requires a `GpuHostcallFuture` that can be `.await`ed from `#[warp_cooperative]` async fn, with the PAL layer providing both sync and async APIs.

## Findings

### Q: How can a sync fn (std::io::Write::write) yield to warp scheduler?
A: **It can't return Poll::Pending — but it can yield internally during the spin-wait.**

The current spin loop in `gpu_hostcall_request()` already calls `sys_spin_load_acquire_u32` which emits `nanosleep.u32 64`. This instruction tells the hardware scheduler to deschedule the warp for ~64ns, allowing other eligible warps on the same SM to execute. So the spin-wait already yields — it's not a pure busy-wait.

However, this is still a **blocking** yield — the thread doesn't return control to the caller, it just sleeps inside the loop. The caller (PAL/std) is stuck until the host responds.

**Confidence**: high

### Q: Can the MIR pass transform sync I/O calls into .await points automatically?
A: **Theoretically possible but extremely invasive.** The MIR pass would need to:
1. Identify calls to specific functions (write, read, open, close)
2. Replace them with async versions that return `impl Future`
3. Insert `.await` points at each call site
4. Transform the enclosing function into a coroutine

This is essentially auto-async-ification — converting sync code to async at the IR level. This is a research-grade compiler transformation that's well beyond our current MIR pass (which only adds convergence barriers). **Not recommended for now.**

**Confidence**: high

### Q: What does 'yield to other warps' mean on GPU?
A: On GPU, the hardware scheduler manages multiple warps per SM. When a warp executes `nanosleep`, it is descheduled for the specified duration. During that time, the scheduler issues instructions from other eligible warps. This is the GPU's native "yield" mechanism.

Key insight: **the GPU already does cooperative multitasking at the warp level.** The spin-wait with nanosleep is already yielding. The question is whether we can do BETTER than spinning.

With Independent Thread Scheduling (sm_70+, our target is sm_86), the GPU can deschedule individual threads, not just whole warps. But there's no PTX instruction for "deschedule me until this memory location changes" (like Linux futex). The best we have is nanosleep.

**Confidence**: high

### Q: What's the practical architecture?
A: **Three-layer design:**

**Layer 1: `GpuHostcallFuture` (gpu-runtime)** — already exists as `GpuPrintFuture`
- Poll once: submit request, return Pending
- Poll again: check CONTROL_READY, return Ready or Pending
- No spin loop — caller decides when to re-poll

**Layer 2: Async kernel functions**
```rust
#[warp_cooperative]
async fn data_pipeline(buf: *mut u8) {
    let data = HostcallReadFuture::new(buf, "input.txt").await;   // yields
    let result = heavy_compute(&data);                              // runs
    HostcallWriteFuture::new(buf, "output.txt", &result).await;    // yields
}
```
The `.await` points are where the warp-cooperative MIR pass inserts convergence barriers. Between awaits, all lanes run compute in lockstep.

**Layer 3: Sync PAL stays sync** — `println!()` and `File::write()` keep their spin-wait. They work but don't yield to the async scheduler. Users who want yielding use the async API directly.

**This means:**
- `println!()` works today (sync, spin-wait with nanosleep)
- For real async yield, use `HostcallFuture::new().await` in `#[warp_cooperative] async fn`
- The PAL doesn't need to change — async is opt-in via the Future API

### Q: What new Futures do we need?
A: Generalize `GpuPrintFuture` into a generic `GpuHostcallFuture<const SERVICE: u32>`:
- `GpuHostcallWriteFuture` — SERVICE_WRITE (fd, data) → bytes written
- `GpuHostcallReadFuture` — SERVICE_READ (fd, max_len) → data
- `GpuHostcallOpenFuture` — SERVICE_OPEN (path, flags) → fd
- `GpuHostcallCloseFuture` — SERVICE_CLOSE (fd) → ()
- Plus keep `GpuPrintFuture` for SERVICE_PRINT

Each follows the same pattern: Init → submit packet → Waiting → check CONTROL_READY → Done.

## Unexpected Discoveries
- **nanosleep already yields**: The spin loop is NOT pure busy-wait. `nanosleep.u32 64` deschedules the warp. This means the current sync I/O already allows other warps to run — it's just not as efficient as true async (the thread is still "occupied" even while sleeping).
- **`GpuPrintFuture` is the template**: We already have a working async hostcall Future. The task is to generalize it to all I/O operations, not invent a new mechanism.
- **No need to change PAL**: The sync PAL (println!, File) works fine with spin-wait. Async is a separate, opt-in API for users who want real yield.

## Open Questions
- Should we provide a `std::fs`-like async API? e.g., `gpu_async::fs::File` with async methods?
- Can we wrap the sync PAL functions in a macro that auto-generates the async version?
- Performance: how much throughput improvement does async yield give vs sync spin-wait when multiple warps do I/O?

## Impact on Downstream Tasks
- async-yield.2 unblocked: implement generic GpuHostcallFuture for all I/O services
- async-yield.3 unblocked: data pipeline demo using these futures
- Epic criterion 1 (async hostcall Future) has clear implementation path
- Epic criterion 2 (PAL async bridge) reframed: PAL stays sync, async is separate API
