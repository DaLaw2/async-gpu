# gpu-executor.1: Design — executor API for GPU async
**Cycle**: 276 | **Theme**: gpu-executor | **Kind**: design | **Status**: done

## Summary
Designed the GPU-side async executor API. Key finding: `SpinExecutor` already exists in gpu-runtime (lines 2205-2254) with the exact `run()` method needed. The design task reduces to: (1) expose SpinExecutor in prelude, (2) add a convenience `block_on()` free function, (3) make yield strategy configurable.

## Findings

### Existing Infrastructure

**SpinExecutor** (gpu-runtime/src/lib.rs:2205-2254):
```rust
pub struct SpinExecutor;
impl SpinExecutor {
    pub unsafe fn run<F: Future>(future: &mut F) -> Option<F::Output>
}
```
- Creates no-op RawWakerVTable (GPU has no real wake mechanism)
- Polls up to 10M times with `nanosleep.u32 64` between polls
- Returns `Some(output)` on completion, `None` on timeout
- **NOT exported in prelude** — users must use `gpu_runtime::std_future::SpinExecutor::run()`

**WarpExecutor** (gpu-runtime/src/lib.rs:1449-1490):
- Polls WarpFuture across 32 lanes simultaneously
- Already in prelude
- Different trait (WarpFuture, not Future)

**warp_cooperative module** (gpu-runtime/src/lib.rs:2265-2378):
- `warp_poll_future()` — broadcast single poll result
- `warp_run_future()` — full executor loop with lane 0 polling
- `warp_run_two_futures()` — sequential two-future execution

### Current Boilerplate (30 lines per kernel)

Every kernel manually creates:
```rust
static VTABLE: RawWakerVTable = RawWakerVTable::new(
    |p| RawWaker::new(p, &VTABLE), |_| {}, |_| {}, |_| {},
);

let waker = Waker::from_raw(RawWaker::new(core::ptr::null(), &VTABLE));
let mut cx = Context::from_waker(&waker);
let mut fut = my_async_fn(args);
let mut pinned = Pin::new_unchecked(&mut fut);
let mut result = 0xDEADu32;
let mut polls = 0u32;
while polls < 10_000_000 {
    match pinned.as_mut().poll(&mut cx) {
        Poll::Ready(val) => { result = val; break; }
        Poll::Pending => { core::arch::asm!("nanosleep.u32 1000;"); }
    }
    polls += 1;
}
```

### Proposed API

#### 1. Free function `block_on` (convenience wrapper)

```rust
/// Run a Future to completion on the current GPU thread.
/// Returns Some(output) if the future completes within max_polls iterations.
/// Returns None if the future times out.
pub unsafe fn block_on<F: Future>(future: F) -> Option<F::Output>
```

This is a thin wrapper around SpinExecutor::run that pins the future internally.

#### 2. Configurable SpinExecutor

```rust
pub struct SpinExecutor {
    max_polls: u32,       // default: 10_000_000
    nanosleep_ns: u32,    // default: 1000 (1 µs)
}

impl SpinExecutor {
    pub const fn new() -> Self;
    pub const fn with_max_polls(self, max: u32) -> Self;
    pub const fn with_nanosleep(self, ns: u32) -> Self;
    pub unsafe fn run<F: Future>(&self, future: F) -> Option<F::Output>;
}
```

#### 3. Prelude additions

```rust
pub use crate::std_future::block_on;
// SpinExecutor stays in std_future module (not prelude) — advanced users only
```

### Design Decisions

1. **`block_on` takes `F` by value (not `&mut F`)**: The function pins internally. Callers just pass the future directly: `block_on(my_async_fn(args))`. This eliminates the need for `Pin::new_unchecked` at call sites.

2. **Default nanosleep = 1000 ns (1 µs)**: The async-pipeline uses 1000ns and it works well. SpinExecutor currently uses 64ns which is too aggressive for I/O-bound futures.

3. **No warp-cooperative variant in block_on**: `block_on` is for single-thread use. Warp-cooperative execution uses the existing `warp_run_future()` family. They serve different use cases and shouldn't be unified.

4. **Return `Option<F::Output>` not `F::Output`**: Timeout is a real failure mode on GPU (TDR). Callers must handle the `None` case.

## Implementation Plan

1. Refactor SpinExecutor to hold config fields (max_polls, nanosleep_ns)
2. Add `block_on<F: Future>(future: F) -> Option<F::Output>` free function
3. Export `block_on` in prelude
4. Refactor async-pipeline kernel to use `block_on(data_pipeline(buf))`
5. Update other example kernels

Estimated: ~50 lines of library code + kernel simplifications.

## Open Questions
- Should `block_on` return `Result<F::Output, TimeoutError>` instead of `Option`?
- Should there be a `block_on_or_trap` variant that traps on timeout instead of returning None?

## Impact on Downstream Tasks
- **gpu-executor.2**: Unblocked — implementation is straightforward
- **async-pipeline example**: Will shrink from 149 lines to ~20 lines in kernel
