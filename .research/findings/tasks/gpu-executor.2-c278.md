# gpu-executor.2: Experiment — implement block_on in gpu-runtime, refactor async-pipeline
**Cycle**: 278 | **Theme**: gpu-executor | **Kind**: experiment | **Status**: done

## Summary
Implemented `block_on` free function in gpu-runtime and refactored async-pipeline to use it. The kernel entry point shrank from 30 lines of manual poll boilerplate to a single `block_on(data_pipeline(buf))` call. PTX output verified: 7x `bar.warp.sync`, 1x `shfl.sync`, 12x `nanosleep` — all warp-cooperative behavior preserved.

## Changes Made

### gpu-runtime/src/lib.rs (std_future module)
1. Extracted shared `NOOP_VTABLE` and `noop_waker()` helper
2. Added `block_on<F: Future>(future: F) -> Option<F::Output>` — convenience wrapper
3. Added `block_on_with<F: Future>(future, max_polls, nanosleep_ns)` — configurable variant
4. Refactored `SpinExecutor::run` to use shared `noop_waker()`
5. Changed default nanosleep from 64ns to 1000ns (1µs) — better for I/O-bound futures
6. Exported `block_on` in `prelude`

### async-pipeline kernel
Before (30 lines boilerplate):
```rust
static VTABLE: RawWakerVTable = ...;
let waker = Waker::from_raw(RawWaker::new(...));
let mut cx = Context::from_waker(&waker);
let mut fut = data_pipeline(buf);
let mut pinned = Pin::new_unchecked(&mut fut);
while polls < 10_000_000 { match pinned.as_mut().poll(&mut cx) { ... } }
```

After (1 line):
```rust
let result = block_on(data_pipeline(buf)).unwrap_or(0xDEAD);
```

### Removed imports
- `core::future::Future`
- `core::pin::Pin`
- `core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker}`
- `static VTABLE: RawWakerVTable = ...;`

## Verification
- gpu-runtime compiles for nvptx64: ✓
- async-pipeline kernel compiles with patched rustc: ✓
- MIR pass output: `data_pipeline::{closure#0}` — 0 yield(s), 6 poll(s), 6 suspension(s), 7 return(s) ✓
- PTX: 7x bar.warp.sync, 1x shfl.sync.idx, 12x nanosleep ✓

## Impact on Downstream Tasks
- **gpu-executor theme**: Both criteria met (block_on exists, async-pipeline refactored)
- **gpu-executor theme can be marked completed**
