# Theme Synthesis: coro-impl — Generator implementation + streaming pipeline

## Progress
- [x] coro-impl.1: GpuGenerator trait + WarpBroadcast + for_each_yield + CounterGenerator
- [ ] coro-impl.2: Streaming pipeline demo (fibonacci producer + consumer on GPU)

## Verified Conclusions
1. `GpuGenerator<R=()>` trait compiles for nvptx64 (sm_75) — debug and release
2. `WarpBroadcast` trait covers all scalar types via shfl.sync (1-2 cycles per broadcast)
3. `for_each_yield` combinator provides zero-buffered streaming inline loop
4. `GeneratorTask` Future adapter enables generators in GpuExecutor work queue
5. `CounterGenerator` reference implementation validates the trait pattern
6. No MIR pass changes needed — confirmed by both design analysis and compilation

## Key Decisions
- Manual `GpuGenerator` impls (not `#[coroutine]` closures) for MVP
- `unsafe trait` (not `unsafe fn`) — matches WarpFuture pattern
- `Copy` bound on Yield/Return — required for shfl.sync broadcast
- No `Pin` — GPU generators live in fixed-address global memory
- No `spawn_generator` yet — deferred to avoid touching executor.rs prematurely

## Next Steps
1. coro-impl.2: Build streaming pipeline demo kernel with visible output
2. Consider fibonacci or prefix-sum producer to demonstrate compute-driven yield
