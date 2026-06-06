# Theme Synthesis: coro-design — Generator trait + MIR pass design

## Progress
- [x] coro-design.1: Investigation — map Rust coroutine semantics to GPU warps
- [x] coro-design.2: Design — GpuGenerator trait + MIR pass assessment

## Verified Conclusions
1. `WarpCooperativeTransform` requires **NO changes** — discriminant broadcast + return barrier already handle generator coroutine bodies
2. `GpuGenerator<R=()>` trait: `resume_warp() -> WarpCoroutineState<Y, R>`, lane-0 resume + broadcast
3. `WarpBroadcast<T>` trait: shfl.sync for <=128-bit (1-4 cycles), shared memory fallback for larger
4. Executor integration via `GeneratorTask` Future adapter — generators coexist with futures in WorkQueue
5. Streaming pipeline via `for_each_yield(gen, consumer, wcx)` — direct inline loop, zero-buffered

## Rejected Approaches
- Per-lane yield values (changes SIMT model, requires MIR pass changes)
- Separate generator task kind in executor (doubles complexity, Future adapter works)
- Channel-based pipeline (adds ring buffer, not zero-buffered)
- MIR pass yield-value broadcast (runtime trait dispatch is simpler)

## Open Questions
- `gen fn` vs `#[coroutine]` syntax (using raw coroutine — more expressive)
- Async generators `async gen` and per-lane divergent generators (T2+)

## Key Metrics
- Broadcast: shfl.sync ~1 cycle/32-bit; shared memory ~5 cycles
- Generator overhead: 1 discriminant + 1 yield-value broadcast per iteration

## Next Steps
1. coro-impl.1: Implement GpuGenerator + WarpBroadcast + for_each_yield in gpu-runtime
2. coro-impl.2: Streaming pipeline demo (fibonacci producer + consumer)
