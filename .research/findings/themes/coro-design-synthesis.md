# Theme Synthesis: coro-design — Generator trait + MIR pass design

## Progress
- [x] coro-design.1: Investigation — map Rust coroutine semantics to GPU warps (DONE)
- [ ] coro-design.2: GpuGenerator trait design
- [ ] coro-design.3: MIR pass extension for generators
- [ ] coro-design.4: Streaming pipeline demo

## Verified Conclusions
1. Rust's `Coroutine` trait and `async fn` share the **same** `StateTransform` MIR pass — identical state machine infrastructure (discriminant, suspension points, dispatch switch)
2. `WarpCooperativeTransform` already processes generator coroutine bodies — discriminant broadcast and return barriers work unchanged
3. After `StateTransform`, `yield` is erased to discriminant-write + `Return` — no `Yield` terminators remain in MIR
4. Yield-value broadcast is a **runtime** concern (shfl.sync for ≤128-bit, shared memory for larger), not a MIR pass concern
5. The `GpuExecutor` infrastructure supports multiple generator tasks per warp via the existing work queue

## Rejected Approaches
- Creating a separate MIR pass for generators (unnecessary — existing pass already handles them)
- Using `Option<Y>` as the yield type (that's `gen fn` desugaring; raw `CoroutineState<Y,R>` is more expressive)
- Channel-based zero-buffering pipeline (adds ring buffer overhead; direct inline loop is simpler and truly zero-buffered)

## Open Questions
- Per-lane vs uniform yield values (uniform for I/O, per-lane for compute)
- Resume argument type R (start with `()`, extend later); `async gen` support (T2+)

## Key Metrics
- Broadcast cost: shfl.sync ~1 cycle/32-bit word; shared memory ~5 cycles (48KB on sm_75)

## Next Steps
1. Design `GpuGenerator` trait with `resume_warp(&mut self, arg: R, wcx: &mut WarpContext) -> WarpCoroutineState<Y, R>`
2. Implement yield-value broadcast (shfl.sync path) + inline producer-consumer demo (zero buffering)
