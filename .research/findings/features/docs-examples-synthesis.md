# docs-examples synthesis

30 standalone examples: 19 in examples/std/, 11 in examples/hostcall/.

## Verification (docs-examples.4)

- **30/30 compile** via `cargo check` — zero errors
- **30/30 have README.md** — 2 were missing (gpu-channels, structured-concurrency), now added
- Minor unused-variable warnings in 3 std examples (cifar-train, diff-physics, gpt2-lora) — non-blocking

## New examples created in this feature (docs-examples.1-3)

1. **auto-fusion** (std) — FusionOptimizer tape analysis, FusionCodegen JIT, fused vs unfused benchmark
2. **par-iter** (hostcall) — GPU parallel iterators: map/filter/fold/zip/enumerate/collect
3. **gpu-test** (std) — #[gpu_test] proc macro usage, assert!/panic! propagation
4. **transparent-data** (std) — GpuArray<T> auto-sync, zero explicit cudaMemcpy
5. **dyn-dispatch** (std) — &dyn Trait + Box<dyn Trait> vtable dispatch on GPU
6. **auto-tuning** (std) — AutoTuner warmup-based block size search + TuningCache

## Missing READMEs added (docs-examples.4)

- `examples/hostcall/gpu-channels/README.md` — oneshot/MPSC channels + async executor
- `examples/hostcall/structured-concurrency/README.md` — BlockScope/GridScope, spawn_all, nested scopes
