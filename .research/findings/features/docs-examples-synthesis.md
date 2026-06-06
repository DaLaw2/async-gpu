# docs-examples synthesis

27+3 = 30 standalone examples. 18 in examples/std/, 12 in examples/hostcall/.

## Created in docs-examples.3 (3 examples)
1. **auto-fusion** (std) — FusionOptimizer tape analysis, FusionCodegen JIT, fused vs unfused benchmark
2. **par-iter** (hostcall) — GPU parallel iterators: map/filter/fold/zip/enumerate/collect demos
3. **gpu-test** (std) — #[gpu_test] proc macro usage, assert!/panic! propagation

## Previously created (3 examples)
4. **transparent-data** — GpuArray<T> auto-sync, zero explicit cudaMemcpy
5. **dyn-dispatch** — &dyn Trait + Box<dyn Trait> vtable dispatch on GPU
6. **auto-tuning** — AutoTuner warmup-based block size search + TuningCache

## All 6 new examples verified: cargo check passes with zero warnings
