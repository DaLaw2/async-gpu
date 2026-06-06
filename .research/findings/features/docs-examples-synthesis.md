# docs-examples synthesis

24 standalone examples audited (14 std/, 10 hostcall/). All compile-ready.

## Missing standalone examples (5 features)
1. **transparent-data** — GpuArray<T> with zero explicit transfers
2. **auto-fusion** — elementwise chain fused into single kernel
3. **dyn-dispatch** — Box<dyn Trait> polymorphism on GPU
4. **auto-tuning** — warmup-based kernel parameter search
5. **par_iter** — GPU parallel iterator (map/filter/fold/collect)

## Existing but incomplete (2 examples)
- `gpu-channels` — missing README.md
- `structured-concurrency` — missing README.md, content is current

## Borderline gaps
- **gpu_test macro** — used in test harness, no "how-to" example
- **tiered-memory** (SharedRef/GlobalRef) — no standalone showcase

## Dep hygiene: 11/24 examples import gpu_host directly (not async_gpu facade)

## Priority order for new examples
1. par_iter (kernel demos exist, just needs host driver)
2. transparent-data (GpuArray API is simple, high user value)
3. dyn-dispatch (kernel tests exist, needs showcase wrapper)
4. auto-tuning (API is self-contained)
5. auto-fusion (needs nn feature, more complex)
