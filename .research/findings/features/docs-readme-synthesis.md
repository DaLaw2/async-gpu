# docs-readme — Feature Synthesis

## Scope
README overhaul to match 62+ shipped stories.

## Key Gaps (priority order)
1. **12 capabilities absent**: GpuArray, AutoTuner, AutoScheduler, tiered memory, dyn dispatch, GpuHashMap, FlightRecorder, #[gpu_test], gradient checkpointing, tape fusion, unified channels, cross-block work dispatch
2. **6 capabilities under-documented**: par_iter, structured concurrency, coroutines, cost model, generics, GPU panic
3. **3 factual errors**: ThreadIndex (should be WarpIndex), "7 test crates" (should be 9), "130+ kernels" (actually 245)
4. **Crate map stale**: missing async-gpu facade, wrong test crate count, missing GpuVec/Pipeline/FlightRecorder from core types
5. **docs/ not linked**: ARCHITECTURE.md, CHANGELOG.md, getting-started.md exist in docs/ but README doesn't reference them

## Approach
- Feature matrix: add rows for GpuArray, AutoTuner, AutoScheduler, tiered memory, dyn dispatch, GpuHashMap, FlightRecorder, #[gpu_test]
- Add standalone code examples or details sections for par_iter, dyn dispatch, auto-tuning, transparent data
- Fix all 3 factual errors
- Update crate map with async-gpu facade and correct test crate count
- Link to docs/ARCHITECTURE.md and docs/CHANGELOG.md

## Risk
README length already ~514 lines. Adding 12 features needs careful use of collapsed sections.
