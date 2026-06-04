# lib-docs Theme Synthesis

## Status
lib-docs.1 (guide structure design) complete. lib-docs.2 (writing) not started.

## Key Design Decisions
- 5-step guide: Setup -> Run hello-gpu -> Write kernel -> Write host -> Run & next
- Targets Scenario A (stock nightly, core-only kernels) as primary path (~17 min)
- Scenario B (patched std) deferred to "Level Up" appendix to stay under 30 min
- SAXPY as first-write kernel (canonical GPU compute, not I/O)
- Show one-liner API first (Step 2), builder API second (Step 4)
- Build.rs provided as copy-paste template, not written from scratch

## Critical Path
- build.rs template for kernel PTX compilation is the highest-friction element
- Guide should use async-gpu facade consistently (not gpu-host directly)
- Troubleshooting section needed for nightly toolchain errors

## Dependencies
- setup.sh (lib-toolchain.2): exists and works, no blockers
- Example migration to async-gpu (lib-cleanup.6): guide uses async-gpu regardless
- Actual writing deferred to lib-docs.2

## Open Items
- Whether to create examples/template/ as copy-and-modify starter
- Guide location: recommend docs/getting-started.md linked from README
- Raw nvptx intrinsics vs gpu_runtime::index for thread indexing in guide
