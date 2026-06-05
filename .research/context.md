## Current Focus
**Cycle 621 SAVED** (2026-06-05). kernel-split all themes complete (13/13 tasks).
Litmus test FAILED for test crate: 30 min (ptxas scales with complexity, not size).
4/5 epic criteria met. Criterion 5 (≤5 min) unmet for largest crate.
Decision needed: further split test crate, or adjust criterion to exclude ptxas.

## Recent Decisions
- 2026-06-05: opt-level 1 default, release-prod preserves opt-level 3
- 2026-06-05: ptxas bottleneck identified: 76 entry points with std complexity → 30 min
- 2026-06-05: PTX shrank 47% (11.4→6.0 MB) but ptxas time unchanged
- 2026-06-05: Smaller crates (core 1.3MB, compute 1.9MB, io 2.3MB) likely meet target

## Tried & Rejected
- PTX size reduction as ptxas optimization: ptxas scales with code complexity, not byte count
- opt-level 1: reduces PTX size but not ptxas time significantly

## Active Constraints
- GTX 1660 (sm_75): ptxas ~30 min for 76-entry-point crate
- test-integration.2 kernels stashed in git stash@{0}
- ptxas is an NVIDIA black box — cannot optimize its runtime

## Key Metrics
- 4 kernel crates: core (17), compute (84), io (55), test (76) entry points
- PTX sizes: core 1.3MB, compute 1.9MB, io 2.3MB, test 6.0MB
- Single-crate PTX build: 27s. ptxas: 30 min (test), likely <5 min for others
- 776 tasks completed, 48 epics archived

## Next
1. Resolve kernel-split criterion 5: further split test crate OR adjust criterion
2. Unstash test-integration.2 + verify
3. Continue with T1 epics (gpu-type-safety, gpu-generics)
