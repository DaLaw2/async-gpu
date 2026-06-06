# docs-readme.2: README Hero, Feature Matrix, Performance Table, Progressive Examples Rewrite
**Cycle**: 643 | **Feature**: docs-readme | **Kind**: experiment | **Status**: done

## Summary

Rewrote four major README sections plus fixed all 3 factual errors from the audit.
Final README: 631 lines (under 700 limit), up from 518.

### Changes Made

1. **Hero section**: Rewrote to lead with "Write plain Rust. The compiler handles GPU."
   Mentions GpuArray<T>, auto-fusion, Box<dyn Trait>, auto-tuning, par_iter, and 245+ kernels.
   Added Conv2D 54.8% peak stat.

2. **Feature matrix**: Expanded from 16 rows to 29 rows. Added all 12 missing capabilities:
   GpuArray<T>, SharedRef/GlobalRef, GpuHashMap, AutoScheduler, par_iter, cross-block work
   dispatch, dynamic dispatch, AutoTuner, tape-level fusion, gradient checkpointing,
   FlightRecorder, #[gpu_test]. Improved 6 under-documented items with links and details.
   Grouped by category (Data, Runtime, Safety, Perf, Debug, I/O, Compute, ML).

3. **Performance table**: Added Conv2D Winograd (2753 GFLOPS, 54.8% peak), auto-tuning
   (1.4x best-vs-worst), dyn dispatch (<1.15x overhead). Added two Winograd rows to
   full benchmark details table.

4. **Progressive examples**: Expanded from 3 to 6 snippets. Added: Transparent Data
   (GpuArray<T>), Auto-Tuning (AutoTuner), Dynamic Dispatch (Box<dyn Trait> on GPU).

5. **Factual errors fixed**:
   - L131: ThreadIndex<'kernel> -> WarpIndex<'scope>
   - L498: "7 test crates" -> "9 test crates"
   - L143: "130+ kernels" -> "245+ kernels"

6. **Crate map**: Added async-gpu facade crate, expanded gpu-host with auto_tune.rs,
   scheduler.rs, resource_report.rs. Expanded gpu-runtime with tiered_mem.rs,
   collections.rs, unified_channel.rs, grid_work.rs, generator.rs, safety.rs,
   flight_recorder.rs. Added docs/ directory. Updated kernel-test description.

7. **Core types table**: Added GpuArray<T>, GpuVec<T>, Pipeline, FlightRecorder.

8. **Documentation section**: Added links to docs/ARCHITECTURE.md, CHANGELOG.md,
   getting-started.md, VALIDATION.md.

9. **Additional features**: Added unified channels, AutoScheduler par_map to runtime features.

## Findings

- All 12 absent capabilities from the audit are now in the feature matrix with source links
- All 6 under-documented capabilities now have expanded descriptions
- All 3 factual errors are corrected
- Crate map now reflects actual project structure
- docs/ is linked from README

## Open Questions

1. Some feature matrix links are relative paths to source files — these work on GitHub
   but not in other markdown renderers. Consider whether to use full GitHub URLs.
2. The auto-tuning and dyn dispatch progressive examples are based on API docs and test
   code rather than standalone runnable examples in examples/. Creating dedicated example
   crates would make these more discoverable.
