# docs-readme — Feature Synthesis

## Scope
README overhaul to match 62+ shipped stories.

## Status: DONE (docs-readme.2)
All audit findings resolved:
- 12 absent capabilities: added to feature matrix with source links
- 6 under-documented capabilities: expanded with details
- 3 factual errors: fixed (WarpIndex, 9 test crates, 245+ kernels)
- Crate map: updated with async-gpu facade, all gpu-runtime modules, docs/
- Core types: added GpuArray, GpuVec, Pipeline, FlightRecorder
- Progressive examples: 3 new (transparent data, auto-tuning, dyn dispatch)
- Performance: Conv2D Winograd 54.8%, auto-tuning 1.4x, dyn dispatch <1.15x
- Documentation section: links to docs/ directory

## Remaining Work
- docs-readme.3 (if needed): standalone example crates for auto-tuning, dyn dispatch
- ARCHITECTURE.md and CHANGELOG.md updates tracked under separate features

## Risk
None. README at 631 lines (under 700 limit). All content verified against source.
