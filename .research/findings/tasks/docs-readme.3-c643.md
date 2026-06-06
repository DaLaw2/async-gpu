# docs-readme.3 — Crate Map, Limitations, Architecture Sections Update

## Summary

Updated README.md crate map, limitations, and architecture sections. Added a
new Architecture section with a 5-line build model summary linking to
docs/ARCHITECTURE.md. Rewrote the crate map with correct 19-crate count
organized by layer (Facade/Core/Kernel/Test), added missing key modules
(gpu_array.rs, scope.rs, par_iter.rs), and listed all 9 test crates by name.
Updated limitations to remove dyn dispatch (no longer a limitation), add
multi-GPU and FP8 notes, and clarify partial std scope.

## Findings

### Crate Map Changes
- Corrected total count: 19 crates (was implicitly undercounted)
- Layer headers: Facade (1), Core (5), Kernel (4), Test (9)
- Added gpu_array.rs to gpu-host key modules (was missing — the most important type)
- Added scope.rs and par_iter.rs to gpu-runtime key modules (were missing)
- Listed all 9 test crates by name instead of just a count
- Added full crate paths (e.g., crates/core/gpu-host/ not just gpu-host/)

### Limitations Changes
- REMOVED: dyn dispatch — works fully now (Box<dyn Trait>, &dyn Fn, hashbrown)
- ADDED: Single GPU limitation (device selection works, no cross-device orchestration)
- UPDATED: f32+f16 → also note FP8 not implemented
- UPDATED: Partial std → list what works (Vec, HashMap, Mutex, File, println!)

### Architecture Section
- New section between "Neural network module" details and "Crate Map"
- 5-line summary of two-workspace build model
- Links to docs/ARCHITECTURE.md for full details

### Line Count
- Before: 631 lines
- After: 652 lines (under 700 limit)

## Open Questions

1. The task description said "15 crates" but actual count is 19. Used the real count.
2. Documentation section already existed from docs-readme.2 — no changes needed.
