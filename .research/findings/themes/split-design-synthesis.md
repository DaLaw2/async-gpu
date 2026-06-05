# split-design — Theme Synthesis

**Epic**: kernel-split (break gpu-kernel-std into per-feature crates)
**Status**: active | **Tasks completed**: 1/? | **Updated**: 2026-06-05

## Key Finding
The proposed 4-crate split (core / compute / io / test) has **zero circular dependencies**. All 18 source files map cleanly to exactly one crate. Cross-file imports flow strictly downward: 10 files import from helpers.rs (→ core), 1 file imports from hybrid.rs (→ same io crate). No file straddles boundaries.

## Entry Point Distribution
core: 18 kernels + 4 infra | compute: 81 kernels | io: 38 kernels | test: 61 kernels

## Critical Path Items
1. `dynamic_smem` global_asm must be duplicated in each crate using shared memory
2. lib.rs stdio infrastructure (~174 lines, 3 statics) must be extracted to core as public API
3. Crate type decision: core as `rlib` (linkable) vs `cdylib` (separate PTX) drives the entire architecture
4. Host loader must support multi-cubin loading (investigate in next task)

## Risk Summary
Medium: global_asm duplication + stdio infra extraction. Low: all else. No blockers found.
