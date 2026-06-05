# split-loader.4: Dev-mode opt-level reduction

## Summary

Reduced PTX compile profile from opt-level 3 + fat LTO to opt-level 1 + no LTO
for default dev builds. Added `release-prod` profile for benchmarks/shipping.

## Changes

1. **4 kernel Cargo.toml files**: `[profile.release]` now uses opt-level 1, no
   LTO. New `[profile.release-prod]` inherits release with opt-level 3, fat LTO.
2. **build-kernels.sh**: Added `--prod` flag. Default uses `--release` (dev),
   `--prod` uses `--profile release-prod`. Output directory handled correctly.
3. **build-kernel-test.sh**: Same `--prod` flag treatment.

## PTX size comparison

| Crate             | opt-level 3 + fat LTO | opt-level 1 + no LTO | Delta  |
|-------------------|------------------------|-----------------------|--------|
| gpu-kernel-core   | 1,054,286 B (1.0 MB)  | 1,405,562 B (1.3 MB)  | +33%   |
| gpu-kernel-compute| 1,467,467 B (1.4 MB)  | 2,037,383 B (1.9 MB)  | +39%   |
| gpu-kernel-io     | 1,849,390 B (1.8 MB)  | 2,412,930 B (2.3 MB)  | +30%   |
| gpu-kernel-test   | 7,221,049 B (6.9 MB)  | 6,224,000 B (5.9 MB)  | -14%   |

Smaller crates grow because fat LTO was eliminating dead code aggressively.
The test crate shrinks because opt-level 3 inlining inflates large codebases.

## Entry point verification

All 4 crates produce identical entry point counts at both optimization levels:
- core: 17, compute: 84, io: 55, test: 76

## Build time observation

The test crate built in 1m08s at opt-level 1. The primary time savings come from
LLVM doing less optimization work (fewer inlining passes, no LTO link phase).
The real payoff is in iterative development where dependencies are already cached.

## Key insight

The build speed improvement matters more than PTX size. PTX size increase for
smaller crates is acceptable in dev mode — ptxas compilation time (which dominates)
scales with complexity, not raw PTX byte count. Production builds via `--prod`
retain full optimization.
