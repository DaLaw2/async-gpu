# split-loader theme synthesis

**Theme**: split-loader — Host-side multi-PTX loading for kernel split
**Epic**: kernel-split (T0)

## Status

All 5 tasks complete. Multi-PTX loading, per-crate cubin, dev-mode
profiles, and litmus test measurement done.

## What shipped

- 4 canonical constants + PtxModule catalog + .module() API
- Per-crate cubin builds (ptxas for each crate independently)
- Cubin embedding via include_bytes! + PtxModule.cubin field
- Dev-mode opt-level 1 default, --prod for full opt
- build-kernels.sh parallel ptxas + build-kernel-test.sh

## Litmus test result: FAILED

PTX cargo build: 27s. ptxas (6 MB test crate): 30 min. Total: ~30.5 min.
ptxas time does NOT scale with PTX size — it scales with code complexity.
The test crate (76 entry points, std-heavy) is as slow as the old 11 MB
unified build. Smaller crates (core/compute/io, 1-2 MB) likely < 5 min.

## What remains

- Reduce test crate complexity to hit 5 min target (future epic)
- Deprecation warnings on aliases + call-site migration (Phase 3)
