# ci-protect.2: Add test kernel PTX builds and doc-check to CI
**Cycle**: 120 | **Theme**: ci-protect | **Kind**: experiment | **Status**: done

## Summary

Extended CI pipeline to cover gpu-atomics and gpu-runtime (fmt, clippy, doc), plus PTX builds for 5 additional kernel crates. Removed std-build-test from CI since it requires patched-std directory not available in CI environment.

## Findings

### Q: Can all GPU kernel crates build PTX in CI?
A: All kernel crates except std-build-test can build in CI. std-build-test requires `build-std = ["std", "core", "panic_abort"]` which depends on a `patched-std` directory containing a custom std library — this is not available in CI and would require special setup (submodule or artifact download). Removed from CI for now.
**Confidence**: high

### Q: Can gpu-atomics and gpu-runtime pass clippy/fmt/doc on x86_64?
A: Yes, after the cfg stubs added in api-docs.2. All 4 public crates now pass full lint + doc generation on x86_64 stable toolchain.
**Confidence**: high

## Changes Made
- `.github/workflows/build.yml`:
  - Added fmt/clippy for gpu-atomics and gpu-runtime
  - Added doc generation for gpu-atomics and gpu-runtime
  - Added PTX builds: async-hostcall-test, async-pipeline-test, embassy-test, multi-warp-test, gpu-std-test
  - Removed std-build-test PTX build (patched-std dependency)
  - Fixed warp-macro clippy needless_borrows_for_generic_args (CI Rust 1.94 vs local 1.88)

## Verification
- CI #44 lint + host jobs passed; build-ptx failed only on std-build-test (now removed)
- CI #45 (post-fix) pending verification
