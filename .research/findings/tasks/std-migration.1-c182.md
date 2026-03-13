# std-migration.1: Create gpu-kernel-std crate + verify println/Vec/format work
**Cycle**: 182 | **Theme**: std-migration | **Kind**: experiment | **Status**: blocked

## Summary
Created gpu-kernel-std crate with gpu-runtime dependency (eliminates 430+ lines of
duplicated hostcall PTX from std-build-test). Build fails because patched-std is
incompatible with current nightly (2026-03-11) — core library has breaking API changes.

## Findings

### Q: Can a new crate build with -Zbuild-std=std and produce valid PTX?
A: Not yet. The patched-std library (in `patched-std/`) was created for an older nightly
and is now incompatible with nightly-2026-03-11. Errors include:
- `feature(doc_cfg_hide)` removed in 1.92.0
- `UnsizedConstParamTy` built-in macro not found
- Various `rustc_*` attributes no longer recognized

The build mechanism works (`__CARGO_TESTS_ONLY_SRC_ROOT` env var points cargo to patched-std),
but the patched std source must be rebased onto the current nightly's core/std.

**Confidence**: high

### Q: Do println!, Vec, String, format! all work without duplicating hostcall code?
A: The crate was designed to use `gpu_runtime::hostcall::gpu_hostcall_print()` for the
PAL `gpu_stdout_write()` function, which would eliminate the 430+ lines of duplicated
inline PTX in std-build-test. This approach should work once the patched std builds.

**Confidence**: medium (untested due to blocker)

### Q: What is the PTX size difference vs no_std kernel?
A: Cannot measure — blocked by patched std incompatibility.

## Blocker
**patched-std must be rebased onto nightly-2026-03-11 core/std.** This requires:
1. Copy nightly-2026-03-11 sysroot library to `patched-std/library/`
2. Re-apply the CUDA-specific patches:
   - `sys/alloc/cuda.rs` — slab allocator for GPU
   - `sys/alloc/mod.rs` — add nvptx64 cfg for cuda allocator
   - `sys/stdio/cuda.rs` — PAL stdout/stdin routing to extern functions
   - Various cfg changes for `target_arch = "nvptx64"`
3. Verify std-build-test still passes

This is a user task (modifies toolchain sources outside the repo).

## Impact on Downstream Tasks
- All std-migration and std-fs tasks are blocked until patched-std is updated
- gpu-error-propagation tasks are NOT blocked (work with no_std)
- host-sdk tasks are NOT blocked
