# split-execute.5: Refactor gpu-kernel-std -> gpu-kernel-test

## Status: DONE

## Summary

Renamed the crate directory and package from `gpu-kernel-std` to `gpu-kernel-test`
to reflect its actual purpose: test and demo kernels. This is the final task in
the kernel-split epic (split-execute theme).

## What changed

### Crate rename
- `crates/kernel/gpu-kernel-std/` -> `crates/kernel/gpu-kernel-test/`
- `Cargo.toml`: name = "gpu-kernel-test", updated description
- `lib.rs`: Updated module-level doc comment

### Build infrastructure
- `scripts/build-kernel-std.sh` -> `scripts/build-kernel-test.sh` (renamed + updated)
- `crates/core/gpu-host/build.rs`: Updated kernel dir path, PTX source filename
  (`gpu_kernel_test.ptx` instead of `gpu_kernel_std.ptx`), rerun-if-changed paths
- `scripts/setup.sh`: All 3 references updated
- `scripts/ci-lint.sh`: Exclusion comment updated

### Host-side references
- `crates/core/gpu-host/src/lib.rs`: Updated ptx module doc comments
- PTX/cubin filenames in gpu-host/ kept as `kernel_std.ptx` / `kernel_std.cubin`
  for backward compatibility (the build script copies to both names)

### Comment updates across codebase
- `crates/core/gpu-runtime/src/entry.rs`
- `crates/kernel/gpu-kernel-io/src/lib.rs`
- `crates/test/gpu-test-harness/` (tests_std.rs, tests_par_iter.rs, gpu_tests.rs)
- `crates/test/gpu-test-macro/src/lib.rs`
- `crates/test/async-hostcall-test/src/lib.rs`
- `crates/test/std-build-test/src/lib.rs`
- `examples/hostcall/warp-cooperative/` (main.rs, run.sh)
- `examples/hostcall/structured-concurrency/` (main.rs, run.sh)
- `examples/hostcall/gpu-channels/` (main.rs, run.sh)
- `examples/std/thread-demo/src/main.rs`
- `std-patches/sys_stdio_cuda.rs`
- `.claude/commands/dev-dispatch.md`
- `README.md` (also added gpu-kernel-core, gpu-kernel-compute, gpu-kernel-io entries)

## Verification
- gpu-kernel-test: PTX build succeeded (7.2 MB, 1m00s)
- gpu-kernel-core: builds OK
- gpu-kernel-compute: builds OK
- gpu-kernel-io: builds OK
- fmt checks: all pass
- No orphan references to old name remain (verified via grep)

## Design decision
- Kept PTX/cubin output filenames as `kernel_std.ptx` / `kernel_std.cubin` in gpu-host/
  for backward compatibility. The `ptx::KERNEL_STD` constant name is kept since it's
  used by gpu-test-macro and gpu-test-harness. Renaming these would be a separate
  breaking change with no functional benefit.
