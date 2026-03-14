# toolchain-auto.1: Create scripts/build-toolchain.sh
**Cycle**: 273 | **Theme**: toolchain-auto | **Kind**: experiment | **Status**: done

## Summary
Created `scripts/build-toolchain.sh` — a single script that builds the full patched toolchain from source, including nvptx64 target support. The script handles rustc source cloning, compiler patch application, std patch application into the rustc source tree, bootstrap configuration, and x.py build invocation.

## Design Decisions

### Q: Should we build stage1 or stage2?
A: **Stage1 is sufficient.** Stage2 rebuilds the compiler with itself (for validation), but for our purposes the stage1 compiler already has the WarpCooperativeTransform MIR pass. Stage2 would double build time for no benefit.

**Confidence**: high

### Q: How to apply std patches into the rustc source tree?
A: The existing `apply-std-patches.sh` copies from `rustc-src/library/std/` to an output directory. For the toolchain build, we need patches applied to `patched-rustc/library/std/` instead. Solution: run `apply-std-patches.sh` to a temp directory, then overlay the `src/` directory onto `patched-rustc/library/std/src/`. A marker file (`.async_gpu_std_patched`) prevents re-application on subsequent runs.

**Confidence**: high

### Q: How to handle nvptx64 target?
A: Set `target = ["x86_64-pc-windows-msvc", "nvptx64-nvidia-cuda"]` in `bootstrap.toml`. x.py automatically builds core/alloc (but not full std) for nvptx64. The `download-ci-llvm = true` option downloads pre-built LLVM to avoid building it from source (saves ~30 min).

**Confidence**: medium — untested end-to-end, x.py nvptx64 support may have quirks

## Script Features
- `--from-scratch`: Clean rebuild from fresh rustc clone
- `--print-sysroot`: Output sysroot path for scripting
- `--targets=...`: Customize build targets
- Incremental build support (subsequent builds are fast)
- Build log saved to `.research/toolchain-build.log`
- Automatic host triple detection
- Sysroot verification (checks nvptx64 libs exist)

## Open Questions
- Will `x.py build compiler library` with nvptx64 target actually produce usable core/alloc rlibs?
- Does the patched WarpCooperativeTransform MIR pass survive the bootstrap process without errors?
- Is `download-ci-llvm = true` reliable on Windows with MSVC?

## Impact on Downstream Tasks
- async-yield.3 partially unblocked: script exists, but user needs to actually run it to build the toolchain
- toolchain-auto.2 (PTX post-processing) is independent and can proceed in parallel
