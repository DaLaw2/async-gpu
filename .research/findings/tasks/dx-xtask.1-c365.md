# dx-xtask.1: Audit current build flows
**Cycle**: 365 | **Theme**: dx-xtask | **Kind**: investigation | **Status**: done

## Summary
Audited all build.rs patterns, ci-lint.sh, PTX post-processing, and kernel crate structure
to inform cargo xtask design.

## Findings

### Q: What build.rs patterns exist across examples?
A: All 6 example build.rs files share a 4-step pattern:
1. Read nightly version from `rust-toolchain.toml` (walk up 3 levels)
2. Run `cargo +{toolchain} build --release` in `../kernel/` with clean env vars
3. Patch PTX: `.target sm_30` → `.target sm_86`, `.version 6.0` → `.version 7.1`
4. Copy patched PTX to `OUT_DIR/kernel.ptx` for `include_str!`

Patched rustc detection exists in async-pipeline and parallel-search only.
Fallback: if build fails, use cached PTX (warning, not error).

**Confidence**: high

### Q: How does ci-lint.sh discover and build kernel crates?
A: Hard-coded list of 11 kernel paths (6 examples + 5 test crates).
Reads nightly from `rust-toolchain.toml`. No explicit PTX post-processing in CI.

**Confidence**: high

### Q: What PTX post-processing steps are needed?
A: Three transforms:
1. `.ptr .align N` removal (CUDA PTX JIT rejects these)
2. Panic extern stubbing (`.extern .func panic_*` → `.visible .func { trap; ret; }`)
3. Target version patch (`.target sm_30` → `.target sm_86`)

Currently only #3 is done (in build.rs). #1 and #2 were in deleted `postprocess-ptx.sh`.

**Confidence**: high

### Q: What is the minimal xtask interface?
A: `cargo xtask gpu-build [--all | NAME]` with:
- Kernel discovery (scan for `.cargo/config.toml` with nvptx64 target)
- Toolchain resolution from `rust-toolchain.toml`
- Build invocation with clean env
- Optional PTX post-processing
- `--list` to show discovered kernels

## Impact on Downstream Tasks
- dx-xtask.2: implement xtask crate based on these patterns
- dx-xtask.3: add PTX post-processing to xtask
