# build-fix.1: Investigation — audit build scripts
**Cycle**: 285 | **Theme**: build-fix | **Kind**: investigation | **Status**: done

## Summary
Build infrastructure is solid (scripts well-structured, platform split clean). Main gap: README lacks patched toolchain documentation. No broken scripts — postprocess-ptx.sh was deleted intentionally (sed post-processing is done inline). Git status has only test output files and Cargo.lock files as noise.

## Findings

### Q: Which scripts are broken?
A: **None are broken.** All scripts function correctly:
- build-toolchain.sh (Linux) — robust, 187 lines
- build-toolchain.bat (Windows) — working, hardcoded VS2022 Community path
- ci-lint.sh — auto-stubs PTX, clear reporting
- apply-rustc-patches.sh / apply-std-patches.sh — correct
- gen-rustc-patches.sh / gen-std-patches.sh — correct
- pre-push.sh — minimal but correct (not tracked, in .gitignore)

### Q: What's the minimal set of build steps?
A:
- **Quick (stock nightly)**: clone → `rustup install nightly-2026-03-11` → `cargo run` examples ✓ documented
- **Full (patched toolchain)**: clone → `bash scripts/build-toolchain.sh` → set RUSTC → build ✗ NOT documented

### Q: Git status cleanup needed?
A: Minor. Untracked files: 2x Cargo.lock (examples), 2x test output .txt files. These should be gitignored or cleaned.

**Confidence**: high

## Impact
- build-fix.2: Focus on README update (add patched toolchain section) + gitignore cleanup. No broken scripts to fix.
