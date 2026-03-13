# build-auto.2: Single-source CI — ci-lint.sh as source of truth
**Cycle**: 201 | **Theme**: build-automation | **Kind**: design | **Status**: done

## Summary
Simplified `.github/workflows/build.yml` to call `bash scripts/ci-lint.sh` directly
for the lint job, eliminating duplicated fmt/clippy/doc steps. Removed the separate
build-ptx job since ci-lint.sh already builds all PTX kernels. Auto-discovered PTX
stubs in the build-host job using the same grep pattern from build-auto.1.

## Design Decision
**ci-lint.sh is the single source of truth for lint and PTX build steps.**

Before: build.yml had ~60 lines of duplicated fmt/clippy/doc steps that drifted
out of sync (missing gpu-libc fmt, missing gpu-host doc enforcement, missing
async-io/vector-math checks). After: build.yml calls ci-lint.sh and stays in sync
automatically.

## Findings
### Q: What is the best way to keep CI and local lint in sync?
A: CI calls the same script developers run locally. One script, two environments.
**Confidence**: high

### Q: Should ci-lint.sh be the source of truth, or a shared crate list?
A: ci-lint.sh. It's executable, testable, and already works. A declarative config
would add indirection without benefit — the script IS the config.
**Confidence**: high
