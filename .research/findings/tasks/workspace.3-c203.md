# workspace.3: Single-source nightly toolchain version
**Cycle**: 203 | **Theme**: workspace-consolidation | **Kind**: experiment | **Status**: done

## Summary
Nightly toolchain version now defined in one place: `rust-toolchain.toml`.
Both `scripts/ci-lint.sh` and `.github/workflows/build.yml` read from it
instead of hardcoding the version.

## Findings
### Q: Can ci-lint.sh read NIGHTLY from rust-toolchain.toml instead of hardcoding?
A: Yes. `grep '^channel' rust-toolchain.toml | sed` extracts the version string.
Replaced `NIGHTLY="nightly-2026-03-11"` with dynamic extraction.
**Confidence**: high

### Q: Can CI workflow reference the same file?
A: Yes. Added a step that reads `rust-toolchain.toml` and outputs the channel
via `$GITHUB_OUTPUT`, then references it in the toolchain install step.
**Confidence**: high
