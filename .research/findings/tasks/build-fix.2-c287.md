# build-fix.2: README update + git status cleanup
**Cycle**: 287 | **Theme**: build-fix | **Kind**: experiment | **Status**: done

## Summary
Updated README.md with `#[warp_cooperative]` MIR pass documentation, patched toolchain build instructions, and async-pipeline example. Cleaned git status by updating .gitignore to exclude test output files, example Cargo.lock files, and toolchain build logs.

## Changes

### README.md
- **Hero example**: Replaced old `#[warp_async]` macro example with `#[warp_cooperative] async fn` using standard Future types
- **Quick Start**: Added `async-pipeline` example to run commands
- **Patched Toolchain**: New section with build instructions for Linux (.sh) and Windows (.ps1)
- **Examples**: Added `async-pipeline` description (small I/O + bulk sideband I/O)
- **Warp-Cooperative section**: Rewrote to document both approaches (#[warp_cooperative] MIR pass vs #[warp_async] proc macro) with comparison table
- **Architecture diagram**: Updated to show both MIR pass and proc macro
- **Crate map**: Added `async-pipeline/`, `scripts/` entries
- **Capabilities table**: Updated async runtime and warp-cooperative entries
- **Limitations**: Added note about patched rustc requirement for #[warp_cooperative]

### .gitignore
- Added: `gpu_autonomous.txt`, `gpu_roundtrip.txt` (test output files)
- Added: `examples/*/host/Cargo.lock`, `examples/*/kernel/Cargo.lock`
- Added: `.research/toolchain-build*.log`
- Result: `git status` is now clean (0 untracked files)

## Verification
- `bash scripts/ci-lint.sh` — all checks passed
- `git status -u` — only modified files are .gitignore, README.md, state.toml

**Confidence**: high

## Impact
- build-fix theme: ALL 3 criteria met → theme completed
  - C1: build-toolchain scripts work (confirmed in build-fix.1)
  - C2: README has build instructions ✓
  - C3: git status is clean ✓
