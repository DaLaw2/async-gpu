# onboarding.1: Create single-command build+run for hello-gpu
**Cycle**: 109 | **Theme**: onboarding | **Kind**: experiment | **Status**: done

## Summary

hello-gpu already had a build.rs that auto-compiles the kernel crate to PTX. The gap was README presentation: the multi-step gpu-host flow was the primary quickstart, making the project look harder to run than it is. Added root-level `run-hello-gpu.sh`/`.bat` scripts and restructured README to make hello-gpu the one-command entry point.

## Findings

### Q: What is the simplest way to automate kernel build + PTX copy + host run?
A: hello-gpu's build.rs already handles it — `cargo run --release` from `examples/hello-gpu/host` does everything. The gap was discoverability, not automation. Added `run-hello-gpu.sh` and `run-hello-gpu.bat` at repo root for zero-friction onboarding. These just `cd` into the hello-gpu host dir and run cargo.
**Confidence**: high

### Q: Should this be a build.rs, a cargo-make task, or a shell script?
A: Shell script (+ .bat for Windows) is the simplest and most portable. A build.rs already exists in hello-gpu/host. cargo-make would add a dependency. The shell scripts are 5 lines each and require no additional tooling.
**Confidence**: high

## Changes Made
- Created `run-hello-gpu.sh` (bash, executable) at repo root
- Created `run-hello-gpu.bat` (Windows batch) at repo root
- Restructured README Quick Start: hello-gpu is now primary, multi-step flow is in collapsible details
- Prerequisites simplified: just Rust + NVIDIA GPU (nightly auto-installs via rust-toolchain.toml)

## Unexpected Discoveries
- The prerequisites section was misleading — it listed manual nightly/component installation, but `rust-toolchain.toml` at repo root handles all of that automatically via rustup.

## Open Questions
- Should gpu-host also get a build.rs for auto-compiling all 6 kernel crates? (Complex — 6 different kernel crates with different build configs. Not needed for onboarding, but would improve DX for the full test suite.)

## Impact on Downstream Tasks
- onboarding.2 (README update) is partially done — README Quick Start already updated. May still need polish.
