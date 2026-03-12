# ci.2: rust-toolchain.toml and toolchain requirements documentation
**Cycle**: 49 | **Theme**: ci | **Kind**: experiment | **Status**: done

## Summary
Created `rust-toolchain.toml` pinning the known-good nightly version (2025-08-25)
with all required components: rust-src, llvm-tools, llvm-bitcode-linker, and the
nvptx64-nvidia-cuda target.

## Findings

### Q: Which nightly version is currently known-good?
A: **nightly-2025-08-25** (rustc 1.91.0-nightly, commit 54c581243). This is the
version used throughout all 47 completed research tasks. All PTX output, Embassy
compilation, and std patching have been verified on this version.

**Confidence**: high

### Q: What components are required?
A: Four rustup components:
- `rust-src` — required for `-Zbuild-std=core` and `-Zbuild-std=std`
- `llvm-tools` — provides LLVM utilities
- `llvm-bitcode-linker` — required linker for nvptx64 cdylib output
- Target: `nvptx64-nvidia-cuda`

**Confidence**: high

### Q: How to document CUDA toolkit and driver requirements?
A: Updated in README.md: CUDA 12.0+ toolkit and SM70+ GPU required for running
(not for compiling). The host crate uses cudarc which dynamically loads CUDA at
runtime. No compile-time CUDA dependency.

**Confidence**: high

## Files Created
- `rust-toolchain.toml` — Pins nightly-2025-08-25 with all components
