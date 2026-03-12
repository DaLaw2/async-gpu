# ci.1: GitHub Actions workflow for PTX compilation
**Cycle**: 49 | **Theme**: ci | **Kind**: experiment | **Status**: done

## Summary
Created GitHub Actions workflow at `.github/workflows/build.yml` with two jobs:
1. `build-ptx`: Compiles gpu-kernel to PTX on ubuntu-latest using nightly-2025-08-25
   with nvptx64-nvidia-cuda target. No GPU required (PTX is text output).
2. `build-host`: Checks gpu-host compiles on stable Rust. Does not require CUDA
   toolkit since cudarc uses runtime loading.

## Findings

### Q: Can GitHub Actions compile nvptx64 PTX without a GPU?
A: **Yes.** PTX compilation only needs the LLVM PTX backend (included in rustc) and
llvm-bitcode-linker (installed via rustup component). No NVIDIA driver or GPU needed.
The output is PTX text that can be loaded by the host at runtime via cudarc.

**Confidence**: high (verified locally — build succeeds without CUDA toolkit)

### Q: What CUDA toolkit version is available in Actions runners?
A: Not needed for PTX compilation. For host compilation, cudarc uses runtime loading
(dlopen) so no CUDA SDK is needed at compile time. Only at runtime (which CI doesn't
test — no GPU in standard runners).

**Confidence**: high

### Q: How to cache the nightly toolchain and CUDA SDK?
A: Using `actions/cache@v4` for `~/.cargo/registry`, `~/.cargo/git`, and the target
directory. The `dtolnay/rust-toolchain@master` action handles toolchain installation
and is cached by default.

**Confidence**: high

## Files Created
- `.github/workflows/build.yml` — CI workflow
