# Contributing to async_gpu

Thanks for your interest in contributing! This guide covers the development workflow for the project.

## Prerequisites

- [Rust](https://rustup.rs/) with stable and nightly toolchains
- nvptx64 target: `rustup target add nvptx64-nvidia-cuda --toolchain nightly-2026-03-11`
- Rust nightly src: `rustup component add rust-src --toolchain nightly-2026-03-11`
- NVIDIA GPU (SM 70+) with CUDA 12.x driver
- (Optional) Patched rustc for `#[warp_cooperative]` — see README for build instructions

## Project Structure

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system design. The key directories:

```
crates/
  core/        Host SDK (gpu-host), protocol, runtime, atomics, libc
  kernel/      GPU kernel crates (require nightly + nvptx64)
  macro/       Proc macros (#[warp_async])
  test/        GPU test kernels
examples/      Self-contained examples with kernel/ and host/ subdirs
rustc-patches/ Custom MIR pass patches for the Rust compiler
scripts/       Build and CI scripts
```

## Development Workflow

### Running CI Locally

Before pushing, always run the local CI lint script:

```bash
bash scripts/ci-lint.sh
```

This checks formatting, clippy, docs, PTX kernel builds, and host crate compilation. It mirrors the GitHub Actions CI.

### Adding a New Example

1. Create `examples/my-example/kernel/` with:
   - `Cargo.toml` (crate-type = ["cdylib"], depends on gpu-runtime)
   - `.cargo/config.toml` (target = nvptx64, build-std = ["core"])
   - `src/lib.rs` with `#[no_mangle] pub unsafe extern "ptx-kernel" fn ...`

2. Create `examples/my-example/host/` with:
   - `Cargo.toml` (depends on gpu-host, cudarc)
   - `build.rs` (calls cargo to build the kernel PTX)
   - `src/main.rs` (loads PTX, launches kernel, verifies output)

3. Add run scripts: `run.sh` and `run.bat`
4. Add a `README.md` with expected output
5. Add the kernel to `PTX_KERNELS` and the host to the host check section in `scripts/ci-lint.sh`
6. Reference the example in the main README.md

Look at `examples/hello-gpu/` as a template.

### Adding a New GPU Kernel to gpu-kernel

1. Add the kernel function in `crates/kernel/gpu-kernel/src/lib.rs`
2. Register the kernel name in `gpu-host` so it can be loaded
3. Add `include_str!` for the PTX file in the host crate — CI automatically creates stubs

### Modifying the Hostcall Protocol

The hostcall protocol is in `crates/core/gpu-protocol/`. Changes here affect both GPU and host sides:

- `gpu-protocol` defines service IDs, packet layout, and constants
- `gpu-runtime` has the GPU-side hostcall submission functions
- `gpu-host` has the host-side listener that processes requests

Changes to the protocol require updating all three crates.

### Working with the MIR Pass

The custom rustc MIR pass lives in `rustc-patches/`. To modify it:

1. Apply patches to a rustc source checkout
2. Build the patched toolchain: `bash scripts/build-toolchain.sh`
3. Test with `examples/warp-cooperative/`

The MIR pass inserts `bar.warp.sync` barriers and `shfl.sync.idx` discriminant broadcasts into async state machines.

## Code Style

- **Error handling**: Use `thiserror` for custom error types. **`anyhow` is prohibited.**
- **GPU functions**: All cross-crate GPU functions must be `#[inline(always)]` (no PTX linker)
- **Unsafe**: GPU/CUDA interop requires `unsafe`. Add `// SAFETY:` comments explaining invariants
- **Formatting**: `cargo +stable fmt` — enforced by CI
- **Linting**: `cargo +stable clippy -- -D warnings` — enforced by CI
- **Docs**: Public items in `gpu-host` and `gpu-protocol` require doc comments (`-D missing_docs`)

## Architecture Notes

- NVVM intrinsics are broken on current LLVM — use inline PTX assembly only
- `#[thread_local]` is not supported on nvptx64 (no LLVM GlobalTLSAddress)
- LLVM treats inline asm as opaque barrier, so warp-cooperative code is safe
- PTX post-processing is needed for some builds (remove `.ptr .align`, stub panic symbols)

## License

By contributing, you agree that your contributions will be licensed under both MIT and Apache-2.0, matching the project's dual license.
