# lib-docs.2: Getting-Started Guide + Rustdoc Examples

Written the getting-started guide at `docs/getting-started.md` and linked it
from `README.md`.

## Summary

- **Guide location**: `docs/getting-started.md`
- **Word count**: ~1,738 words
- **Line count**: 456 lines (within 500-line target)
- **Estimated reading time**: ~9 minutes (reading) + 8-12 minutes (hands-on) = ~17-21 minutes total
- **Structure**: 5 steps + troubleshooting + Level Up appendix

## Findings

### Q1: Code snippets verified against existing examples?

**Confidence: 95%**

All code snippets were verified against the actual codebase:

1. **Kernel SAXPY** — matches `examples/hostcall/vector-math/kernel/src/lib.rs` lines 28-36
   exactly. Same features (`abi_gpu_kernel`, `stdarch_nvptx`, `asm_experimental_arch`),
   same panic handler (with `trap;` inline asm), same thread indexing pattern.

2. **Host SAXPY** — matches `examples/hostcall/vector-math/host/src/main.rs` Demo 1
   (lines 19-45). Uses same `gpu::custom()` builder flow: `.ptx()`, `.threads(256)`,
   `.elements(N)`, `.prepare()`, `upload`, `launch`, `download`.

3. **Build script** — guide instructs users to copy `examples/hostcall/vector-math/host/build.rs`
   directly and change only the PTX filename. The build.rs is 93 lines and handles
   toolchain detection, env cleaning, and PTX patching.

4. **Kernel .cargo/config.toml** — matches `examples/hostcall/vector-math/kernel/.cargo/config.toml`
   exactly: `target = "nvptx64-nvidia-cuda"`, `linker = "llvm-bitcode-linker"`,
   `build-std = ["core"]`.

5. **Hello-GPU one-liner API** — matches `examples/hostcall/hello-gpu/host/src/main.rs`.
   Uses `async_gpu::gpu` (the facade crate), not `gpu_host::gpu` directly.

### Q2: Guide uses `async_gpu` consistently?

**Confidence: 100%**

Yes. The guide uses `async_gpu::gpu` for imports and `async_gpu::Result` for
return types, matching the facade crate. The host Cargo.toml depends on
`async-gpu`, not `gpu-host`. Only the Cargo.toml also lists `cudarc` as a
direct dependency (needed for `CudaSlice` types in `upload`/`download`).

### Q3: Any API gaps discovered?

**Confidence: 85%**

1. **`cudarc` as transitive dependency**: The `upload()` and `download()` methods
   return `CudaSlice<T>` from `cudarc`, but `async-gpu` does not re-export `cudarc`.
   Users must add `cudarc` to their Cargo.toml even though they never interact with
   cudarc directly. This is a friction point but not a blocker — the guide includes
   cudarc in the dependencies.

2. **No `gpu::custom().ptx()` without external PTX**: The `ptx()` method takes a
   `&'static str`, which means the PTX must be compiled separately and embedded via
   `include_str!`. There is no built-in "compile my kernel crate for me" — the
   build.rs is required. This is the biggest friction point in the guide.

3. **README uses `gpu_host::gpu` in code samples**: The README.md Quick Start shows
   `use gpu_host::gpu` while the guide uses `use async_gpu::gpu`. This inconsistency
   exists in the repo but is not introduced by the guide.

### Q4: README link added?

**Confidence: 100%**

Added a callout line in README.md under the "## Quick Start" heading:

> **New here?** The [Getting Started Guide](docs/getting-started.md) walks you
> through writing and running your first GPU kernel in under 30 minutes.

## Impact on Downstream Tasks

- The guide does NOT create example directories (inline code only, per constraints).
  A future task could create `examples/getting-started/` as a runnable template.
- The README inconsistency (`gpu_host` vs `async_gpu`) should be addressed by
  lib-cleanup.6 (migrate examples to use facade crate).
