# toolchain.4: Minimal GPU Kernel Compilation and Execution
**Date**: 2026-03-11
**Cycle**: 1
**Theme**: toolchain
**Kind**: experiment
**Status**: done

## Summary

Successfully compiled two Rust GPU kernels (`vector_add` and `write_thread_idx`) targeting
`nvptx64-nvidia-cuda` with `#![no_std]` + `#![feature(abi_ptx)]`, generated valid PTX (`.target sm_86`),
loaded it via `cudarc` using the CUDA driver API, and confirmed correct execution on an RTX 3060.
Both kernels passed verification.

The standard `cargo build` path via `llvm-bitcode-linker` is blocked by a missing `llvm-tools`
component (`llvm-link` not yet on disk). A workaround using `--emit=asm -C linker=echo` produces
functionally equivalent PTX directly from `rustc` and was used successfully.

---

## Compilation Process

### Step 1: GPU Kernel Crate Setup

Created `crates/gpu-kernel/` as a standalone workspace (not in the root workspace, because it uses
a different target and `-Zbuild-std`):

```toml
# crates/gpu-kernel/Cargo.toml
[workspace]

[package]
name = "gpu-kernel"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[profile.release]
panic = "abort"
opt-level = 3
lto = false
```

`.cargo/config.toml`:
```toml
[build]
target = "nvptx64-nvidia-cuda"

[target.nvptx64-nvidia-cuda]
linker = "llvm-bitcode-linker"
rustflags = ["-C", "target-cpu=sm_86"]

[unstable]
build-std = ["core"]
build-std-features = ["compiler-builtins-mem"]
```

`src/lib.rs` features used:
- `#![no_std]` — mandatory; GPU has no OS
- `#![feature(abi_ptx)]` — enables `extern "ptx-kernel"` function ABI
- `#![feature(stdarch_nvptx)]` — enables `core::arch::nvptx` intrinsics (`_thread_idx_x`, etc.)
- Manual `#[panic_handler]` that loops forever (required by `no_std`)

### Step 2: PTX Compilation

**Intended path** (`cargo build` with `llvm-bitcode-linker`):
```
cargo +nightly build --release --target nvptx64-nvidia-cuda -Zbuild-std=core
```
This path **failed** — see Errors section below.

**Workaround** (`--emit=asm` with echo linker):
```
cargo +nightly rustc --release --target nvptx64-nvidia-cuda -Zbuild-std=core \
    -- --emit=asm -C linker=echo -C target-cpu=sm_86
```
The `--emit=asm` flag instructs `rustc` to emit PTX assembly per CGU before the link step.
Using `echo` as the linker suppresses the link step entirely. The result is placed at:
```
target/nvptx64-nvidia-cuda/release/deps/gpu_kernel.s
```
The `.s` file is valid PTX and was copied to `crates/gpu-host/kernel.ptx` for use by the host.

### Step 3: Host Crate Setup

Created `crates/gpu-host/` in the root Cargo workspace. Dependencies:
- `cudarc = "0.12.1"` with feature `cuda-12060` — provides CUDA driver API bindings
- `anyhow = "1"` — error handling

The host binary:
1. Initializes `CudaDevice` (device 0)
2. Loads the PTX using `cudarc::nvrtc::Ptx::from_src` + `dev.load_ptx(...)`
3. Retrieves kernel function handles via `dev.get_func(...)`
4. Allocates device buffers, copies input data via `dev.htod_sync_copy`
5. Launches kernels via `f.launch(cfg, args)`
6. Copies results back via `dev.dtoh_sync_copy`
7. Verifies correctness

The PTX string is embedded at compile time with `include_str!("../kernel.ptx")`.

### Step 4: Kernel Execution

Execution on RTX 3060 (SM 8.6, CUDA driver 13.0):

```
=== GPU Kernel Execution Test ===

CUDA device initialized successfully
write_thread_idx output (64 elements):
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
  Verification PASSED: all 64 elements correct

vector_add output (first 16 of 128 elements):
  [128.0, 128.0, 128.0, 128.0, 128.0, 128.0, 128.0, 128.0, ...]
  Verification PASSED: all 128 elements equal 128

All tests PASSED.
```

---

## PTX Output Analysis

PTX generated for `sm_86` (`.version 7.1`):

**`write_thread_idx` kernel** — reads `%tid.x`, `%ctaid.x`, `%ntid.x`, computes linear index,
bounds-checks, and issues a single `st.global.b32`:
```ptx
.visible .entry write_thread_idx(
    .param .u64 .ptr .align 1 write_thread_idx_param_0,
    .param .u32 write_thread_idx_param_1
)
{
    mov.u32  %r3, %tid.x;
    mov.u32  %r4, %ctaid.x;
    mov.u32  %r5, %ntid.x;
    mad.lo.s32 %r1, %r4, %r5, %r3;  // global idx
    setp.lt.u32 %p1, %r1, %r2;      // bounds check
    ...
    st.global.b32  [%rd1], %r1;     // write idx to output
}
```

**`vector_add` kernel** — two `ld.global.b32`, one `add.rn.f32`, one `st.global.b32`.
The `add.rn.f32` uses round-to-nearest-even mode, which is the correct IEEE 754 default.

**Observations**:
- `.visible .entry` marks public kernel entry points — this is the correct attribute for
  `extern "ptx-kernel"` functions.
- Parameters are passed as `ld.param.b64` (pointer) and `ld.param.b32` (u32), matching the
  Rust function signatures.
- No warp synchronization instructions were emitted (as expected for embarrassingly parallel kernels).
- `panic_handler` compiles to `bra.uni $L__BB0_1` (infinite loop) — minimal overhead.
- `sm_86` PTX `.version 7.1` is loaded correctly by CUDA driver 13.0 (driver supports PTX up
  to PTX ISA version matching the installed toolkit version, but the driver itself can JIT-compile
  older PTX versions; CUDA 13.0 driver accepts PTX 7.x without issue).

**Limitation**: PTX was emitted per CGU (`gpu_kernel.s`), which only covers functions in the
`gpu-kernel` crate. If `core` intrinsics were called, they would be inlined or resolved in the
CGU. The `--emit=asm` workaround does not link multiple CGUs, so multi-crate kernels would need
`llvm-tools` for the bitcode link step.

---

## Errors Encountered and Workarounds

### Error 1: Missing workspace Cargo.toml
**Symptom**: `cargo build` from the root workspace failed when `gpu-kernel` was listed as a member
but its `Cargo.toml` did not exist.
**Fix**: Created `crates/gpu-host/Cargo.toml` immediately.

### Error 2: gpu-kernel inside root workspace
**Symptom**:
```
error: current package believes it's in a workspace when it's not
```
**Root cause**: The root `Cargo.toml` detected the `gpu-kernel` package but it used different
target/build flags incompatible with the workspace.
**Fix**: Added an empty `[workspace]` table to `crates/gpu-kernel/Cargo.toml` to make it a
standalone workspace, not a member of the root workspace.

### Error 3: Missing panic handler
**Symptom**:
```
error: `#[panic_handler]` function required, but not found
```
**Fix**: Added a minimal panic handler to `src/lib.rs`:
```rust
#[panic_handler]
fn panic(_info: &PanicInfo) -> ! { loop {} }
```

### Error 4: Missing `rust-ptx-linker`
**Symptom**:
```
error: linker `rust-ptx-linker` not found
```
**Root cause**: The default PTX linker is `rust-ptx-linker`, which is a separate external binary
not included in the Rust toolchain. It must be installed separately via `cargo install rust-ptx-linker`.
**Fix**: Switched to `llvm-bitcode-linker` (already installed as a rustup component) by adding
`linker = "llvm-bitcode-linker"` under `[target.nvptx64-nvidia-cuda]` in `.cargo/config.toml`.

### Error 5: `llvm-tools` component not installed (llvm-link missing)
**Symptom**:
```
error: linking with `llvm-bitcode-linker` failed: exit code: 1
Error: An error occured when calling llvm-link. Make sure the llvm-tools component is installed.
Caused by: program not found
```
**Root cause**: `llvm-bitcode-linker` delegates the bitcode link step to `llvm-link`, which is
part of the `llvm-tools` rustup component. That component was not installed.
**Attempted fix**: `rustup +nightly component add llvm-tools` — download started but was still
in progress during the experiment (llvm-tools is ~300 MB).
**Workaround**: Emit PTX directly without the linker step:
```
cargo +nightly rustc --release --target nvptx64-nvidia-cuda -Zbuild-std=core \
    -- --emit=asm -C linker=echo -C target-cpu=sm_86
```
`--emit=asm` produces `gpu_kernel.s` (valid PTX) per CGU.
`-C linker=echo` causes the linker invocation to be `echo <args>`, which succeeds immediately
and writes nothing, so cargo reports success.
The resulting `.s` file is copied to `kernel.ptx` and used by the host.

**Limitation of workaround**: Produces per-CGU PTX only. For multi-crate kernels, proper linking
of multiple LLVM bitcode files would be required, which needs `llvm-tools`.

### Error 6: PTX `.target sm_30` vs hardware sm_86
**Symptom**: Not an error, but initial PTX emitted `.target sm_30` (the nvptx64 target's default
base architecture). While the PTX ISA is forward-compatible and CUDA's JIT compiler would still
accept it, it misses SM 8.6 specific optimizations.
**Fix**: Added `-C target-cpu=sm_86` to emit `.target sm_86, .version 7.1` PTX, optimal for
the RTX 3060. This flag is passed via `rustflags` in `.cargo/config.toml`.

---

## Key Conclusions

1. **Compilation is possible without nvcc**: The `rustc` NVPTX backend + `--emit=asm` produces
   valid PTX that can be loaded and executed via the CUDA driver API. `nvcc` is not required.

2. **`llvm-tools` is required for the standard `cargo build` path**: The `llvm-bitcode-linker`
   linker depends on `llvm-link` from `llvm-tools`. Without it, the workaround
   `--emit=asm -C linker=echo` is effective for single-CGU kernels.

3. **PTX quality is good**: The emitted PTX is clean, uses correct `ld.param`/`st.global`
   patterns, and correctly accesses special registers (`%tid.x`, `%ctaid.x`, `%ntid.x`).

4. **cudarc 0.12 works with CUDA 13.0 driver**: The CUDA driver API is backward-compatible.
   `cudarc::nvrtc::Ptx::from_src` loads inline PTX strings successfully.

5. **`extern "ptx-kernel"` ABI is correct**: Functions are emitted as `.visible .entry`,
   which is the PTX entry point declaration required for kernel launching.

6. **`-C target-cpu=sm_86` is important**: Targets the actual hardware for better code generation.

7. **PTX is forward-compatible in the driver**: CUDA 13.0 driver accepted `.target sm_86`
   PTX 7.1 without issues.

---

## Open Questions

1. **After `llvm-tools` installs**: Does `cargo build` with `llvm-bitcode-linker` produce
   a linked `.ptx` that differs from the per-CGU `--emit=asm` output? (Expected: same, since
   there's only one CGU here.)

2. **Multi-crate kernels**: Can a GPU kernel import functions from another `no_std` crate
   compiled for nvptx64? The bitcode linker should handle this, but is untested.

3. **`atomics.1` dependency**: Do Rust `core::sync::atomic` operations emit `.sys` scope PTX
   instructions on SM 8.6, or fall back to `.gpu` scope? This is critical for GPU-CPU atomics.

4. **Inline PTX (asm!)**: Can `core::arch::asm!` be used on nvptx64 for custom PTX instructions
   (e.g., `membar.sys`, custom atomics)? Previous research indicated inline asm was problematic.

5. **`crate-type = ["cdylib"]` vs `["rlib"]`**: The `cdylib` type was used; would `rlib` be
   more appropriate for linking with future multi-crate GPU kernels?

6. **LTO interaction**: With `lto = false`, each CGU is compiled separately. Enabling LTO
   would merge CGUs before PTX emission — would this improve code quality or cause issues?

---

## Impact on Downstream Tasks

- **`toolchain` theme**: The compilation path is validated. The SM 8.6 PTX compiles and executes
  correctly. `toolchain` theme first success criterion is met.
- **`atomics.2`**: Depends on this task being done. The `write_thread_idx` kernel infrastructure
  can be reused to write an atomics stress-test kernel.
- **`hostcall.1`**: Can now experiment with mapped/pinned memory shared between GPU and host.
  The kernel compilation path is available.
- **`async-runtime.3`**: Future executor experiments can use the same kernel compilation + cudarc
  launch infrastructure established here.

---

## Theme Progress

**toolchain**: 3/5 tasks done or active. First success criterion ("Can compile and run a minimal
Rust kernel on GPU") is now **satisfied**. Second criterion (ADR) is pending — see `toolchain.3`.
The `llvm-tools` component install should be completed by the user to unblock the standard
`cargo build` path for multi-crate kernels.
