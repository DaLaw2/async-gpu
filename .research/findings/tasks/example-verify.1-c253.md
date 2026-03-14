# example-verify.1: Build and run hello-gpu, async-io, vector-math — document status
**Cycle**: 253 | **Theme**: example-verify | **Kind**: investigation | **Status**: done

## Summary

All three examples (`hello-gpu`, `async-io`, `vector-math`) pass `cargo +stable check` with zero errors and zero warnings. The host-side imports and API usage are fully consistent with the current `gpu-host` public API. Kernel crates are properly structured. Build scripts handle PTX compilation with appropriate fallback paths.

## Per-Example Status

### hello-gpu
- **cargo +stable check**: PASS (0 errors, 0 warnings)
- **Host Cargo.toml**: Depends on `cudarc 0.12` and `gpu-host` (path, default-features=false). Correct.
- **Host main.rs imports**: `GpuHostError`, `GpuRuntime`, `HostcallBuffer`, `MappedBuffer` — all match current `gpu-host` re-exports.
- **API usage verified**:
  - `GpuRuntime::new(0)`, `load_ptx()`, `get_func()`, `launch_config()`, `htod_sync_copy()`, `alloc_zeros()`, `dtoh_sync_copy()`, `synchronize()` — all present in `runtime.rs`
  - `GpuHostError::KernelNotFound(&'static str)` — present in `error.rs`
  - `HostcallBuffer::new(8)`, `.dev_ptr`, `.sideband_dev_ptr`, `.listen()`, `.signal_shutdown()` — all present in `hostcall.rs`
  - `MappedBuffer::<u32>::new_zeroed(1)`, `.dev_ptr()`, `.read()`, `.write()` — all present in `memory.rs`
- **PTX handling**: `build.rs` compiles `hello-gpu-kernel` via nightly toolchain, copies PTX to OUT_DIR. Host uses `include_str!(concat!(env!("OUT_DIR"), "/kernel.ptx"))`.
- **Kernel Cargo.toml**: Depends on `gpu-runtime` (path `../../../crates/gpu-runtime`). Path is valid.
- **Kernel .cargo/config.toml**: Targets `nvptx64-nvidia-cuda`, `sm_86`, uses `llvm-bitcode-linker`, builds `core`.

### async-io
- **cargo +stable check**: PASS (0 errors, 0 warnings)
- **Host Cargo.toml**: Same dependency pattern as hello-gpu. Correct.
- **Host main.rs imports**: Same as hello-gpu (`GpuHostError`, `GpuRuntime`, `HostcallBuffer`, `MappedBuffer`). All valid.
- **API usage**: Identical patterns to hello-gpu — `HostcallBuffer::new(8)`, `.dev_ptr`, `.sideband_dev_ptr`, `.listen()`, `.signal_shutdown()`. All valid.
- **PTX handling**: Same build.rs pattern, expects `async_io_kernel.ptx`.
- **Kernel Cargo.toml**: Depends on `gpu-runtime`. Path valid.

### vector-math
- **cargo +stable check**: PASS (0 errors, 0 warnings)
- **Host Cargo.toml**: Same dependency pattern. Correct.
- **Host main.rs imports**: `GpuHostError`, `GpuRuntime` only (no `HostcallBuffer`/`MappedBuffer` — pure compute, no hostcall). Correct.
- **API usage**: `GpuRuntime` methods only. Uses `div_ceil` on u32 for grid dimensions. All valid on stable Rust.
- **PTX handling**: Same build.rs pattern, expects `vector_math_kernel.ptx`.
- **Kernel Cargo.toml**: No runtime dependency (empty `[dependencies]`). Kernel uses raw `core::arch::nvptx` intrinsics and inline PTX asm. Correct for a pure compute kernel.

## Build Script Architecture (shared by all three)
- All three build scripts share the same pattern: invoke `cargo +{nightly} build --release` on the sibling kernel crate, copy resulting PTX to `OUT_DIR/kernel.ptx`.
- Nightly toolchain is read from repo root's `rust-toolchain.toml`.
- If kernel compilation fails, falls back to cached PTX in the kernel's target dir.
- Patches PTX if it targets sm_30 (upgrades to sm_86 + PTX version 7.1).
- Environment variables (`CARGO`, `RUSTC`, `RUSTFLAGS`, `CARGO_ENCODED_RUSTFLAGS`, `CARGO_TARGET_DIR`, `CARGO_BUILD_TARGET`) are removed to prevent parent cargo from affecting the kernel build.

## Open Questions

1. **No end-to-end build test in CI**: The examples depend on a working nightly + nvptx toolchain + `llvm-bitcode-linker`. It's unclear whether CI ever tests these examples. If the kernel crate's dependency on `gpu-runtime` breaks, `cargo check` on the host won't catch it since the build.rs would just use cached PTX.
2. **Cached PTX staleness**: The fallback to cached PTX means `cargo check` can pass even when the kernel code has diverged. There's no mechanism to detect stale PTX.
3. **gpu-host default-features=false**: All examples disable default features (which would pull in `safetensors`/`tiktoken-rs` for the `gpt2` feature). This is intentional and correct.
