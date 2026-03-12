# clean-example.2: Implement hello-gpu example
**Cycle**: 74 | **Theme**: clean-example | **Kind**: experiment | **Status**: done

## Summary
Implemented and verified the hello-gpu example with 4 kernels demonstrating the full
async_gpu stack. All demos pass: vector_add (pure compute), hello_gpu (PRINT hostcall),
file_io_demo (file OPEN+WRITE+CLOSE), bulk_read_demo (sideband BULK_READ).

## Findings

### Q: Does the example compile and run out of the box?
A: Yes, after two fixes:
1. Kernel needed `#![feature(asm_experimental_arch)]` for inline PTX asm used by
   `gpu_runtime::panic_handler!()` macro.
2. `gpu-host` was binary-only (no lib.rs). Added `lib.rs` re-exporting `pub mod error;
   pub mod hostcall;` so the example can `use gpu_host::hostcall::HostcallBuffer`.

After fixes, `cargo run --release` from `examples/hello-gpu/host/` compiles the kernel
via build.rs and runs all 4 demos successfully.
**Confidence**: high

### Q: Does build.rs correctly invoke cargo for nvptx64 target?
A: Yes. build.rs clears CARGO/RUSTC/RUSTFLAGS/CARGO_TARGET_DIR env vars to prevent
parent cargo from interfering, uses `cargo +nightly-2026-03-11 build --release` in the
kernel directory, and copies the PTX from `target/nvptx64-nvidia-cuda/release/hello_gpu_kernel.ptx`.
Includes sm_30→sm_86 patching as a safety net.
**Confidence**: high

### Q: Is the output clear and demonstrates key capabilities?
A: Yes. Output shows each demo with clear labels and PASSED/FAILED status:
- vector_add: pure compute correctness check
- hello_gpu: GPU prints message via hostcall
- file_io_demo: creates and writes file from GPU, verifies content
- bulk_read_demo: reads back the file via sideband, shows bytes read count
**Confidence**: high

## Key Changes Made
- `examples/hello-gpu/kernel/src/lib.rs`: Added `asm_experimental_arch` feature gate
- `crates/gpu-host/src/lib.rs`: NEW — exports error and hostcall modules as library
- `crates/gpu-host/Cargo.toml`: No changes needed (Cargo auto-detects lib.rs + main.rs)

## Impact on Downstream Tasks
- gpu-host is now usable as a library dependency, enabling any future example or test
  crate to `use gpu_host::hostcall::HostcallBuffer` directly.
