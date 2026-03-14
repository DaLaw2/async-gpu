/// Build script for gpu-host.
///
/// This script re-runs when the GPU kernel source changes.
/// The actual PTX compilation is done manually via:
///
///   cd crates/gpu-kernel
///   cargo +nightly rustc --release --target nvptx64-nvidia-cuda -Zbuild-std=core \
///       -- --emit=asm -C linker=echo -C target-cpu=sm_86
///
/// The generated PTX at:
///   crates/gpu-kernel/target/nvptx64-nvidia-cuda/release/deps/gpu_kernel.s
/// is then copied to:
///   crates/gpu-host/kernel.ptx
///
/// NOTE: Full automated compilation via `cargo build` requires the `llvm-tools`
/// rustup component (for `llvm-link`).  Install it with:
///   rustup +nightly component add llvm-tools
///
/// Until then, the --emit=asm + echo-linker workaround produces equivalent PTX.
fn main() {
    // Tell Cargo to re-run this build script if the PTX file changes.
    println!("cargo:rerun-if-changed=kernel.ptx");
    println!("cargo:rerun-if-changed=../gpu-kernel/src/lib.rs");
}
