//! Build script: compiles the kernel crate to PTX and embeds it.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let kernel_dir = manifest_dir.join("..").join("kernel");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    println!("cargo:rerun-if-changed=../kernel/src/lib.rs");
    println!("cargo:rerun-if-changed=../kernel/Cargo.toml");

    let status = Command::new("cargo")
        .args(["+nightly-2026-03-11", "build", "--release"])
        .current_dir(&kernel_dir)
        .env_remove("CARGO")
        .env_remove("RUSTC")
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .env_remove("CARGO_TARGET_DIR")
        .env_remove("CARGO_BUILD_TARGET")
        .status()
        .expect("Failed to run cargo for kernel compilation. Is nightly-2026-03-11 installed?");

    if !status.success() {
        panic!("Kernel PTX compilation failed");
    }

    let ptx_src = kernel_dir
        .join("target")
        .join("nvptx64-nvidia-cuda")
        .join("release")
        .join("vector_math_kernel.ptx");

    if !ptx_src.exists() {
        panic!(
            "PTX file not found at {:?}. Check kernel compilation output.",
            ptx_src
        );
    }

    let ptx_content = std::fs::read_to_string(&ptx_src).expect("Failed to read PTX");
    if ptx_content.contains(".target sm_30") {
        let patched = ptx_content
            .replace(".target sm_30", ".target sm_86")
            .replace(".version 6.0", ".version 7.1");
        let ptx_dst = out_dir.join("kernel.ptx");
        std::fs::write(&ptx_dst, patched).expect("Failed to write patched PTX");
    } else {
        let ptx_dst = out_dir.join("kernel.ptx");
        std::fs::copy(&ptx_src, &ptx_dst).expect("Failed to copy PTX file");
    }
}
