//! Build script: compiles the kernel crate to PTX and embeds it.
//!
//! This build script invokes `cargo build` on the kernel crate targeting
//! nvptx64-nvidia-cuda. The resulting PTX file is placed in OUT_DIR
//! so the host binary can `include_str!` it.

use std::env;
use std::path::PathBuf;
use std::process::Command;

/// Read the nightly toolchain channel from the repo root's rust-toolchain.toml.
/// Falls back to "+nightly" if the file cannot be parsed.
fn nightly_toolchain() -> String {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    // Walk up from examples/<name>/host/ to repo root
    let repo_root = manifest_dir.join("..").join("..").join("..");
    let toolchain_file = repo_root.join("rust-toolchain.toml");
    if let Ok(content) = std::fs::read_to_string(&toolchain_file) {
        for line in content.lines() {
            let line = line.trim();
            if line.starts_with("channel") {
                if let Some(val) = line.split('=').nth(1) {
                    let channel = val.trim().trim_matches('"');
                    return format!("+{channel}");
                }
            }
        }
    }
    "+nightly".to_string()
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let kernel_dir = manifest_dir.join("..").join("kernel");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let toolchain = nightly_toolchain();

    println!("cargo:rerun-if-changed=../kernel/src/lib.rs");
    println!("cargo:rerun-if-changed=../kernel/Cargo.toml");
    println!("cargo:rerun-if-changed=../../../rust-toolchain.toml");

    // Build the kernel crate for nvptx64.
    // IMPORTANT: We clear CARGO to prevent the parent cargo from influencing
    // the child build. We also clear RUSTC and RUSTFLAGS for the same reason.
    // The kernel's .cargo/config.toml and rust-toolchain.toml handle everything.
    let status = Command::new("cargo")
        .args([&toolchain, "build", "--release"])
        .current_dir(&kernel_dir)
        .env_remove("CARGO")
        .env_remove("RUSTC")
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .env_remove("CARGO_TARGET_DIR")
        .env_remove("CARGO_BUILD_TARGET")
        .status()
        .unwrap_or_else(|_| {
            panic!(
                "Failed to run cargo for kernel compilation. Is {} installed?",
                toolchain
            )
        });

    if !status.success() {
        // During `cargo clippy` or `cargo check`, the kernel toolchain may not
        // be available.  Fall back to a previously-built PTX if one exists so
        // that the host crate can still be checked without the full kernel
        // build pipeline.
        let ptx_fallback = kernel_dir
            .join("target")
            .join("nvptx64-nvidia-cuda")
            .join("release")
            .join("hello_gpu_kernel.ptx");
        if ptx_fallback.exists() {
            eprintln!(
                "cargo:warning=Kernel compilation failed; using cached PTX at {:?}",
                ptx_fallback
            );
        } else {
            panic!("Kernel PTX compilation failed and no cached PTX found");
        }
    }

    // Find the generated PTX file
    let ptx_src = kernel_dir
        .join("target")
        .join("nvptx64-nvidia-cuda")
        .join("release")
        .join("hello_gpu_kernel.ptx");

    if !ptx_src.exists() {
        panic!(
            "PTX file not found at {:?}. Check kernel compilation output.",
            ptx_src
        );
    }

    // Verify the PTX target is correct
    let ptx_content = std::fs::read_to_string(&ptx_src).expect("Failed to read PTX");
    if ptx_content.contains(".target sm_30") {
        eprintln!("WARNING: PTX has .target sm_30, expected sm_86. Patching...");
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
