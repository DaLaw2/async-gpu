/// Build script for gpu-host.
///
/// Automatically compiles the gpu-kernel crate for nvptx64 and copies the PTX
/// to this crate's directory. This enables single-command builds:
///
///     cargo build -p gpu-host
///
/// If the kernel compilation fails (e.g., nightly toolchain not installed),
/// falls back to the existing kernel.ptx file. This allows `cargo check` and
/// `cargo clippy` to work without the full kernel build pipeline.
///
/// The generated PTX is placed at `crates/core/gpu-host/kernel.ptx` where
/// `include_str!("../kernel.ptx")` picks it up.
use std::env;
use std::path::PathBuf;
use std::process::Command;

/// Read the nightly toolchain channel from the repo root's rust-toolchain.toml.
fn nightly_toolchain(repo_root: &std::path::Path) -> String {
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
    // Repo root is 3 levels up from crates/core/gpu-host/
    let repo_root = manifest_dir.join("..").join("..").join("..");
    let kernel_dir = repo_root.join("crates").join("kernel").join("gpu-kernel");
    let ptx_dst = manifest_dir.join("kernel.ptx");
    let toolchain = nightly_toolchain(&repo_root);

    // Rerun if kernel source or PTX file changes
    println!("cargo:rerun-if-changed=kernel.ptx");
    println!("cargo:rerun-if-changed=../../kernel/gpu-kernel/src/lib.rs");
    println!("cargo:rerun-if-changed=../../kernel/gpu-kernel/src/hostcall_kernels.rs");
    println!("cargo:rerun-if-changed=../../kernel/gpu-kernel/Cargo.toml");
    println!("cargo:rerun-if-changed=../../../rust-toolchain.toml");

    // Skip kernel build if AUTO_BUILD_KERNEL=0 (for CI or manual workflows)
    if env::var("AUTO_BUILD_KERNEL").as_deref() == Ok("0") {
        return;
    }

    // Check if kernel_dir exists
    if !kernel_dir.exists() {
        eprintln!(
            "cargo:warning=gpu-kernel directory not found at {kernel_dir:?}, using existing PTX"
        );
        return;
    }

    // Build the kernel crate for nvptx64
    let status = Command::new("cargo")
        .args([&toolchain, "build", "--release"])
        .current_dir(&kernel_dir)
        .env_remove("CARGO")
        .env_remove("RUSTC")
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .env_remove("CARGO_TARGET_DIR")
        .env_remove("CARGO_BUILD_TARGET")
        .status();

    match status {
        Ok(s) if s.success() => {
            // Copy PTX from kernel build output
            let ptx_src = kernel_dir
                .join("target")
                .join("nvptx64-nvidia-cuda")
                .join("release")
                .join("gpu_kernel.ptx");

            if ptx_src.exists() {
                if let Err(e) = std::fs::copy(&ptx_src, &ptx_dst) {
                    eprintln!("cargo:warning=Failed to copy PTX: {e}. Using existing PTX.");
                }
            } else {
                eprintln!(
                    "cargo:warning=PTX not found at {ptx_src:?} after successful build. Using existing PTX."
                );
            }
        }
        Ok(_) => {
            // Build failed (non-zero exit) — fall back to existing PTX
            eprintln!(
                "cargo:warning=Kernel compilation failed (exit code). Using existing kernel.ptx."
            );
        }
        Err(e) => {
            // Could not run cargo at all
            eprintln!("cargo:warning=Could not run kernel build: {e}. Using existing kernel.ptx.");
        }
    }
}
