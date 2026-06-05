/// Build script for gpu-host.
///
/// Automatically compiles the gpu-kernel-test crate for nvptx64 and copies the PTX
/// to this crate's directory. gpu-kernel-test contains test/demo kernels that
/// exercise std features on GPU (println!, Vec, File I/O, thread::spawn, etc.).
///
/// The PTX is copied to both `kernel.ptx` and `kernel_std.ptx` for backward
/// compatibility — all code that references either constant gets the same PTX.
///
/// If the kernel compilation fails (e.g., nightly toolchain not installed),
/// falls back to the existing kernel.ptx file. This allows `cargo check` and
/// `cargo clippy` to work without the full kernel build pipeline.
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
    let kernel_dir = repo_root
        .join("crates")
        .join("kernel")
        .join("gpu-kernel-test");
    let ptx_dst = manifest_dir.join("kernel.ptx");
    let ptx_std_dst = manifest_dir.join("kernel_std.ptx");
    let toolchain = nightly_toolchain(&repo_root);

    // Rerun if kernel source or PTX file changes
    println!("cargo:rerun-if-changed=kernel.ptx");
    println!("cargo:rerun-if-changed=kernel_std.ptx");
    println!("cargo:rerun-if-changed=../../kernel/gpu-kernel-test/src/lib.rs");
    println!("cargo:rerun-if-changed=../../kernel/gpu-kernel-test/src/warp.rs");
    println!("cargo:rerun-if-changed=../../kernel/gpu-kernel-test/src/thread_test.rs");
    println!("cargo:rerun-if-changed=../../kernel/gpu-kernel-test/src/sc_demo.rs");
    println!("cargo:rerun-if-changed=../../kernel/gpu-kernel-test/src/par_iter_demo.rs");
    println!("cargo:rerun-if-changed=../../kernel/gpu-kernel-test/Cargo.toml");
    println!("cargo:rerun-if-changed=../../../rust-toolchain.toml");

    // Skip kernel build if AUTO_BUILD_KERNEL=0 (for CI or manual workflows)
    if env::var("AUTO_BUILD_KERNEL").as_deref() == Ok("0") {
        return;
    }

    // Check if kernel_dir exists
    if !kernel_dir.exists() {
        eprintln!(
            "cargo:warning=gpu-kernel-test directory not found at {kernel_dir:?}, using existing PTX"
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
                .join("gpu_kernel_test.ptx");

            if ptx_src.exists() {
                // Copy to kernel.ptx (primary)
                if let Err(e) = std::fs::copy(&ptx_src, &ptx_dst) {
                    eprintln!("cargo:warning=Failed to copy PTX to kernel.ptx: {e}.");
                }
                // Copy to kernel_std.ptx (backward compat — same content)
                if let Err(e) = std::fs::copy(&ptx_src, &ptx_std_dst) {
                    eprintln!("cargo:warning=Failed to copy PTX to kernel_std.ptx: {e}.");
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
