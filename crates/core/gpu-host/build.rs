/// Build script for gpu-host.
///
/// Automatically compiles all 4 kernel crates for nvptx64 and copies their PTX
/// to this crate's directory:
///   - gpu-kernel-core    → kernel_core.ptx
///   - gpu-kernel-compute → kernel_compute.ptx
///   - gpu-kernel-io      → kernel_io.ptx
///   - gpu-kernel-test    → kernel_test.ptx
///
/// For backward compatibility, kernel.ptx and kernel_std.ptx are also produced
/// as copies of kernel_test.ptx (the former gpu-kernel-std).
///
/// If any kernel compilation fails (e.g., nightly toolchain not installed),
/// falls back to the existing PTX files. This allows `cargo check` and
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

/// Kernel crate descriptor for the build loop.
struct KernelCrate {
    /// Directory name under crates/kernel/ (e.g., "gpu-kernel-core").
    dir_name: &'static str,
    /// Cargo artifact name (underscored, e.g., "gpu_kernel_core").
    artifact: &'static str,
    /// Output PTX filename (e.g., "kernel_core.ptx").
    ptx_name: &'static str,
}

const KERNEL_CRATES: &[KernelCrate] = &[
    KernelCrate {
        dir_name: "gpu-kernel-core",
        artifact: "gpu_kernel_core",
        ptx_name: "kernel_core.ptx",
    },
    KernelCrate {
        dir_name: "gpu-kernel-compute",
        artifact: "gpu_kernel_compute",
        ptx_name: "kernel_compute.ptx",
    },
    KernelCrate {
        dir_name: "gpu-kernel-io",
        artifact: "gpu_kernel_io",
        ptx_name: "kernel_io.ptx",
    },
    KernelCrate {
        dir_name: "gpu-kernel-test",
        artifact: "gpu_kernel_test",
        ptx_name: "kernel_test.ptx",
    },
];

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    // Repo root is 3 levels up from crates/core/gpu-host/
    let repo_root = manifest_dir.join("..").join("..").join("..");
    let toolchain = nightly_toolchain(&repo_root);

    // Rerun-if-changed for all kernel crate sources
    for kc in KERNEL_CRATES {
        println!("cargo:rerun-if-changed={}", kc.ptx_name);
        let src_dir = format!("../../kernel/{}/src", kc.dir_name);
        // Watch the src directory for any .rs changes
        if let Ok(entries) = std::fs::read_dir(manifest_dir.join(&src_dir)) {
            for entry in entries.flatten() {
                if entry.path().extension().is_some_and(|e| e == "rs") {
                    let rel = format!("{}/{}", src_dir, entry.file_name().to_string_lossy());
                    println!("cargo:rerun-if-changed={rel}");
                }
            }
        }
        println!(
            "cargo:rerun-if-changed=../../kernel/{}/Cargo.toml",
            kc.dir_name
        );
    }
    // Backward-compat aliases
    println!("cargo:rerun-if-changed=kernel.ptx");
    println!("cargo:rerun-if-changed=kernel_std.ptx");
    println!("cargo:rerun-if-changed=../../../rust-toolchain.toml");

    // Skip kernel build if AUTO_BUILD_KERNEL=0 (for CI or manual workflows)
    if env::var("AUTO_BUILD_KERNEL").as_deref() == Ok("0") {
        return;
    }

    // Build each kernel crate and copy its PTX
    for kc in KERNEL_CRATES {
        let kernel_dir = repo_root.join("crates").join("kernel").join(kc.dir_name);
        let ptx_dst = manifest_dir.join(kc.ptx_name);

        if !kernel_dir.exists() {
            eprintln!(
                "cargo:warning={} directory not found at {kernel_dir:?}, using existing PTX",
                kc.dir_name
            );
            continue;
        }

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
                let ptx_src = kernel_dir
                    .join("target")
                    .join("nvptx64-nvidia-cuda")
                    .join("release")
                    .join(format!("{}.ptx", kc.artifact));

                if ptx_src.exists() {
                    if let Err(e) = std::fs::copy(&ptx_src, &ptx_dst) {
                        eprintln!("cargo:warning=Failed to copy PTX to {}: {e}.", kc.ptx_name);
                    }
                } else {
                    eprintln!(
                        "cargo:warning=PTX not found at {ptx_src:?} after successful build of {}. Using existing PTX.",
                        kc.dir_name
                    );
                }
            }
            Ok(_) => {
                eprintln!(
                    "cargo:warning={} compilation failed (exit code). Using existing {}.",
                    kc.dir_name, kc.ptx_name
                );
            }
            Err(e) => {
                eprintln!(
                    "cargo:warning=Could not run {} build: {e}. Using existing {}.",
                    kc.dir_name, kc.ptx_name
                );
            }
        }
    }

    // Backward-compat: copy kernel_test.ptx → kernel.ptx and kernel_std.ptx
    let kernel_test_ptx = manifest_dir.join("kernel_test.ptx");
    if kernel_test_ptx.exists() {
        let _ = std::fs::copy(&kernel_test_ptx, manifest_dir.join("kernel.ptx"));
        let _ = std::fs::copy(&kernel_test_ptx, manifest_dir.join("kernel_std.ptx"));
    }
}
