//! Build script: compiles the warp-cooperative kernel crate to PTX.
//!
//! This example requires the patched rustc toolchain (MIR pass for async
//! warp convergence). The build script tries the patched rustc first, then
//! falls back to stock nightly (which will likely fail for async kernels).

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    manifest_dir.join("..").join("..").join("..").join("..")
}

fn nightly_toolchain() -> String {
    let toolchain_file = repo_root().join("rust-toolchain.toml");
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

fn patched_rustc() -> Option<PathBuf> {
    let candidates = [
        repo_root().join("patched-rustc/build/x86_64-unknown-linux-gnu/stage1/bin/rustc"),
        repo_root().join("patched-rustc/build/host/stage1/bin/rustc"),
    ];
    candidates.into_iter().find(|p| p.exists())
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let kernel_dir = manifest_dir.join("..");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let toolchain = nightly_toolchain();

    println!("cargo:rerun-if-changed=../src/lib.rs");
    println!("cargo:rerun-if-changed=../Cargo.toml");
    println!("cargo:rerun-if-changed=../../../../rust-toolchain.toml");

    let mut cmd = Command::new("cargo");
    cmd.args([
            &toolchain, "build", "--release",
            "--target", "nvptx64-nvidia-cuda",
            "-Zbuild-std=core",
        ])
        .current_dir(&kernel_dir)
        .env_remove("CARGO")
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .env_remove("CARGO_TARGET_DIR")
        .env_remove("CARGO_BUILD_TARGET")
        .env("CARGO_ENCODED_RUSTFLAGS", "-Ctarget-cpu=sm_75\x1f-Ctarget-feature=+ptx78");

    if let Some(rustc) = patched_rustc() {
        eprintln!("cargo:warning=Using patched rustc: {rustc:?}");
        cmd.env("RUSTC", &rustc);
    } else {
        cmd.env_remove("RUSTC");
    }

    // Ensure llvm-bitcode-linker is on PATH (lives in the stock nightly sysroot)
    let sysroot_bin = std::process::Command::new("rustup")
        .args(["run", toolchain.trim_start_matches('+'), "rustc", "--print", "sysroot"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| {
            PathBuf::from(s.trim())
                .join("lib/rustlib/x86_64-unknown-linux-gnu/bin/self-contained")
        });
    if let Some(bin_dir) = sysroot_bin.filter(|p| p.exists()) {
        let path = env::var("PATH").unwrap_or_default();
        cmd.env("PATH", format!("{}:{path}", bin_dir.display()));
    }

    let status = cmd.status();

    let ptx_src = kernel_dir
        .join("target")
        .join("nvptx64-nvidia-cuda")
        .join("release")
        .join("test_warp_crate.ptx");

    match status {
        Ok(s) if s.success() && ptx_src.exists() => {}
        _ => {
            if ptx_src.exists() {
                eprintln!("cargo:warning=Kernel build failed; using cached PTX");
            } else {
                panic!(
                    "Kernel PTX compilation failed and no cached PTX at {:?}. \
                     This example requires the patched rustc toolchain \
                     (run scripts/build-toolchain.sh first).",
                    ptx_src
                );
            }
        }
    }

    let ptx_content = std::fs::read_to_string(&ptx_src).expect("Failed to read PTX");
    if ptx_content.contains(".target sm_30") {
        let patched = ptx_content
            .replace(".target sm_30", ".target sm_75")
            .replace(".version 6.0", ".version 7.1");
        std::fs::write(out_dir.join("kernel.ptx"), patched).expect("Failed to write PTX");
    } else {
        std::fs::copy(&ptx_src, out_dir.join("kernel.ptx")).expect("Failed to copy PTX");
    }
}
