//! Build script: compiles the kernel crate to PTX using the patched rustc.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let kernel_dir = manifest_dir.join("..").join("kernel");
    let repo_root = manifest_dir.join("..").join("..").join("..");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    println!("cargo:rerun-if-changed=../kernel/src/lib.rs");
    println!("cargo:rerun-if-changed=../kernel/Cargo.toml");

    let patched_sysroot = repo_root
        .join("patched-rustc")
        .join("build")
        .join("x86_64-pc-windows-msvc")
        .join("stage1");
    let patched_rustc = patched_sysroot.join("bin").join("rustc.exe");

    let use_patched = patched_rustc.exists();
    if use_patched {
        eprintln!(
            "cargo:warning=Using patched rustc: {}",
            patched_rustc.display()
        );
    } else {
        eprintln!("cargo:warning=Patched rustc not found, using stock nightly");
    }

    let mut cmd = Command::new("cargo");
    let toolchain = read_nightly_toolchain(&repo_root);
    cmd.arg(&toolchain);
    cmd.args(["build", "--release"]);
    cmd.current_dir(&kernel_dir);

    cmd.env_remove("CARGO_ENCODED_RUSTFLAGS");
    cmd.env_remove("CARGO_TARGET_DIR");
    cmd.env_remove("CARGO_BUILD_TARGET");

    if use_patched {
        cmd.env("RUSTC", &patched_rustc);
    } else {
        cmd.env_remove("RUSTC");
    }
    cmd.env_remove("RUSTFLAGS");
    cmd.env_remove("CARGO");

    let status = cmd.status().expect("Failed to run cargo for kernel build");

    let ptx_name = "parallel_search_kernel.ptx";
    if !status.success() {
        let ptx_fallback = kernel_dir
            .join("target")
            .join("nvptx64-nvidia-cuda")
            .join("release")
            .join(ptx_name);
        if ptx_fallback.exists() {
            eprintln!(
                "cargo:warning=Kernel compilation failed; using cached PTX at {:?}",
                ptx_fallback
            );
        } else {
            panic!("Kernel PTX compilation failed and no cached PTX found");
        }
    }

    let ptx_src = kernel_dir
        .join("target")
        .join("nvptx64-nvidia-cuda")
        .join("release")
        .join(ptx_name);

    if !ptx_src.exists() {
        panic!(
            "PTX file not found at {:?}. Check kernel compilation output.",
            ptx_src
        );
    }

    let ptx_content = std::fs::read_to_string(&ptx_src).expect("Failed to read PTX");
    let ptx_dst = out_dir.join("kernel.ptx");
    if ptx_content.contains(".target sm_30") {
        let patched = ptx_content
            .replace(".target sm_30", ".target sm_86")
            .replace(".version 6.0", ".version 7.1");
        std::fs::write(&ptx_dst, patched).expect("Failed to write patched PTX");
    } else {
        std::fs::copy(&ptx_src, &ptx_dst).expect("Failed to copy PTX file");
    }
}

fn read_nightly_toolchain(repo_root: &std::path::Path) -> String {
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
