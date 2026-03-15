//! cargo xtask — build automation for async_gpu
//!
//! Provides a single entry point for GPU kernel builds, PTX post-processing,
//! and host crate compilation.
//!
//! Usage:
//!   cargo xtask gpu-build              # build all GPU kernels
//!   cargo xtask gpu-build hello-gpu    # build a specific example kernel
//!   cargo xtask gpu-build --list       # list discovered kernels
//!   cargo xtask gpu-build --postprocess  # apply PTX post-processing

use std::env;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

// ---------------------------------------------------------------------------
// Error handling (no anyhow!)
// ---------------------------------------------------------------------------

#[derive(Debug)]
enum XtaskError {
    UnknownCommand(String),
    KernelNotFound(String),
    BuildFailed { kernel: String, detail: String },
    IoError(String, std::io::Error),
}

impl fmt::Display for XtaskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownCommand(cmd) => write!(f, "Unknown command: {cmd}"),
            Self::KernelNotFound(name) => write!(f, "Kernel not found: {name}"),
            Self::BuildFailed { kernel, detail } => {
                write!(f, "Build failed for {kernel}: {detail}")
            }
            Self::IoError(ctx, e) => write!(f, "{ctx}: {e}"),
        }
    }
}

type Result<T> = std::result::Result<T, XtaskError>;

// ---------------------------------------------------------------------------
// Kernel discovery
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct KernelCrate {
    name: String,
    path: PathBuf,
    needs_patched_rustc: bool,
    needs_std: bool,
}

fn repo_root() -> PathBuf {
    let manifest = env::var("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| env::current_dir().unwrap());

    // Walk up until we find Cargo.toml with [workspace]
    let mut dir = manifest.as_path();
    loop {
        let cargo_toml = dir.join("Cargo.toml");
        if cargo_toml.exists() {
            if let Ok(content) = fs::read_to_string(&cargo_toml) {
                if content.contains("[workspace]") {
                    return dir.to_path_buf();
                }
            }
        }
        dir = match dir.parent() {
            Some(p) => p,
            None => return manifest,
        };
    }
}

fn discover_kernels(root: &Path) -> Vec<KernelCrate> {
    let mut kernels = Vec::new();

    // Known kernel locations
    let search_dirs = [
        ("examples", false),
        ("crates/kernel", true),
        ("crates/test", true),
    ];

    for (base, is_internal) in &search_dirs {
        let base_path = root.join(base);
        if !base_path.exists() {
            continue;
        }

        let entries = match fs::read_dir(&base_path) {
            Ok(e) => e,
            Err(_) => continue,
        };

        for entry in entries.flatten() {
            let entry_path = entry.path();
            if !entry_path.is_dir() {
                continue;
            }

            // For examples: kernel is in <example>/kernel/
            // For crates/kernel and crates/test: the directory itself is the kernel
            let kernel_path = if !is_internal {
                entry_path.join("kernel")
            } else {
                entry_path.clone()
            };

            let config_path = kernel_path.join(".cargo").join("config.toml");
            if !config_path.exists() {
                continue;
            }

            // Check if it has nvptx64 target
            let config_content = match fs::read_to_string(&config_path) {
                Ok(c) => c,
                Err(_) => continue,
            };
            if !config_content.contains("nvptx64-nvidia-cuda") {
                continue;
            }

            let name = entry_path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .to_string();

            // Check if it needs patched rustc or std
            let needs_patched_rustc = name == "async-pipeline"
                || name == "warp-cooperative"
                || name == "async-pipeline-test";
            let needs_std = config_content.contains("\"std\"");

            kernels.push(KernelCrate {
                name,
                path: kernel_path,
                needs_patched_rustc,
                needs_std,
            });
        }
    }

    kernels.sort_by(|a, b| a.name.cmp(&b.name));
    kernels
}

// ---------------------------------------------------------------------------
// Toolchain resolution
// ---------------------------------------------------------------------------

fn read_nightly_toolchain(root: &Path) -> String {
    let toolchain_file = root.join("rust-toolchain.toml");
    if let Ok(content) = fs::read_to_string(&toolchain_file) {
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

// ---------------------------------------------------------------------------
// PTX post-processing
// ---------------------------------------------------------------------------

fn postprocess_ptx(ptx: &str) -> String {
    let mut result = String::with_capacity(ptx.len());

    for line in ptx.lines() {
        // Fix 1: Remove `.ptr .align N` parameter annotations
        // Before: .param .u64 .ptr .align 1 name
        // After:  .param .u64 name
        let processed = remove_ptr_align(line);

        // Fix 2: Stub extern panic/abort functions
        // Before: .extern .func panic_const_async_fn_resumed (...);
        // After:  .visible .func panic_const_async_fn_resumed (...) { trap; ret; }
        if is_extern_panic_decl(&processed) {
            let stubbed = stub_extern_panic(&processed);
            result.push_str(&stubbed);
        } else {
            result.push_str(&processed);
        }
        result.push('\n');
    }

    // Fix 3: Patch old target version
    let result = result.replace(".target sm_30", ".target sm_86");
    result.replace(".version 6.0", ".version 7.1")
}

fn remove_ptr_align(line: &str) -> String {
    // Remove all occurrences of `.ptr .align N` (where N is a decimal number)
    // Example: `.param .u64 .ptr .align 1 name` → `.param .u64 name`
    let mut result = line.to_string();
    while let Some(ptr_pos) = result.find(".ptr") {
        let rest = &result[ptr_pos + 4..];
        let rest_trimmed = rest.trim_start();
        if !rest_trimmed.starts_with(".align") {
            break;
        }
        let after_align = &rest_trimmed[6..]; // skip ".align"
        let after_align_trimmed = after_align.trim_start();
        let num_len = after_align_trimmed
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(after_align_trimmed.len());
        if num_len == 0 {
            break;
        }
        // Calculate the byte position where the number ends
        let remainder = &after_align_trimmed[num_len..];
        result = format!("{}{}", &result[..ptr_pos], remainder);
    }
    result
}

fn is_extern_panic_decl(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.starts_with(".extern .func")
        && trimmed.ends_with(';')
        && (trimmed.contains("panic") || trimmed.contains("abort"))
}

fn stub_extern_panic(line: &str) -> String {
    // .extern .func name (...); → .visible .func name (...) { trap; ret; }
    let trimmed = line.trim();

    // Remove ".extern" prefix, replace with ".visible"
    let without_extern = trimmed
        .strip_prefix(".extern")
        .unwrap_or(trimmed)
        .trim_start();

    // Remove trailing semicolon
    let without_semi = without_extern.trim_end_matches(';').trim_end();

    format!(".visible {without_semi}\n{{\n\ttrap;\n\tret;\n}}")
}

// ---------------------------------------------------------------------------
// Build logic
// ---------------------------------------------------------------------------

fn build_kernel(kernel: &KernelCrate, toolchain: &str, do_postprocess: bool) -> Result<()> {
    println!("  Building {}...", kernel.name);

    let mut cmd = Command::new("cargo");
    cmd.args([toolchain, "build", "--release"])
        .current_dir(&kernel.path)
        .env_remove("CARGO")
        .env_remove("RUSTC")
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .env_remove("CARGO_TARGET_DIR")
        .env_remove("CARGO_BUILD_TARGET");

    let status = cmd
        .status()
        .map_err(|e| XtaskError::IoError(format!("Failed to run cargo for {}", kernel.name), e))?;

    if !status.success() {
        return Err(XtaskError::BuildFailed {
            kernel: kernel.name.clone(),
            detail: format!("cargo build exited with {status}"),
        });
    }

    // Find PTX output
    let ptx_dir = kernel
        .path
        .join("target")
        .join("nvptx64-nvidia-cuda")
        .join("release");

    if do_postprocess {
        // Find .ptx files in the output directory
        if let Ok(entries) = fs::read_dir(&ptx_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|e| e == "ptx") {
                    let content = fs::read_to_string(&path)
                        .map_err(|e| XtaskError::IoError(format!("Read PTX {:?}", path), e))?;
                    let processed = postprocess_ptx(&content);
                    if processed != content {
                        fs::write(&path, &processed)
                            .map_err(|e| XtaskError::IoError(format!("Write PTX {:?}", path), e))?;
                        println!("    Post-processed: {}", path.display());
                    }
                }
            }
        }
    }

    println!("    OK");
    Ok(())
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

fn cmd_gpu_build(args: &[String]) -> Result<()> {
    let root = repo_root();
    let toolchain = read_nightly_toolchain(&root);
    let all_kernels = discover_kernels(&root);

    let mut list_only = false;
    let mut postprocess = false;
    let mut filter: Option<String> = None;
    let mut skip_patched = true;

    for arg in args {
        match arg.as_str() {
            "--list" => list_only = true,
            "--postprocess" => postprocess = true,
            "--include-patched" => skip_patched = false,
            _ if !arg.starts_with('-') => filter = Some(arg.clone()),
            _ => {
                eprintln!("Unknown flag: {arg}");
            }
        }
    }

    if list_only {
        println!("Discovered {} kernel crates:", all_kernels.len());
        for k in &all_kernels {
            let flags = match (k.needs_patched_rustc, k.needs_std) {
                (true, true) => " [patched-rustc, std]",
                (true, false) => " [patched-rustc]",
                (false, true) => " [std]",
                (false, false) => "",
            };
            println!("  {:<30} {}{}", k.name, k.path.display(), flags);
        }
        return Ok(());
    }

    let kernels_to_build: Vec<&KernelCrate> = if let Some(ref name) = filter {
        let found: Vec<_> = all_kernels.iter().filter(|k| k.name == *name).collect();
        if found.is_empty() {
            return Err(XtaskError::KernelNotFound(name.clone()));
        }
        found
    } else {
        all_kernels
            .iter()
            .filter(|k| !skip_patched || !k.needs_patched_rustc)
            .collect()
    };

    println!(
        "==> Building {} kernel(s) with toolchain {}",
        kernels_to_build.len(),
        toolchain
    );

    let mut failed = Vec::new();
    let mut succeeded = 0;

    for kernel in &kernels_to_build {
        match build_kernel(kernel, &toolchain, postprocess) {
            Ok(()) => succeeded += 1,
            Err(e) => {
                eprintln!("    FAILED: {e}");
                failed.push(kernel.name.clone());
            }
        }
    }

    println!();
    if failed.is_empty() {
        println!("==> All {succeeded} kernel(s) built successfully!");
    } else {
        println!(
            "==> {succeeded} succeeded, {} failed: {}",
            failed.len(),
            failed.join(", ")
        );
    }

    if failed.is_empty() {
        Ok(())
    } else {
        Err(XtaskError::BuildFailed {
            kernel: failed.join(", "),
            detail: "One or more kernels failed to build".into(),
        })
    }
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();

    if args.is_empty() {
        print_help();
        return;
    }

    let result = match args[0].as_str() {
        "gpu-build" => cmd_gpu_build(&args[1..]),
        "help" | "--help" | "-h" => {
            print_help();
            Ok(())
        }
        cmd => Err(XtaskError::UnknownCommand(cmd.to_string())),
    };

    if let Err(e) = result {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn print_help() {
    println!(
        "cargo xtask — build automation for async_gpu

COMMANDS:
    gpu-build              Build GPU kernel crates to PTX
    help                   Show this help

GPU-BUILD OPTIONS:
    <name>                 Build a specific kernel by name
    --list                 List discovered kernel crates
    --postprocess          Apply PTX post-processing (ptr align removal, panic stubs)
    --include-patched      Include kernels that need patched rustc (skipped by default)

EXAMPLES:
    cargo xtask gpu-build                  # Build all stock-nightly kernels
    cargo xtask gpu-build hello-gpu        # Build just hello-gpu kernel
    cargo xtask gpu-build --list           # Show all discovered kernels
    cargo xtask gpu-build --postprocess    # Build + post-process PTX"
    );
}
