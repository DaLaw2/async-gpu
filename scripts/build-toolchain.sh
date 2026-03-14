#!/bin/bash
# Build a patched Rust toolchain with warp-cooperative async/await support.
#
# Usage:
#   ./scripts/build-toolchain.sh [--from-scratch]
#
# This script:
#   1. Ensures rustc-src/ exists (clone if --from-scratch)
#   2. Creates/updates patched-rustc/ with compiler patches applied
#   3. Applies std patches into the patched-rustc/library/std/
#   4. Builds the patched compiler + library (including nvptx64 sysroot)
#   5. Reports the sysroot path for use with RUSTC / --sysroot
#
# Prerequisites:
#   - Python 3 (for x.py)
#   - LLVM/Clang or MSVC build tools
#   - ~30GB disk space for build artifacts
#
# After building, use the toolchain:
#   export RUSTC_SYSROOT="$(./scripts/build-toolchain.sh --print-sysroot)"
#   # or point rustup to the stage2 sysroot

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
RUSTC_SRC="$REPO_DIR/rustc-src"
PATCHED_RUSTC="$REPO_DIR/patched-rustc"
PATCH_DIR_RUSTC="$REPO_DIR/rustc-patches"
PATCH_DIR_STD="$REPO_DIR/std-patches"

# Parse arguments
FROM_SCRATCH=false
PRINT_SYSROOT=false
TARGETS="x86_64-pc-windows-msvc,nvptx64-nvidia-cuda"

for arg in "$@"; do
    case "$arg" in
        --from-scratch) FROM_SCRATCH=true ;;
        --print-sysroot) PRINT_SYSROOT=true ;;
        --targets=*) TARGETS="${arg#--targets=}" ;;
        --help|-h)
            echo "Usage: $0 [--from-scratch] [--print-sysroot] [--targets=t1,t2]"
            echo ""
            echo "Options:"
            echo "  --from-scratch   Clone fresh rustc-src and rebuild patched-rustc from scratch"
            echo "  --print-sysroot  Print the sysroot path and exit (requires prior build)"
            echo "  --targets=...    Comma-separated build targets (default: x86_64-pc-windows-msvc,nvptx64-nvidia-cuda)"
            exit 0
            ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# --- Print sysroot mode ---
if [ "$PRINT_SYSROOT" = true ]; then
    SYSROOT="$PATCHED_RUSTC/build/host/stage2"
    if [ ! -d "$SYSROOT" ]; then
        # Try stage1
        SYSROOT="$PATCHED_RUSTC/build/host/stage1"
    fi
    if [ ! -d "$SYSROOT" ]; then
        echo "ERROR: No sysroot found. Run build-toolchain.sh first." >&2
        exit 1
    fi
    echo "$SYSROOT"
    exit 0
fi

echo "╔══════════════════════════════════════════════╗"
echo "║  async_gpu — Patched Toolchain Builder       ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ============================================================
# Step 1: Ensure rustc-src/ exists
# ============================================================

if [ "$FROM_SCRATCH" = true ] || [ ! -d "$RUSTC_SRC/compiler" ]; then
    echo "=== Step 1: Cloning rustc source ==="
    if [ -d "$RUSTC_SRC" ]; then
        echo "  Removing existing rustc-src/..."
        rm -rf "$RUSTC_SRC"
    fi
    echo "  Cloning rust-lang/rust (depth 1)..."
    git clone --depth 1 https://github.com/rust-lang/rust.git "$RUSTC_SRC"
    echo "  Done."
else
    echo "=== Step 1: rustc-src/ already present — skipping clone ==="
    echo "  (Use --from-scratch to reclone)"
fi

RUSTC_VERSION=$(cat "$RUSTC_SRC/src/version" 2>/dev/null || echo "unknown")
echo "  rustc source version: $RUSTC_VERSION"
echo ""

# ============================================================
# Step 2: Create/refresh patched-rustc/ with compiler patches
# ============================================================

echo "=== Step 2: Applying compiler patches ==="

if [ "$FROM_SCRATCH" = true ] || [ ! -d "$PATCHED_RUSTC/compiler" ]; then
    echo "  Copying rustc-src/ → patched-rustc/..."
    rm -rf "$PATCHED_RUSTC"
    # Copy everything except the .git directory and build artifacts
    mkdir -p "$PATCHED_RUSTC"
    # Use rsync if available, otherwise cp
    if command -v rsync &>/dev/null; then
        rsync -a --exclude='.git' --exclude='build' "$RUSTC_SRC/" "$PATCHED_RUSTC/"
    else
        cp -r "$RUSTC_SRC"/* "$PATCHED_RUSTC/"
        cp -r "$RUSTC_SRC"/.* "$PATCHED_RUSTC/" 2>/dev/null || true
        rm -rf "$PATCHED_RUSTC/.git"
    fi

    echo "  Applying rustc patches..."
    bash "$SCRIPT_DIR/apply-rustc-patches.sh" "$PATCHED_RUSTC"
else
    echo "  patched-rustc/ already exists with compiler patches — skipping"
    echo "  (Use --from-scratch to reapply from clean state)"
fi
echo ""

# ============================================================
# Step 3: Apply std patches into patched-rustc/library/std/
# ============================================================

echo "=== Step 3: Applying std patches into patched-rustc/library/std/ ==="

# The apply-std-patches.sh script copies from rustc-src/library/std/ to an output dir.
# We need to apply patches directly to patched-rustc/library/std/ instead.
# We do this by running apply-std-patches.sh with patched-rustc/library/std as output.

PATCHED_STD="$PATCHED_RUSTC/library/std"

# Check if std patches are already applied by looking for a marker file
MARKER="$PATCHED_STD/.async_gpu_std_patched"

if [ "$FROM_SCRATCH" = true ] || [ ! -f "$MARKER" ]; then
    echo "  Applying std patches directly to patched-rustc/library/std/..."

    # Reset std/src to stock first (from rustc-src), then apply patches in-place.
    STOCK_STD="$REPO_DIR/rustc-src/library/std"
    rm -rf "$PATCHED_STD/src"
    cp -r "$STOCK_STD/src" "$PATCHED_STD/src"

    cd "$PATCHED_STD"

    # Apply each patch with -p1 (patches use a/src/... b/src/... format)
    for pf in "$PATCH_DIR_STD"/*.patch; do
        pname=$(basename "$pf")
        echo "    [PATCH] $pname"
        patch -p1 --binary < "$pf"
    done

    # Copy new .rs files (explicit mapping — filenames don't map cleanly to paths)
    copy_new() { mkdir -p "$(dirname "$2")"; cp "$PATCH_DIR_STD/$1" "$2"; echo "    [NEW]   $2"; }
    copy_new sys_alloc_cuda.rs           src/sys/alloc/cuda.rs
    copy_new sys_fs_cuda.rs              src/sys/fs/cuda.rs
    copy_new sys_io_error_cuda.rs        src/sys/io/error/cuda.rs
    copy_new sys_stdio_cuda.rs           src/sys/stdio/cuda.rs
    copy_new sys_thread_local_gpu_threads.rs src/sys/thread_local/gpu_threads.rs

    cd "$REPO_DIR"

    # Write marker
    echo "Patched by build-toolchain.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$MARKER"
    echo "  Done."
else
    echo "  std patches already applied (marker found) — skipping"
    echo "  (Use --from-scratch to reapply)"
fi
echo ""

# ============================================================
# Step 4: Configure and build
# ============================================================

echo "=== Step 4: Building patched toolchain ==="

# Generate bootstrap config (bootstrap.toml / config.toml)
BOOTSTRAP_CONFIG="$PATCHED_RUSTC/bootstrap.toml"

# Detect host triple
HOST_TRIPLE=$(rustc -vV 2>/dev/null | grep '^host:' | awk '{print $2}')
if [ -z "$HOST_TRIPLE" ]; then
    HOST_TRIPLE="x86_64-pc-windows-msvc"
fi

echo "  Host triple: $HOST_TRIPLE"
echo "  Build targets: $TARGETS"

# Write bootstrap config
cat > "$BOOTSTRAP_CONFIG" << EOF
# Generated by build-toolchain.sh — safe to regenerate
# See bootstrap.example.toml for all options

[build]
# Build both compiler and library
build = "$HOST_TRIPLE"
host = ["$HOST_TRIPLE"]
target = [$(echo "$TARGETS" | sed 's/,/", "/g; s/^/"/; s/$/"/' )]

# Use the system's installed nightly as the bootstrap compiler
# (This avoids needing to download a separate stage0)
# cargo = "cargo"
# rustc = "rustc"

[rust]
# Use incremental compilation for faster rebuilds
incremental = true
# Debug assertions help catch bugs in the compiler
debug-assertions = true
# Don't optimize heavily — faster builds
optimize = 1

[llvm]
# Download pre-built LLVM instead of building from source (MUCH faster)
download-ci-llvm = true

[target.nvptx64-nvidia-cuda]
# nvptx64 only needs core/alloc, not full std
# The build system handles this automatically
EOF

echo "  Generated bootstrap.toml"
echo ""

# On Windows, ensure MSVC environment is available (cl.exe, link.exe)
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$(uname -s)" == *"MINGW"* || "$(uname -s)" == *"MSYS"* ]]; then
    if ! command -v cl.exe &>/dev/null; then
        echo "  MSVC cl.exe not in PATH — searching for vcvarsall.bat..."
        VCVARSALL=""
        for vs_year in 2022 2019 2017; do
            for vs_edition in Community Professional Enterprise BuildTools; do
                candidate="C:/Program Files/Microsoft Visual Studio/$vs_year/$vs_edition/VC/Auxiliary/Build/vcvarsall.bat"
                if [ -f "$candidate" ]; then
                    VCVARSALL="$candidate"
                    break 2
                fi
            done
        done
        # Also check x86 Program Files
        if [ -z "$VCVARSALL" ]; then
            for vs_year in 2022 2019 2017; do
                for vs_edition in Community Professional Enterprise BuildTools; do
                    candidate="C:/Program Files (x86)/Microsoft Visual Studio/$vs_year/$vs_edition/VC/Auxiliary/Build/vcvarsall.bat"
                    if [ -f "$candidate" ]; then
                        VCVARSALL="$candidate"
                        break 2
                    fi
                done
            done
        fi

        if [ -n "$VCVARSALL" ]; then
            echo "  Found VS at: $(dirname "$(dirname "$(dirname "$VCVARSALL")")")"
            VSDIR=$(dirname "$(dirname "$(dirname "$VCVARSALL")")")

            # Auto-detect MSVC tools version and Windows SDK version
            MSVC_VER=$(ls "$VSDIR/Tools/MSVC/" 2>/dev/null | sort -V | tail -1)
            WINSDK_VER=$(ls "C:/Program Files (x86)/Windows Kits/10/Include/" 2>/dev/null | sort -V | tail -1)
            echo "  MSVC version: $MSVC_VER"
            echo "  Windows SDK: $WINSDK_VER"

            MSVC_BASE="$VSDIR/Tools/MSVC/$MSVC_VER"
            WINSDK_INC="C:/Program Files (x86)/Windows Kits/10/Include/$WINSDK_VER"
            WINSDK_LIB="C:/Program Files (x86)/Windows Kits/10/Lib/$WINSDK_VER"

            # Set PATH (add MSVC bin dir)
            export PATH="$(cygpath -u "$MSVC_BASE/bin/Hostx64/x64"):$PATH"

            # Set INCLUDE (MSVC headers + Windows SDK headers)
            export INCLUDE="$(cygpath -w "$MSVC_BASE/include");$(cygpath -w "$WINSDK_INC/ucrt");$(cygpath -w "$WINSDK_INC/shared");$(cygpath -w "$WINSDK_INC/um");$(cygpath -w "$WINSDK_INC/winrt")"

            # Set LIB (MSVC libs + Windows SDK libs)
            export LIB="$(cygpath -w "$MSVC_BASE/lib/x64");$(cygpath -w "$WINSDK_LIB/ucrt/x64");$(cygpath -w "$WINSDK_LIB/um/x64")"

            # Set CC/CXX for the cc crate
            export CC="cl.exe"
            export CXX="cl.exe"

            if command -v cl.exe &>/dev/null; then
                echo "  MSVC loaded: $(cl.exe 2>&1 | head -1)"
            else
                echo "  WARNING: cl.exe still not in PATH."
            fi
        else
            echo "  ERROR: Cannot find vcvarsall.bat. Install Visual Studio Build Tools."
            echo "  Or run this script from a Developer Command Prompt."
            exit 1
        fi
    else
        echo "  MSVC already in PATH: $(cl.exe 2>&1 | head -1)"
    fi
fi

# Run the build
echo "  Starting x.py build (this may take 20-60 minutes on first build)..."
echo "  Building: compiler + library for [$TARGETS]"
echo ""

cd "$PATCHED_RUSTC"

# Build stage1 compiler + library (stage2 is only needed for full dist)
# For our purposes, stage1 with the patched MIR pass is sufficient.
python x.py build compiler library 2>&1 | tee "$REPO_DIR/.research/toolchain-build.log" || {
    echo ""
    echo "╔══════════════════════════════════════════════╗"
    echo "║  BUILD FAILED                                ║"
    echo "╚══════════════════════════════════════════════╝"
    echo ""
    echo "Build log: .research/toolchain-build.log"
    echo "Common fixes:"
    echo "  - Missing Python: install Python 3"
    echo "  - Missing MSVC: install Visual Studio Build Tools"
    echo "  - LLVM download failed: check internet connection"
    echo ""
    exit 1
}

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  BUILD SUCCEEDED                             ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# Find the sysroot
SYSROOT=""
for stage in stage2 stage1; do
    # On Windows, build dir uses the host triple name
    for build_dir in "$PATCHED_RUSTC/build/$HOST_TRIPLE/$stage" "$PATCHED_RUSTC/build/host/$stage"; do
        if [ -d "$build_dir" ]; then
            SYSROOT="$build_dir"
            break 2
        fi
    done
done

if [ -z "$SYSROOT" ]; then
    echo "WARNING: Could not determine sysroot path automatically."
    echo "Check: $PATCHED_RUSTC/build/"
else
    echo "Sysroot: $SYSROOT"
    echo ""

    # Verify nvptx64 target is present
    NVPTX_LIB="$SYSROOT/lib/rustlib/nvptx64-nvidia-cuda/lib"
    if [ -d "$NVPTX_LIB" ]; then
        echo "nvptx64 target: PRESENT"
        ls "$NVPTX_LIB"/*.rlib 2>/dev/null | head -5
        echo "  ..."
    else
        echo "WARNING: nvptx64 target libraries not found at $NVPTX_LIB"
        echo "The build may not have included the nvptx64 target."
    fi

    echo ""
    echo "Usage:"
    echo "  # Set as default toolchain for this project:"
    echo "  export RUSTC=\"$SYSROOT/bin/rustc\""
    echo ""
    echo "  # Or use --sysroot:"
    echo "  rustc --sysroot \"$SYSROOT\" --target nvptx64-nvidia-cuda ..."
    echo ""
    echo "  # Get sysroot programmatically:"
    echo "  ./scripts/build-toolchain.sh --print-sysroot"
fi
