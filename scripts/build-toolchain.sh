#!/bin/bash
# Build a patched Rust toolchain with warp-cooperative async/await support.
# For Linux/macOS only. On Windows, use build-toolchain.ps1 instead.
#
# Usage:
#   ./scripts/build-toolchain.sh [--from-scratch] [--print-sysroot] [--targets=t1,t2]
#
# Prerequisites:
#   - Python 3, git, cmake, ninja (or make), clang/gcc
#   - ~30GB disk space for build artifacts

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
RUSTC_SRC="$REPO_DIR/rustc-src"
PATCHED_RUSTC="$REPO_DIR/patched-rustc"
PATCH_DIR_STD="$REPO_DIR/std-patches"

# Detect host triple
HOST_TRIPLE=$(rustc -vV 2>/dev/null | grep '^host:' | awk '{print $2}')
if [ -z "$HOST_TRIPLE" ]; then
    HOST_TRIPLE="x86_64-unknown-linux-gnu"
fi

# Parse arguments
FROM_SCRATCH=false
PRINT_SYSROOT=false
TARGETS="$HOST_TRIPLE,nvptx64-nvidia-cuda"

for arg in "$@"; do
    case "$arg" in
        --from-scratch) FROM_SCRATCH=true ;;
        --print-sysroot) PRINT_SYSROOT=true ;;
        --targets=*) TARGETS="${arg#--targets=}" ;;
        --help|-h)
            echo "Usage: $0 [--from-scratch] [--print-sysroot] [--targets=t1,t2]"
            echo ""
            echo "Options:"
            echo "  --from-scratch   Clone fresh rustc-src and rebuild from scratch"
            echo "  --print-sysroot  Print the sysroot path and exit (requires prior build)"
            echo "  --targets=...    Comma-separated build targets (default: $TARGETS)"
            exit 0
            ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# --- Print sysroot mode ---
if [ "$PRINT_SYSROOT" = true ]; then
    for stage in stage2 stage1; do
        for build_dir in "$PATCHED_RUSTC/build/host/$stage" "$PATCHED_RUSTC/build/$HOST_TRIPLE/$stage"; do
            if [ -d "$build_dir" ]; then echo "$build_dir"; exit 0; fi
        done
    done
    echo "ERROR: No sysroot found. Run build-toolchain.sh first." >&2
    exit 1
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
    rm -rf "$RUSTC_SRC"
    git clone --depth 1 https://github.com/rust-lang/rust.git "$RUSTC_SRC"
else
    echo "=== Step 1: rustc-src/ already present (use --from-scratch to reclone) ==="
fi
echo "  Version: $(cat "$RUSTC_SRC/src/version" 2>/dev/null || echo "unknown")"
echo ""

# ============================================================
# Step 2: Create/refresh patched-rustc/ with compiler patches
# ============================================================

echo "=== Step 2: Applying compiler patches ==="

if [ "$FROM_SCRATCH" = true ] || [ ! -d "$PATCHED_RUSTC/compiler" ]; then
    rm -rf "$PATCHED_RUSTC"
    mkdir -p "$PATCHED_RUSTC"
    rsync -a --exclude='.git' --exclude='build' "$RUSTC_SRC/" "$PATCHED_RUSTC/"
    bash "$SCRIPT_DIR/apply-rustc-patches.sh" "$PATCHED_RUSTC"
else
    echo "  Already present — skipping (use --from-scratch to reapply)"
fi
echo ""

# ============================================================
# Step 3: Apply std patches into patched-rustc/library/std/
# ============================================================

echo "=== Step 3: Applying std patches ==="

PATCHED_STD="$PATCHED_RUSTC/library/std"
MARKER="$PATCHED_STD/.async_gpu_std_patched"

if [ "$FROM_SCRATCH" = true ] || [ ! -f "$MARKER" ]; then
    # Reset std/src to stock, then apply patches
    rm -rf "$PATCHED_STD/src"
    cp -r "$RUSTC_SRC/library/std/src" "$PATCHED_STD/src"

    cd "$PATCHED_STD"
    for pf in "$PATCH_DIR_STD"/*.patch; do
        echo "    [PATCH] $(basename "$pf")"
        patch -p1 --binary < "$pf"
    done

    copy_new() { mkdir -p "$(dirname "$2")"; cp "$PATCH_DIR_STD/$1" "$2"; echo "    [NEW]   $2"; }
    copy_new sys_alloc_cuda.rs           src/sys/alloc/cuda.rs
    copy_new sys_fs_cuda.rs              src/sys/fs/cuda.rs
    copy_new sys_io_error_cuda.rs        src/sys/io/error/cuda.rs
    copy_new sys_stdio_cuda.rs           src/sys/stdio/cuda.rs
    copy_new sys_thread_local_gpu_threads.rs src/sys/thread_local/gpu_threads.rs
    cd "$REPO_DIR"

    echo "Patched on $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$MARKER"
else
    echo "  Already applied — skipping (use --from-scratch to reapply)"
fi
echo ""

# ============================================================
# Step 4: Configure and build
# ============================================================

echo "=== Step 4: Building patched toolchain ==="
echo "  Host: $HOST_TRIPLE"
echo "  Targets: $TARGETS"

cat > "$PATCHED_RUSTC/bootstrap.toml" << EOF
change-id = "ignore"

[build]
build = "$HOST_TRIPLE"
host = ["$HOST_TRIPLE"]
target = [$(echo "$TARGETS" | sed 's/,/", "/g; s/^/"/; s/$/"/' )]

[rust]
incremental = true
debug-assertions = true
optimize = 1

[llvm]
download-ci-llvm = true

[target.nvptx64-nvidia-cuda]
EOF

cd "$PATCHED_RUSTC"
python3 x.py build compiler library 2>&1 | tee "$REPO_DIR/.research/toolchain-build.log" || {
    echo ""
    echo "BUILD FAILED — see .research/toolchain-build.log"
    exit 1
}

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  BUILD SUCCEEDED                             ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# Find sysroot
SYSROOT=""
for stage in stage2 stage1; do
    for build_dir in "$PATCHED_RUSTC/build/$HOST_TRIPLE/$stage" "$PATCHED_RUSTC/build/host/$stage"; do
        [ -d "$build_dir" ] && SYSROOT="$build_dir" && break 2
    done
done

if [ -n "$SYSROOT" ]; then
    echo "Sysroot: $SYSROOT"
    NVPTX_LIB="$SYSROOT/lib/rustlib/nvptx64-nvidia-cuda/lib"
    [ -d "$NVPTX_LIB" ] && echo "nvptx64: PRESENT" || echo "WARNING: nvptx64 libs not found"
    echo ""
    echo "Usage:  export RUSTC=\"$SYSROOT/bin/rustc\""
else
    echo "WARNING: Could not find sysroot. Check: $PATCHED_RUSTC/build/"
fi
