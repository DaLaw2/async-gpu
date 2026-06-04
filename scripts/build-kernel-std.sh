#!/bin/bash
# Build the unified gpu-kernel-std PTX and pre-compile to cubin for fast loading.
#
# After the kernel crate merge, gpu-kernel-std is the ONLY kernel crate.
# It contains all kernel entry points (compute, hostcall, warp, thread, std).
#
# The kernel_std.ptx is 5+ MB and takes 10+ minutes to JIT compile.
# Pre-compiling to cubin with ptxas reduces load time to <1 second.
#
# Prerequisites:
#   - Patched std in sysroot (run apply-std-patches.sh first)
#   - CUDA toolkit (ptxas)
#
# Usage: ./scripts/build-kernel-std.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
KERNEL_STD_DIR="$REPO_DIR/crates/kernel/gpu-kernel-std"
HOST_DIR="$REPO_DIR/crates/core/gpu-host"

# Find toolchain
TOOLCHAIN_FILE="$REPO_DIR/rust-toolchain.toml"
CHANNEL=$(grep 'channel' "$TOOLCHAIN_FILE" | sed 's/.*= *"\(.*\)"/\1/')
echo "Using toolchain: +$CHANNEL"

# Find ptxas
PTXAS=""
for dir in /usr/local/cuda*/bin /opt/cuda/bin; do
    if [ -x "$dir/ptxas" ]; then
        PTXAS="$dir/ptxas"
        break
    fi
done
if [ -z "$PTXAS" ]; then
    echo "ERROR: ptxas not found. Install CUDA toolkit."
    exit 1
fi
echo "Using ptxas: $PTXAS"

# Step 1: Ensure patched std is in sysroot
SYSROOT=$(rustup run "$CHANNEL" rustc --print sysroot)
STD_SRC="$SYSROOT/lib/rustlib/src/rust/library/std"
if [ ! -f "$STD_SRC/src/sys/thread/cuda.rs" ]; then
    echo "Patched std not found in sysroot. Applying patches..."
    PATCHED_STD="$REPO_DIR/patched-std"
    if [ ! -d "$PATCHED_STD/src" ]; then
        echo "ERROR: patched-std/ not found. Run apply-std-patches.sh first."
        exit 1
    fi
    # Back up stock std
    if [ ! -d "$STD_SRC.bak" ]; then
        cp -r "$STD_SRC" "$STD_SRC.bak"
    fi
    rsync -a "$PATCHED_STD/src/" "$STD_SRC/src/"
    # Copy latest std-patches new files over (they may be newer than patched-std)
    for f in sys_thread_cuda.rs sys_alloc_cuda.rs sys_fs_cuda.rs sys_io_error_cuda.rs sys_stdio_cuda.rs sys_thread_local_gpu_threads.rs; do
        target=$(echo "$f" | sed 's/^sys_/src\/sys\//' | sed 's/_cuda\.rs$/\/cuda.rs/' | sed 's/_gpu_threads\.rs$/\/gpu_threads.rs/')
        if [ "$f" = "sys_thread_local_gpu_threads.rs" ]; then
            target="src/sys/thread_local/gpu_threads.rs"
        fi
        if [ -f "$REPO_DIR/std-patches/$f" ]; then
            cp "$REPO_DIR/std-patches/$f" "$STD_SRC/$target"
        fi
    done
    echo "Patched std applied to sysroot."
fi

# Step 2: Build gpu-kernel-std (unified kernel crate)
echo ""
echo "=== Building gpu-kernel-std (unified kernel crate) ==="
cd "$KERNEL_STD_DIR"
cargo "+$CHANNEL" build --release 2>&1 | grep -E "Compiling|Finished|error|warning.*gpu-kernel"
PTX_SRC="$KERNEL_STD_DIR/target/nvptx64-nvidia-cuda/release/gpu_kernel_std.ptx"
if [ ! -f "$PTX_SRC" ]; then
    echo "ERROR: PTX not generated at $PTX_SRC"
    exit 1
fi
echo "PTX: $(wc -c < "$PTX_SRC") bytes"

# Step 3: Copy PTX to gpu-host (both names for backward compat)
cp "$PTX_SRC" "$HOST_DIR/kernel_std.ptx"
cp "$PTX_SRC" "$HOST_DIR/kernel.ptx"
echo "Copied PTX to gpu-host/ (kernel.ptx + kernel_std.ptx)"

# Step 4: Pre-compile to cubin
echo ""
echo "=== Pre-compiling PTX to cubin (this takes ~10 minutes) ==="
echo "Running: $PTXAS --gpu-name sm_75 ..."
time "$PTXAS" --gpu-name sm_75 -o "$HOST_DIR/kernel_std.cubin" "$PTX_SRC" 2>&1
echo "Cubin: $(wc -c < "$HOST_DIR/kernel_std.cubin") bytes"
echo ""
echo "Done! kernel.ptx, kernel_std.ptx and kernel_std.cubin ready in $HOST_DIR/"
