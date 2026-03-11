#!/bin/bash
# Apply std patches for nvptx64 GPU compilation.
# Usage: ./std-patches/apply.sh [output_dir]
#
# This script copies the nightly rust-src library source to a local directory
# and applies the minimal patches needed for -Zbuild-std=std on nvptx64-nvidia-cuda.
#
# After running, set the env var when building:
#   __CARGO_TESTS_ONLY_SRC_ROOT="$OUTPUT_DIR" cargo +nightly build --release

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${1:-$(dirname "$SCRIPT_DIR")/patched-std}"

# Find the nightly sysroot
SYSROOT=$(rustc +nightly --print sysroot)
SRC_DIR="$SYSROOT/lib/rustlib/src/rust/library"

if [ ! -d "$SRC_DIR" ]; then
    echo "ERROR: rust-src component not found at $SRC_DIR"
    echo "Install it with: rustup +nightly component add rust-src"
    exit 1
fi

echo "Copying std library source from sysroot..."
mkdir -p "$OUTPUT_DIR"
cp -r "$SRC_DIR" "$OUTPUT_DIR/library"

echo "Applying patches..."

# Patch 1: sys/alloc/mod.rs — add nvptx64 to MIN_ALIGN + cuda cfg_select case
cd "$OUTPUT_DIR/library"
patch -p0 < "$SCRIPT_DIR/alloc_mod.patch"

# Patch 2: sys/thread_local/mod.rs — add cuda to no_threads + guard cases
patch -p0 < "$SCRIPT_DIR/thread_local_mod.patch"

# Patch 3: sys/random/mod.rs — add cuda to unsupported case
patch -p0 < "$SCRIPT_DIR/random_mod.patch"

# New file: sys/alloc/cuda.rs — bump allocator for GPU
cp "$SCRIPT_DIR/cuda.rs" "std/src/sys/alloc/cuda.rs"

echo ""
echo "Patched std source is at: $OUTPUT_DIR/library"
echo ""
echo "To build a crate with -Zbuild-std=std for nvptx64:"
echo "  __CARGO_TESTS_ONLY_SRC_ROOT=\"$OUTPUT_DIR/library\" cargo +nightly build --release"
