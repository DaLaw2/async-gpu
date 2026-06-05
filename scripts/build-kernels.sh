#!/usr/bin/env bash
# Build all kernel crates to PTX and pre-compile to cubin in parallel.
#
# Builds the 4 kernel crates (core, compute, io, test) sequentially for PTX
# (so shared dependencies compile once), then runs ptxas in parallel to produce
# cubins for all crates simultaneously.
#
# Two build modes:
#   Default (dev):  opt-level 1, no LTO  — fast iteration (~2x faster)
#   --prod:         opt-level 3, fat LTO — maximum optimization for benchmarks
#
# Prerequisites:
#   - Patched std in sysroot (run apply-std-patches.sh first)
#   - CUDA toolkit (ptxas)
#
# Usage:
#   ./scripts/build-kernels.sh              # Build all 4 crates (dev mode)
#   ./scripts/build-kernels.sh --prod       # Build all 4 crates (production)
#   ./scripts/build-kernels.sh core test    # Build only specified crates (dev)
#   ./scripts/build-kernels.sh --prod core  # Build specified crates (production)
#
# For single-crate test iteration, see build-kernel-test.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
HOST_DIR="$REPO_DIR/crates/core/gpu-host"

# ── All kernel crates ───────────────────────────────────────────
ALL_CRATES=(core compute io test)

# ── Parse --prod flag ─────────────────────────────────────────────
CARGO_PROFILE="--release"
BUILD_MODE="dev"
ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--prod" ]; then
        CARGO_PROFILE="--profile release-prod"
        BUILD_MODE="prod"
    else
        ARGS+=("$arg")
    fi
done
set -- "${ARGS[@]+"${ARGS[@]}"}"

# ── Parse arguments: optional crate filter ──────────────────────
if [ $# -gt 0 ]; then
    CRATES=("$@")
    # Validate each argument
    for crate in "${CRATES[@]}"; do
        found=0
        for valid in "${ALL_CRATES[@]}"; do
            if [ "$crate" = "$valid" ]; then
                found=1
                break
            fi
        done
        if [ "$found" -eq 0 ]; then
            echo "ERROR: Unknown crate '$crate'. Valid: ${ALL_CRATES[*]}"
            exit 1
        fi
    done
else
    CRATES=("${ALL_CRATES[@]}")
fi

echo "==> Building kernel crates: ${CRATES[*]} (mode: $BUILD_MODE)"

# ── Read toolchain from rust-toolchain.toml ─────────────────────
TOOLCHAIN_FILE="$REPO_DIR/rust-toolchain.toml"
CHANNEL=$(grep '^channel' "$TOOLCHAIN_FILE" | sed 's/.*= *"\(.*\)"/\1/')
if [ -z "$CHANNEL" ]; then
    echo "ERROR: Could not parse nightly channel from $TOOLCHAIN_FILE"
    exit 1
fi
echo "Using toolchain: +$CHANNEL"

# ── Find ptxas ──────────────────────────────────────────────────
PTXAS=""
for dir in /usr/local/cuda*/bin /opt/cuda/bin; do
    if [ -x "$dir/ptxas" ] 2>/dev/null; then
        PTXAS="$dir/ptxas"
        break
    fi
done
if command -v ptxas >/dev/null 2>&1; then
    PTXAS="$(command -v ptxas)"
fi
if [ -z "$PTXAS" ]; then
    echo "ERROR: ptxas not found. Install CUDA toolkit."
    exit 1
fi
echo "Using ptxas: $PTXAS"

# ── Detect SM architecture ──────────────────────────────────────
SM="sm_75"  # default (Turing)
if command -v nvidia-smi >/dev/null 2>&1; then
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
    if [ -n "$CC" ] && [ "$CC" -ge 70 ] 2>/dev/null; then
        SM="sm_$CC"
    fi
fi
echo "Target architecture: $SM"

# ── Determine PTX output directory based on profile ────────────
if [ "$BUILD_MODE" = "prod" ]; then
    PTX_PROFILE_DIR="release-prod"
else
    PTX_PROFILE_DIR="release"
fi

# ── Step 1: Build PTX for each crate (sequential — shared deps compile once)
echo ""
echo "=== Step 1: Building PTX for ${#CRATES[@]} crate(s) ==="
for crate in "${CRATES[@]}"; do
    crate_dir="$REPO_DIR/crates/kernel/gpu-kernel-$crate"
    echo ""
    echo "--- Building gpu-kernel-$crate ---"
    if [ ! -d "$crate_dir" ]; then
        echo "ERROR: Crate directory not found: $crate_dir"
        exit 1
    fi
    # shellcheck disable=SC2086
    (cd "$crate_dir" && cargo "+$CHANNEL" build $CARGO_PROFILE 2>&1 | grep -E "Compiling|Finished|error|warning.*gpu-kernel")

    ptx_src="$crate_dir/target/nvptx64-nvidia-cuda/$PTX_PROFILE_DIR/gpu_kernel_${crate}.ptx"
    if [ ! -f "$ptx_src" ]; then
        echo "ERROR: PTX not generated at $ptx_src"
        exit 1
    fi
    echo "PTX ready: $(du -h "$ptx_src" | cut -f1)"
done

# ── Step 2: Copy PTX files to gpu-host directory ────────────────
echo ""
echo "=== Step 2: Copying PTX files to gpu-host ==="
for crate in "${CRATES[@]}"; do
    ptx_src="$REPO_DIR/crates/kernel/gpu-kernel-$crate/target/nvptx64-nvidia-cuda/$PTX_PROFILE_DIR/gpu_kernel_${crate}.ptx"
    ptx_dst="$HOST_DIR/kernel_${crate}.ptx"
    cp "$ptx_src" "$ptx_dst"
    echo "  kernel_${crate}.ptx ($(du -h "$ptx_dst" | cut -f1))"
done

# Backward-compat aliases: kernel_test.ptx → kernel.ptx + kernel_std.ptx
for crate in "${CRATES[@]}"; do
    if [ "$crate" = "test" ]; then
        cp "$HOST_DIR/kernel_test.ptx" "$HOST_DIR/kernel.ptx"
        cp "$HOST_DIR/kernel_test.ptx" "$HOST_DIR/kernel_std.ptx"
        echo "  kernel.ptx + kernel_std.ptx (backward-compat copies of kernel_test.ptx)"
    fi
done

# ── Step 3: Run ptxas in PARALLEL for all crates ───────────────
echo ""
echo "=== Step 3: Compiling cubins in parallel (ptxas $SM) ==="
pids=()
crate_for_pid=()
for crate in "${CRATES[@]}"; do
    ptx_file="$HOST_DIR/kernel_${crate}.ptx"
    cubin_file="$HOST_DIR/kernel_${crate}.cubin"
    "$PTXAS" --gpu-name "$SM" -o "$cubin_file" "$ptx_file" &
    pids+=($!)
    crate_for_pid+=("$crate")
    echo "  Started ptxas for kernel_${crate} (PID $!)"
done

echo ""
echo "Waiting for ${#pids[@]} ptxas process(es)..."

failed=0
for i in "${!pids[@]}"; do
    pid="${pids[$i]}"
    crate="${crate_for_pid[$i]}"
    if wait "$pid"; then
        cubin_file="$HOST_DIR/kernel_${crate}.cubin"
        echo "  OK: kernel_${crate}.cubin ($(du -h "$cubin_file" | cut -f1))"
    else
        echo "  FAILED: kernel_${crate}.cubin (ptxas exit code $?)"
        failed=$((failed + 1))
    fi
done

if [ "$failed" -gt 0 ]; then
    echo ""
    echo "ERROR: $failed ptxas build(s) failed"
    exit 1
fi

# Backward-compat: kernel_test.cubin → kernel_std.cubin
for crate in "${CRATES[@]}"; do
    if [ "$crate" = "test" ]; then
        cp "$HOST_DIR/kernel_test.cubin" "$HOST_DIR/kernel_std.cubin"
        echo "  kernel_std.cubin (backward-compat copy of kernel_test.cubin)"
    fi
done

echo ""
echo "=== All ${#CRATES[@]} kernel(s) built successfully ==="
echo ""
echo "PTX files:"
for crate in "${CRATES[@]}"; do
    printf "  %-24s %s\n" "kernel_${crate}.ptx" "$(du -h "$HOST_DIR/kernel_${crate}.ptx" | cut -f1)"
done
echo ""
echo "Cubin files:"
for crate in "${CRATES[@]}"; do
    printf "  %-24s %s\n" "kernel_${crate}.cubin" "$(du -h "$HOST_DIR/kernel_${crate}.cubin" | cut -f1)"
done
