#!/bin/bash
# async_gpu — environment check
#
# Usage: bash scripts/env-check.sh
#
# Checks all prerequisites for building and running async_gpu examples.
# Does NOT modify system state — only reports what is present/missing.

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

ok()   { printf "${GREEN}[OK]${NC} %s\n" "$1"; }
warn() { printf "${YELLOW}[WARN]${NC} %s\n" "$1"; }
fail() { printf "${RED}[MISSING]${NC} %s\n" "$1"; }

echo "====================================="
echo "  async_gpu — Environment Check"
echo "====================================="
echo ""

ISSUES=0

# ── 1. Rust toolchain ─────────────────────────────────────
echo "--- Rust Toolchain ---"

if command -v rustup >/dev/null 2>&1; then
    ok "rustup $(rustup --version 2>/dev/null | head -1 | awk '{print $2}')"
else
    fail "rustup not found — install from https://rustup.rs"
    ISSUES=$((ISSUES + 1))
fi

if command -v rustc >/dev/null 2>&1; then
    ok "rustc $(rustc --version | awk '{print $2}')"
else
    fail "rustc not found"
    ISSUES=$((ISSUES + 1))
fi

# Check nightly toolchain from rust-toolchain.toml
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NIGHTLY=""
if [ -f "$REPO_ROOT/rust-toolchain.toml" ]; then
    NIGHTLY=$(grep '^channel' "$REPO_ROOT/rust-toolchain.toml" | sed 's/.*= *"\(.*\)"/\1/')
fi

if [ -n "$NIGHTLY" ]; then
    if rustup toolchain list 2>/dev/null | grep -q "$NIGHTLY"; then
        ok "Nightly toolchain: $NIGHTLY"
    else
        fail "Nightly toolchain $NIGHTLY not installed"
        echo "      Run: rustup toolchain install $NIGHTLY"
        ISSUES=$((ISSUES + 1))
    fi
fi

# Check nvptx64 target
if rustup target list --installed 2>/dev/null | grep -q "nvptx64-nvidia-cuda"; then
    ok "nvptx64-nvidia-cuda target installed"
else
    fail "nvptx64-nvidia-cuda target not installed"
    echo "      Run: rustup target add nvptx64-nvidia-cuda --toolchain $NIGHTLY"
    ISSUES=$((ISSUES + 1))
fi

# Check rust-src component
if rustup component list --installed 2>/dev/null | grep -q "rust-src"; then
    ok "rust-src component installed"
else
    fail "rust-src component not installed (needed for -Zbuild-std)"
    echo "      Run: rustup component add rust-src --toolchain $NIGHTLY"
    ISSUES=$((ISSUES + 1))
fi

# Check llvm-bitcode-linker
if rustup component list --installed 2>/dev/null | grep -q "llvm-bitcode-linker"; then
    ok "llvm-bitcode-linker component installed"
elif command -v llvm-bitcode-linker >/dev/null 2>&1; then
    ok "llvm-bitcode-linker found in PATH"
else
    fail "llvm-bitcode-linker not installed (needed for nvptx64 linking)"
    echo "      Run: rustup component add llvm-bitcode-linker --toolchain $NIGHTLY"
    ISSUES=$((ISSUES + 1))
fi

echo ""

# ── 2. CUDA ───────────────────────────────────────────────
echo "--- CUDA ---"

if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    ok "GPU: $GPU_NAME (driver $DRIVER)"

    # Check compute capability
    SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
    if [ -n "$SM" ] && [ "$SM" -ge 70 ] 2>/dev/null; then
        ok "Compute capability: $(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1) (SM 70+ required)"
    elif [ -n "$SM" ]; then
        fail "Compute capability $SM is below minimum (SM 70+ required)"
        ISSUES=$((ISSUES + 1))
    fi
else
    fail "nvidia-smi not found — NVIDIA GPU driver not installed"
    ISSUES=$((ISSUES + 1))
fi

if command -v nvcc >/dev/null 2>&1; then
    CUDA_VER=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    ok "CUDA toolkit: $CUDA_VER"
else
    warn "nvcc not found — CUDA toolkit may not be in PATH (runtime driver is usually sufficient)"
fi

echo ""

# ── 3. Build tools ────────────────────────────────────────
echo "--- Build Tools ---"

if command -v cargo >/dev/null 2>&1; then
    ok "cargo $(cargo --version | awk '{print $2}')"
else
    fail "cargo not found"
    ISSUES=$((ISSUES + 1))
fi

if command -v git >/dev/null 2>&1; then
    ok "git $(git --version | awk '{print $3}')"
else
    fail "git not found"
    ISSUES=$((ISSUES + 1))
fi

echo ""

# ── Summary ───────────────────────────────────────────────
echo "====================================="
if [ "$ISSUES" -eq 0 ]; then
    printf "${GREEN}All prerequisites met!${NC}\n"
    echo ""
    echo "Quick start:"
    echo "  cargo run --manifest-path examples/hostcall/hello-gpu/host/Cargo.toml"
    echo ""
    echo "Or use xtask to build all GPU kernels:"
    echo "  cargo xtask gpu-build"
else
    printf "${RED}$ISSUES issue(s) found.${NC} Fix the items above, then re-run this script.\n"
fi
echo "====================================="
