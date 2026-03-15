#!/bin/bash
# Download and install the pre-built async_gpu patched Rust toolchain.
#
# Usage:
#   ./scripts/install-toolchain.sh              # install from repo's rust-toolchain.toml
#   ./scripts/install-toolchain.sh nightly-2026-03-11  # install specific version
#   bash <(curl -sL https://raw.githubusercontent.com/DaLaw2/async-gpu/main/scripts/install-toolchain.sh)
#
# The toolchain is installed to ~/.rustup/toolchains/async-gpu/
# and can be used with: cargo +async-gpu build ...

set -euo pipefail

REPO="DaLaw2/async-gpu"
TOOLCHAIN_NAME="async-gpu"
HOST_TRIPLE=$(rustc -vV 2>/dev/null | grep '^host:' | awk '{print $2}')

if [ -z "$HOST_TRIPLE" ]; then
    echo "ERROR: rustc not found. Please install Rust first: https://rustup.rs"
    exit 1
fi

# Only Linux x86_64 pre-built toolchains are available for now
if [ "$HOST_TRIPLE" != "x86_64-unknown-linux-gnu" ]; then
    echo "ERROR: Pre-built toolchain is only available for x86_64-unknown-linux-gnu"
    echo "  Your host triple: $HOST_TRIPLE"
    echo ""
    echo "For other platforms, build from source:"
    echo "  ./scripts/build-toolchain.sh    # Linux/macOS"
    echo "  scripts\\build-toolchain.bat     # Windows"
    exit 1
fi

# ── Determine nightly version ──────────────────────────────────
NIGHTLY_CHANNEL="${1:-}"

if [ -z "$NIGHTLY_CHANNEL" ]; then
    # Try to read from rust-toolchain.toml in current directory or script directory
    for toml_path in "./rust-toolchain.toml" "$(dirname "$0")/../rust-toolchain.toml"; do
        if [ -f "$toml_path" ]; then
            NIGHTLY_CHANNEL=$(grep '^channel' "$toml_path" | sed 's/.*= *"\(.*\)"/\1/')
            echo "Read nightly channel from $toml_path: $NIGHTLY_CHANNEL"
            break
        fi
    done
fi

if [ -z "$NIGHTLY_CHANNEL" ]; then
    echo "ERROR: Could not determine nightly version."
    echo "  Either run from the async_gpu repo directory, or pass version explicitly:"
    echo "    $0 nightly-2026-03-11"
    exit 1
fi

echo "╔══════════════════════════════════════════════╗"
echo "║  async_gpu — Toolchain Installer             ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "  Nightly:  $NIGHTLY_CHANNEL"
echo "  Host:     $HOST_TRIPLE"
echo ""

# ── Paths ──────────────────────────────────────────────────────
RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}"
INSTALL_DIR="$RUSTUP_HOME/toolchains/$TOOLCHAIN_NAME"
ARCHIVE_NAME="async-gpu-toolchain-${HOST_TRIPLE}-${NIGHTLY_CHANNEL}.tar.gz"
RELEASE_TAG="toolchain-${NIGHTLY_CHANNEL}"
DOWNLOAD_URL="https://github.com/${REPO}/releases/download/${RELEASE_TAG}/${ARCHIVE_NAME}"
TMP_DIR=$(mktemp -d)

cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

# ── Check if already installed ─────────────────────────────────
if [ -d "$INSTALL_DIR" ] && [ -x "$INSTALL_DIR/bin/rustc" ]; then
    INSTALLED_VERSION=$("$INSTALL_DIR/bin/rustc" --version 2>/dev/null || echo "unknown")
    echo "Existing installation found: $INSTALLED_VERSION"
    echo ""
    read -rp "Reinstall? [y/N] " REPLY
    if [[ ! "$REPLY" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
    echo ""
fi

# ── Download ───────────────────────────────────────────────────
echo "=== Downloading toolchain ==="
echo "  URL: $DOWNLOAD_URL"
echo ""

if command -v curl >/dev/null 2>&1; then
    curl -fSL --progress-bar -o "$TMP_DIR/$ARCHIVE_NAME" "$DOWNLOAD_URL"
elif command -v wget >/dev/null 2>&1; then
    wget -q --show-progress -O "$TMP_DIR/$ARCHIVE_NAME" "$DOWNLOAD_URL"
else
    echo "ERROR: Neither curl nor wget found. Please install one."
    exit 1
fi

echo ""

# ── Extract ────────────────────────────────────────────────────
echo "=== Extracting to $INSTALL_DIR ==="

# Remove old installation if present
if [ -d "$INSTALL_DIR" ]; then
    rm -rf "$INSTALL_DIR"
fi

mkdir -p "$INSTALL_DIR"

# The archive contains a stage directory (e.g., stage2/).
# Extract and move contents directly into the toolchain directory.
tar -xzf "$TMP_DIR/$ARCHIVE_NAME" -C "$TMP_DIR"

# Find the extracted stage directory
EXTRACTED=$(find "$TMP_DIR" -maxdepth 1 -type d -name "stage*" | head -1)
if [ -z "$EXTRACTED" ]; then
    echo "ERROR: Archive did not contain expected stage directory."
    echo "  Contents of archive:"
    tar -tzf "$TMP_DIR/$ARCHIVE_NAME" | head -20
    exit 1
fi

# Move all contents into the install directory
cp -a "$EXTRACTED/." "$INSTALL_DIR/"

echo ""

# ── Verify ─────────────────────────────────────────────────────
echo "=== Verifying installation ==="

if [ ! -x "$INSTALL_DIR/bin/rustc" ]; then
    echo "ERROR: rustc not found at $INSTALL_DIR/bin/rustc"
    exit 1
fi

RUSTC_VERSION=$("$INSTALL_DIR/bin/rustc" --version)
echo "  rustc: $RUSTC_VERSION"

# Check for nvptx64 target support
NVPTX_LIB="$INSTALL_DIR/lib/rustlib/nvptx64-nvidia-cuda/lib"
if [ -d "$NVPTX_LIB" ]; then
    echo "  nvptx64-nvidia-cuda: PRESENT"
else
    echo "  WARNING: nvptx64-nvidia-cuda libs not found"
fi

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Installation complete!                      ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "Usage:"
echo "  cargo +$TOOLCHAIN_NAME build --target nvptx64-nvidia-cuda"
echo ""
echo "Or set as default for this project:"
echo "  rustup override set $TOOLCHAIN_NAME"
echo ""
echo "Toolchain path: $INSTALL_DIR"
