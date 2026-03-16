#!/usr/bin/env bash
# Download CIFAR-10 binary format to models/cifar10/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CIFAR_DIR="$REPO_ROOT/models/cifar10"

mkdir -p "$CIFAR_DIR"

URL="https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
TARBALL="$CIFAR_DIR/cifar-10-binary.tar.gz"

if [ -f "$CIFAR_DIR/data_batch_1.bin" ]; then
    echo "CIFAR-10 already exists at $CIFAR_DIR"
else
    echo "Downloading CIFAR-10 (~170MB)..."
    curl -L --progress-bar -o "$TARBALL" "$URL"
    echo "Extracting..."
    tar xzf "$TARBALL" -C "$CIFAR_DIR" --strip-components=1
    rm -f "$TARBALL"
    echo "Done."
fi

echo ""
echo "CIFAR-10 dataset at $CIFAR_DIR:"
ls -lh "$CIFAR_DIR"/*.bin 2>/dev/null | head -10
