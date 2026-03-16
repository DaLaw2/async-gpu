#!/usr/bin/env bash
# Download MNIST dataset to models/mnist/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MNIST_DIR="$REPO_ROOT/models/mnist"

mkdir -p "$MNIST_DIR"

BASE_URL="https://storage.googleapis.com/cvdf-datasets/mnist"

for FILE in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz; do
    OUT="$MNIST_DIR/$FILE"
    UNPACKED="${OUT%.gz}"
    if [ -f "$UNPACKED" ]; then
        echo "Already exists: $UNPACKED"
        continue
    fi
    echo "Downloading $FILE..."
    curl -L --progress-bar -o "$OUT" "$BASE_URL/$FILE"
    gunzip -f "$OUT"
    echo "  → $UNPACKED ($(du -h "$UNPACKED" | cut -f1))"
done

echo ""
echo "MNIST dataset ready at $MNIST_DIR"
ls -lh "$MNIST_DIR"
