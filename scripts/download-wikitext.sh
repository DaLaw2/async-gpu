#!/usr/bin/env bash
# Download WikiText-2 raw text (small subset for LoRA demo)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$REPO_ROOT/models/wikitext2"

mkdir -p "$DATA_DIR"

URL="https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"
OUT="$DATA_DIR/train.txt"

if [ -f "$OUT" ]; then
    echo "Already exists: $OUT"
else
    echo "Downloading WikiText-2 train.txt..."
    curl -L --progress-bar -o "$OUT" "$URL"
    echo "Done: $(wc -l < "$OUT") lines, $(du -h "$OUT" | cut -f1)"
fi
