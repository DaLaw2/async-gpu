#!/usr/bin/env bash
# Pre-push check: regenerate patches (if needed) + run local CI lint.
# NOT tracked by git — listed in .gitignore.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

cd "$REPO_DIR"

# Step 1: Regenerate std patches if patched-std/ exists
if [ -d "patched-std/library/std" ]; then
    echo "=== Regenerating std patches ==="
    bash scripts/gen-std-patches.sh
    echo ""
fi

# Step 2: Regenerate rustc patches if both rustc-src/ and patched-rustc/ exist
if [ -d "rustc-src/compiler" ] && [ -d "patched-rustc/compiler" ]; then
    echo "=== Regenerating rustc patches ==="
    bash scripts/gen-rustc-patches.sh
    echo ""
fi

# Step 3: Run local CI lint
echo "=== Running CI lint ==="
bash scripts/ci-lint.sh
