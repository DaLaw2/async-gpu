#!/usr/bin/env bash
# Compile-fail test runner for tiered_mem borrow safety.
#
# Verifies that cross-scope memory access violations are rejected at compile time.
# Run from the repository root: bash crates/core/gpu-runtime/tests/compile_fail_runner.sh
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'
FAIL=0
PASS=0

TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

# Create a temporary crate that depends on gpu-runtime
mkdir -p "$TMPDIR/src"
cat > "$TMPDIR/Cargo.toml" <<'TOML'
[workspace]

[package]
name = "compile-fail-tests"
version = "0.0.0"
edition = "2021"

[dependencies]
gpu-runtime = { path = "GPU_RUNTIME_PATH" }
TOML

# Resolve absolute path to gpu-runtime
GPU_RUNTIME_ABS="$(cd crates/core/gpu-runtime && pwd)"
sed -i "s|GPU_RUNTIME_PATH|$GPU_RUNTIME_ABS|" "$TMPDIR/Cargo.toml"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEST_DIR="$SCRIPT_DIR/compile_fail"

# expect_fail: test should NOT compile
expect_fail() {
    local name="$1"
    local file="$2"
    cp "$file" "$TMPDIR/src/main.rs"
    if cargo +stable check --manifest-path "$TMPDIR/Cargo.toml" 2>/dev/null; then
        echo -e "  ${RED}FAIL${NC} $name (expected compile error, but it compiled)"
        FAIL=$((FAIL + 1))
    else
        echo -e "  ${GREEN}OK${NC}   $name (correctly rejected)"
        PASS=$((PASS + 1))
    fi
}

# expect_pass: test SHOULD compile
expect_pass() {
    local name="$1"
    local file="$2"
    cp "$file" "$TMPDIR/src/main.rs"
    if cargo +stable check --manifest-path "$TMPDIR/Cargo.toml" 2>/dev/null; then
        echo -e "  ${GREEN}OK${NC}   $name (compiles as expected)"
        PASS=$((PASS + 1))
    else
        echo -e "  ${RED}FAIL${NC} $name (expected to compile, but got errors)"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== Compile-fail tests: tiered_mem borrow safety ==="
echo ""
echo "--- Tests that MUST fail to compile ---"

expect_fail "shared_ref_escape_scope"        "$TEST_DIR/shared_ref_escape_scope.rs"
expect_fail "shared_ref_not_send"            "$TEST_DIR/shared_ref_not_send.rs"
expect_fail "shared_ref_not_sync"            "$TEST_DIR/shared_ref_not_sync.rs"
expect_fail "shared_ref_in_global_container" "$TEST_DIR/shared_ref_in_global_container.rs"
expect_fail "shared_ref_use_after_scope"     "$TEST_DIR/shared_ref_use_after_scope.rs"
expect_fail "shared_ref_return_from_scope"   "$TEST_DIR/shared_ref_return_from_scope.rs"

echo ""
echo "--- Tests that MUST compile ---"

expect_pass "valid_shared_ref_within_scope"  "$TEST_DIR/valid_shared_ref_within_scope.rs"
expect_pass "valid_global_ref_is_send"       "$TEST_DIR/valid_global_ref_is_send.rs"

echo ""
if [ "$FAIL" -eq 0 ]; then
    echo -e "${GREEN}All $PASS/$((PASS + FAIL)) compile-fail tests passed!${NC}"
else
    echo -e "${RED}$FAIL tests failed, $PASS passed.${NC}"
    exit 1
fi
