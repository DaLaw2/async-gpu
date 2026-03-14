#!/usr/bin/env bash
# Local CI lint check — mirrors .github/workflows/build.yml lint job.
# Run this before pushing to catch issues early.
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

FAIL=0

run() {
    echo "==> $1"
    if eval "$2" > /dev/null 2>&1; then
        echo -e "    ${GREEN}OK${NC}"
    else
        echo -e "    ${RED}FAIL${NC}"
        eval "$2" 2>&1 | tail -20
        FAIL=1
    fi
}

# PTX stubs (auto-discovered from gpu-host include_str! calls)
PTX_STUBS=$(grep -roh 'include_str!("\.\.\/[^"]*\.ptx")' crates/gpu-host/src/ | sed 's/include_str!("\.\.\/\(.*\)")/\1/' | sort -u)
for f in $PTX_STUBS; do
    [ -f "crates/gpu-host/$f" ] || echo "// stub" > "crates/gpu-host/$f"
done

# --- fmt checks ---
CRATES_FMT="gpu-host gpu-protocol warp-macro gpu-atomics gpu-runtime gpu-libc"
for c in $CRATES_FMT; do
    run "fmt $c" "cargo +stable fmt --manifest-path crates/$c/Cargo.toml -- --check"
done

# --- clippy checks (stable-compatible crates only) ---
CRATES_CLIPPY="gpu-host gpu-protocol warp-macro gpu-atomics gpu-runtime"
for c in $CRATES_CLIPPY; do
    run "clippy $c" "cargo +stable clippy --manifest-path crates/$c/Cargo.toml -- -D warnings"
done

# --- doc-tests ---
run "doc-tests gpu-protocol" "cargo +stable test --manifest-path crates/gpu-protocol/Cargo.toml --doc"

# --- cargo doc with -D missing_docs ---
run "doc gpu-host" "RUSTDOCFLAGS='-D missing_docs' cargo +stable doc --manifest-path crates/gpu-host/Cargo.toml --lib --no-default-features --no-deps"
run "doc gpu-protocol" "RUSTDOCFLAGS='-D missing_docs' cargo +stable doc --manifest-path crates/gpu-protocol/Cargo.toml --no-deps"
run "doc warp-macro" "RUSTDOCFLAGS='-D missing_docs' cargo +stable doc --manifest-path crates/warp-macro/Cargo.toml --no-deps"

# --- cargo doc (no missing_docs) ---
run "doc gpu-atomics" "cargo +stable doc --manifest-path crates/gpu-atomics/Cargo.toml --no-deps"
run "doc gpu-runtime" "cargo +stable doc --manifest-path crates/gpu-runtime/Cargo.toml --no-deps"

# --- host check ---
run "check gpu-host" "cargo +stable check --manifest-path crates/gpu-host/Cargo.toml"

# --- PTX kernel builds (nightly + nvptx64) ---
# Must cd into each dir so .cargo/config.toml (target, build-std) is picked up.
# Read nightly version from rust-toolchain.toml (single source of truth)
NIGHTLY=$(grep '^channel' rust-toolchain.toml | sed 's/.*= *"\(.*\)"/\1/')
PTX_KERNELS="crates/gpu-kernel crates/async-hostcall-test crates/async-pipeline-test crates/embassy-test crates/multi-warp-test crates/gpu-std-test crates/gpu-kernel-std examples/hello-gpu/kernel examples/async-io/kernel examples/vector-math/kernel"
# Note: crates/std-build-test excluded — requires patched std (apply-patches.sh) to build
for k in $PTX_KERNELS; do
    name=$(basename "$k")
    run "ptx $name" "(cd $k && cargo +$NIGHTLY build --release)"
done

# --- example hosts ---
run "check hello-gpu-host" "cargo +stable check --manifest-path examples/hello-gpu/host/Cargo.toml"
run "check async-io-host" "cargo +stable check --manifest-path examples/async-io/host/Cargo.toml"
run "check vector-math-host" "cargo +stable check --manifest-path examples/vector-math/host/Cargo.toml"

echo ""
if [ "$FAIL" -eq 0 ]; then
    echo -e "${GREEN}All CI lint checks passed!${NC}"
else
    echo -e "${RED}Some checks failed — fix before pushing.${NC}"
    exit 1
fi
