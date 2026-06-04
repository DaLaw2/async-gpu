#!/usr/bin/env bash
# async_gpu — One-command setup
#
# Usage:
#   ./scripts/setup.sh              # Default: --std mode
#   ./scripts/setup.sh --quick      # Stock nightly only, no patched std (~2 min)
#   ./scripts/setup.sh --std        # Nightly + patched std + kernel_std build (~15 min)
#   ./scripts/setup.sh --full       # Everything + patched compiler (2-4 hours)
#   ./scripts/setup.sh --check      # Verify current setup is working
#
# Modes:
#   --quick   Nightly toolchain + components. Core-only kernels work.
#   --std     (default) Also patches std in sysroot for File I/O, thread::spawn, println!
#   --full    Also builds patched compiler for #[warp_cooperative] MIR pass.
#   --check   Verify setup: environment + compile + run a trivial kernel.

set -euo pipefail

# ── Colors & helpers ─────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

ok()   { printf "  ${GREEN}✓${NC} %s\n" "$1"; }
fail() { printf "  ${RED}✗${NC} %s\n" "$1"; }
warn() { printf "  ${YELLOW}!${NC} %s\n" "$1"; }
info() { printf "  ${CYAN}→${NC} %s\n" "$1"; }
step() { printf "\n${BOLD}${BLUE}[%s]${NC} ${BOLD}%s${NC}\n" "$1" "$2"; }

die() {
    printf "\n${RED}ERROR:${NC} %s\n" "$1"
    [ -n "${2:-}" ] && printf "       %s\n" "$2"
    exit 1
}

# ── Paths ────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TOOLCHAIN_FILE="$REPO_DIR/rust-toolchain.toml"

# ── Parse mode ───────────────────────────────────────────────

MODE="std"  # default

for arg in "$@"; do
    case "$arg" in
        --quick) MODE="quick" ;;
        --std)   MODE="std" ;;
        --full)  MODE="full" ;;
        --check) MODE="check" ;;
        --help|-h)
            printf "${BOLD}async_gpu setup${NC}\n\n"
            echo "Usage: ./scripts/setup.sh [--quick | --std | --full | --check]"
            echo ""
            echo "Modes:"
            echo "  --quick   Stock nightly only. Core-only kernels. (~2 min)"
            echo "  --std     (default) + patched std for File I/O, threads, println! (~15 min)"
            echo "  --full    + patched compiler for #[warp_cooperative]. (2-4 hours)"
            echo "  --check   Verify current setup is working."
            echo ""
            echo "Examples:"
            echo "  ./scripts/setup.sh              # Default std mode"
            echo "  ./scripts/setup.sh --quick      # Fastest, core-only"
            echo "  ./scripts/setup.sh --check      # Just verify"
            exit 0
            ;;
        *)
            die "Unknown argument: $arg" "Run with --help for usage."
            ;;
    esac
done

# ── Banner ───────────────────────────────────────────────────

printf "\n${BOLD}"
echo "╔══════════════════════════════════════════════╗"
echo "║     async_gpu — Setup                        ║"
echo "╚══════════════════════════════════════════════╝"
printf "${NC}\n"
printf "  Mode: ${BOLD}${CYAN}%s${NC}\n" "$MODE"

# ── Read nightly version from rust-toolchain.toml ────────────

if [ ! -f "$TOOLCHAIN_FILE" ]; then
    die "rust-toolchain.toml not found at $TOOLCHAIN_FILE" \
        "Are you running from the repository root?"
fi

NIGHTLY=$(grep '^channel' "$TOOLCHAIN_FILE" | sed 's/.*= *"\(.*\)"/\1/')
if [ -z "$NIGHTLY" ]; then
    die "Could not parse nightly version from rust-toolchain.toml"
fi
info "Toolchain: $NIGHTLY (from rust-toolchain.toml)"

# ══════════════════════════════════════════════════════════════
# CHECK MODE — verify everything, modify nothing
# ══════════════════════════════════════════════════════════════

if [ "$MODE" = "check" ]; then
    ISSUES=0

    step "1/4" "Environment"

    # rustup
    if command -v rustup >/dev/null 2>&1; then
        ok "rustup $(rustup --version 2>/dev/null | head -1 | awk '{print $2}')"
    else
        fail "rustup not found — install from https://rustup.rs"
        ISSUES=$((ISSUES + 1))
    fi

    # nightly toolchain
    if rustup toolchain list 2>/dev/null | grep -q "$NIGHTLY"; then
        ok "Nightly toolchain: $NIGHTLY"
    else
        fail "Nightly toolchain $NIGHTLY not installed"
        ISSUES=$((ISSUES + 1))
    fi

    # components
    for comp in rust-src llvm-bitcode-linker; do
        if rustup component list --installed --toolchain "$NIGHTLY" 2>/dev/null | grep -q "$comp"; then
            ok "Component: $comp"
        else
            fail "Component: $comp not installed"
            ISSUES=$((ISSUES + 1))
        fi
    done

    # nvptx64 target
    if rustup target list --installed --toolchain "$NIGHTLY" 2>/dev/null | grep -q "nvptx64-nvidia-cuda"; then
        ok "Target: nvptx64-nvidia-cuda"
    else
        fail "Target: nvptx64-nvidia-cuda not installed"
        ISSUES=$((ISSUES + 1))
    fi

    # GPU
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
        ok "GPU: $GPU_NAME (driver $DRIVER)"
    else
        fail "nvidia-smi not found — no NVIDIA GPU driver"
        ISSUES=$((ISSUES + 1))
    fi

    # CUDA toolkit (ptxas)
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
    if [ -n "$PTXAS" ]; then
        ok "ptxas: $PTXAS"
    else
        warn "ptxas not found (needed for --std and --full modes)"
    fi

    step "2/4" "Patched std status"

    # patched-std directory
    if [ -d "$REPO_DIR/patched-std/src" ]; then
        ok "patched-std/ directory present"
    else
        warn "patched-std/ not found (run --std mode to create)"
    fi

    # sysroot patched?
    if command -v rustup >/dev/null 2>&1 && rustup toolchain list 2>/dev/null | grep -q "$NIGHTLY"; then
        SYSROOT=$(rustup run "$NIGHTLY" rustc --print sysroot 2>/dev/null || true)
        if [ -n "$SYSROOT" ] && [ -f "$SYSROOT/lib/rustlib/src/rust/library/std/src/sys/thread/cuda.rs" ]; then
            ok "Sysroot std is patched"
        else
            warn "Sysroot std is NOT patched (run --std mode to patch)"
        fi
    fi

    # kernel_std.ptx / kernel_std.cubin
    HOST_DIR="$REPO_DIR/crates/core/gpu-host"
    if [ -f "$HOST_DIR/kernel_std.ptx" ]; then
        ok "kernel_std.ptx present ($(wc -c < "$HOST_DIR/kernel_std.ptx") bytes)"
    else
        warn "kernel_std.ptx not found"
    fi
    if [ -f "$HOST_DIR/kernel_std.cubin" ]; then
        ok "kernel_std.cubin present ($(wc -c < "$HOST_DIR/kernel_std.cubin") bytes)"
    else
        warn "kernel_std.cubin not found"
    fi

    step "3/4" "Patched compiler status"

    if [ -d "$REPO_DIR/patched-rustc/build" ]; then
        PATCHED_RUSTC_BIN=""
        HOST_TRIPLE=$(rustc -vV 2>/dev/null | grep '^host:' | awk '{print $2}')
        for stage in stage1 stage2; do
            for build_dir in "$REPO_DIR/patched-rustc/build/$HOST_TRIPLE/$stage" "$REPO_DIR/patched-rustc/build/host/$stage"; do
                if [ -x "$build_dir/bin/rustc" ]; then
                    PATCHED_RUSTC_BIN="$build_dir/bin/rustc"
                    break 2
                fi
            done
        done
        if [ -n "$PATCHED_RUSTC_BIN" ]; then
            ok "Patched compiler: $PATCHED_RUSTC_BIN"
        else
            warn "patched-rustc/ exists but no rustc binary found"
        fi
    else
        info "No patched compiler (only needed for --full mode)"
    fi

    step "4/4" "Compile & run smoke test"

    if [ "$ISSUES" -gt 0 ]; then
        warn "Skipping smoke test — $ISSUES prerequisite issue(s) above"
    else
        # Try building the simplest PTX kernel
        info "Building gpu-kernel-std (PTX)..."
        KERNEL_DIR="$REPO_DIR/crates/kernel/gpu-kernel-std"
        if (cd "$KERNEL_DIR" && cargo +"$NIGHTLY" build --release) >/dev/null 2>&1; then
            ok "gpu-kernel-std PTX build succeeded"

            # Try running hello-gpu example
            info "Running hello-gpu example..."
            HELLO_HOST="$REPO_DIR/examples/hostcall/hello-gpu/host"
            if [ -d "$HELLO_HOST" ]; then
                OUTPUT=$(cd "$REPO_DIR" && cargo +stable run --manifest-path "$HELLO_HOST/Cargo.toml" 2>&1) || true
                if echo "$OUTPUT" | grep -q "PASSED"; then
                    ok "hello-gpu example ran successfully"
                else
                    warn "hello-gpu example did not produce expected output"
                    echo "$OUTPUT" | tail -5 | while read -r line; do
                        info "  $line"
                    done
                fi
            fi
        else
            fail "gpu-kernel-std PTX build failed"
            ISSUES=$((ISSUES + 1))
        fi
    fi

    # ── Summary ──────────────────────────────────────────────

    printf "\n${BOLD}────────────────────────────────────────────────${NC}\n"
    if [ "$ISSUES" -eq 0 ]; then
        printf "${GREEN}${BOLD}Setup looks good!${NC}\n"
    else
        printf "${RED}${BOLD}$ISSUES issue(s) found.${NC} Fix the items above.\n"
    fi
    printf "${BOLD}────────────────────────────────────────────────${NC}\n\n"
    exit "$( [ "$ISSUES" -eq 0 ] && echo 0 || echo 1 )"
fi

# ══════════════════════════════════════════════════════════════
# SETUP MODES: --quick, --std, --full
# ══════════════════════════════════════════════════════════════

TOTAL_STEPS=0
case "$MODE" in
    quick) TOTAL_STEPS=3 ;;
    std)   TOTAL_STEPS=6 ;;
    full)  TOTAL_STEPS=7 ;;
esac

CURRENT_STEP=0
next_step() {
    CURRENT_STEP=$((CURRENT_STEP + 1))
    step "$CURRENT_STEP/$TOTAL_STEPS" "$1"
}

# ── Step: Prerequisites ──────────────────────────────────────

next_step "Checking prerequisites"

# rustup
if ! command -v rustup >/dev/null 2>&1; then
    die "rustup is required but not found." \
        "Install from: https://rustup.rs"
fi
ok "rustup found"

# NVIDIA GPU
if ! command -v nvidia-smi >/dev/null 2>&1; then
    die "NVIDIA GPU driver not found (nvidia-smi missing)." \
        "Install the NVIDIA driver for your GPU first."
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
ok "GPU: $GPU_NAME"

# Compute capability check
SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
if [ -n "$SM" ] && [ "$SM" -lt 70 ] 2>/dev/null; then
    die "GPU compute capability $(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1) is below minimum." \
        "SM 7.0+ (Volta or newer) is required."
fi

# ptxas (needed for --std and --full)
if [ "$MODE" = "std" ] || [ "$MODE" = "full" ]; then
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
        die "ptxas not found (required for --$MODE mode)." \
            "Install the CUDA toolkit: https://developer.nvidia.com/cuda-downloads"
    fi
    ok "ptxas: $PTXAS"
fi

# Full mode extra prereqs
if [ "$MODE" = "full" ]; then
    for cmd in python3 cmake; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            die "$cmd is required for --full mode but not found."
        fi
        ok "$cmd found"
    done
    # ninja or make
    if command -v ninja >/dev/null 2>&1; then
        ok "ninja found"
    elif command -v make >/dev/null 2>&1; then
        ok "make found (ninja preferred but make works)"
    else
        die "ninja or make required for --full mode but neither found."
    fi
    # C compiler
    if command -v clang >/dev/null 2>&1; then
        ok "clang found"
    elif command -v gcc >/dev/null 2>&1; then
        ok "gcc found"
    else
        die "C compiler (clang or gcc) required for --full mode but not found."
    fi
fi

# ── Step: Install nightly toolchain ──────────────────────────

next_step "Installing nightly toolchain ($NIGHTLY)"

if rustup toolchain list 2>/dev/null | grep -q "$NIGHTLY"; then
    ok "Toolchain $NIGHTLY already installed"
else
    info "Installing $NIGHTLY..."
    rustup toolchain install "$NIGHTLY" --profile minimal
    ok "Toolchain installed"
fi

# Components
COMPONENTS="rust-src llvm-tools llvm-bitcode-linker rustfmt"
for comp in $COMPONENTS; do
    if rustup component list --installed --toolchain "$NIGHTLY" 2>/dev/null | grep -q "$comp"; then
        ok "Component $comp already installed"
    else
        info "Adding component $comp..."
        rustup component add "$comp" --toolchain "$NIGHTLY"
        ok "Component $comp installed"
    fi
done

# nvptx64 target
if rustup target list --installed --toolchain "$NIGHTLY" 2>/dev/null | grep -q "nvptx64-nvidia-cuda"; then
    ok "Target nvptx64-nvidia-cuda already installed"
else
    info "Adding target nvptx64-nvidia-cuda..."
    rustup target add nvptx64-nvidia-cuda --toolchain "$NIGHTLY"
    ok "Target installed"
fi

# ── Step: Build PTX kernel (smoke test) ──────────────────────

next_step "Building gpu-kernel-std PTX (smoke test)"

KERNEL_DIR="$REPO_DIR/crates/kernel/gpu-kernel-std"
info "Building gpu-kernel-std for nvptx64..."
if (cd "$KERNEL_DIR" && cargo +"$NIGHTLY" build --release) 2>&1 | tail -5; then
    ok "PTX kernel build succeeded"
else
    fail "PTX kernel build failed"
    die "gpu-kernel-std failed to build for nvptx64." \
        "Check that the nightly toolchain and components are correctly installed."
fi

# ── Quick mode done ──────────────────────────────────────────

if [ "$MODE" = "quick" ]; then
    printf "\n${BOLD}${GREEN}╔══════════════════════════════════════════════╗${NC}\n"
    printf "${BOLD}${GREEN}║  Quick setup complete!                        ║${NC}\n"
    printf "${BOLD}${GREEN}╚══════════════════════════════════════════════╝${NC}\n\n"
    echo "What works now:"
    ok "All core-only GPU kernels (no_std + core)"
    ok "All hostcall examples (hello-gpu, async-io, vector-math, ...)"
    ok "Host crates (gpu-host, gpu-protocol, async-gpu)"
    echo ""
    echo "For std::thread, File I/O, println! on GPU, run:"
    echo "  ./scripts/setup.sh --std"
    echo ""
    echo "Verify with:"
    echo "  ./scripts/setup.sh --check"
    exit 0
fi

# ══════════════════════════════════════════════════════════════
# --std and --full continue here
# ══════════════════════════════════════════════════════════════

# ── Step: Clone rustc source (for std patches) ───────────────

next_step "Preparing rustc source for std patches"

RUSTC_SRC="$REPO_DIR/rustc-src"
if [ -d "$RUSTC_SRC/library/std" ]; then
    ok "rustc-src/ already present"
else
    info "Cloning rustc source (depth 1, ~500MB)..."
    git clone --depth 1 https://github.com/rust-lang/rust.git "$RUSTC_SRC"
    ok "rustc source cloned"
fi

# ── Step: Apply std patches ──────────────────────────────────

next_step "Applying std patches"

PATCHED_STD="$REPO_DIR/patched-std"
if [ -f "$PATCHED_STD/src/sys/thread/cuda.rs" ]; then
    ok "patched-std/ already present and looks correct"
    info "To force re-patch: rm -rf patched-std/ && re-run"
else
    info "Running apply-std-patches.sh..."
    bash "$SCRIPT_DIR/apply-std-patches.sh"
    if [ -f "$PATCHED_STD/src/sys/thread/cuda.rs" ]; then
        ok "Std patches applied"
    else
        die "apply-std-patches.sh completed but patched-std/ looks incomplete."
    fi
fi

# ── Step: Build kernel_std PTX + cubin ───────────────────────

next_step "Building kernel_std (PTX + cubin)"

HOST_DIR="$REPO_DIR/crates/core/gpu-host"
if [ -f "$HOST_DIR/kernel_std.cubin" ] && [ -f "$HOST_DIR/kernel_std.ptx" ]; then
    ok "kernel_std.ptx and kernel_std.cubin already present"
    info "To force rebuild: rm crates/core/gpu-host/kernel_std.{ptx,cubin} && re-run"
else
    info "This may take 10-15 minutes (PTX build + ptxas compilation)..."
    bash "$SCRIPT_DIR/build-kernel-std.sh"
    if [ -f "$HOST_DIR/kernel_std.cubin" ]; then
        ok "kernel_std.ptx + cubin built successfully"
    else
        die "build-kernel-std.sh completed but kernel_std.cubin not found."
    fi
fi

# ── Std mode done ────────────────────────────────────────────

if [ "$MODE" = "std" ]; then
    printf "\n${BOLD}${GREEN}╔══════════════════════════════════════════════╗${NC}\n"
    printf "${BOLD}${GREEN}║  Std setup complete!                          ║${NC}\n"
    printf "${BOLD}${GREEN}╚══════════════════════════════════════════════╝${NC}\n\n"
    echo "What works now:"
    ok "Everything from --quick mode"
    ok "std::thread::spawn on GPU"
    ok "File I/O from GPU kernels"
    ok "println! from GPU"
    ok "All examples/std/* demos"
    echo ""
    echo "For patched compiler (#[warp_cooperative]), run:"
    echo "  ./scripts/setup.sh --full"
    echo ""
    echo "Verify with:"
    echo "  ./scripts/setup.sh --check"
    exit 0
fi

# ══════════════════════════════════════════════════════════════
# --full continues here
# ══════════════════════════════════════════════════════════════

# ── Step: Build patched compiler ─────────────────────────────

next_step "Building patched compiler (this takes 2-4 hours)"

PATCHED_RUSTC_DIR="$REPO_DIR/patched-rustc"
HOST_TRIPLE=$(rustc -vV 2>/dev/null | grep '^host:' | awk '{print $2}')
PATCHED_RUSTC_BIN=""

# Check if already built
for stage in stage1 stage2; do
    for build_dir in "$PATCHED_RUSTC_DIR/build/$HOST_TRIPLE/$stage" "$PATCHED_RUSTC_DIR/build/host/$stage"; do
        if [ -x "$build_dir/bin/rustc" ]; then
            PATCHED_RUSTC_BIN="$build_dir/bin/rustc"
            break 2
        fi
    done
done

if [ -n "$PATCHED_RUSTC_BIN" ]; then
    ok "Patched compiler already built: $PATCHED_RUSTC_BIN"
    info "To force rebuild: ./scripts/build-toolchain.sh --from-scratch"
else
    info "Building patched toolchain... (go get coffee)"
    bash "$SCRIPT_DIR/build-toolchain.sh"
    # Re-check
    for stage in stage1 stage2; do
        for build_dir in "$PATCHED_RUSTC_DIR/build/$HOST_TRIPLE/$stage" "$PATCHED_RUSTC_DIR/build/host/$stage"; do
            if [ -x "$build_dir/bin/rustc" ]; then
                PATCHED_RUSTC_BIN="$build_dir/bin/rustc"
                break 2
            fi
        done
    done
    if [ -n "$PATCHED_RUSTC_BIN" ]; then
        ok "Patched compiler built: $PATCHED_RUSTC_BIN"
    else
        die "build-toolchain.sh completed but no rustc binary found."
    fi
fi

printf "\n${BOLD}${GREEN}╔══════════════════════════════════════════════╗${NC}\n"
printf "${BOLD}${GREEN}║  Full setup complete!                         ║${NC}\n"
printf "${BOLD}${GREEN}╚══════════════════════════════════════════════╝${NC}\n\n"
echo "What works now:"
ok "Everything from --std mode"
ok "#[warp_cooperative] attribute"
ok "WarpCooperativeTransform MIR pass"
ok "All examples including warp-cooperative and async-pipeline"
echo ""
echo "To use the patched compiler:"
echo "  export RUSTC=\"$PATCHED_RUSTC_BIN\""
echo ""
echo "Verify with:"
echo "  ./scripts/setup.sh --check"
