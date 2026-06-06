#!/usr/bin/env bash
# Per-kernel GPU resource analysis via ptxas -v.
#
# Parses ptxas -v output for each PTX file to extract per-kernel:
#   - Physical register count
#   - Spill stores/loads (bytes)
#   - Stack frame size
#   - Shared memory usage
#   - Constant memory usage
#
# Calculates sm_75 occupancy and emits warnings:
#   - CRITICAL: occupancy < 25%
#   - WARN:     occupancy < 50%
#   - WARN:     register spills detected
#   - INFO:     occupancy >= 50% (healthy)
#
# Usage:
#   ./scripts/kernel-resources.sh                    # Analyze all 4 kernel PTX files
#   ./scripts/kernel-resources.sh kernel_core.ptx    # Analyze specific PTX file(s)
#   ./scripts/kernel-resources.sh --json             # Output JSON instead of table
#
# Requires: ptxas (CUDA toolkit)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
HOST_DIR="$REPO_DIR/crates/core/gpu-host"

# ── sm_75 hardware limits ─────────────────────────────────────────
SM_REGS_PER_SM=65536
SM_MAX_THREADS=1024
SM_MAX_BLOCKS=16
SM_SMEM_PER_SM=49152
SM_WARP_SIZE=32
SM_NAME="sm_75"

# ── Colors ─────────────────────────────────────────────────────────
RED='\033[0;31m'
YELLOW='\033[0;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# ── Parse arguments ───────────────────────────────────────────────
OUTPUT_JSON=0
PTX_FILES=()
for arg in "$@"; do
    if [ "$arg" = "--json" ]; then
        OUTPUT_JSON=1
    else
        PTX_FILES+=("$arg")
    fi
done

# Default: analyze all kernel PTX files
if [ ${#PTX_FILES[@]} -eq 0 ]; then
    for f in "$HOST_DIR"/kernel_core.ptx "$HOST_DIR"/kernel_compute.ptx "$HOST_DIR"/kernel_io.ptx "$HOST_DIR"/kernel_test.ptx; do
        if [ -f "$f" ]; then
            PTX_FILES+=("$f")
        fi
    done
fi

if [ ${#PTX_FILES[@]} -eq 0 ]; then
    echo "ERROR: No PTX files found. Run build-kernels.sh first."
    exit 1
fi

# ── Find ptxas ─────────────────────────────────────────────────────
PTXAS=""
for dir in /usr/local/cuda*/bin /opt/cuda/bin; do
    if [ -x "$dir/ptxas" ] 2>/dev/null; then
        PTXAS="$dir/ptxas"
        break
    fi
done
if [ -z "$PTXAS" ] && command -v ptxas >/dev/null 2>&1; then
    PTXAS="$(command -v ptxas)"
fi
if [ -z "$PTXAS" ]; then
    echo "ERROR: ptxas not found. Install CUDA toolkit."
    exit 1
fi

# ── Occupancy calculation for sm_75 ───────────────────────────────
# Given registers per thread and block size, compute occupancy.
# Assumes 256 threads/block (8 warps) as default.
calculate_occupancy() {
    local regs_per_thread=$1
    local block_size=${2:-256}

    # Register granularity: allocated in units of 256 regs per warp on sm_75
    # Each warp needs regs_per_thread * 32 registers, rounded up to 256
    local regs_per_warp=$(( regs_per_thread * SM_WARP_SIZE ))
    # Round up to next multiple of 256
    local regs_per_warp_alloc=$(( ((regs_per_warp + 255) / 256) * 256 ))

    local warps_per_block=$(( (block_size + SM_WARP_SIZE - 1) / SM_WARP_SIZE ))

    # Max warps limited by registers
    local max_warps_by_regs=0
    if [ "$regs_per_warp_alloc" -gt 0 ]; then
        max_warps_by_regs=$(( SM_REGS_PER_SM / regs_per_warp_alloc ))
    fi

    # Max blocks limited by registers
    local max_blocks_by_regs=0
    if [ "$warps_per_block" -gt 0 ] && [ "$max_warps_by_regs" -gt 0 ]; then
        max_blocks_by_regs=$(( max_warps_by_regs / warps_per_block ))
    fi

    # Clamp to hardware max blocks per SM
    if [ "$max_blocks_by_regs" -gt "$SM_MAX_BLOCKS" ]; then
        max_blocks_by_regs=$SM_MAX_BLOCKS
    fi

    # Active threads
    local active_threads=$(( max_blocks_by_regs * block_size ))
    if [ "$active_threads" -gt "$SM_MAX_THREADS" ]; then
        active_threads=$SM_MAX_THREADS
    fi

    # Occupancy as percentage (integer)
    echo $(( active_threads * 100 / SM_MAX_THREADS ))
}

# ── Analyze one PTX file ──────────────────────────────────────────
# Outputs one line per kernel: name|regs|spill_stores|spill_loads|stack|cumul_stack|cmem|occupancy
analyze_ptx() {
    local ptx_file="$1"
    local raw_file parsed_file
    raw_file=$(mktemp)
    parsed_file=$(mktemp)

    # Run ptxas -v; output goes to stderr
    "$PTXAS" -v --gpu-name "$SM_NAME" -o /dev/null "$ptx_file" 2>"$raw_file" || {
        echo "ERROR: ptxas failed on $ptx_file" >&2
        rm -f "$raw_file" "$parsed_file"
        return 1
    }

    # Two-pass approach:
    # Pass 1: Extract entry function names with their "Used N registers" and spill info
    # Pass 2: Calculate occupancy
    #
    # ptxas -v output structure for each entry function:
    #   ptxas info    : Compiling entry function 'NAME' for 'sm_75'
    #   ptxas info    : Function properties for NAME
    #       N bytes stack frame, N bytes spill stores, N bytes spill loads
    #   ptxas info    : Used N registers, ...
    #   [optional device function properties...]
    #
    # For entry functions with device function dependencies, the entry function's
    # own properties come first, followed by device function properties.

    awk '
    /Compiling entry function/ {
        # Extract kernel name between single quotes
        match($0, /entry function '"'"'([^'"'"']+)'"'"'/, arr)
        if (arr[1] != "") {
            current_entry = arr[1]
            found_props = 0
            spill_st = 0
            spill_ld = 0
            stack = 0
        }
        next
    }

    current_entry != "" && !found_props && /Function properties for / {
        # Check if this is the entry function properties (first one after Compiling)
        if (index($0, "Function properties for " current_entry) > 0) {
            found_props = 1
        }
        next
    }

    current_entry != "" && found_props == 1 && /bytes stack frame/ {
        match($0, /([0-9]+) bytes stack frame, ([0-9]+) bytes spill stores, ([0-9]+) bytes spill loads/, arr)
        if (arr[1] != "") {
            stack = arr[1]
            spill_st = arr[2]
            spill_ld = arr[3]
        }
        found_props = 2
        next
    }

    current_entry != "" && /Used [0-9]+ registers/ {
        match($0, /Used ([0-9]+) registers/, arr)
        regs = arr[1]

        cmem = 0
        if (match($0, /([0-9]+) bytes cmem\[0\]/, arr2)) {
            cmem = arr2[1]
        }

        cumul_stack = 0
        if (match($0, /([0-9]+) bytes cumulative stack/, arr3)) {
            cumul_stack = arr3[1]
        }

        print current_entry "|" regs "|" spill_st "|" spill_ld "|" stack "|" cumul_stack "|" cmem
        current_entry = ""
        next
    }
    ' "$raw_file" > "$parsed_file"

    # Pass 2: add occupancy calculation
    while IFS='|' read -r kernel regs spill_st spill_ld stack cumul_stack cmem; do
        local occ
        occ=$(calculate_occupancy "$regs")
        echo "${kernel}|${regs}|${spill_st}|${spill_ld}|${stack}|${cumul_stack}|${cmem}|${occ}"
    done < "$parsed_file"

    rm -f "$raw_file" "$parsed_file"
}

# ── Main ──────────────────────────────────────────────────────────

# Collect all results
declare -a ALL_RESULTS=()
declare -a ALL_FILES=()

total_kernels=0
warn_count=0
crit_count=0
spill_count=0

for ptx_file in "${PTX_FILES[@]}"; do
    # Resolve path
    if [[ "$ptx_file" != /* ]]; then
        if [ -f "$HOST_DIR/$ptx_file" ]; then
            ptx_file="$HOST_DIR/$ptx_file"
        fi
    fi

    if [ ! -f "$ptx_file" ]; then
        echo "WARNING: PTX file not found: $ptx_file" >&2
        continue
    fi

    ptx_name=$(basename "$ptx_file")
    echo "Analyzing $ptx_name..." >&2

    results=$(analyze_ptx "$ptx_file")
    if [ -n "$results" ]; then
        ALL_FILES+=("$ptx_name")
        while IFS= read -r line; do
            ALL_RESULTS+=("${ptx_name}|${line}")
        done <<< "$results"
    fi
done

echo "" >&2

# ── Output ────────────────────────────────────────────────────────

if [ "$OUTPUT_JSON" -eq 1 ]; then
    # JSON output
    echo "["
    first=1
    for result in "${ALL_RESULTS[@]}"; do
        IFS='|' read -r file kernel regs spill_st spill_ld stack cumul_stack cmem occ <<< "$result"
        if [ "$first" -eq 0 ]; then echo ","; fi
        first=0
        local_level="ok"
        if [ "$occ" -lt 25 ]; then local_level="critical"
        elif [ "$occ" -lt 50 ]; then local_level="warn"
        fi
        printf '  {"file":"%s","kernel":"%s","registers":%d,"spill_stores":%d,"spill_loads":%d,"stack_frame":%d,"cumulative_stack":%d,"cmem":%d,"occupancy":%d,"level":"%s"}' \
            "$file" "$kernel" "$regs" "$spill_st" "$spill_ld" "$stack" "$cumul_stack" "$cmem" "$occ" "$local_level"
    done
    echo ""
    echo "]"
    exit 0
fi

# Table output
echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  GPU Kernel Resource Report  (target: $SM_NAME, block: 256 threads)${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════════════════════════════════${NC}"

current_file=""
for result in "${ALL_RESULTS[@]}"; do
    IFS='|' read -r file kernel regs spill_st spill_ld stack cumul_stack cmem occ <<< "$result"

    total_kernels=$((total_kernels + 1))

    # Print file header on change
    if [ "$file" != "$current_file" ]; then
        current_file="$file"
        echo ""
        echo -e "${CYAN}── $file ──${NC}"
        printf "  ${DIM}%-40s %5s %5s %8s %6s %4s${NC}\n" "KERNEL" "REGS" "OCC%" "SPILL" "STACK" "WARN"
        echo -e "  ${DIM}$(printf '%.0s─' {1..75})${NC}"
    fi

    # Determine warning level
    level=""
    color="$GREEN"
    if [ "$occ" -lt 25 ]; then
        level="CRIT"
        color="$RED"
        crit_count=$((crit_count + 1))
    elif [ "$occ" -lt 50 ]; then
        level="WARN"
        color="$YELLOW"
        warn_count=$((warn_count + 1))
    fi

    # Spill indicator
    spill_info="-"
    if [ "$spill_st" -gt 0 ] || [ "$spill_ld" -gt 0 ]; then
        spill_info="${spill_st}/${spill_ld}B"
        if [ -z "$level" ]; then
            level="spill"
        fi
        spill_count=$((spill_count + 1))
    fi

    # Stack info
    stack_info="-"
    if [ "$cumul_stack" -gt 0 ]; then
        stack_info="${cumul_stack}B"
    elif [ "$stack" -gt 0 ]; then
        stack_info="${stack}B"
    fi

    # Print kernel line
    if [ -n "$level" ]; then
        printf "  ${color}%-40s %5d %4d%% %8s %6s %4s${NC}\n" \
            "$kernel" "$regs" "$occ" "$spill_info" "$stack_info" "$level"
    else
        printf "  %-40s %5d %4d%%  %7s %6s\n" \
            "$kernel" "$regs" "$occ" "$spill_info" "$stack_info"
    fi
done

# ── Bank Conflict Detection (PTX pattern analysis) ───────────────
bank_conflict_count=0
declare -a BANK_WARNINGS=()

detect_bank_conflicts() {
    local ptx_file="$1"
    local ptx_name
    ptx_name=$(basename "$ptx_file")

    # Single-pass awk for performance (avoids per-line grep on large PTX files)
    local awk_output
    awk_output=$(awk '
    /\.entry / {
        match($0, /\.entry ([A-Za-z_][A-Za-z0-9_]*)/, arr)
        if (arr[1] != "") {
            current_kernel = arr[1]
            uses_smem = 0
        }
        next
    }

    current_kernel != "" && /cvta\.shared/ {
        uses_smem = 1
        next
    }

    current_kernel != "" && uses_smem == 1 && /mad\.lo\.s32/ {
        # Extract numeric operands after the second comma
        n = split($0, parts, ",")
        for (i = 2; i <= n; i++) {
            gsub(/^[ \t]+|[ \t;]+$/, "", parts[i])
            if (parts[i] ~ /^[0-9]+$/) {
                stride = parts[i] + 0
                if (stride > 0 && stride % 128 == 0) {
                    key = current_kernel ":" stride
                    if (!(key in seen)) {
                        seen[key] = 1
                        padded = stride + 4
                        printf "%s|%d|%d\n", current_kernel, stride, padded
                    }
                }
            }
        }
    }
    ' "$ptx_file")

    while IFS='|' read -r kern stride padded; do
        if [ -n "$kern" ]; then
            BANK_WARNINGS+=("  [WARN] bank-conflict: '$kern' ($ptx_name) shared memory stride $stride (multiple of 128 → bank conflict). Use stride $padded.")
            bank_conflict_count=$((bank_conflict_count + 1))
        fi
    done <<< "$awk_output"
}

# Run bank conflict detection on all analyzed PTX files
for ptx_file in "${PTX_FILES[@]}"; do
    resolve_file="$ptx_file"
    if [[ "$resolve_file" != /* ]]; then
        if [ -f "$HOST_DIR/$resolve_file" ]; then
            resolve_file="$HOST_DIR/$resolve_file"
        fi
    fi
    if [ -f "$resolve_file" ]; then
        detect_bank_conflicts "$resolve_file"
    fi
done

# ── Actionable Warnings ─────────────────────────────────────────
declare -a ACTIONABLE_WARNINGS=()

for result in "${ALL_RESULTS[@]}"; do
    IFS='|' read -r file kernel regs spill_st spill_ld stack cumul_stack cmem occ <<< "$result"

    if [ "$occ" -lt 25 ]; then
        ACTIONABLE_WARNINGS+=("  ${RED}[CRIT] low-occupancy: '$kernel' ($file) uses $regs regs → ${occ}% occ. Reduce to ≤128 regs for 50%. Consider: launch_bounds, loop tiling, reducing live variables.${NC}")
    elif [ "$occ" -lt 50 ]; then
        target_msg=""
        if [ "$regs" -gt 128 ]; then
            target_msg=" Reduce to ≤128 regs for 50%."
        elif [ "$regs" -gt 64 ]; then
            target_msg=" Reduce to ≤64 regs for 75%."
        fi
        ACTIONABLE_WARNINGS+=("  ${YELLOW}[WARN] low-occupancy: '$kernel' ($file) uses $regs regs → ${occ}% occ.${target_msg}${NC}")
    fi

    if [ "$spill_st" -gt 0 ] || [ "$spill_ld" -gt 0 ]; then
        total_spill=$((spill_st + spill_ld))
        ACTIONABLE_WARNINGS+=("  ${YELLOW}[WARN] register-spill: '$kernel' ($file) spills ${total_spill}B (${spill_st} store + ${spill_ld} load). Each spill adds ~100 cycles.${NC}")
    fi
done

# ── Summary ───────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  Summary${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════════════════════════════════${NC}"
echo "  Total kernels analyzed: $total_kernels"

if [ "$crit_count" -gt 0 ]; then
    echo -e "  ${RED}CRITICAL (<25% occupancy): $crit_count kernel(s)${NC}"
fi
if [ "$warn_count" -gt 0 ]; then
    echo -e "  ${YELLOW}WARNING  (<50% occupancy): $warn_count kernel(s)${NC}"
fi
if [ "$spill_count" -gt 0 ]; then
    echo -e "  ${YELLOW}SPILLS detected:           $spill_count kernel(s)${NC}"
fi
if [ "$bank_conflict_count" -gt 0 ]; then
    echo -e "  ${YELLOW}BANK CONFLICTS detected:   $bank_conflict_count pattern(s)${NC}"
fi
healthy=$((total_kernels - crit_count - warn_count))
if [ "$healthy" -gt 0 ]; then
    echo -e "  ${GREEN}Healthy  (>=50% occupancy): $healthy kernel(s)${NC}"
fi

# Print actionable warnings
if [ ${#ACTIONABLE_WARNINGS[@]} -gt 0 ] || [ ${#BANK_WARNINGS[@]} -gt 0 ]; then
    echo ""
    echo -e "${BOLD}  Actionable Warnings${NC}"
    echo -e "  ${DIM}$(printf '%.0s─' {1..75})${NC}"
    for msg in "${ACTIONABLE_WARNINGS[@]}"; do
        echo -e "$msg"
    done
    for msg in "${BANK_WARNINGS[@]}"; do
        echo -e "${YELLOW}${msg}${NC}"
    done
fi

echo ""

# Exit code: 1 if any critical, 0 otherwise
if [ "$crit_count" -gt 0 ]; then
    exit 1
fi
exit 0
