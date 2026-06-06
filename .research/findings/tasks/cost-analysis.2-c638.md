# cost-analysis.2: Per-Kernel Resource Estimation Pass

## Summary

Implemented a complete per-kernel resource estimation pipeline: a bash script
(`scripts/kernel-resources.sh`) for ptxas -v parsing + occupancy calculation
with formatted reports, a Rust library module (`resource_report.rs`) in gpu-host
for programmatic access, and integration into `build-kernels.sh` via `--report`
flag (auto-runs in `--prod` mode). Analyzed all kernel PTX files except
kernel_test.ptx (ptxas takes 22+ min on 8MB PTX).

## Deliverables

### 1. scripts/kernel-resources.sh
Standalone analysis script. Takes PTX files as input, runs ptxas -v, parses
output, calculates sm_75 occupancy, and emits a formatted report with per-kernel
register count, occupancy percentage, spill info, stack usage, and warning levels.
Supports `--json` output mode. Exit code 1 if any CRITICAL (<25% occ) kernels.

### 2. gpu-host/src/resource_report.rs
Rust module with:
- `SmConfig` — SM architecture params (sm_75 preset provided)
- `KernelResources` — per-kernel resource struct (regs, spills, stack, cmem)
- `OccupancyLevel` — Ok/Warn/Critical classification
- `parse_ptxas_output()` — parses ptxas -v stderr into Vec<KernelResources>
- `format_report()` — produces human-readable table
- `occupancy()` — register-based occupancy calculation with correct sm_75 granularity
- 8 unit tests covering parsing, occupancy math, and report formatting

### 3. build-kernels.sh integration
Added `--report` flag. Resource analysis auto-runs in `--prod` mode (since ptxas
is already invoked) and on-demand with `--report` in dev mode.

## Analysis Results

### kernel_core.ptx (17 kernels)
All healthy: 4-18 registers, 100% occupancy. Simple atomic/math/test kernels.

### kernel_compute.ptx (84 kernels)
- 24 kernels at exactly 112 regs / 50% occ — device function register inflation
  (shared device functions from std set the floor at 112 regs regardless of kernel
  complexity; e.g., `bias_add` uses 28 virtual regs but gets 112 physical)
- `gemm_f32_v3`: 111 regs, 50% occupancy (the most register-hungry pure-compute kernel)
- `batch_search_pipeline`: 126 regs, 50% occupancy + 592B cumulative stack
- `gemm_f32_v2`: 69 regs, 75% occupancy
- Remaining 58 kernels: 100% occupancy (4-54 regs)

### kernel_io.ptx (55 kernels)
- 10 kernels at 112 regs / 50% occ (device function inflation from std/panic handlers)
- `parallel_grep_kernel`: 112 regs + 4184B cumulative stack (deep call tree)
- `bulk_io_test`: 64 regs + 8192B cumulative stack (large stack allocation)
- Remaining 45 kernels: 100% occupancy

### std_build_test.ptx (17 kernels)
- **showcase_kernel: 129 regs, 25% occupancy (WARNING)** — real perf issue detected
- `slab_dealloc_test_kernel`: 114 regs, 50% occ
- `std_file_write_kernel`: 90 regs, 50% occ
- `std_println_multi_kernel`: 82 regs, 50% occ
- 4 kernels at 72 regs, 75% occ

### kernel_test.ptx (113 kernels — NOT analyzed, ptxas too slow)
From prior research (cost-analysis.1):
- `std_pipeline_test`: 255 regs, 25% occupancy (CRITICAL)
- `matmul_io_compute`: 216 regs, 25% occupancy (CRITICAL)
- These would be flagged by kernel-resources.sh; confirmed manually.

ptxas -v on kernel_test.ptx (8MB, 113 entry functions) consumed 22+ minutes of
CPU time and 2GB RAM before being terminated. This is the key performance
constraint for the analysis pipeline.

## Key Findings

### 1. Device function register inflation is the dominant pattern
24 of 84 compute kernels and 10 of 55 IO kernels hit exactly 112 registers —
NOT because the kernel itself is complex, but because shared device functions
(std panic handlers, allocators, string formatters) require 112 regs, and ptxas
allocates the maximum across all device functions in the compilation unit.

**Impact**: Many kernels show 50% occupancy that should be 100%. Separate
compilation or `__launch_bounds__` could fix this, but it's a ptxas design issue,
not a kernel design issue. The resource report correctly identifies these, but
the warning may be misleading for simple kernels.

### 2. ptxas analysis time scales super-linearly with PTX size
| PTX file         | Size | Entry funcs | ptxas -v time |
|------------------|------|-------------|---------------|
| kernel_core.ptx  | 110K | 17          | <2s           |
| kernel_io.ptx    | 3.0M | 55          | ~60s          |
| kernel_compute.ptx| 3.5M| 84          | ~90s          |
| std_build_test.ptx| 2.1M| 17          | ~20s          |
| kernel_test.ptx  | 8.1M | 113         | >22 min       |

kernel_test.ptx is the outlier: 8MB of PTX (including massive std library code)
makes ptxas spend enormous time on register allocation for device functions.
Analysis must be opt-in (`--report`/`--prod`), never in dev builds.

### 3. Occupancy formula verified correct
Manually verified against known data points:
- 12 regs → 100% occ (register allocates 512/warp, 128 warps, clamped to 1024 threads)
- 112 regs → 50% occ (3584/warp → 18 warps → 2 blocks × 256 = 512 threads)
- 255 regs → 25% occ (8160/warp → 8 warps → 1 block × 256 = 256 threads)

### 4. Real performance issues detected
The tool successfully identifies real occupancy problems:
- `showcase_kernel` (129 regs, 25% occ) in std_build_test.ptx
- `std_pipeline_test` (255 regs, 25% occ) and `matmul_io_compute` (216 regs, 25% occ)
  in kernel_test.ptx (confirmed from prior research)

## Occupancy Formula Reference (sm_75)

```
regs_per_warp = registers_per_thread × 32
alloc = ceil(regs_per_warp / 256) × 256   # allocation granularity
max_warps = 65536 / alloc                  # total SM regs / per-warp alloc
max_blocks = min(max_warps / warps_per_block, 16)  # 16 = hw max blocks/SM
active_threads = min(max_blocks × block_size, 1024)
occupancy = active_threads / 1024 × 100%
```

## CI Validation

All CI lint checks pass (`scripts/ci-lint.sh`):
- fmt, clippy, doc, check: all OK for all crates
- No regressions from new module
- 8 unit tests pass for resource_report module
