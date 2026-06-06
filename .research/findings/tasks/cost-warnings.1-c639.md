# cost-warnings.1: Compile-time warnings for low occupancy + bank conflicts

## Summary

Enhanced the compile-time GPU performance warning system with actionable
diagnostics for low occupancy and bank conflict detection. Both the Rust
library (`resource_report.rs`) and bash script (`kernel-resources.sh`) now
emit specific, remediation-oriented warnings.

## Changes

### 1. resource_report.rs — Enhanced warning system

New public API additions:
- `WarningConfig` — configurable thresholds (default: CRIT <25%, WARN <50%)
- `KernelWarning` — structured warning with kernel name, severity, kind, message
- `WarningSeverity` — Info/Warn/Critical enum
- `WarningKind` — LowOccupancy/RegisterSpill/BankConflict enum
- `analyze_warnings()` — generates actionable warnings from kernel resources
- `format_warnings()` — human-readable warning report
- `detect_bank_conflicts()` — PTX pattern analysis for bank-conflict strides

Actionable warning messages include:
- Register reduction targets: "Reduce to <=128 registers for 50% occupancy"
- Spill byte counts: "spills 192 bytes (128 store + 64 load)"
- Bank conflict remediation: "stride 128 → use stride 132 instead"
- Occupancy improvement steps for next tier

Bank conflict detection identifies:
- `mad.lo.s32`/`mul.lo.s32` with stride multiples of 128 in kernels using shared memory
- `shl.b32` with shift >= 7 producing 128/256-byte strides
- Correctly skips padded strides (e.g., 132 = 128 + 4 in gemm_f32_v3)

`format_report()` now includes an "Actionable Warnings" section when issues are found.

10 new unit tests added (18 total, all passing).

### 2. kernel-resources.sh — Enhanced bash script

Added after the per-kernel table:
- Bank conflict detection via single-pass awk (fast even on 60K+ line PTX files)
- Per-kernel actionable warning messages with register reduction targets
- Spill warning messages with byte counts and performance impact
- Summary now includes bank conflict count
- "Actionable Warnings" section at the bottom with all warnings

### 3. format_report() auto-appends warnings

When `format_report()` detects critical/warning/spill kernels, it automatically
calls `analyze_warnings()` and appends the actionable section to the report.

## Validation

### Warnings correctly emitted

| PTX file          | Kernel          | Regs | Occ% | Warning                                    |
|-------------------|-----------------|------|------|--------------------------------------------|
| std_build_test    | showcase_kernel | 129  | 25%  | WARN: reduce to <=128 for 50%              |
| kernel_test*      | std_pipeline_test| 255 | 25%  | WARN: reduce to <=128 for 50%              |
| kernel_test*      | matmul_io_compute| 216 | 25%  | WARN: reduce to <=128 for 50%              |

*From prior research (not re-run due to ptxas time constraint on 8MB PTX)

### Bank conflicts correctly detected/skipped

| PTX file         | Kernel       | Stride | Result                              |
|------------------|--------------|--------|-------------------------------------|
| kernel_compute   | gemm_f32_v2  | 132    | OK (not flagged — padded)           |
| kernel_compute   | gemm_f32_v3  | 132    | OK (not flagged — padded)           |
| test PTX (unit)  | bad_stride   | 128    | WARN (flagged — bank conflict)      |
| test PTX (unit)  | good_stride  | 132    | OK (not flagged — padded)           |

### Healthy kernels — no false positives

- kernel_core.ptx (17 kernels): all 100% occupancy, no warnings
- 84 compute kernels: no warnings (gemm_f32_v3 at exactly 50% = threshold boundary, not warned)

## Key Findings

1. **Register target calculation works correctly**: For 25% occupancy, the system
   recommends reducing to <=128 registers for 50% — verified via occupancy formula.

2. **Bank conflict detection is conservative**: Only flags strides that are exact
   multiples of 128, which guarantees conflict. The gemm_f32_v3 padding pattern
   (stride 132) is correctly recognized as safe.

3. **Threshold boundaries match existing classification**: The `<` comparison
   means 25% occupancy is classified as WARN (not CRITICAL), and 50% as OK
   (not WARN). This matches the existing OccupancyLevel classification.

4. **Performance of PTX analysis**: Bank conflict detection via awk is fast
   (~0.1s on 60K line PTX) vs the old per-line grep approach (would take minutes).

## Tests

18 unit tests in `resource_report::tests`:
- 8 existing (parsing, occupancy, format_report)
- 10 new (WarningConfig, analyze_warnings critical/warn/healthy/spill/custom,
  bank_conflict bad_stride/good_stride, next_occupancy_target, format_warnings)
