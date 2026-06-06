# Theme Synthesis: cost-warnings — Compile-time GPU performance lints

## Status
1/N tasks complete. Warning emission system implemented and validated.

## Key Decisions
- Actionable warnings over bare labels: every warning includes specific register targets and remediation steps
- Bank conflict detection via PTX stride analysis (not MIR): simple, fast, catches common patterns
- Configurable thresholds via WarningConfig (default: CRIT <25%, WARN <50%)
- Conservative bank conflict detection: only flag strides that are exact multiples of 128

## Architecture
1. `gpu-host/src/resource_report.rs` — WarningConfig, KernelWarning, analyze_warnings(), detect_bank_conflicts()
2. `scripts/kernel-resources.sh` — awk-based bank conflict detection + actionable warning output section
3. `format_report()` auto-appends warnings when issues are found

## Validated Results
- showcase_kernel (129 regs, 25% occ) correctly warned with register reduction target
- gemm_f32_v3 stride 132 correctly NOT flagged (padded to avoid bank conflicts)
- No false positives on 101 healthy kernels across kernel_core + kernel_compute

## Risks
- Bank conflict detection is heuristic; data-dependent access patterns are invisible
- Device function register inflation may cause misleading warnings for simple kernels
