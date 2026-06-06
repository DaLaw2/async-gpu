# Theme Synthesis: cost-warnings — Compile-time GPU performance lints

## Status
2/N tasks complete. Warning system proven on real kernel.

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
- showcase_kernel (129 regs, 25% occ) correctly warned — reducing to 128 regs doubles occupancy to 50%
- Proven on 187+ kernels across 8 PTX files: 1 true warning, 0 false positives
- gemm_f32_v3 stride 132 correctly NOT flagged (padded to avoid bank conflicts)
- Bash and Rust paths produce consistent results (18 unit tests pass)

## Risks
- Bank conflict detection is heuristic; data-dependent access patterns are invisible
- Device function register inflation may cause misleading warnings for simple kernels
