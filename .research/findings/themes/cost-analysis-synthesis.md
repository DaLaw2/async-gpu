# Theme Synthesis: cost-analysis — MIR-level GPU Resource Estimation

## Status
2/N tasks complete. ptxas -v parsing pipeline implemented and validated.

## Key Decisions
- **ptxas -v is the only source of truth** for physical registers; MIR/PTX virtual counts are unreliable
- Analysis pipeline: bash script (kernel-resources.sh) + Rust library (resource_report.rs)
- Integrated into build-kernels.sh (auto in --prod, opt-in --report in dev)
- kernel_test.ptx (8MB) takes 22+ min for ptxas -v; analysis is --prod only

## Architecture
1. `scripts/kernel-resources.sh` — standalone ptxas -v parser with occupancy report + JSON mode
2. `gpu-host/src/resource_report.rs` — Rust parser: SmConfig, KernelResources, occupancy calculator
3. `build-kernels.sh --report` — integrated resource analysis step after PTX compilation

## Validated Results
- 173+ kernels analyzed across 7 PTX files (all except kernel_test.ptx)
- Real perf issues found: showcase_kernel (129 regs, 25% occ), plus known std_pipeline_test/matmul_io_compute
- Device function register inflation: 34 kernels at 112 regs due to shared std device functions
- sm_75 occupancy formula verified correct against all known data points

## Risks
- ptxas analysis time prohibitive for large PTX (22+ min for 8MB); limit to --prod mode
- Device function inflation causes misleading 50% occupancy for simple kernels

## Next Steps
- MIR pass for bank conflict stride analysis (cost-warnings theme)
- Consider per-file caching of ptxas -v results to avoid re-analysis
