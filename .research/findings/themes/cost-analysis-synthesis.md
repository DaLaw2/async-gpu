# Theme Synthesis: cost-analysis — MIR-level GPU Resource Estimation

## Status
1/N tasks complete. Foundation investigation done.

## Key Decisions
- **Hybrid approach**: ptxas -v for registers/occupancy, MIR for bank conflicts
- MIR local count cannot predict physical registers (>3x error margin)
- PTX virtual regs also unreliable (2-5x overcount vs physical)
- ptxas -v is the ONLY source of truth for register allocation

## Architecture
Two integration points:
1. ptxas -v parsing in build scripts (register/occupancy/spill reporting)
2. MIR pass in rustc (bank conflict stride analysis, following warp_cooperative.rs pattern)

## Risks
- Device function register inflation: kernels inherit worst-case regs from shared device fns (112-reg floor)
- All shared memory is dynamic — host-side analysis needed for smem size
- ptxas adds 10-30s to build; may limit to --prod builds only

## Evidence
Real kernels with occupancy issues:
- std_pipeline_test: 255 regs → 25% occupancy (immediate test case)
- matmul_io_compute: 216 regs → 25% occupancy; gemm_f32_v3: 111 regs → 50%

## Next Steps
- Implement ptxas -v parsing + occupancy calculator with sm_75 warning thresholds
- Design MIR pass for shared memory stride analysis
