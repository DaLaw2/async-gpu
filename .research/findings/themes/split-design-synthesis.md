# split-design — Theme Synthesis

**Epic**: kernel-split (break gpu-kernel-std into per-feature crates)
**Status**: active | **Tasks completed**: 2/? | **Updated**: 2026-06-05

## Key Findings
- 4-crate split is dependency-safe with zero circular deps (task .1)
- Host loader multi-cubin API designed with full backward compat (task .3)
- Alias strategy: `KERNEL->KERNEL_COMPUTE`, `KERNEL_STD->KERNEL_TEST` -- zero call-site changes in phase 2
- Per-crate `PtxModule` struct + `ALL` array enables auto-discovery fallback

## Architecture Summary
ptx module gets 4 per-crate constants + cubin pairs + deprecated aliases. gpu.rs needs only `.module()` builder addition. KernelRegistry needs zero changes (all ML kernels are in compute). gpu-test-macro migrates from disk cubin to embedded cubin.

## Critical Path Remaining
1. Crate-type decision for gpu-kernel-core (rlib vs cdylib) -- drives linkage model
2. Binary size strategy for embedded cubins (features vs disk vs include_bytes)
3. Build script rewrite (parallel per-crate ptxas)
4. Actual crate split implementation + host loader code changes
5. LTO force-link: `#[no_mangle]` in rlib may need `#[used]` array to survive LTO

## Risk Summary
Medium: cubin binary size (~190 MB embedded). Low: all API changes backward-compat. No blockers.
