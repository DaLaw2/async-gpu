## Current Focus
**Cycle 642 — gpu-panic and feature-audit stories ready for verification** (2026-06-06). All 12 tasks across both stories completed. gpu_assert! fully removed. PTX auto-discovery added. Story Verification Gates next.

## Recent Decisions
- 2026-06-06: gpu_assert! entirely deleted (~200 lines). Standard assert! is now the only assertion path. SERVICE_ASSERT protocol opcode removed.
- 2026-06-06: PTX auto-discovery: get_kernel() now searches all modules (ptx::ALL) with text pre-filter, fixing 8 examples that loaded wrong PTX.
- 2026-06-06: HostcallBuffer encapsulated (pub→pub(crate) + accessors). Facade re-exports added.
- 2026-06-06: Patched std default_hook outputs GPU block/warp/lane in panic messages.

## Tried & Rejected
- MIR-only register estimation: >3x error margin vs physical registers
- PTX virtual register counting: 2-5x overcount vs ptxas allocation

## Active Constraints
- GTX 1660 (sm_75): 192 GB/s, 5 TFLOPS FP32, 48KB smem, 64K regs
- Max 2 concurrent heavy subagents
- kernel_test.ptx JIT takes 20+ min
- Priority gate: gpu-panic + feature-audit (high) block compile-time-cost (medium)

## Key Metrics
- 816 tasks completed, 56 stories completed (57th and 58th pending verification)
- gpu-panic: 6/6 tasks done → Story Verification Gate pending
- feature-audit: 6/6 tasks done → Story Verification Gate pending
- Known gap: std abort path doesn't call set_warp_trapped()/write_panic_to_result()

## Next
1. Story Verification Gate: gpu-panic — verify all 4 success criteria met
2. Story Verification Gate: feature-audit — verify all 4 success criteria met
3. If both pass → cascade close → priority gate lifts → cost-warnings.2 becomes eligible
4. Brainstorm for next stories (gpu-dyn-dispatch, transparent-data, etc.)
