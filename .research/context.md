## Current Focus
**Cycle 611 SAVED** (2026-06-05). fusion-codegen.3 done — ElemMul op, vectorized BiasAdd, 2.05x fused speedup.
test-integration.2 code written (11 new kernels) but stashed pending kernel-split epic.
New T0 epic kernel-split (highest priority) — split gpu-kernel-std to fix 30min cubin builds.
fusion-codegen theme completed (3/3). test-integration theme paused on cubin bottleneck.

## Recent Decisions
- 2026-06-05: kernel-split epic created (T0, highest) — split gpu-kernel-std into ≥3 crates
- 2026-06-05: test-integration.2 stashed — 11 kernels written, verify after kernel-split
- 2026-06-05: ElemMul op added to fusion codegen — enables multiply+add+activation chains
- 2026-06-05: BiasAdd float4 vectorized path for n_cols%4==0 — single 128-bit loads
- 2026-06-05: Pre-existing clippy warnings fixed across nn/ (dead code, div_ceil, unused vars)

## Tried & Rejected
- bar.sync for scope join: deadlocks if not all warps participate
- Re-exporting nvptx64 types from host facade: architectural mismatch
- Custom GPU assert for Phase 1: unnecessary, std panic handler already sends coordinates
- GPU par_iter single-block vs Rayon: 4.5% SM utilization, volatile loads bypass cache
- Float4-vectorized bias reads in BiasAdd: requires col%4==0 alignment (kept scalar fallback)
- GPU atomics on stack: ptxas rejects atom.acquire.sys.local.cas — must use global memory statics

## Active Constraints
- GTX 1660 (sm_75): no tensor cores, 192 GB/s, 5 TFLOPS FP32
- 48KB shared memory per block
- Max 2 concurrent subagents (OOM risk)
- **cubin build bottleneck**: unified 11.4MB PTX → 30min ptxas. kernel-split epic addresses this.
- SAVE only after dispatched tasks verified-done, never mid-flight
- test-integration.2 kernels stashed in git stash@{0}

## Key Metrics
- SGEMM V4.1: 2691 GFLOPS at 4096³ (90% cuBLAS)
- FusionCodegen: 7 ops (incl. ElemMul), float4 vectorized, 2.05x fused speedup
- #[gpu_test]: 3 verified tests (11 more stashed, pending kernel-split)
- 761 tasks completed, 48 epics archived

## Next
1. **kernel-split** (T0, highest): brainstorm → themes/tasks → split gpu-kernel-std
2. fusion-integrate.1: wire FusionCodegen into nn API (after kernel-split unblocks builds)
3. test-integration.2: unstash + verify after kernel-split
4. par_iter multi-block: needs brainstorm for gpu-iterator performance tasks
