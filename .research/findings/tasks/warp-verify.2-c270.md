# warp-verify.2: GPU execution — run warp-cooperative kernel and verify correctness
**Cycle**: 270 | **Theme**: warp-verify | **Kind**: experiment | **Status**: done

## Summary
Both warp-cooperative async kernels run correctly on real GPU hardware (sm_86). `test_simple_warp` (no .await, just `bar.warp.sync`) produces correct output for all 32 lanes. `test_multi_await` (2 .await points, full `activemask + shfl.sync.idx.b32 + bar.warp.sync` pattern) also produces correct output for all 32 lanes.

## Findings

### Q: Does #[warp_cooperative] async fn produce correct results on GPU?
A: **Yes.** Both test kernels pass all 32 lanes:
- `test_simple_warp`: `output[tid] = tid + 1` — PASSED
- `test_multi_await`: `output[tid] = 2*tid + 12` — PASSED

**Confidence**: high

### Q: What PTX post-processing is needed for the kernel to load?
A: Two issues required patching:
1. **`.ptr .align 1`** parameter annotation — LLVM generates `.param .u64 .ptr .align 1 name` but the CUDA PTX JIT rejects this. Removing `.ptr .align 1` fixes it. This is a known LLVM NVPTX backend issue where the alignment annotation is too conservative.
2. **`.extern .func panic_const_async_fn_resumed`** — the coroutine state machine includes an unreachable panic path for "async fn resumed after completion". Since we compile with `-Zbuild-std=core` there's no `core::panicking` implementation. Fix: replace `.extern .func` with a stub containing `trap;`.

Neither issue affects correctness — the `.ptr .align 1` is just a type annotation, and the panic path is unreachable in correct code.

**Confidence**: high

### Q: Does shfl.sync broadcast cause warp divergence issues?
A: No divergence issues observed. In our test, all 32 lanes execute the same `multi_await` function with the same coroutine state at each poll, so the `shfl.sync.idx.b32` broadcast from lane 0 matches what all lanes would compute independently. The `brx.idx` dispatch sends all lanes to the same branch. The real test of divergent behavior would require different lanes to have different coroutine states.

**Confidence**: medium (uniform case only — divergent case untested)

## Unexpected Discoveries
- LLVM's `.ptr .align 1` annotation on `extern "ptx-kernel"` parameters causes CUDA JIT failure. This doesn't happen with our existing kernels that use `*mut u32` parameters directly (those get `.param .u64`). The difference is that `extern "ptx-kernel"` with raw pointer params generates the extra `.ptr .align 1`.
- The `panic_const_async_fn_resumed` extern is always emitted for async fn coroutines even when the panic path is provably unreachable. A future MIR optimization could eliminate this.

## Open Questions
- Divergent warp behavior: what happens when different lanes are in different coroutine states?
- Can the PTX post-processing be automated in the build pipeline?
- Can we eliminate the `panic_const_async_fn_resumed` extern at the MIR/LLVM level?

## Impact on Downstream Tasks
- warp-verify theme SUCCESS — both criteria met
- native-warp-async epic criterion 3 MET: "#[warp_cooperative] async fn compiles via rustc MIR pass and runs on GPU"
- gpu-autonomous v3 criterion 7 MET (same)
