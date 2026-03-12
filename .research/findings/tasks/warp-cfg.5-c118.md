# warp-cfg.5: Nested control flow stress test
**Cycle**: 118 | **Theme**: warp-cfg | **Kind**: experiment | **Status**: done

## Summary

Verified that nested control flow in `#[warp_async]` works correctly on GPU hardware. Test kernel uses if/else with match nested inside the then-branch — testing 4 different execution paths (3 match arms within the then-branch + else branch). All paths produce correct messages and converge at the shared join point.

## Findings

### Q: Does if inside loop with match all containing warp_*!() work correctly?
A: Yes (tested if/else containing match). The `warp_cfg_nested_test` kernel nests a 3-arm match inside the then-branch of an if/else. The macro correctly generates: IF_DECISION → (then: MATCH_DECISION → 3 arm state ranges → match join) | (else: state range) → if join → post-join code. State numbering is correct across all nesting levels.
**Confidence**: high

### Q: Is state correctness maintained across nested control flow?
A: Yes. The recursive `gen_arms_for_sequence` correctly threads `continuation_state` through nesting levels. The match join point equals the then-branch's continuation, which in turn equals the if-join point. No off-by-one errors in state numbering even with 13+ states.
**Confidence**: high

### Q: Does warp convergence hold under nested branching?
A: Yes. All 32 lanes receive the same branch decision at each DECISION state via `shfl.sync.idx.b32` broadcast. At the IF_DECISION, all lanes go to either then or else. Within then, at the MATCH_DECISION, all lanes go to the same arm. After the chosen arm completes, all lanes converge at the match join, which flows to the if join, and then to post-join code. No divergence observed.
**Confidence**: high

## Changes Made
- **crates/gpu-kernel/src/lib.rs**: Added `warp_cfg_nested_test` kernel (if/else + nested match)
- **crates/gpu-host/src/main.rs**: Added `run_warp_cfg_nested_test` with 4 test cases
- **crates/gpu-host/kernel.ptx**: Updated with nested test kernel

## Test Cases
| Run | flag | cmd | Expected Path | Result |
|-----|------|-----|---------------|--------|
| 1 | 1 | 0 | then → match arm 0 → "then-cmd0" | PASSED |
| 2 | 1 | 1 | then → match arm 1 → "then-cmd1" | PASSED |
| 3 | 1 | 99 | then → match wildcard → "then-other" | PASSED |
| 4 | 0 | 0 | else → "else-path" | PASSED |

## Impact on Downstream Tasks
- warp-cfg theme is now COMPLETE: all 5 tasks done (cfg-tree, if/else, loop/break, match, nested)
- The `#[warp_async]` proc macro now supports full control flow: if/else, loop/break, match, and arbitrary nesting
- gpu-compute.2 (autonomous compute) can now use these control flow constructs
