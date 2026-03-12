# warp-cfg.4: Implement match support in #[warp_async]
**Cycle**: 117 | **Theme**: warp-cfg | **Kind**: experiment | **Status**: done

## Summary

Implemented match expression support in the `#[warp_async]` proc macro. Added `CfgNode::Match { scrutinee, arms }` variant to the CFG tree. The macro generates a MATCH_DECISION state where lane 0 evaluates the scrutinee, maps it to an arm index via a nested `match`, and broadcasts the index to all 32 lanes via `shfl.sync.idx.b32`. Each arm gets its own state range, all converging at a join point. Verified on GPU hardware with 3 different command values.

## Findings

### Q: Can the macro broadcast discriminant and generate per-arm state ranges?
A: Yes. The MATCH_DECISION state evaluates the scrutinee only on lane 0 (`wcx.is_leader()`), maps it to a u32 arm index via a generated `match` expression, broadcasts the index, then dispatches to the correct arm's start state via another `match` on the index. State numbering: 1 (DECISION) + sum of all arm state counts. Each arm converges at the same join point (next_state after the Match node).
**Confidence**: high

### Q: Does match arm dispatch work correctly with warp convergence?
A: Yes. All 32 lanes receive the same arm index via broadcast, so they all enter the same arm's state range simultaneously. After the arm completes, all lanes converge at the join point. The pattern `arm_index → arm_start_state` dispatch ensures correct SIMT execution.
**Confidence**: high

### Q: Hardware test: dispatch-command match pipeline runs on GPU?
A: Yes. Test kernel `warp_cfg_match_test` takes a `cmd: u64` parameter. Three runs:
- cmd=0 → arm 0 prints "cmd: zero" + "match: done" ✓
- cmd=1 → arm 1 prints "cmd: one" + "match: done" ✓
- cmd=99 → wildcard arm prints "cmd: other" + "match: done" ✓
All runs return status=1 (success) and produce exactly 2 messages each.
**Confidence**: high

## Changes Made
- **crates/warp-macro/src/lib.rs**: Extended CFG system
  - Added `CfgNode::Match { scrutinee: syn::Expr, arms: Vec<(syn::Pat, Vec<CfgNode>)> }` variant
  - Updated `count_node_states`, `collect_all_vars`, `expr_contains_warp_call`, `contains_break_if` for Match
  - Added match parsing in `build_cfg` — validates all arms have warp calls, no guards
  - Added MATCH_DECISION codegen in `gen_arms_for_sequence` — dual match: scrutinee→index, index→state
  - Updated module docs with match example
- **crates/gpu-kernel/src/lib.rs**: Added `warp_cfg_match_test` kernel
- **crates/gpu-host/src/main.rs**: Added `run_warp_cfg_match_test` with 3-run verification
- **crates/gpu-host/kernel.ptx**: Updated with match test kernel

## State Machine Example

For `match cmd { 0 => { warp_print!(A); } 1 => { warp_print!(B); } _ => { warp_print!(C); } } warp_print!(D);`:
```
State 0: MATCH_DECISION → broadcast(match cmd { 0=>0, 1=>1, _=>2 }) → goto arm start
State 1: INIT print(A)     [arm 0]
State 2: WAIT print(A)     → goto 7 (join)
State 3: INIT print(B)     [arm 1]
State 4: WAIT print(B)     → goto 7 (join)
State 5: INIT print(C)     [arm 2]
State 6: WAIT print(C)     → goto 7 (join)
State 7: INIT print(D)     [post-match]
State 8: WAIT print(D)     → DONE (9)
State 9: DONE
```

## Unexpected Discoveries
- Match arm guards (`if` conditions) are not supported because they require additional broadcast logic. This is an acceptable limitation for now — most match dispatches use simple patterns.
- The dual-match approach (scrutinee→index, index→state) adds one level of indirection but ensures the broadcast value is a simple u32, which is efficient for `shfl.sync.idx.b32`.

## Open Questions
- Should empty arms (no warp calls) be allowed with a no-op pass-through? Currently all arms must contain warp calls.

## Impact on Downstream Tasks
- warp-cfg.5 (nested control flow) now has all 3 control flow constructs available: if/else, loop/break, match
- The CfgNode tree architecture handles all constructs uniformly via recursive descent
