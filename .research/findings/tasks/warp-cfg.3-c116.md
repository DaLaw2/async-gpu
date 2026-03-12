# warp-cfg.3: Implement loop/break support in #[warp_async]
**Cycle**: 116 | **Theme**: warp-cfg | **Kind**: experiment | **Status**: done

## Summary

Implemented loop/break support in the `#[warp_async]` proc macro. Added `CfgNode::Loop` and `CfgNode::BreakIf` variants to the CFG tree. The macro generates back-edge transitions (end of body → loop start) and BREAK_DECISION states where lane 0 evaluates the break condition and broadcasts via `shfl.sync.idx.b32`. Verified on GPU hardware with counter=0 (immediate break).

## Findings

### Q: Can the macro handle loop bodies with warp_*!() and generate cycle-back transitions?
A: Yes. `build_cfg()` now handles `Stmt::Expr(Expr::Loop(...))` by recursively descending into the loop body with `in_loop=true`. In `gen_arms_for_sequence`, `CfgNode::Loop` passes `continuation_state = loop_start` (back-edge) so the last statement in the body transitions back to the loop's first state instead of proceeding linearly. State count for a Loop node is `count_sequence_states(body)`.
**Confidence**: high

### Q: Does break correctly exit the loop state machine?
A: Yes. `is_break_if()` detects the pattern `if cond { break; }` (no else, single break statement) and converts it to `CfgNode::BreakIf { cond }`. The BREAK_DECISION state evaluates the condition on lane 0, broadcasts, and jumps to `break_target` (the post-loop state) if true, or `next_state` (next body statement) if false. The `break_target: Option<u32>` parameter threads through recursive codegen.
**Confidence**: high

### Q: Hardware test: read-until-done loop pipeline runs on GPU?
A: Yes. Test kernel `warp_cfg_loop_test` with counter=0 produces exactly 2 messages: "iter" then "done". The loop executes one iteration (INIT+WAIT for print "iter"), hits the BREAK_DECISION (counter==0 → break), exits to post-loop print "done", then reaches DONE state. Status=1 (success).
**Confidence**: high

## Changes Made
- **crates/warp-macro/src/lib.rs**: Extended CFG system
  - Added `CfgNode::Loop { body: Vec<CfgNode> }` and `CfgNode::BreakIf { cond: syn::Expr }`
  - Added `break_target: Option<u32>` parameter to `gen_arms_for_sequence`
  - Added `is_break_if()` — detects `if cond { break; }` pattern
  - Added `contains_break_if()` — verifies loop has exit path
  - Loop body codegen uses `continuation_state = loop_start` for back-edge
  - BreakIf generates BREAK_DECISION state with broadcast
- **crates/gpu-kernel/src/lib.rs**: Added `warp_cfg_loop_test` kernel
- **crates/gpu-host/src/main.rs**: Added `run_warp_cfg_loop_test` with counter=0 verification
- **crates/gpu-host/kernel.ptx**: Updated with loop test kernel

## State Machine Example

For `loop { warp_print!(iter); if counter == 0 { break; } } warp_print!(done);`:
```
State 0: INIT print("iter")       → submit PRINT
State 1: WAIT print("iter")       → goto 2
State 2: BREAK_DECISION           → broadcast(counter==0) → if true: goto 3 (post-loop), else: goto 0 (back-edge)
State 3: INIT print("done")       → submit PRINT
State 4: WAIT print("done")       → DONE (5)
State 5: DONE
```

## Unexpected Discoveries
- Since `#[warp_async]` doesn't yet support compute-only statements (only warp_*!() calls), the loop counter cannot be decremented. The break condition is constant per invocation, limiting testability to immediate-break (counter=0) or infinite loop (counter!=0). Full loop testing requires gpu-compute support (warp-cfg.5 or later).

## Open Questions
- Should loop unrolling hints be supported for bounded loops?
- Can compute-only statements (assignments, arithmetic) be added as pass-through in the state machine?

## Impact on Downstream Tasks
- warp-cfg.4 (match) can extend CfgNode with `Match { scrutinee, arms }` variant
- warp-cfg.5 (nested control flow) now has both if/else and loop/break available
- The break_target threading pattern scales to nested loops
