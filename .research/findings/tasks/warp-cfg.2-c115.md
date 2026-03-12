# warp-cfg.2: Implement if/else support in #[warp_async]
**Cycle**: 115 | **Theme**: warp-cfg | **Kind**: experiment | **Status**: done

## Summary

Implemented if/else support in the `#[warp_async]` proc macro. The macro now builds a CFG tree (`CfgNode::Call` | `CfgNode::IfElse`) instead of a flat call list, and generates DECISION states for branches. Lane 0 evaluates the condition and broadcasts the result to all 32 lanes via `shfl.sync.idx.b32`. Both branches verified on GPU hardware with parameter-controlled test.

## Findings

### Q: Can the macro parse if blocks containing warp_*!() and generate forked state ranges?
A: Yes. The new `build_cfg()` function recursively descends into if/else blocks, building a tree of `CfgNode` variants. State numbering uses `count_node_states()` to precompute offsets: each `Call` gets 2 states (INIT+WAIT), each `IfElse` gets 1 state (DECISION) plus the sum of its branches' states. The recursive `gen_arms_for_sequence()` generates match arms with correct state transitions, including join-point convergence after branches.
**Confidence**: high

### Q: Does lane-0 broadcast decision work correctly for if/else branching?
A: Yes. The DECISION state captures function parameters and known variables from prior warp calls, evaluates the condition only on lane 0 (`wcx.is_leader()`), broadcasts the result as a u32 (0/1), and sets `self.state` to the correct branch's first state. All 32 lanes then enter the same branch arm on the next poll.
**Confidence**: high

### Q: Hardware test: if-based branching pipeline runs on GPU?
A: Yes. Test kernel `warp_cfg_if_else_test` takes a `flag: u64` parameter. Two runs:
- flag=1 → then branch prints "branch: then", joins to "branch: done" ✓
- flag=0 → else branch prints "branch: else", joins to "branch: done" ✓
Both runs return status=1 (success) and produce exactly 2 messages each.
**Confidence**: high

## Changes Made
- **crates/warp-macro/src/lib.rs**: Major refactoring
  - Added `CfgNode` enum (`Call`, `IfElse`) replacing flat `Vec<WarpCall>`
  - Added `build_cfg()` replacing `extract_warp_calls()` — handles if/else recursively
  - Added `count_node_states()`, `count_sequence_states()`, `collect_all_vars()` helpers
  - Added `stmts_contain_warp_call()` / `expr_contains_warp_call()` for checking if blocks
  - Added `extract_else_stmts()` for handling `else { }` and `else if { }` blocks
  - Added `gen_arms_for_sequence()` — recursive match arm generator with DECISION states
  - Updated `warp_async` to use CFG-based pipeline
- **crates/gpu-kernel/src/lib.rs**: Added `warp_cfg_if_else_test` kernel
- **crates/gpu-host/src/main.rs**: Added `run_warp_cfg_if_else_test` with 2-run verification
- **crates/gpu-host/kernel.ptx**: Updated with new kernel

## State Machine Example

For `if flag != 0 { warp_print!(A); } else { warp_print!(B); } warp_print!(C);`:
```
State 0: DECISION → broadcast(flag != 0) → goto 1 or 3
State 1: INIT print(A)
State 2: WAIT print(A) → goto 5 (join)
State 3: INIT print(B)
State 4: WAIT print(B) → goto 5 (join)
State 5: INIT print(C)
State 6: WAIT print(C) → DONE (7)
State 7: DONE
```

## Unexpected Discoveries
- Function parameters need explicit capture in DECISION states — the condition expression uses variable names that resolve to `self.field`, requiring `let name = self.name;` captures just like warp call result variables.
- `else if` chains work naturally through recursive descent: `else if cond { ... }` becomes a single `Stmt::Expr(Expr::If(...))` in the else branch, which `build_cfg` handles recursively.

## Open Questions
- Should DECISION states be inlined into the preceding WAIT arm to save a poll cycle? Currently adds one extra poll per branch point.

## Impact on Downstream Tasks
- warp-cfg.3 (loop/break) can extend CfgNode with `Loop { body }` variant
- warp-cfg.4 (match) can extend CfgNode with `Match { scrutinee, arms }` variant
- The recursive gen_arms_for_sequence architecture scales cleanly to nested control flow
