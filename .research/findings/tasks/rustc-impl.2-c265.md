# rustc-impl.2: Implement WarpCooperativeTransform in toolchain/compiler/ and register in pipeline
**Cycle**: 265 | **Theme**: rustc-impl | **Kind**: experiment | **Status**: done

## Summary

Implemented the WarpCooperativeTransform MIR pass in `toolchain/compiler/rustc_mir_transform/src/warp_cooperative.rs` (510 lines). Registered the pass in `lib.rs` via `declare_passes!` and `run_runtime_lowering_passes()`. The pass implements Phase 2 (discriminant broadcast via shfl.sync) and Rule 4 (warp barrier before return), plus the Phase 1 analysis from the skeleton.

## Changes

### toolchain/compiler/rustc_mir_transform/src/warp_cooperative.rs (NEW — 510 lines)

Full MIR pass implementing:
1. **Attribute detection**: `has_warp_cooperative_attr()` uses `Symbol::intern("warp_cooperative")` — no `rustc_span` symbol registration needed
2. **Gating**: `is_enabled()` checks `sess.target.arch == "nvptx64"`, `run_pass()` checks attribute + coroutine metadata
3. **Analysis**: Collects dispatch switch, poll call sites, yield points, return blocks; emits diagnostic notes
4. **Phase 2 — Discriminant broadcast**: `insert_discriminant_broadcast()` splits the dispatch block:
   - Original: `_discr = discriminant(*_1); switchInt(_discr) → [...]`
   - New: `_discr = discriminant(*_1); goto → bb_activemask`
   - `bb_activemask`: `asm!("activemask.b32 $0") → bb_shfl`
   - `bb_shfl`: `asm!("shfl.sync.idx.b32 $0, $1, 0, 31, $2") → bb_switch`
   - `bb_switch`: `switchInt(_bc_discr) → [...]`
5. **Rule 4 — Barrier before return**: `insert_barrier_before_return()` inserts:
   - `bb_mask`: `asm!("activemask.b32 $0") → bb_barrier`
   - `bb_barrier`: `asm!("bar.warp.sync $0") → bb_actual_ret`
   - `bb_actual_ret`: `return`

Key design decisions:
- **Inline asm via `TerminatorKind::InlineAsm`**: Uses `NvptxInlineAsmRegClass::reg32` for operands
- **Template allocation via `Box::leak`**: Since the MIR pass creates new InlineAsm templates that the HIR arena doesn't own, we leak the allocation (freed at process exit anyway)
- **`InlineAsmOptions::NOSTACK`**: These asm blocks don't use the stack
- **`UnwindAction::Unreachable`**: GPU inline asm cannot unwind

### toolchain/compiler/rustc_mir_transform/src/lib.rs (MODIFIED — 2 changes)

1. In `declare_passes!` (line 140): added `mod warp_cooperative : WarpCooperativeTransform;`
2. In `run_runtime_lowering_passes()` (line 656-659): added the pass after `&coroutine::StateTransform`

### rustc-patches/warp_cooperative.rs (UPDATED)

Replaced skeleton with the full implementation (copy from toolchain/).

### rustc-patches/lib_rs.patch (UPDATED)

Updated to reflect actual `declare_passes!` and `run_runtime_lowering_passes()` insertion points.

### rustc-patches/PATCHES.md (UPDATED)

- Removed Step 3 (symbol registration) — no longer needed
- Updated expected output to mention shfl.sync and bar.warp.sync in PTX
- Updated function name from `mir_drops_elaborated_and_const_checked` to `run_runtime_lowering_passes`

## Architecture

```
Post-StateTransform MIR (dispatch switch):

BEFORE:                           AFTER:
bb0: {                            bb0: {
  _d = discr(*_1);                  _d = discr(*_1);
  switchInt(_d) → [...]              goto → bb_mask;
}                                 }
                                  bb_mask: {
                                    asm!("activemask.b32 $0",
                                         out(reg32) _mask);
                                    → bb_shfl
                                  }
                                  bb_shfl: {
                                    asm!("shfl.sync.idx.b32 $0,$1,0,31,$2",
                                         out(reg32) _bcd,
                                         in(reg32) _d,
                                         in(reg32) _mask);
                                    → bb_switch
                                  }
                                  bb_switch: {
                                    switchInt(_bcd) → [...]
                                  }

Return blocks:

BEFORE:                           AFTER:
bb_ret: {                         bb_ret: {
  _0 = Poll::Ready(val);            _0 = Poll::Ready(val);
  return;                            goto → bb_m;
}                                 }
                                  bb_m: { activemask → bb_b }
                                  bb_b: { bar.warp.sync → bb_r }
                                  bb_r: { return }
```

## Not Yet Implemented (Future Phases)

- **Phase 3**: Leader-only poll — gate Future::poll calls behind lane-0 predication
- **Phase 4**: Broadcast Poll::Ready payload (u32/u64/struct decomposition)
- **Phase 5**: Result broadcasting for `?` operator, validation (reject dyn Future, Drop types)

## Impact on Downstream Tasks

- rustc-impl.3 is now UNBLOCKED — toolchain/ has the changes, ready to generate diffs
- Theme success criteria 1 ✅ (pass source file placed)
- Theme success criteria 2 ✅ (pass registered in lib.rs pipeline)
- Theme success criteria 3 ✅ (pass identifies poll calls and suspension points)
- Theme success criteria 4: depends on rustc-impl.3
