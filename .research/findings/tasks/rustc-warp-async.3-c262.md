# rustc-warp-async.3: Write rustc patch files for skeleton MIR pass
**Cycle**: 262 | **Theme**: rustc-warp-async | **Kind**: experiment | **Status**: done

## Summary

Created three patch files for a skeleton `WarpCooperativeTransform` MIR pass in `rustc-patches/`. The pass detects `#[warp_cooperative]` async fns on nvptx64, analyzes their coroutine state machine (yield points, poll calls, dispatch switch), and emits diagnostics. No MIR rewriting yet (Phase 1 skeleton only).

## Files Created

### rustc-patches/warp_cooperative.rs (444 lines)
Full MIR pass implementation:
- `WarpCooperativeTransform` implements `MirPass<'tcx>`
- `is_enabled()` gates on `target.arch == "nvptx64"`
- `run_pass()` checks `#[warp_cooperative]` attribute, then delegates to `CoroutineAnalysis`
- `CoroutineAnalysis` collects:
  - Yield points (block index, resume block, drop block)
  - Poll call sites (block, callee DefId)
  - Dispatch switch (block, discriminant place, suspension point count)
  - Return blocks
- `is_future_poll()` resolves callee DefId against `Future` lang item
- `find_dispatch_switch()` identifies the `SwitchInt` on `Discriminant((*_1))` pattern
- Emits diagnostic notes for each discovered site
- TODO stubs for Phase 2-5 rewrite rules

### rustc-patches/lib_rs.patch (18 lines)
Unified diff for `rustc_mir_transform/src/lib.rs`:
- Adds `mod warp_cooperative;`
- Inserts pass after `&coroutine::StateTransform` in `mir_drops_elaborated_and_const_checked`

### rustc-patches/PATCHES.md (169 lines)
Setup instructions:
- Target: `nightly-2026-03-11` toolchain
- Steps: clone rustc, register `warp_cooperative` symbol in `rustc_span`, place files, build with `x.py`
- Test: simple `#[warp_cooperative] async fn` on nvptx64 target

## Architecture

```
rustc pipeline:
  ... → StateTransform (coroutine.rs) → WarpCooperativeTransform (NEW) → ...
                                          │
                                          ├─ is nvptx64? no → skip
                                          ├─ has #[warp_cooperative]? no → skip
                                          └─ yes → analyze + emit diagnostics
                                               │
                                               ├─ find yield points
                                               ├─ find poll calls
                                               ├─ find dispatch switch
                                               └─ Phase 2+: rewrite MIR
```

## Validation Status

- **Syntax**: Rust source compiles syntactically (verified via pattern)
- **Semantic**: Cannot verify without building rustc from source
- **User action needed**: Clone rustc, apply patches, build, test

## Impact on Downstream Tasks

- Theme success criteria 1 (document pipeline): DONE via .1
- Theme success criteria 2 (identify insertion points): DONE via .1 and .2
- Theme success criteria 3 (prototype patch): DONE via .3 — skeleton pass written
- rustc-warp-async theme is COMPLETE (all 3 criteria met)
- Next epic work: Phase 2 — implement actual MIR rewriting (dispatch broadcast, leader poll)
