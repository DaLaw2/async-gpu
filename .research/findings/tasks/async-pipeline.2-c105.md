# async-pipeline.2: Generalize #[warp_async] proc macro
**Cycle**: 105 | **Theme**: async-pipeline | **Kind**: experiment | **Status**: done

## Summary
Completely rewrote the `#[warp_async]` proc macro from supporting only `warp_print!()` to supporting all 7 hostcall services: `warp_open!`, `warp_close!`, `warp_read!`, `warp_write!`, `warp_bulk_read!`, `warp_bulk_write!`, `warp_print!`. Added `let` variable bindings for return values. Full review (2-agent team) passed after 6 fixes.

## Findings

### Q: How to parse multiple macro types?
A: Generic `MacroArgs` parser handles comma-separated expressions. `ServiceKind::from_name()` dispatches by macro name. Each service validates expected argument count. Unified approach — no per-service parser needed.
**Confidence**: high

### Q: How to track let bindings and infer field types?
A: All hostcall return values are `u64`. Each `let var = warp_xxx!(...)` adds a `var: u64` field to the generated struct. Variables are tracked in `known_vars_so_far` and captured as `let var = self.var;` before closures in subsequent INIT states.
**Confidence**: high

### Q: How to generate per-service payload fill logic?
A: `gen_payload_fill()` function generates the fill closure body per `ServiceKind`. Uses protocol constants for layout. All expressions parenthesized before cast (`(#expr) as u64`) to prevent precedence bugs. Buffer overflow protection added for PRINT (56 bytes), OPEN (56 bytes), WRITE (48 bytes).
**Confidence**: high

### Q: How to handle return types?
A: Restricted to `-> bool` (ready value = `true`) and `-> ()` (ready value = `()`). Other return types produce a clear compile error directing to hand-written WarpFuture. Kernel entry writes `1u32` for success, `0u32` for bool false.
**Confidence**: high

## Key Design Decisions
1. **Uses `warp_hostcall_submit`/`warp_hostcall_wait_u64`** instead of inline packet ops → simpler generated code, single maintenance point
2. **PRINT loses cooperative 32-lane write** → lane 0 writes all bytes via closure. Performance impact negligible (56 bytes max, hostcall round-trip dwarfs fill time)
3. **No branching in macro** — linear pipeline only. Branching requires hand-written WarpFuture (documented)
4. **No sideband_alloc inside macro** — users pass pre-allocated offsets via function parameters

## Review Results (rv5)
Full review with proposer + skeptic. 6 issues found and fixed:
- 3 buffer overflow clamps (PRINT, OPEN, WRITE)
- Duplicate variable detection
- Return type handling (bool vs ())
- Expression cast parenthesization

## Changes Made
- `crates/warp-macro/src/lib.rs`: Complete rewrite (447 → ~430 lines, same size but far more capable)

## Impact on Downstream Tasks
- **async-pipeline.3** (branching example): Ready — macro won't help, hand-written WarpFuture
- **async-pipeline.4** (pipelining example): Ready — same
- **async-pipeline.5** (Embassy scale test): Independent, ready
