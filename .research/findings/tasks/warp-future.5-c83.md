# warp-future.5: #[warp_async] Proc Macro for WarpFuture State Machine Generation
**Cycle**: 83 | **Theme**: warp-future | **Kind**: experiment | **Status**: done

## Summary
Implemented `#[warp_async]` proc macro that transforms a function containing `warp_print!()` calls into a complete WarpFuture state machine. Each `warp_print!` becomes an INIT + WAIT state pair. The generated code maintains warp convergence, correct memory ordering, and matches hand-written quality. Verified on SM86 hardware: 2-call generated state machine produced correct messages in 1.6ms. Full code review (rv4) performed — initial PoC had 4 critical issues, all fixed in rework.

## Findings

### Q: Can a proc macro transform async-like syntax into WarpFuture state machine?
A: **Yes.** The `#[warp_async]` attribute macro parses the function body, identifies `warp_print!()` macro invocations (via `Stmt::Macro` AST nodes), and generates a complete WarpFuture struct + impl + kernel entry point. The key insight: syn 2.0 parses `warp_print!()` as `Stmt::Macro` (not `Stmt::Expr`), so the extraction must handle that variant. The generated state machine is structurally identical to hand-written code.
**Confidence**: high

### Q: How to handle warp_await! expansion with __syncwarp()?
A: Each `warp_print!` becomes TWO states:
- **INIT**: Lane 0 pops packet, all lanes cooperatively write payload (first 32 bytes via lane_id, remainder by lane 0), lane 0 submits. Ends with `syncwarp` + `Pending`.
- **WAIT**: All lanes convergently spin-load the control word. On READY, lane 0 releases packet and transitions to next state. Ends with `syncwarp`.

The `syncwarp` barriers are placed at: (1) after payload writes / before header fill, (2) after state transition / before returning. This matches the hand-written pattern exactly.
**Confidence**: high

### Q: Does the generated code match hand-written WarpFuture quality?
A: **Yes, after rework.** The initial PoC had critical issues identified by code review (rv4):
- String-based macro argument parsing (fragile, breaks on commas in expressions)
- Silently discarding non-warp_print statements
- Hardcoded `bool` return type
- No parameter validation

After rework:
- Uses `syn::parse2<WarpPrintArgs>` for robust structured argument parsing
- Rejects non-warp_print statements with compile errors
- Parses and propagates the return type
- Validates first parameter matches buf identifier
- Supports >32 byte messages (hybrid cooperative + sequential write)
- Single CONTROL_FILLED store instead of double write
- `#[inline(always)]` on poll_warp

## Implementation

### New crate: `crates/warp-macro/`
- Proc macro crate with `syn`, `quote`, `proc-macro2` dependencies
- Single attribute macro: `#[warp_async]`
- Custom `Parse` impl for `WarpPrintArgs` (structured argument extraction)

### Usage
```rust
#[warp_macro::warp_async]
unsafe fn warp_macro_print_test(buf: *mut u8) -> bool {
    warp_print!(buf, b"Macro[1/2]: GENERATED_CODE!!");
    warp_print!(buf, b"Macro[2/2]: PROC_MACRO_WORKS!");
}
```

### Generated (conceptual)
- `WarpMacroPrintTest` struct with `buf`, `state`, `pkt_idx`
- `WarpFuture<Output = bool>` impl with 5 states (INIT0, WAIT0, INIT1, WAIT1, DONE)
- `warp_macro_print_test` kernel entry point

### Hardware Verification
- Target: SM86 (RTX 3060)
- Launch: 1 block × 32 threads
- Messages: "Macro[1/2]: GENERATED_CODE!!" and "Macro[2/2]: PROC_MACRO_WORKS!"
- Result: 1 (success)
- Elapsed: ~1.6ms (2 calls)

### Code Review (rv4)
- **Verdict**: rework → pass (after fixes)
- 4 Critical issues found and fixed
- 6 Important issues found and fixed
- 6 Minor issues found and fixed

## Known Limitations
1. Only supports `warp_print!()` as yield points — no generic `warp_call!()` for other services
2. Return value hardcoded to `true` — cannot return computed values
3. No support for code between warp_print! calls (variable bindings, conditionals)
4. Kernel entry point signature hardcoded to `(buf: *mut u8, result: *mut u32)`
5. These are by-design scope limitations for the PoC, documented in the macro's doc comments

## Impact on Downstream Tasks
- warp-future theme is now complete: feasibility → measurement → intrinsics → single PoC → multi PoC → proc macro
- The proc macro demonstrates that automated WarpFuture generation is viable
- Future work: generic `warp_call!()` for non-PRINT services, computed return values, inter-call code support
