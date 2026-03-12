# Review: warp-future.5 — #[warp_async] Proc Macro

**Reviewer**: single-agent | **Verdict**: rework

## Issues Found

### Critical

**C1. Token-based macro argument parsing is fragile and incorrect for complex expressions**

`try_extract_from_macro` (line 55-58) converts the entire token stream to a string, then splits on the first comma:

```rust
let tokens_str = mac.tokens.to_string();
let args: Vec<_> = tokens_str.splitn(2, ',').map(|s| s.trim().to_string()).collect();
```

This breaks for any message expression containing a comma. For example:
- `warp_print!(buf, if cond { b"a" } else { b"b" })` — the comma inside braces would confuse `splitn`
- `warp_print!(buf, my_func(a, b))` — splits on the wrong comma

More fundamentally, converting tokens to a string and re-parsing them loses span information, which degrades error messages. The proper approach is to use `syn::parse2` with a custom `Parse` impl or `syn::punctuated::Punctuated` to parse the comma-separated arguments structurally.

**C2. The `buf` argument from `warp_print!()` is silently discarded**

The macro extracts only `msg_expr` (the second argument) and throws away the first argument (`buf`). The generated code always uses `self.buf` instead. This means:

1. If the user writes `warp_print!(some_other_buf, msg)`, the macro silently ignores `some_other_buf` and uses `self.buf`. This is a correctness trap — the generated code does not honor what the user wrote.
2. There is no validation that the first argument is actually `buf` or matches the function parameter.

**C3. Function parameters are completely ignored**

The input function signature `unsafe fn warp_macro_print_test(buf: *mut u8) -> bool` declares a `buf` parameter, but the macro never inspects the parameter list. The generated struct always has exactly `buf: *mut u8, state: u32, pkt_idx: u16` regardless of what the user declared. If the user adds additional parameters (e.g., `count: u32`), they are silently dropped with no error.

**C4. Return type is hardcoded to `bool`**

The generated code hardcodes:
- `WarpFuture for #struct_name` with `type Output = bool`
- `WarpPoll<bool>` in the poll function
- `WarpPoll::Ready(true)` as the terminal value

The user-written `-> bool` and `true` return expression are completely ignored. If the user writes `-> u32` or returns a different value, the macro still generates `bool`. This should at minimum parse and propagate the return type and the final expression.

### Important

**I1. The `unwrap()` on line 58 will panic (abort compilation) on malformed input**

```rust
let msg_tokens: proc_macro2::TokenStream = args[1].parse().unwrap();
```

If the second argument cannot be re-parsed as a valid token stream after the string round-trip, this panics the compiler with an unhelpful error. Proc macros should always return `syn::Error::to_compile_error()` instead of panicking.

**I2. Non-`warp_print!` statements are silently discarded**

The macro calls `extract_warp_prints` which only collects `warp_print!()` invocations. Any other statements in the function body — variable bindings, computations, control flow, other function calls — are silently dropped. There is no warning or error for unrecognized statements.

For example:
```rust
#[warp_async]
unsafe fn my_pipeline(buf: *mut u8) -> bool {
    let x = compute_something();  // silently dropped
    warp_print!(buf, b"hello");
    do_cleanup();                  // silently dropped
    true                           // silently dropped
}
```

This is extremely misleading. The user writes sequential code expecting it to execute, but only `warp_print!` calls survive. At minimum, the macro should emit a compile error for any non-`warp_print!` statement, or the documentation should make this dramatically clear.

**I3. The `true` expression at the end of the input function is dead code**

The input function body ends with `true` (line 1511 of gpu-kernel), but the macro never reads the final expression. The generated code always returns `WarpPoll::Ready(true)`. The `true` in the input function is pure decoration — misleading the reader into thinking it controls the return value.

**I4. Missing `calls_completed` field compared to hand-written version**

The hand-written `WarpMultiPrintFuture` tracks `calls_completed: u32`. The macro-generated struct omits this field. While it is not strictly needed for correctness (the state machine implicitly tracks progress), the hand-written version uses it, suggesting it may be important for debugging or future extensions.

**I5. The `visit-mut` syn feature is enabled but unused**

`Cargo.toml` declares `syn = { version = "2", features = ["full", "visit-mut"] }` but the code never uses `syn::visit_mut`. This adds unnecessary compile-time overhead for the proc macro.

**I6. Kernel entry point has a fixed `(buf: *mut u8, result: *mut u32)` signature**

The generated kernel always takes `(buf: *mut u8, result: *mut u32)` regardless of the input function's parameter list. This means the macro can only generate kernels that take a hostcall buffer and a result pointer. There is no way to pass additional kernel parameters.

### Minor

**M1. `_attr` is ignored without validation**

The `_attr` parameter (attribute arguments like `#[warp_async(some_option)]`) is silently ignored. If a user passes unexpected attributes, there is no error. Should either validate that `_attr` is empty or document what attributes are accepted.

**M2. Double store to CONTROL field**

The generated INIT state writes to `PKT_OFF_CONTROL` twice:
```rust
gpu_atomics::sys_store_release_u32(
    pkt.add(gpu_protocol::PKT_OFF_CONTROL) as *mut u32, 0,
);
gpu_atomics::sys_store_release_u32(
    pkt.add(gpu_protocol::PKT_OFF_CONTROL) as *mut u32,
    gpu_protocol::CONTROL_FILLED,
);
```

This matches the hand-written version (clear then set pattern), so it is intentional. However, the "clear to 0 then set to FILLED" two-step is unusual. If the packet was already zeroed on allocation, the first store is redundant. If it was not zeroed, the two-step is a potential race window. This is inherited from the hand-written code and not a macro-specific issue, but worth noting.

**M3. No `#[inline(always)]` on `poll_warp`**

The hand-written version also lacks this, but since this is GPU code where the PTX has no linker, `poll_warp` should be `#[inline(always)]` to ensure it gets inlined into the executor loop. The trait definition may already handle this, but it is worth verifying.

**M4. Message length limited to 32 bytes**

The cooperative payload write uses `lane_id` (0..31) to index into the message:
```rust
if lid < msg_len {
    core::ptr::write_volatile(msg_base.add(lid as usize), msg[lid as usize]);
}
```

Messages longer than 32 bytes will be silently truncated (only the first 32 bytes written). The hand-written version handles this by splitting into prefix/suffix with separate cooperative writes, but the macro makes no such accommodation. The test messages are 28 and 27 bytes, just fitting under the limit, but this is a footgun for users.

**M5. `to_pascal_case` does not handle edge cases**

- Leading/trailing underscores produce empty segments: `_foo_` becomes `Foo`
- Double underscores `foo__bar` produce empty string segments (harmless but odd)
- These are unlikely in practice but show the function was not written defensively.

**M6. The `Expr` import is unused for anything other than the macro match**

`use syn::{parse_macro_input, ItemFn, Stmt, Expr, ExprMacro};` — while these are all used, the `Expr` import only participates in a single destructuring pattern. This is fine but worth noting that `ExprMacro` could be matched directly without importing `Expr` separately.

## Positive Aspects

1. **The core INIT/WAIT state machine generation is correct.** The generated state pairs faithfully reproduce the hand-written pattern: pop free packet, cooperative payload write, header fill, submit, then spin-wait for READY, release, transition. The state numbering (0, 1, 2, 3, ..., 2N) is clean and correct.

2. **Warp convergence is maintained.** The critical syncwarp barriers are in the right places: after payload writes (before header fill) and after state transition (before returning). The broadcast_u32 for state and pkt_idx ensures all lanes see consistent values.

3. **The error for zero warp_print calls is good.** Producing a compile error with a clear message is the right approach.

4. **The struct/impl/kernel generation is clean.** The quote! macro usage is idiomatic and the generated code structure matches the hand-written version closely.

5. **The naming convention (snake_case -> PascalCase) is a nice touch** for generating struct names from function names.

6. **Memory ordering is correct.** Uses `sys_store_release_u32` for CONTROL writes and `sys_spin_load_acquire_u32` for CONTROL reads, matching the hand-written version exactly.

## Verdict

**Rework required.** The proc macro generates correct GPU code for its narrow use case (sequential `warp_print!` calls with byte-literal messages under 32 bytes), and the warp convergence / memory ordering is sound. However, several critical issues must be addressed before this can serve as a general-purpose `#[warp_async]` attribute:

1. **String-based token parsing** (C1) is the most urgent fix — use `syn::parse2` for robust argument parsing.
2. **Silently discarding non-warp_print statements** (I2) is a correctness trap — either support arbitrary code between calls or emit errors.
3. **Hardcoded `bool` return type and ignored parameters** (C3, C4) make the macro unusable for any function signature other than the exact test case.
4. **The 32-byte message limit** (M4) should at minimum be documented or enforced with a compile-time check.

The macro is a solid proof-of-concept demonstrating that proc-macro-generated WarpFuture state machines produce correct GPU code. The path to a production-quality `#[warp_async]` requires making the parsing robust, supporting arbitrary expressions between yield points, and parameterizing the output type.
