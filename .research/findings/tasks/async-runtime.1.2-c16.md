# async-runtime.1.2: Test LTO cross-crate resolution for Embassy on nvptx64
**Cycle**: 16 | **Theme**: async-runtime | **Kind**: experiment | **Status**: done

## Summary
Fat LTO (`lto = "fat"` in Cargo profile) **fully resolves** all cross-crate Embassy executor extern calls on nvptx64. `Executor::spawn`, `Executor::poll`, panic functions, the task arena, and critical-section are all inlined/defined in the final PTX output. The only remaining extern was `__pender` (Embassy's waker callback), which is resolved by providing a no-op `#[no_mangle] extern "C" fn __pender`. No Embassy fork or vendoring is needed. Thin LTO also works identically.

## Findings
### Q: Does `-C lto=fat` resolve Executor::spawn and Executor::poll extern calls?
A: **Yes, completely.** With `lto = "fat"` in `[profile.release]`, both `Executor::spawn` and `Executor::poll` are inlined into the kernel entry point. The spawn logic appears as an inline CAS loop with `fence.sc.sys` + `atom.acquire.global.cas.b64`, and the poll logic appears as an `atom.global.exch.b64` drain loop with indirect function pointer calls via `prototype_4`. Additionally, `panic_fmt`, `panic_const_async_fn_resumed`, and the Embassy ARENA storage are all resolved (defined with bodies, not extern). The only remaining `.extern` is `__pender`, which is Embassy's user-supplied waker callback.

Important: LTO must be set via `Cargo.toml` `[profile.release] lto = "fat"`, NOT via `-C lto=fat` rustc flag (the latter conflicts with Cargo's `-C embed-bitcode=no` default).

**Confidence**: high

### Q: If not, does vendoring Embassy source into gpu-kernel work?
A: **Not needed.** Fat LTO resolves everything. Vendoring is unnecessary.

**Confidence**: high

### Q: What is the minimal fork needed (which functions need `#[inline(always)]`)?
A: **No fork needed at all.** Fat LTO eliminates the cross-crate boundary entirely. All Embassy functions are merged into a single codegen unit and inlined/defined as needed. The only user-supplied symbol is `__pender`, which is part of Embassy's intended API (not a missing inline).

**Confidence**: high

## Compilation Attempts

### Round 1: No LTO (baseline)
- Command: `cargo +nightly rustc --target nvptx64-nvidia-cuda -Zbuild-std=core --release -- --emit=asm -C linker=echo -C target-cpu=sm_86`
- Cargo.toml: `lto = false`
- Result: **Compiles successfully** (254 lines PTX)
- PTX analysis: 4 `.extern .func` declarations:
  - `_ZN16embassy_executor3raw8Executor5spawn...` (Executor::spawn)
  - `_ZN16embassy_executor3raw8Executor4poll...` (Executor::poll)
  - `_ZN4core9panicking9panic_fmt...` (panic_fmt)
  - `_ZN4core9panicking11panic_const28panic_const_async_fn_resumed...`
- Also: `.extern .global .align 8 .b8 ...ARENA...` (embassy arena storage)

### Round 2: `-C lto=fat` via rustc flag
- Command: same with `-C lto=fat` appended
- Result: **ERROR** — `options -C embed-bitcode=no and -C lto are incompatible`
- Root cause: Cargo passes `-C embed-bitcode=no` by default when `lto = false` in profile. Must set LTO in Cargo.toml instead.

### Round 3: Fat LTO via Cargo profile
- Cargo.toml: `lto = "fat"`
- Command: same as Round 1 (no extra `-C lto` flag)
- Result: **Compiles successfully** (287 lines PTX)
- PTX analysis: **Only 1 `.extern`** remaining: `__pender` (Embassy's user-supplied waker callback)
  - `Executor::spawn` — RESOLVED (inlined as CAS loop at lines 191-211)
  - `Executor::poll` — RESOLVED (inlined as exch+drain loop at lines 213-228)
  - `panic_fmt` — RESOLVED (defined as infinite loop, matches `#[panic_handler]`)
  - `panic_const_async_fn_resumed` — RESOLVED (defined, calls panic_fmt)
  - `ARENA` — RESOLVED (defined as `.global .align 8 .b8 ...ARENA...[4104] = {}`)
  - `critical_section_acquire/release` — RESOLVED (defined as no-op `.visible .func`)

### Round 4: Thin LTO via Cargo profile
- Cargo.toml: `lto = "thin"`
- Result: **Same as fat LTO** — only `__pender` remains extern
- PTX size: 286 lines (nearly identical to fat)

### Round 5: Fat LTO + `__pender` provided
- Added `#[no_mangle] unsafe extern "C" fn __pender(_: *mut ()) {}` to source
- Result: **Zero `.extern` declarations** — fully self-contained PTX (283 lines)
- This is the complete solution: no unresolved symbols whatsoever

## Unexpected Discoveries
1. **Both fat and thin LTO work identically** for this use case. Since nvptx64 has no linker anyway, the distinction between fat/thin is moot — both merge all codegen units into a single PTX module.
2. **LTO must be set in Cargo.toml, not as a rustc flag.** The `-C lto=fat` flag conflicts with Cargo's default `-C embed-bitcode=no`. This is a common gotcha.
3. **Embassy's `__pender` is the only user-supplied symbol.** This is by design — Embassy requires the user to define how the executor gets woken up. On GPU, a no-op is correct for synchronous polling.
4. **The PTX output with LTO is actually slightly larger** (283 vs 254 lines) because the previously-extern functions now have inline bodies. But this is correct — the code was always needed, it was just missing before.
5. **Indirect function call for task polling works in PTX.** The poll loop uses `prototype_4 : .callprototype ()_ (.param .b64 _)` for indirect dispatch — this is the mechanism Embassy uses to call task-specific poll functions via stored function pointers.
6. **No Embassy fork or vendoring is needed at all.** This was the best-case scenario and significantly de-risks the async-runtime theme.

## Open Questions
1. **Does the indirect `call %rd34, ..., prototype_4` actually work on GPU hardware?** PTX supports indirect calls, but CUDA may have restrictions on virtual function dispatch. This needs runtime testing.
2. **Is `atom.acquire.global.cas.b64` correct for Embassy's internal task queue?** Embassy uses `AtomicPtr` for its run queue — the `.global` scope might need to be `.sys` scope if the executor spans CPU-GPU boundary. For a pure-GPU executor this should be fine.
3. **Register pressure?** The kernel uses `%rd<42>` (42 64-bit registers) + `%r<3>` + `%p<10>`. This is moderate but should be monitored as task complexity grows.

## Impact on Downstream Tasks
- **async-runtime.1.3** (critical-section provider): Confirmed needed — the no-op implementation works for compilation but the runtime behavior needs validation. The critical-section functions are inlined by LTO, so there is zero overhead for the no-op case.
- **async-runtime.2** (executor design): Massively simplified. The approach is: (1) add `lto = "fat"` to gpu-kernel Cargo.toml, (2) add embassy-executor dependency, (3) provide `__pender` and critical-section impl, (4) done. No fork, no vendoring.
- **gpu-kernel crate**: Needs `lto = "fat"` in its `[profile.release]` section.
- **All future cross-crate GPU code**: LTO is the general solution for nvptx64's lack of a linker. This finding applies beyond Embassy.
