# async-runtime.1.1: Embassy executor compilation for nvptx64
**Cycle**: 12 | **Theme**: async-runtime | **Kind**: experiment | **Status**: done

## Summary
Embassy-executor (both v0.7 and v0.9) with `arch-spin` feature compiles cleanly for nvptx64-nvidia-cuda with `-Zbuild-std=core`. No source modifications required. The generated PTX contains correct atomic CAS, fences, and the full executor spawn/poll codepath. Two gaps remain: (1) `critical-section` extern symbols need a GPU implementation, and (2) cross-crate functions (Executor::spawn, Executor::poll) appear as unresolved `.extern` calls due to the nvptx64 linker limitation (no PTX linker exists).

## Findings
### Q: Does embassy-executor with arch-spin compile for nvptx64?
A: **Yes**, both v0.7 and v0.9 compile without errors. The `arch-spin` feature avoids platform-specific atomics issues. The entire executor machinery — task creation via `#[embassy_executor::task]` macro, `Executor::new()`, `Spawner`, `SpawnToken`, and `Executor::poll()` — all compile to valid PTX assembly targeting sm_86.
**Confidence**: high

### Q: What specific errors occur?
A: **No compilation errors.** The build succeeds cleanly. The only concerns are runtime-level:
1. `_critical_section_1_0_acquire` / `_critical_section_1_0_release` are extern in v0.7 (need a critical-section implementation for GPU). In v0.9, these appear to be inlined/removed.
2. `Executor::spawn` and `Executor::poll` appear as `.extern .func` (not inlined) — this is the known nvptx64 cross-crate linking issue. These would need `#[inline(always)]` in a fork, or the functions must be in the same codegen unit.
3. `core::panicking::panic_fmt` and `panic_const_async_fn_resumed` are extern — standard panic infrastructure that needs a GPU implementation (or `panic = "abort"` + trap).
**Confidence**: high

### Q: Can errors be worked around without modifying Embassy?
A: Partially. Since there are no compilation errors, the question shifts to runtime linking:
- **critical-section**: Can be provided externally via a `critical-section` implementation crate (e.g., disable-interrupts or a custom GPU spin-lock). No Embassy changes needed.
- **Cross-crate inlining**: This is the main issue. `Executor::spawn()` and `Executor::poll()` are not `#[inline(always)]`, so they become unresolved extern calls in PTX. Workarounds without forking:
  - Use LTO (`-C lto=fat`) to merge codegen units — **likely works**.
  - Vendor Embassy source directly into the GPU crate — works but ugly.
  - If LTO doesn't resolve it, a minimal fork adding `#[inline(always)]` to ~3 functions is needed.
**Confidence**: medium (LTO workaround needs verification)

### Q: What minimal Embassy fork changes are needed?
A: If LTO doesn't resolve cross-crate calls, the changes are minimal:
1. Add `#[inline(always)]` to `Executor::spawn()`, `Executor::poll()`, and any other cross-crate public methods.
2. Provide a `critical-section` implementation for GPU (this is external to Embassy, not a fork change).
3. No other changes needed — no TLS, no AtomicU8, no vtable issues detected.

Estimated fork diff: ~5-10 lines changed (adding inline attributes).
**Confidence**: high

## Compilation Attempts
### Round 1
- Command: `cargo +nightly rustc --target nvptx64-nvidia-cuda -Zbuild-std=core --release -- --emit=asm -C linker=echo -C target-cpu=sm_86`
- Errors: Workspace membership conflict (embassy-test found parent workspace)
- Fix: Added `[workspace]` table to embassy-test/Cargo.toml to make it standalone

### Round 2
- Command: Same
- Errors: `Executor` type not found at `embassy_executor::Executor` (it's at `embassy_executor::raw::Executor`)
- Fix: Changed import to `embassy_executor::raw::Executor`

### Round 3
- Command: Same
- Errors: `Executor::new()` requires a `*mut ()` context argument; `Executor` is not `Sync` (contains `*mut ()`)
- Fix: Simplified test to use local variable instead of static, pass `ptr::null_mut()` to `new()`

### Round 4 (success with v0.7)
- Command: Same
- Errors: None — **clean compilation**
- Generated PTX: 355 lines, contains `atom.acq_rel.cas.b32`, `fence.sc.sys`, critical-section calls, full executor spawn+poll codepath
- Extern symbols: `_critical_section_1_0_acquire`, `_critical_section_1_0_release`, `Executor::spawn`, `Executor::poll`, panic functions

### Round 5 (success with v0.9)
- Command: Same, with `embassy-executor = "0.9"`
- Errors: None — **clean compilation**
- Generated PTX: 257 lines (more compact)
- Extern symbols: `Executor::spawn`, `Executor::poll`, panic functions (critical-section no longer extern)
- Notable: v0.9 appears to have improved the critical-section usage

## Unexpected Discoveries
1. **Embassy compiles for GPU out of the box** — This was better than expected. The `arch-spin` feature was designed for bare-metal/embedded, but it maps perfectly to GPU's similar constraints (no OS, no std, spin-based synchronization).
2. **v0.9 is more compact** — The v0.9 PTX is ~28% smaller (257 vs 355 lines), suggesting improved code generation or reduced critical-section overhead.
3. **Atomics work correctly** — PTX output shows proper `atom.acq_rel.cas.b32` and `fence.sc.sys` instructions, confirming that Rust's atomic operations translate correctly to GPU atomics.
4. **The `#[embassy_executor::task]` proc macro works** — Task pool allocation, spawn token creation, and the entire macro-generated task infrastructure compiles for nvptx64.
5. **No vtable dispatch issues** — The executor's function-pointer-based poll dispatch (`st.b64 [%rd27+32], %rd21` storing poll fn pointer) works fine in PTX.

## Open Questions
1. **Does LTO resolve the extern cross-crate calls?** Adding `-C lto=fat` to the build might inline `Executor::spawn` and `Executor::poll`, avoiding the need for a fork.
2. **What critical-section implementation to use on GPU?** Options: (a) disable preemption (GPU threads aren't preempted anyway), (b) warp-level spin lock, (c) no-op if single-warp executor.
3. **Will the executor actually *work* at runtime?** Compilation success doesn't guarantee correctness — the AtomicPtr-based task queue needs to function correctly under GPU's memory model (weaker than x86).
4. **Per-warp vs per-block vs per-grid executor?** The current design assumes a single executor instance. GPU parallelism may need one executor per warp or per block.

## Impact on Downstream Tasks
- **async-runtime.2** (executor port): Massively de-risked. Embassy compiles as-is; the port is primarily about providing critical-section and resolving cross-crate inlining.
- **hostcall theme**: The executor can drive async hostcall futures once the hostcall protocol (hostcall.3 design) is implemented as an async interface.
- **gpu-std theme**: `println!` and I/O can be implemented as async operations polled by this executor.
- **New task needed**: Test LTO cross-crate resolution (`async-runtime.1.2` or similar).
- **New task needed**: Implement GPU critical-section provider.
