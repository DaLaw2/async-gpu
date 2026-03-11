# async-runtime.1.3: GPU critical-section provider for Embassy
**Cycle**: 19 | **Theme**: async-runtime | **Kind**: experiment | **Status**: done

## Summary
Created `gpu-critical-section` crate providing a no-op `critical_section::Impl` for nvptx64. Combined with Embassy executor v0.7 (`arch-spin` feature) and fat LTO, the full executor spawn+poll path compiles to self-contained PTX with zero unresolved externs. The no-op implementation is correct because Embassy on nvptx64 uses atomic-based run queue and state (not critical-section-based), so acquire/release are defined but never called in the hot path.

## Findings
### Q: Can a no-op critical-section work for single-warp executor?
A: **Yes.** The no-op critical-section is fully correct for per-thread Embassy executors on GPU. Embassy v0.7 with `arch-spin` uses `run_queue_atomics.rs` (gated on `target_has_atomic = "ptr"`) and `state_atomics.rs` (gated on `target_has_atomic = "8"`) — both conditions are true on nvptx64. The critical-section functions are compiled into the PTX (satisfying the symbol dependency) but are never called in the executor's spawn or poll paths. The no-op is not just a shortcut — it is the architecturally correct choice for per-thread executors with no shared mutable state.

**Confidence**: high

### Q: Does a spin-lock critical-section work for multi-warp?
A: **Not needed for the current design.** Since Embassy uses atomics (not critical sections) for its run queue on nvptx64, a spin-lock critical-section would not change the executor's behavior. If a future use case requires multi-warp shared state protected by `critical_section::with()`, a spin-lock variant using `gpu-atomics` CAS could be implemented. However, ADR-4 specifies per-thread executors, so this is not on the critical path.

**Confidence**: high

### Q: What is the register pressure impact?
A: **Minimal.** The no-op critical-section adds zero registers (just empty `ret;` functions). The full executor test (spawn + poll) uses:
- Poll-only kernel: `%p<3>`, `%r<2>`, `%rd<9>` (14 registers total)
- Spawn+poll kernel: `%p<7>`, `%r<3>`, `%rd<21>` (31 registers total)

The 31-register count for spawn+poll is moderate and consistent with the async-runtime.1.2 finding of `%rd<42>` for the prior test (which used Embassy macros). Register pressure is driven by the executor's atomic CAS loops and indirect function dispatch, not by the critical-section implementation.

**Confidence**: high

## Compilation Attempts

### Round 1: gpu-critical-section standalone
- Command: `cargo +nightly build --release` (with `.cargo/config.toml` targeting nvptx64)
- Result: **Success** — compiles cleanly for nvptx64
- Time: 11.38s (including core rebuild)

### Round 2: embassy-test with executor-interrupt (error)
- Command: same
- Result: **Error** — `executor-interrupt` is not supported with `arch-spin`
- Fix: Removed `executor-interrupt` feature

### Round 3: embassy-test minimal (poll only, fat LTO)
- Command: `cargo +nightly rustc --release -- --emit=asm`
- Result: **Success** — 72 lines PTX
- PTX analysis: Zero `.extern` declarations. `_critical_section_1_0_acquire` and `_critical_section_1_0_release` defined as `ret;` (no-op). `Executor::poll` fully inlined.

### Round 4: embassy-test without LTO (baseline comparison)
- Cargo.toml: `lto = false`
- Result: Compiles but `Executor::poll` is `.extern .func` (unresolved)
- Note: Critical-section functions do NOT appear as extern even without LTO — confirms they are not referenced from the local crate's codegen unit

### Round 5: embassy-test with task spawn + poll (fat LTO)
- Cargo.toml: `lto = "fat"`
- Result: **Success** — 131 lines PTX, zero `.extern` declarations
- PTX analysis:
  - `_critical_section_1_0_acquire`: defined, no-op `ret;` — never called
  - `_critical_section_1_0_release`: defined, no-op `ret;` — never called
  - `TaskStorage::poll`: inlined (stores `poll_exited`, clears SPAWNED via `atom.and.b32`)
  - `Executor::poll`: inlined (`atom.global.exch.b64` drain loop + indirect call via `prototype_1`)
  - Task spawn: inlined (`atom.acq_rel.global.cas.b32` for state CAS, stores value 42 into future)
  - `write_in_place`: stores `42` to task future field
  - `__pender`: defined as `ret;`

## Unexpected Discoveries

1. **Critical-section is not on Embassy's hot path for nvptx64.** Embassy selects atomic-based run queue and state management when `target_has_atomic = "ptr"` and `target_has_atomic = "8"` are true. nvptx64 satisfies both, so the critical-section acquire/release symbols are defined but never called. This means the no-op implementation is not just "safe enough" — it is the formally correct choice (vacuously true).

2. **Embassy's ImmediateFuture task is optimized to near-zero overhead.** The `poll` function for a `Poll::Ready` future compiles to just two PTX instructions: store `poll_exited` function pointer + atomic AND to clear SPAWNED flag. The future's value (42) is stored during `write_in_place` at spawn time.

3. **Indirect function dispatch works in PTX with LTO.** The executor's poll loop uses `prototype_1 : .callprototype ()_ (.param .b64 _)` for indirect calls — this is how Embassy dispatches to task-specific poll functions. Combined with the async-runtime.1.2 finding, this confirms Embassy's function-pointer-based task dispatch is PTX-compatible.

4. **The `executor-interrupt` feature is incompatible with `arch-spin`.** Embassy enforces this at compile time. This is not a problem — `arch-spin` is the correct choice for GPU (no interrupt-based waking).

## Crates Created

### gpu-critical-section (`crates/gpu-critical-section/`)
- `#![no_std]` library crate (`rlib`)
- Depends on `critical-section = "1.2"`
- Implements `critical_section::Impl` with no-op acquire/release
- Uses `critical_section::set_impl!` macro
- Compiles for both nvptx64 and x86_64

### embassy-test (`crates/embassy-test/`)
- Test crate (`cdylib`) combining Embassy + gpu-critical-section
- Depends on `embassy-executor = "0.7"` with `arch-spin` feature
- Fat LTO enabled (`lto = "fat"`)
- Contains `embassy_test_kernel` entry point that spawns a task and polls it
- Produces fully self-contained PTX with zero unresolved externs

## Open Questions

1. **Does the indirect function call (`call %rd17, (param0), prototype_1`) actually execute correctly on GPU hardware?** PTX supports indirect calls, but CUDA may restrict virtual dispatch. This needs runtime validation (blocked on CUDA driver integration).

2. **Register pressure scaling**: The spawn+poll kernel uses 31 registers for a single trivial task. How does this scale with 4-8 async tasks with non-trivial futures (e.g., HostcallFuture with spin-wait)?

3. **Per-thread executor instantiation**: The test uses a single static executor. In production, each GPU thread needs its own executor. Stack-local allocation with lifetime extension (or per-thread global memory) needs to be designed.

## Impact on Downstream Tasks
- **async-runtime.2.1** (executor design rework): The critical-section provider is complete. The full Embassy stack (spawn + poll + critical-section + pender) compiles to self-contained PTX with fat LTO. The design can proceed with per-thread executor instantiation.
- **async-runtime.3** (minimal async/await on GPU): Unblocked. The test kernel pattern (MaybeUninit executor + spawner + poll) can serve as the template for runtime testing.
- **gpu-kernel crate**: Can add `embassy-executor` and `gpu-critical-section` as dependencies with `lto = "fat"` to get async support.
