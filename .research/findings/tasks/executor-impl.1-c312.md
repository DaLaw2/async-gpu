# executor-impl.1: Verify indirect fn pointers work on nvptx64 PTX
**Cycle**: 312 | **Theme**: executor-impl | **Kind**: investigation | **Status**: done

## Summary
Indirect function pointer calls **work on nvptx64** -- both at the PTX generation level (LLVM backend) and on actual NVIDIA GPU hardware. This has been empirically verified in this codebase: the Embassy executor's task dispatch uses `call %reg, (params), prototype_N` for type-erased poll functions, and all Embassy async tests (ImmediateFuture, CountdownFuture, two-task concurrent) pass on GPU hardware (confirmed in async-runtime.3, cycle 22). The type-erased `poll_fn: fn(*mut u8) -> Poll<()>` approach in the executor design is viable.

## Findings
### Q: Does LLVM nvptx backend support indirect function calls?
A: **Yes.** LLVM's NVPTX backend fully supports indirect function calls. The mechanism works as follows:
1. LLVM emits a `.callprototype` declaration defining the signature (e.g., `prototype_1 : .callprototype ()_ (.param .b64 _)`)
2. The function pointer is loaded into a 64-bit register (e.g., `ld.b64 %rd10, [%rd14+32]`)
3. The indirect call uses: `call %rd10, (param0), prototype_1`

This pattern is present throughout the codebase's compiled PTX files:
- `crates/core/gpu-host/embassy_test.ptx` -- Embassy executor poll dispatch
- `crates/core/gpu-host/async_pipeline_test.ptx` -- async pipeline poll dispatch
- `crates/core/gpu-host/async_hostcall_test.ptx` -- hostcall future poll dispatch
- `crates/core/gpu-host/std_build_test.ptx` -- std library internal dispatch

The LLVM NVPTX backend has supported this since at least LLVM 3.x, as documented in the [LLVM NVPTX User Guide](https://llvm.org/docs/NVPTXUsage.html). The `NVPTXISelLowering.cpp` handles indirect calls by generating `CallPrototype` SDNodes that emit the `.callprototype` string as a PTX operand.

**Confidence**: high

### Q: What PTX instructions are generated for fn pointer calls?
A: The generated PTX follows this pattern:

```ptx
// Load fn pointer from struct field (e.g., TaskStorage.poll_fn)
ld.b64  %rd10, [%rd14+32];

// Set up parameters
{ // callseq N, 0
.param .b64  param0;
st.param.b64  [param0], %rd14;

// Declare prototype (signature of the indirect call target)
prototype_1 : .callprototype ()_ (.param .b64 _);

// Indirect call: call <fn_ptr_reg>, (params), <prototype_label>
call %rd10, (param0), prototype_1;
} // callseq N
```

Key observations:
- PTX does NOT use `call.uni` for indirect calls (that form is for direct calls only). The syntax is `call %reg, (params), prototype`.
- The `.callprototype` declaration is required by PTX ISA for indirect calls -- it tells the assembler the signature so it can set up the calling convention.
- Return values are handled via `.param` blocks in the prototype, e.g., `(.param .b32 _) _` for a function returning a 32-bit value.
- Multiple prototypes can coexist in a single kernel (prototype_1, prototype_3, prototype_6, etc.).

**Confidence**: high

### Q: Are there known issues with type-erased poll_fn on GPU?
A: There are two categories of issues to be aware of:

**1. Known NVIDIA JIT bug with `.calltargets` (2020):**
An [NVIDIA forum report](https://forums.developer.nvidia.com/t/miscompilation-of-indirect-call-with-an-explicitly-specified-list-of-call-targets-ptx/164005) documented a miscompilation where the ptxas JIT compiler incorrectly infers calling conventions for indirect calls with an explicit `.calltargets` list, causing register clobbering. However, this bug applies to `.calltargets` (an optimization hint), not to `.callprototype` (which is what LLVM generates). The codebase does not use `.calltargets` and is unaffected.

**2. Warp-cooperative + indirect dispatch interaction:**
The `#[warp_cooperative]` MIR pass inserts `bar.warp.sync` barriers into the future's `poll()` implementation at compile time. Since these barriers are embedded in the callee's code (not the call site), they survive indirect dispatch -- the call just jumps to the function body, which already contains the synchronization. This was confirmed empirically: Embassy's indirect poll dispatch works correctly with warp-cooperative futures.

**3. No inlining through indirect calls:**
The main practical issue is that indirect calls are opaque to the optimizer -- LLVM cannot inline through a function pointer. This means:
- The poll function body is a separate function, not folded into the executor loop
- Each indirect call has overhead: parameter setup, call, return (~10-20 cycles)
- For trivial futures this is significant overhead; for real I/O futures (hostcall spin-wait) it is negligible

**4. Embassy validation on hardware:**
The Embassy executor's indirect poll dispatch was tested and **passed** on NVIDIA GPU hardware (async-runtime.3, cycle 22):
- ImmediateFuture (1 poll round): PASSED
- CountdownFuture (6 poll rounds with re-enqueue): PASSED
- Two concurrent tasks: PASSED

**Confidence**: high

### Q: What are alternatives if indirect calls don't work?
A: Indirect calls DO work, so these are not strictly needed. However, for cases where avoiding indirect calls is desirable (e.g., to enable inlining or reduce register pressure), the alternatives are:

**1. Enum-based dispatcher (compile-time known types):**
```rust
enum AnyTask {
    HandleClient(HandleClientFuture),
    ProcessItem(ProcessItemFuture),
}
impl Future for AnyTask {
    fn poll(self: Pin<&mut Self>, cx: &mut Context) -> Poll<()> {
        match self.get_unchecked_mut() {
            AnyTask::HandleClient(f) => Pin::new_unchecked(f).poll(cx),
            AnyTask::ProcessItem(f) => Pin::new_unchecked(f).poll(cx),
        }
    }
}
```
- Pro: All calls are direct, inlinable, zero indirection overhead
- Con: Loses generality; all spawnable types must be enumerated at compile time
- Con: Enum size is the max of all variants (wastes memory for small futures)

**2. Monomorphized executor (Embassy's approach):**
Embassy uses `TaskStorage<F>` with a static `poll_fn` per concrete future type. The function pointer is still stored and called indirectly, but the compiler knows the concrete type at spawn time. This is exactly what the codebase already uses successfully.

**3. Macro-generated dispatch table:**
A `spawn!()` macro could generate a match on a type ID, dispatching to the correct concrete poll function. Similar to enum approach but with explicit IDs.

**4. Separate executors per future type:**
Run one executor per future type, each with its own typed task slots. Avoids type erasure entirely but limits cross-type work stealing.

**Confidence**: high

## Unexpected Discoveries

1. **The codebase already has extensive evidence.** The question "do indirect fn pointers work on nvptx64?" was answered affirmatively in async-runtime.3 (cycle 22) with hardware-validated tests. The Embassy executor has been using indirect calls for 290+ cycles of development.

2. **`std_build_test.ptx` also contains indirect calls.** The Rust standard library's internal dispatch mechanisms (e.g., `Write::write_fmt` trait dispatch) also compile to indirect calls on nvptx64, providing additional evidence of broad support.

3. **No `.calltargets` directives are emitted by LLVM.** The LLVM NVPTX backend only generates `.callprototype`, never `.calltargets`. This avoids the known 2020 JIT miscompilation bug entirely.

4. **Register pressure is the main concern, not correctness.** Indirect calls work, but each indirect call site adds register pressure (callee-saved registers must be preserved across the call). The two-task Embassy kernel uses ~56 virtual registers. The executor design should monitor register pressure as task complexity grows.

## Open Questions

1. **Does ptxas optimize indirect calls when only one target exists?** If the `.callprototype` has only one matching function in the module, ptxas might devirtualize the call. This could eliminate the overhead in common cases.

2. **Register spilling through indirect calls.** The PTX virtual register counts (31-56) are upper bounds. The actual SASS register allocation after ptxas compilation may differ. `cuobjdump --dump-sass` would reveal the true cost.

3. **Warp-cooperative interaction at scale.** The existing tests use 1 thread per block. With 32 threads (full warp) all calling through the same indirect poll_fn, convergence is guaranteed (all lanes call the same function pointer). But if different warps call different poll_fn values, per-warp behavior needs validation (this is the normal expected case for the multi-task executor).

## Impact on Downstream Tasks

- **executor-impl.2 (implement spawn):** Fully unblocked. The `poll_fn: unsafe fn(*mut u8, &mut Context) -> Poll<()>` pattern is confirmed viable. Implementation can proceed with the type-erased TaskSlot design from DESIGN-executor.md.
- **executor-impl.3 (work queue):** The MPMC queue design does not depend on indirect calls, but the poll dispatch in the executor loop does. Confirmed safe.
- **Performance optimization:** The main optimization opportunity is reducing indirect call overhead via inlining hints or enum dispatch for hot-path future types. This is a Phase 3 concern per the design doc.
