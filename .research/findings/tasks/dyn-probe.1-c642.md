# dyn-probe.1: Investigation — compile minimal &dyn Trait kernel to PTX, analyze LLVM IR
**Cycle**: 642 | **Theme**: dyn-probe | **Kind**: investigation | **Status**: done

## Summary
`&dyn Trait` compiles successfully to valid PTX on nvptx64 with zero LLVM warnings or errors.
The compiler emits real vtables in `.global` memory, converts pointers to generic address space
via `cvta.global.u64`, and performs indirect calls through register-based `call` instructions
with `.callprototype` declarations. The fat pointer layout matches expectations exactly
(data_ptr + vtable_ptr, each 64-bit).

## Findings

### Q1: Does `&dyn Trait` compile to valid PTX on nvptx64?
**A: Yes.** A minimal kernel with `trait Greeter`, two impls (`GreeterA` returning 42,
`GreeterB` returning 99), and a `call_greeter(&dyn Greeter)` function compiled to valid PTX
without any errors. The kernel entry point `test_gpu_dyn_trait` was emitted as a `.visible .entry`.

Build command: `cargo +nightly-2026-06-03 build --release` in `crates/kernel/gpu-kernel-test/`
with `-Zbuild-std=["std","core","panic_abort"]` and `target = nvptx64-nvidia-cuda`.

### Q2: What call mechanism does the PTX use for dynamic dispatch?
**A: Indirect call via register.** The `call_greeter` function emits:
```ptx
ld.b64      %rd3, [%rd2+24];          // load fn ptr from vtable[3] (offset 24 bytes)
prototype_523 : .callprototype (.param .b32 _) _ (.param .b64 _);
call (retval0), %rd3, (param0), prototype_523;  // indirect call via register
```
This is a register-based indirect call (NOT `call.uni` which is a direct call).
The `.callprototype` declaration tells PTX the signature of the function being called
indirectly: takes one `.b64` param (the `&self` data pointer), returns one `.b32` (the `u32`).

Note: `call.uni` IS used in the caller (the kernel entry point) to call `call_greeter` itself,
which is a direct/uniform call. The dynamic dispatch happens inside `call_greeter`.

### Q3: What address space does the vtable pointer end up in?
**A: `.global` (addrspace 1), converted to generic (addrspace 0) at point of use.**

Vtable declarations:
```ptx
.global .align 8 .u64 anon_..._78[4] = {0, 0, 1, <GreeterA::greet>};
.global .align 8 .u64 anon_..._79[4] = {0, 0, 1, <GreeterB::greet>};
```

At the call site, the vtable address is converted from global to generic:
```ptx
mov.b64         %rd45, anon_..._78;       // global address
cvta.global.u64 %rd46, %rd45;             // convert to generic address space
st.param.b64    [param1], %rd46;           // pass as vtable_ptr
```

Inside `call_greeter`, the vtable is read via generic address space:
```ptx
ld.b64 %rd3, [%rd2+24];   // generic address space load
```

This is correct — PTX generic pointers can access any address space.

### Q4: Are there any LLVM warnings or errors about unsupported features?
**A: None.** No warnings about dyn dispatch, vtables, indirect calls, or address space issues.
The only warnings were unrelated (unnecessary unsafe blocks, unused variable in a different kernel).
The LLVM NVPTX backend handled trait objects without any complaints.

### Q5: Does the fat pointer (&dyn = data_ptr + vtable_ptr) layout match expectations?
**A: Yes, exactly.** The `call_greeter` function takes two `.b64` parameters:
- `param_0` = data pointer (the `&self` pointer to the concrete struct)
- `param_1` = vtable pointer (pointer to the vtable array)

This matches the standard Rust fat pointer layout for trait objects: `(data_ptr, vtable_ptr)`.

### Vtable Layout Analysis
Each vtable is a 4-element `u64` array:
```
vtable[0] = drop_in_place fn ptr     (0 for ZSTs like GreeterA/B — no drop needed)
vtable[1] = size of type              (0 for ZSTs)
vtable[2] = alignment of type         (1)
vtable[3] = greet fn ptr              (pointer to the concrete impl)
```
This matches the standard Rust vtable layout: `[drop, size, align, method0, method1, ...]`.

### Impl Function PTX
Both `GreeterA::greet` and `GreeterB::greet` compiled to trivial functions:
```ptx
// GreeterA::greet
st.param.b32 [func_retval0], 42;
ret;

// GreeterB::greet
st.param.b32 [func_retval0], 99;
ret;
```

## Open Questions
1. **Runtime execution**: Does the indirect `call` instruction actually execute correctly on GPU hardware? The PTX compiled, but PTX-level correctness does not guarantee runtime success — ptxas or the GPU driver could reject indirect calls. This needs a runtime test (dyn-probe.2).
2. **Box<dyn Trait>**: Does heap-allocated trait objects work? The vtable is in `.global` but the data could be in local/shared memory via `Box`. Need to verify address space interactions.
3. **Performance**: What is the overhead of indirect calls vs direct calls on GPU? Indirect calls may disable certain GPU optimizations (instruction prefetch, warp-level scheduling).
4. **Multi-method traits**: Does a trait with multiple methods produce correct vtable offsets?
5. **Trait upcasting / nested dyn**: Do complex trait hierarchies compile?
