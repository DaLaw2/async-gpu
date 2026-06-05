# gen-mono.1: PTX Monomorphization for Generic Kernel Functions

**Status**: Complete  
**Kind**: Investigation  
**Cycle**: 625  

## Summary

Monomorphization for nvptx64 works **identically** to standard Rust targets.
Generic functions, structs, traits, and trait methods are all monomorphized
at the MIR→LLVM IR boundary — the nvptx64 backend receives fully concrete
LLVM IR and emits type-specialized PTX. The patched rustc does not modify
monomorphization behavior; its changes are limited to the `warp_cooperative`
MIR transform and std PAL patches.

## Question 1: How does monomorphization work for nvptx64?

### Answer: Standard Rust monomorphization, no nvptx64-specific behavior

The pipeline is: `Rust source → MIR (monomorphized) → LLVM IR → PTX`

Monomorphization happens at the MIR level, before any target-specific code
generation. By the time LLVM sees the IR, all generic parameters have been
replaced with concrete types. The nvptx64 LLVM backend then emits PTX with
the appropriate type-specific instructions.

**Evidence from PTX output** — a single `generic_add<T>` function produced
three distinct PTX functions:

| Rust type | PTX symbol (demangled)          | PTX instruction | Register width |
|-----------|---------------------------------|-----------------|----------------|
| `f32`     | `generic_add::<f32>`            | `add.rn.f32`    | `.b32`         |
| `u32`     | `generic_add::<u32>`            | `add.s32`       | `.b32`         |
| `i64`     | `generic_add::<i64>`            | `add.s64`       | `.b64`         |

The compiler even applies type-specific optimizations: `apply_ops::<u32>`
compiled `x.double()` (i.e., `x + x`) to `shl.b32 %r2, %r1, 1` (left shift
by 1), while `apply_ops::<f32>` used `add.rn.f32 %r2, %r1, %r1`.

### Patched rustc impact

The 4 rustc patches (`rustc_feature`, `rustc_mir_transform`, `rustc_passes`,
`rustc_span`) add:
- `warp_cooperative` symbol to the compiler's symbol table
- `WarpCooperativeTransform` MIR pass (only active on nvptx64 for annotated functions)
- Attribute validation for `#[warp_cooperative]`

None of these modify monomorphization. The MIR pass runs after monomorphization
in the pipeline (`RuntimePhase::Initial`), so it operates on already-concrete functions.

## Question 2: Can user-defined generic KERNEL ENTRY POINTS work?

### Answer: Not directly. Use concrete wrappers with generic bodies.

**PTX `.entry` points must be concrete**. A `extern "gpu-kernel"` function
annotated with `#[no_mangle]` must produce a single, deterministic symbol name.
Generic functions produce type-mangled symbols per instantiation, which conflicts
with `#[no_mangle]`.

**What happens if you try**: The compiler emits warnings
(`improper_gpu_kernel_arg`) for generic pointer types like `*const T` in
gpu-kernel ABI, and the function is **never emitted** to PTX because it has
no concrete instantiation. Even if instantiated, all monomorphizations would
clash on the `#[no_mangle]` symbol name.

**The correct pattern** (verified working):

```rust
// Concrete entry point — appears as .entry in PTX
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn kernel_f32(input: *const f32, output: *mut f32, n: u32) {
    generic_kernel_body(input, output, n);
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn kernel_u32(input: *const u32, output: *mut u32, n: u32) {
    generic_kernel_body(input, output, n);
}

// Generic body — monomorphized per-type, inlined into entries
#[inline(always)]
fn generic_kernel_body<T: Copy + Add<Output = T>>(input: *const T, output: *mut T, n: u32) {
    let tid = global_thread_idx();
    let stride = global_thread_count();
    let mut i = tid as usize;
    while i < n as usize {
        let x = unsafe { core::ptr::read(input.add(i)) };
        unsafe { core::ptr::write(output.add(i), x + x) };
        i += stride as usize;
    }
}
```

**PTX verification**: Both entries emitted as `.visible .entry` with correct
parameter types. With `#[inline(always)]`, the generic body was fully inlined.
With `#[inline(never)]`, it was emitted as a separate `.func` per monomorphization.

## Question 3: What are the limitations?

### dyn Trait (dynamic dispatch): WORKS

**Confirmed working on nvptx64.** The compiler:
1. Emits vtables in `.global` memory (e.g., `{drop_fn, size, align, method_fn}`)
2. Uses PTX indirect function calls via `.callprototype` for dynamic dispatch
3. Supports `&dyn Trait`, `Box<dyn Trait>`, and `Vec<Box<dyn Trait>>`

PTX for `call_dyn(&dyn DynTestTrait)`:
```ptx
ld.b64 %rd3, [%rd2+24]            // load fn ptr from vtable
prototype_522 : .callprototype (.param .b32 _) _ (.param .b64 _);
call (retval0), %rd3, (param0), prototype_522;  // indirect call
```

**Performance caveat**: Dynamic dispatch on GPU is expensive because:
- Indirect calls prevent warp-level instruction scheduling optimization
- All threads in a warp must call the same function for SIMT efficiency
- If threads have different vtable targets, they serialize (warp divergence)

**Recommendation**: Prefer monomorphization (static dispatch) for GPU code.
Use `dyn Trait` only when truly needed (heterogeneous collections with
unknown-at-compile-time types).

### Recursive generics: LIMITED by GPU stack

GPU threads have limited stack space (default ~1KB per thread, configurable
up to ~16KB). Deeply recursive generic code (e.g., recursive data structures,
recursive trait implementations) can overflow the GPU stack. This is not a
monomorphization issue — it's a runtime stack depth issue.

Tail recursion optimization by LLVM can help, but is not guaranteed.

### Associated types: WORK

Associated types are resolved at compile time during monomorphization,
so they work identically on GPU. The `GpuParallelIterator` trait already
uses `type Item` extensively and compiles successfully.

### Const generics: WORK

The existing `block_mpsc<'scope, T: Copy, const N: usize>` in the codebase
already proves const generics compile to PTX correctly. Each const generic
instantiation produces a separate monomorphized function.

## Question 4: Practical test results

### Test setup

Added to `gpu-kernel-test/src/lib.rs`:
- `generic_add<T: Copy + Add<Output=T>>` — basic generic function
- `generic_mul<T: Copy + Mul<Output=T>>` — basic generic function
- `Accumulator<T>` — generic struct with methods
- `NumericOps` trait with `impl` for `f32`, `u32`, `i64`
- `apply_ops<T: NumericOps>` — trait-bound generic function
- Concrete kernel entries calling `generic_kernel_body<T>`
- `dyn Trait` test with vtable-based dispatch

### Compilation results

All tests compiled successfully with `cargo build --release` (nvptx64 target).
Build time: ~30s (normal for this crate).

### PTX symbol inventory

8 monomorphized functions emitted:

```
generic_add::<f32>    → add.rn.f32
generic_add::<u32>    → add.s32
generic_add::<i64>    → add.s64
generic_mul::<f32>    → mul.rn.f32
generic_mul::<u32>    → mul.lo.s32
apply_ops::<f32>      → add.rn.f32 + mul.rn.f32 + call generic_add::<f32>
apply_ops::<u32>      → shl.b32 + mul.lo.s32 + call generic_add::<u32>
apply_ops::<i64>      → add.s64 + mul.lo.s64 + call generic_add::<i64>
```

### Type-specific optimizations observed

The LLVM nvptx backend applied target-specific optimizations to monomorphized code:
- `u32` doubling (`x + x`) → `shl.b32` (shift left by 1)
- `f32` operations use `.rn` (round-to-nearest) suffix
- `i64` operations correctly use 64-bit registers (`.b64`)

## Recommendations for gen-mono.2 (implementation)

### Pattern: Concrete entry + generic body

The recommended implementation pattern for user-facing generic kernels:

1. **Macro-generated entry points**: A proc macro or declarative macro that
   generates concrete `extern "gpu-kernel"` entries for each type:
   ```rust
   gpu_kernel! {
       fn double_kernel<T: Copy + Add>(input: &GpuSlice<T>, output: &mut GpuSliceMut<T>)
       where T in [f32, u32, i64];
   }
   // Expands to: double_kernel_f32, double_kernel_u32, double_kernel_i64
   ```

2. **Library-level generics are free**: The existing pattern in `gpu-runtime`
   (generic types like `GpuSlice<T>`, `GpuParIter<T>`, `Mutex<T>`) works
   perfectly. These get monomorphized when used in concrete kernel code.

3. **Avoid dyn Trait in hot paths**: Use static dispatch (generics + traits)
   for performance-critical GPU code. Reserve dyn Trait for control flow
   that isn't warp-sensitive.

4. **No special toolchain changes needed**: The existing nightly toolchain
   with the patched std handles all forms of generics correctly.

### What already works (no changes needed)

- Generic library types: `GpuSlice<T>`, `GpuParIter<T>`, `GpuMap<I, F>`, etc.
- Generic functions called from kernel code
- Trait implementations and trait-bound generics
- Const generics
- Associated types
- Dynamic dispatch (dyn Trait) with vtables in global memory

### What needs design work (gen-mono.2)

- Ergonomic macro for generating type-specific kernel entries
- Host-side type dispatch (mapping runtime type info to kernel name)
- Integration with the par_iter API for user-defined element types
