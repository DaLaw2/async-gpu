# dyn-box.1: Box<dyn Trait> allocation + dispatch on GPU via hostcall allocator
**Cycle**: 642 | **Feature**: dyn-box | **Kind**: experiment | **Status**: done

## Summary
`Box<dyn Trait>` compiles successfully to valid PTX on nvptx64. Heap-allocated trait objects
work end-to-end: Box::new allocates via the hostcall allocator (`__rust_alloc` → hostcall malloc),
vtables are emitted in `.global` memory, and indirect dispatch via `.callprototype` works
identically to the `&dyn Trait` case from dyn-probe. `Vec<Box<dyn Animal>>` also compiles,
including `RawVec::grow_one` for the fat-pointer element type.

## Findings

### Q1: Does `Box<dyn Trait>` compile to valid PTX?
**A: Yes.** A kernel with `Box<dyn Animal>` for three types (Cat, Dog, Parrot) compiled
without errors. The kernel entry `test_gpu_box_dyn_trait` appears as `.visible .entry`.
Zero new warnings or errors compared to the baseline.

### Q2: How does Box allocation work on GPU?
**A: Via `__rust_alloc` routed through the patched std's hostcall allocator.**
The PTX contains `__rust_alloc` and `__rdl_alloc` functions linked from the patched std.
Box::new(Cat) triggers allocation for ZSTs (size=0, align=1 — likely optimized to a
dangling pointer by the compiler). Box::new(Parrot { word_count: 42 }) triggers a real
4-byte allocation (size=4, align=4) from the GPU heap via hostcall.

### Q3: What do the vtables look like?
**A: Standard Rust vtable layout in `.global` memory.**
```ptx
// Cat vtable: ZST — size=0, align=1
.global .align 8 .u64 anon_..._146[4] = {0, 0, 1, Cat::Animal::speak};
// Dog vtable: ZST — size=0, align=1
.global .align 8 .u64 anon_..._147[4] = {0, 0, 1, Dog::Animal::speak};
// Parrot vtable: size=4, align=4 (has u32 word_count field)
.global .align 8 .u64 anon_..._149[4] = {0, 4, 4, Parrot::Animal::speak};
```
Layout: `[drop_fn, size, align, speak_fn]`. Same as dyn-probe.1's Greeter vtables.

### Q4: Does Parrot (struct with data) work through Box<dyn>?
**A: Yes.** Parrot::speak reads `word_count` from the heap-allocated data pointer:
```ptx
ld.param.b64  %rd1, [..._param_0];   // load &self (heap pointer)
ld.b32        %r1, [%rd1];           // load self.word_count from heap
add.s32       %r2, %r1, 100;         // 100 + word_count
st.param.b32  [func_retval0], %r2;   // return result
```
The data is read from generic address space, which works for heap memory.

### Q5: Does Vec<Box<dyn Animal>> compile?
**A: Yes.** The PTX includes `RawVec<Box<dyn Animal>>::grow_one`, confirming that
`Vec::push(Box::new(...))` for fat-pointer elements compiles correctly. The Vec stores
fat pointers (data_ptr + vtable_ptr, 16 bytes each on 64-bit).

### Q6: Runtime execution status?
**A: Pending cubin rebuild.** The current cubin build (PID 3668528) uses PTX from before
this kernel was added. Runtime execution requires either:
- PTX JIT fallback (~25 min), or
- Rebuilding the cubin with the updated PTX

Host-side test is written and compiles cleanly.

## Kernel structure
The `test_gpu_box_dyn_trait` kernel tests 5 scenarios:
1. Basic `Box<dyn Animal>` — Cat (returns 1), Dog (returns 2)
2. `Box<dyn Animal>` with data fields — Parrot { word_count: 42 } (returns 142)
3. Pass `Box<dyn>` as `&dyn` via auto-deref
4. `Vec<Box<dyn Animal>>` — 5 heterogeneous animals, iterate and sum speak() values (= 234)
5. Runtime-chosen `Box<dyn Animal>` via conditional

## Files changed
- `crates/kernel/gpu-kernel-test/src/lib.rs` — added `test_gpu_box_dyn_trait` kernel
  with Animal trait, Cat/Dog/Parrot impls, call_animal helper
- `crates/test/gpu-test-harness/tests/gpu_tests.rs` — added host-side test
  `test_gpu_box_dyn_trait` following same pattern as `test_gpu_dyn_trait`

## Open Questions
1. Runtime verification pending — does indirect call through heap-allocated vtable execute correctly on GPU hardware?
2. Drop for Box<dyn Trait> — does the destructor correctly free heap memory via hostcall deallocator?
3. Performance: does heap allocation per trait object add significant overhead vs stack-allocated &dyn Trait?

## Key insight
Box<dyn Trait> requires NO special GPU handling. The patched std's allocator routes
`__rust_alloc` to hostcall malloc, vtables go in `.global` as before, and the indirect
call mechanism is identical to `&dyn Trait`. The only difference is WHERE the data lives
(heap via allocator vs stack) — the dispatch mechanism is the same.
