# dyn-box.2: &dyn Fn() closures and Drop for Box<dyn Trait> on GPU
**Cycle**: 642 | **Feature**: dyn-box | **Kind**: experiment | **Status**: done

## Summary
`&dyn Fn()` closures (both simple and capturing) and `Box<dyn Fn()>` compile to valid PTX
on nvptx64. Drop for `Box<dyn Trait>` also compiles correctly: the vtable contains a
`drop_in_place` function pointer at slot 0, and dropping a `Box<dyn Trait>` dispatches
through it to call the correct type-specific destructor. `Vec<Box<dyn Droppable>>` with
heterogeneous types (HasDrop, HasDrop2) compiles, including correct per-type drop glue
dispatch.

## Findings

### Q1: Do &dyn Fn() closures compile to valid PTX?
**A: Yes.** Non-capturing closures (`|x| x + 1`, `|x| x * 2`) compile as ZST types
(size=0, align=1) with vtables in `.global` memory. The vtable layout for `dyn Fn(u32)->u32`
has 6 entries: `[drop_fn, size, align, call_once, call_fn, call_fn_mut]`.

### Q2: Do capturing closures work through &dyn Fn?
**A: Yes.** Closures that capture variables (`|x| x + offset`, `|x| x * factor`) compile
with correct size/align metadata in the vtable. Captured `u32` values produce closures
with size=8 (or size=4 for the `move` case). The closure's captured environment is stored
in the closure struct and passed as `&self` through the vtable call.

### Q3: Does Box<dyn Fn()> (heap-allocated closure) compile?
**A: Yes.** `Box::new(move |x| x + bias)` compiles correctly. The closure struct (containing
captured `bias: u32`) is heap-allocated via `__rust_alloc`, and the `Fn::call` method is
dispatched through the vtable's indirect call, same as `&dyn Fn`. The vtable shows size=4,
align=4 for this closure type.

### Q4: Does call_fn_dyn (calling through &dyn Fn indirectly) work?
**A: Yes.** The `call_fn_dyn` helper function (marked `#[inline(never)]`) that takes
`&dyn Fn(u32)->u32` compiles with a `.callprototype` for the indirect call. The PTX
shows 4 call sites to `call_fn_dyn`, matching the 4 uses in the kernel.

### Q5: Does Drop for Box<dyn Trait> compile correctly?
**A: Yes.** Two vtables are emitted for the `Droppable` trait:
```ptx
// HasDrop vtable: size=16, align=8 (val: u32 + drop_counter: *mut u32)
.global .align 8 .u64 anon_..._231[4] = {drop_glue_HasDrop, 16, 8, HasDrop::Droppable::value};
// HasDrop2 vtable: size=16, align=8 (val: u32 + drop_counter: *mut u32)
.global .align 8 .u64 anon_..._232[4] = {drop_glue_HasDrop2, 16, 8, HasDrop2::Droppable::value};
```
Each vtable's slot 0 points to the type's `drop_in_place` function. When `Box<dyn Droppable>`
is dropped, the compiler loads the drop function pointer from the vtable and calls it via
`.callprototype`.

### Q6: Does Vec<Box<dyn Droppable>> compile with heterogeneous drop?
**A: Yes.** The PTX includes `RawVec<Box<dyn Droppable>>::grow_one` for the Vec allocation.
Dropping the Vec iterates each element and dispatches the correct destructor via vtable —
HasDrop increments the counter by 1, HasDrop2 increments by 100.

### Q7: Fn trait vtable layout vs regular trait vtable?
**A: Fn vtables have 6 slots vs 4 for regular traits.**
- Regular trait: `[drop_fn, size, align, method_1, ...]`
- Fn trait: `[drop_fn, size, align, call_once, call_fn, call_fn_mut]`
The extra slots are for the three `Fn*` traits (FnOnce, Fn, FnMut).

## Kernel structure
The `test_gpu_dyn_fn_and_drop` kernel tests 5 scenarios:
1. Simple `&dyn Fn()` closures — add_one and double (no captures)
2. Capturing `&dyn Fn()` closures — add_captured(offset=10) and mul_captured(factor=7)
3. `Box<dyn Fn()>` — heap-allocated closure with captured `bias=5`
4. `Box<dyn Droppable>` single drop — HasDrop(val=42), verify drop_counter incremented
5. `Vec<Box<dyn Droppable>>` — heterogeneous drops (1 HasDrop + 2 HasDrop2), counter=201

## Files changed
- `crates/kernel/gpu-kernel-test/src/lib.rs` — added `test_gpu_dyn_fn_and_drop` kernel,
  Droppable trait, HasDrop/HasDrop2 impls, call_fn_dyn helper
- `crates/test/gpu-test-harness/tests/gpu_tests.rs` — added host-side test
  `test_gpu_dyn_fn_and_drop` with 8-element output buffer verification

## Open Questions
1. Runtime verification pending — does the drop counter actually increment on GPU hardware?
2. Performance: closure dispatch through &dyn Fn vs direct call overhead?
3. Does `FnMut` work through `&mut dyn FnMut`? (likely yes, same mechanism)

## Key insight
Closures are just structs that implement `Fn/FnMut/FnOnce` traits. On GPU, the vtable
mechanism is identical to any other trait — the compiler generates a `.callprototype` for
the indirect call and stores function pointers in `.global` memory. Capturing closures
simply have a non-zero-sized struct. Box<dyn Fn> works the same as Box<dyn AnyTrait> —
the closure struct is heap-allocated, and dispatch goes through the vtable. Drop is the
first vtable slot (index 0), and the compiler automatically calls it when the Box goes
out of scope.
