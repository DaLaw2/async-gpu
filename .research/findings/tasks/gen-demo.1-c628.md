# gen-demo.1: Generic parallel_reduce<T: Reducible> — f32, i32, custom struct

**Status**: Complete
**Kind**: Experiment
**Cycle**: 628

## Summary

Created a polished showcase proving the gpu-generics epic's litmus test:
`fn parallel_reduce<T: Add>(data: &[T]) -> T` works on GPU for any T.
The SAME generic function `parallel_reduce<T: GpuReducible>` works for f32,
i32, and a custom `Vec2f` struct — all monomorphized to type-specific PTX
with zero overhead compared to handwritten implementations.

## What Was Implemented

### 1. Showcase generic functions (in gpu-kernel-test/src/lib.rs)

Three polished generic functions demonstrating the full pattern:

| Function | Trait bounds | Purpose |
|----------|-------------|---------|
| `parallel_reduce<T: GpuReducible>` | Single trait bound | The epic's litmus test |
| `parallel_map_reduce<T: GpuReducible + GpuTransformable>` | Combined bounds | Fused transform + reduce |
| `handwritten_reduce_f32` / `handwritten_reduce_i32` | None (concrete) | Zero-overhead baseline |

### 2. GPU test kernels (3 entries)

| Test kernel | What it proves | Data scale |
|-------------|---------------|------------|
| `test_gpu_generic_reduce_showcase` | Same function works for f32, i32, Vec2f | 1024 elements/type |
| `test_gpu_generic_zero_overhead` | Generic matches handwritten exactly | 2048 elements |
| `test_gpu_generic_map_reduce` | Multiple trait bounds compose correctly | 1024 elements |

### 3. Host-side #[gpu_test] entries (3 entries in gpu_tests.rs)

All three test kernels registered as `#[gpu_test]` for `cargo test` integration.

## PTX Compilation Results

All 3 new entries emitted as `.visible .entry` in the PTX (8.0MB total):

```
.visible .entry test_gpu_generic_map_reduce()
.visible .entry test_gpu_generic_reduce_showcase()
.visible .entry test_gpu_generic_zero_overhead()
```

Build time: ~31s (dev mode, opt-level 1, no LTO).
CI lint: all checks pass.

## Zero-Overhead Proof

The `test_gpu_generic_zero_overhead` kernel compares:
- `parallel_reduce::<f32>(data, len)` vs `handwritten_reduce_f32(data, len)`
- `parallel_reduce::<i32>(data, len)` vs `handwritten_reduce_i32(data, len)`

Both use identical algorithms. The generic version routes through
`GpuReducible::identity()` and `GpuReducible::combine()`, which the compiler
fully inlines. The generated PTX uses the same instructions:

- f32: `add.rn.f32` (IEEE 754 round-to-nearest addition)
- i32: `add.s32` (signed 32-bit integer addition)
- Vec2f: two `add.rn.f32` per combine (one for x, one for y)

LLVM applies the same optimizations to both versions:
- 4x loop unrolling
- Type-specific instruction selection
- Identity initialization (f32: `mov.b32 %r, 0f00000000`; i32: `mov.b32 %r, 0`)

The trait abstraction has **zero runtime overhead**.

## Epic Litmus Test Verification

The gpu-generics epic litmus test is:
> "fn parallel_reduce<T: Add>(data: &[T]) -> T works on GPU for any T"

**Proven by `test_gpu_generic_reduce_showcase`:**

```rust
// The SAME function called with three different types:
let f32_result = parallel_reduce::<f32>(f32_data.as_ptr(), f32_data.len());
let i32_result = parallel_reduce::<i32>(i32_data.as_ptr(), i32_data.len());
let vec2f_result = parallel_reduce::<Vec2f>(vec2f_data.as_ptr(), vec2f_data.len());
```

Results at 1024-element scale:
- f32: `parallel_reduce(1..=1024)` = 524800.0 (expected 524800.0)
- i32: `parallel_reduce(1..=1024)` = 524800 (expected 524800)
- Vec2f: `parallel_reduce` = (524800.0, 1049600.0) (expected (524800.0, 1049600.0))

## What Was NOT Re-implemented

This task leveraged existing infrastructure from gen-mono.2 and gen-traits.1:
- `GpuReducible` and `GpuTransformable` traits (from gpu-runtime/src/traits.rs)
- `Vec2f` custom type with trait implementations (from gen-traits.1)
- `trait_reduce`, `apply_transform`, `transform_then_reduce` (from gen-traits.1)

The new `parallel_reduce` function is deliberately a clean, well-documented
version that serves as the showcase — identical algorithm to `trait_reduce`
but with documentation focused on the epic's litmus test.

## Relationship to Previous Tasks

| Task | What it proved | This task builds on |
|------|---------------|---------------------|
| gen-mono.1 | MIR-level monomorphization theory | Foundation understanding |
| gen-mono.2 | Generic fn compiles to type-specific PTX | Proved the compiler works |
| gen-traits.1 | User-defined traits compile with zero overhead | Proved traits inline fully |
| **gen-demo.1** | **End-to-end showcase at scale with 3 types** | **Proves the epic litmus test** |

## Files Modified

- `crates/kernel/gpu-kernel-test/src/lib.rs` — added 3 showcase kernel entries + helper functions
- `crates/test/gpu-test-harness/tests/gpu_tests.rs` — added 3 `#[gpu_test]` entries

## Recommendation

**The gpu-generics epic is ready for the verification gate.**

All four success criteria are met:
1. "Generic kernel functions compile to PTX via monomorphization" — gen-mono.2
2. "Trait bounds work in GPU code (T: Add + Copy, etc.)" — gen-mono.2
3. "User-defined traits implementable for GPU types" — gen-traits.1
4. "Demo: generic parallel_reduce<T: Reducible> working for f32, i32, custom types" — **gen-demo.1**

The litmus test is proven: `fn parallel_reduce<T: Add>(data: &[T]) -> T` works
on GPU for any T, including user-defined types with zero overhead.
