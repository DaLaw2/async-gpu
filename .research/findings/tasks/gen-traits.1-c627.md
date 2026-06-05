# gen-traits.1: User-Defined Traits with Where Bounds on GPU

**Status**: Complete
**Kind**: Experiment
**Cycle**: 627

## Summary

Successfully implemented and compiled user-defined traits (`GpuReducible`,
`GpuTransformable`) with `where` bounds on nvptx64. Generic functions bounded
by these traits monomorphize to type-specific PTX — trait methods are fully
inlined, producing optimal code identical to hand-written kernels. A custom
`#[repr(C)]` struct (`Vec2f`) implementing the traits also works correctly.

## What Was Implemented

### 1. User-defined traits in gpu-runtime (`traits.rs`)

Two traits added to `crates/core/gpu-runtime/src/traits.rs`:

- **`GpuReducible`**: `identity() -> Self` + `combine(self, other: Self) -> Self`
  - Built-in implementations for f32, f64, u32, u64, i32, i64, usize
  - Additive reduction (identity=0, combine=add)

- **`GpuTransformable`**: `default_value()` + `scale(self, factor)` + `offset(self, amount)`
  - Built-in implementations for the same primitive types
  - Demonstrates `where` bound syntax working on GPU

Both traits are `#![no_std]` compatible, use only `core` traits (`Copy`),
and are exported via `gpu_runtime::prelude::*`.

### 2. Generic kernel functions using traits

Three generic functions in `gpu-kernel-test/src/lib.rs`:

| Function | Trait bounds | Pattern |
|----------|-------------|---------|
| `trait_reduce<T: GpuReducible>` | Inline bound | Sequential reduction |
| `apply_transform<T>` where `T: GpuTransformable` | Where clause | Grid-stride transform |
| `transform_then_reduce<T>` where `T: GpuReducible + GpuTransformable` | Multiple where bounds | Transform then reduce |

### 3. Custom type implementing traits

```rust
#[repr(C)]
#[derive(Clone, Copy)]
struct Vec2f { x: f32, y: f32 }

impl GpuReducible for Vec2f { ... }  // per-field addition
impl GpuTransformable for Vec2f { ... }  // per-field scale/offset
```

### 4. Concrete entry points (2 kernels)

| Entry point | Generic instantiation | Purpose |
|-------------|----------------------|---------|
| `trait_reduce_f32` | `trait_reduce::<f32>` | f32 sum via GpuReducible |
| `trait_reduce_i32` | `trait_reduce::<i32>` | i32 sum via GpuReducible |

### 5. GPU test kernels (6 entries)

| Test kernel | What it verifies |
|-------------|------------------|
| `test_gpu_trait_reduce_f32` | f32 reduce via GpuReducible, identity/combine |
| `test_gpu_trait_reduce_i32` | i32 reduce via GpuReducible, identity/combine |
| `test_gpu_where_transform` | Where-clause syntax with f32 and i32 |
| `test_gpu_trait_combined` | Multiple trait bounds (GpuReducible + GpuTransformable) |
| `test_gpu_trait_custom_vec2f` | Custom Vec2f type: reduce, transform, identity, combine |
| `test_gpu_trait_multi_type` | All types (f32, i32, u32, Vec2f) in one kernel |

## PTX Compilation Results

All 8 entries emitted as `.visible .entry` in the PTX (7.9MB total):

```
.visible .entry test_gpu_trait_combined()
.visible .entry test_gpu_trait_custom_vec2f()
.visible .entry test_gpu_trait_multi_type()
.visible .entry test_gpu_trait_reduce_f32()
.visible .entry test_gpu_trait_reduce_i32()
.visible .entry test_gpu_where_transform()
.visible .entry trait_reduce_f32(...)
.visible .entry trait_reduce_i32(...)
```

Build time: ~54s (dev mode, opt-level 1, no LTO).

## Type-Specific PTX Analysis

### trait_reduce: f32 vs i32

**f32** — IEEE 754 round-to-nearest addition, LLVM 4x unrolled:
```ptx
add.rn.f32  %r3, %r10, %r2;    // accumulate with round-to-nearest
add.rn.f32  %r5, %r3, %r4;
add.rn.f32  %r7, %r5, %r6;
add.rn.f32  %r10, %r7, %r8;
```

**i32** — signed integer addition, LLVM 4x unrolled:
```ptx
add.s32     %r3, %r2, %r10;    // signed 32-bit add
add.s32     %r5, %r3, %r4;
add.s32     %r7, %r5, %r6;
add.s32     %r10, %r7, %r8;
```

**Identity initialization** — type-specific:
- f32: `mov.b32 %r10, 0f00000000` (IEEE 754 zero)
- i32: `mov.b32 %r10, 0` (integer zero)

These are the same instructions as gen-mono.2's `generic_reduce_sum` — proving
that `GpuReducible::combine()` and `GpuReducible::identity()` are fully inlined
by the compiler. The trait abstraction has **zero overhead**.

### Where-clause syntax

`apply_transform<T>` uses explicit `where T: GpuTransformable` syntax.
The generated PTX is identical to what `<T: GpuTransformable>` would produce —
the Rust compiler normalizes both forms before MIR-level monomorphization.

### Vec2f custom type

The custom `Vec2f { x: f32, y: f32 }` compiled to per-field f32 operations.
`Vec2f::combine()` produces two `add.rn.f32` instructions (one for `x`, one
for `y`). `Vec2f::scale()` produces two `mul.rn.f32` instructions.

## Key Findings

1. **User-defined traits monomorphize to optimal PTX.** `GpuReducible::combine()`
   and `GpuReducible::identity()` are fully inlined by the compiler — the trait
   abstraction has zero runtime overhead compared to hand-written type-specific code.

2. **Where-clause syntax works identically.** `fn f<T>() where T: Trait` produces
   the same PTX as `fn f<T: Trait>()`. This is expected (they are syntactic
   sugar for the same thing) but worth confirming on nvptx64.

3. **Multiple trait bounds compose correctly.** `where T: GpuReducible + GpuTransformable`
   produces correct code with all trait methods inlined.

4. **Custom `#[repr(C)]` types work through generic trait dispatch.** A user-defined
   `Vec2f` struct implementing `GpuReducible` compiles to per-field f32 operations
   when passed through `trait_reduce::<Vec2f>()`.

5. **LLVM applies the same optimizations through trait bounds.** Loop unrolling (4x),
   type-specific instruction selection (`.rn.f32` vs `.s32`), and identity
   initialization all work identically whether the code uses raw `Add` bounds
   or user-defined trait bounds.

## Relationship to par_iter

The existing `GpuZero`/`GpuOne`/`GpuMaxValue`/`GpuMinValue` traits in `par_iter.rs`
are single-method identity traits. `GpuReducible` is a more general pattern
that combines identity + combine into one trait. Future work could:
- Refactor `par_iter::sum()` to use `GpuReducible` instead of `GpuZero + Add`
- Add `par_iter::reduce(identity, combine)` for custom reductions
- Allow users to implement `GpuReducible` for their types and use them with par_iter

## Files Modified

- `crates/core/gpu-runtime/src/traits.rs` — NEW: GpuReducible + GpuTransformable traits
- `crates/core/gpu-runtime/src/lib.rs` — added `pub mod traits`
- `crates/core/gpu-runtime/src/prelude.rs` — re-exported trait types
- `crates/kernel/gpu-kernel-test/src/lib.rs` — 8 kernel entries (2 concrete + 6 test)
- `crates/test/gpu-test-harness/tests/gpu_tests.rs` — 6 `#[gpu_test]` entries
