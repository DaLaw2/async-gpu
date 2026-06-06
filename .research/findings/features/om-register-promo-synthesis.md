# om-register-promo: Register Promotion for Move Semantics — Synthesis

## Core Finding

GpuRef::read() produces optimal GPU codegen: a single `ld.shared.u32`
returning by-value, keeping the loaded value in a register. Slice access
adds address conversion overhead. Volatile pointer access forces 1024x
more memory traffic (1 load per iteration vs 1 total).

## What Works Already

- GpuRef.read() → `ld.shared.u32`, 1 instruction, value in register
- Value stays in register across all loop iterations (O1 + O3+LTO)
- `#[inline(always)]` on GpuRef methods prevents ABI-forced spills
- LLVM unrolls 8-16x when value is register-held

## Demonstrated Speedup (om-register-promo.2)

Three kernels, 1024-iteration inner loop (acc += val * i):
- **GpuRef::read()**: 1 load, 0 in loop, 16x unroll (O3)
- **as_generic_slice()**: 1 load, 0 in loop, 16x unroll, +5 addr-conv instrs
- **read_volatile(ptr)**: 1024 loads, 1/iter, 8x unroll (O3)

## Guidance for Kernel Authors

1. Use `let val = ref.read(i)` before hot loops, not `&slice[i]` inside.
2. Structs cross call boundaries via pointer regardless of ownership —
   destructure to scalars or ensure inlining for hot paths.
3. Prefer GpuRef::read() over as_generic_slice() for inner loops.

## No Production Code Changes Needed

GpuRef/TieredAccess design is already optimal. Main deliverable is
codegen knowledge for kernel authors writing compute loops.
