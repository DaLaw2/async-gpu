# om-register-promo: Register Promotion for Move Semantics — Synthesis

## Core Finding

Rust ownership directly controls GPU register allocation. Owned/moved scalars
stay in registers; borrowed values spill to local memory (~100-cycle DRAM).
LTO can recover borrowed scalars only when callees are inlineable.

## What Works Already

- GpuRef.read() returns by-value (Copy) — loaded values stay in registers.
- Inner-loop scalars with no address taken: pure register, LLVM unrolls.
- `#[inline(always)]` on GpuRef methods: correct, prevents ABI-forced spills.

## What to Optimize

1. **Document the "borrow penalty"** for GPU kernel authors: taking `&x` in a
   hot loop forces `st.local` per iteration. Prefer `let val = x` (copy).
2. **Structs cross call boundaries via pointer regardless of ownership.**
   For hot paths: destructure to scalars or ensure inlining.
3. **Lint/clippy-like guidance**: flag `&scalar` in GPU inner loops as
   performance antipattern. Consider a `#[gpu_hot]` attribute someday.

## No Production Code Changes Needed

The current GpuRef/TieredAccess design is already optimal: read() returns
by-value, all methods are `#[inline(always)]`, no unnecessary borrows. The
main deliverable is knowledge for kernel authors writing compute loops.
