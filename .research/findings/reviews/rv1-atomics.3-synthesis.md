# Review rv1: atomics.3 — Synthesis
**Date**: 2026-03-11
**Task**: atomics.3 — Implement gpu-atomics crate with system-scope primitives
**Overall Verdict**: issues_found → **rework**

## Individual Verdicts
| Reviewer      | Verdict       |
|---------------|---------------|
| Correctness   | issues_found  |
| Architecture  | issues_found  |
| Performance   | issues_found  |

## Cross-Cutting Summary

The core PTX instruction selection is **correct and sound** — all three reviewers agree
on this. The `.sys` scope, acquire/release qualifiers, register class assignments, and
`#[inline(always)]` annotations are all appropriate for SM70+ GPU-CPU communication.
The `gpu-atomics` crate as an abstraction boundary is the right architectural decision.

However, the crate has significant issues in three categories that must be addressed
before building the hostcall protocol on top:

## Critical Issues (Must Fix)

### C1: Panic handler in library crate (Architecture #7)
`gpu-atomics/src/lib.rs` defines `#[panic_handler]` — a program-global symbol. Any
downstream crate that also defines a panic handler will fail to link. This is a **hard
blocker** for using `gpu-atomics` as a dependency of `gpu-kernel` or any future crate.
**Fix**: Remove panic handler from `gpu-atomics`; keep it only in binary/kernel crates.

### C2: Use-after-free on timeout (Correctness #3)
In `gpu-host/src/main.rs`, on timeout the host frees mapped memory before synchronizing
the GPU. The asynchronously-launched kernel may still be writing to the freed memory.
**Fix**: Call `dev.synchronize()` before `cuMemFreeHost` in the timeout path.

### C3: Kernel argument mismatch (Correctness #1)
Host passes 4 args to `integration_sys_store` which only accepts 3. While the extra arg
is likely harmlessly ignored by the PTX ABI, it's an API contract violation.
**Fix**: Either add `thread_count` parameter to the kernel, or remove the extra argument
from the host launch call.

## Structural Issues (Should Fix Before Hostcall)

### S1: PTX duplication between gpu-kernel and gpu-atomics (Architecture #2)
Both crates implement the same inline PTX primitives independently. Changes must be made
in two places. `gpu-kernel` should depend on `gpu-atomics` and call its functions.

### S2: Test kernel embedded in library (Architecture #1)
`kernel_sys_store_and_signal` is an entry point, not a library function. Move it to
`gpu-kernel` or a dedicated test crate.

### S3: Redundant membar.sys (Performance #1)
The `st.release.sys` → `membar.sys` → `st.release.sys` pattern has a redundant fence.
The second `st.release.sys` already provides release ordering over the first store.
The `membar.sys` costs hundreds of cycles on Ampere with zero correctness benefit.
**Fix**: Remove the `membar.sys` between the two release stores.

### S4: NVVM intrinsics as unexplained public API (Architecture #4)
Two parallel sets of primitives (inline PTX + NVVM intrinsics) without selection criteria.
**Fix**: Gate behind a feature flag or make `pub(crate)` with documentation.

### S5: No safe wrapper type (Architecture #3)
All operations are bare `unsafe fn` with raw pointers. A `SysAtomic<T>` wrapper would
reduce `unsafe` surface area at call sites and enforce pairing of acquire/release.
**Defer**: Can be added when designing hostcall protocol (not blocking).

## Minor Issues (Fix When Convenient)

### M1: Host polling loop lacks `spin_loop()` hint (Performance #2)
Add `std::hint::spin_loop()` to avoid CPU power waste and improve SMT behavior.

### M2: CAS kernel writes via plain Rust deref (Correctness #4)
`test_asm_cas_sys` uses `*output = result;` instead of `st.global.b32`. Inconsistent
with other kernels and may fail address-space inference in edge cases.

### M3: `CU_MEMHOSTALLOC_PORTABLE` not set (Performance #3)
Zero cost to add, prevents issues in multi-context scenarios.

### M4: `readonly` on acquire loads may suppress reloads (Correctness #7)
If `sys_load_acquire_u32` is ever used in a GPU-side spin loop, LLVM may hoist the load.
Add a comment documenting this hazard.

### M5: Asymmetric type coverage (Architecture #5, #6)
Missing u64 variants for CAS and fetch_add, and no exchange primitive. Should be
designed as a complete type matrix before hostcall protocol.

### M6: Per-kernel PTX re-loading (Performance #4)
Load PTX module once and pass handles around instead of re-loading per test.

### M7: Integration test bypasses gpu-atomics (Architecture #8)
The end-to-end test exercises `gpu-kernel`'s inline PTX, not the `gpu-atomics` API.

## Consensus Points (All 3 Reviewers Agree)
1. Core PTX instruction selection is correct for SM70+ GPU-CPU communication
2. `.sys` scope is the right choice for hostcall use case
3. `#[inline(always)]` is appropriate for single-instruction wrappers
4. The crate separation is the right architectural direction
5. The producer/consumer flag protocol (data + flag with release stores) is sound
6. `Ordering::Acquire` on host side is the correct pairing with GPU `st.release.sys`

## Rework Recommendation

Create task `atomics.3.1` to address C1-C3 and S1-S4. The rework should:
1. Remove panic handler from gpu-atomics
2. Fix use-after-free in timeout path
3. Fix kernel argument mismatch
4. Remove redundant membar.sys from integration kernels
5. Move test kernel out of gpu-atomics into gpu-kernel
6. Make gpu-kernel depend on gpu-atomics (eliminate PTX duplication)
7. Gate or internalize NVVM intrinsic fallbacks

S5 (SysAtomic<T> wrapper) and M5 (type completeness) can be deferred to the hostcall
design phase where the actual API requirements will be clearer.
