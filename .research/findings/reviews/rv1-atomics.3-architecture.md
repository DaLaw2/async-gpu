# Review rv1: atomics.3 — Architecture
**Verdict**: issues_found

## Summary

The `gpu-atomics` crate establishes a working foundation for system-scope GPU atomic
primitives using inline PTX assembly. The PTX instruction selection is technically
correct (`.sys` scope, acquire/release semantics where applicable), and the separation
into its own crate is directionally sound. However, the crate suffers from significant
architectural problems: it embeds a test kernel that does not belong in a library crate,
it duplicates inline PTX extensively with `gpu-kernel`, the raw-function API exposes
every operation as `unsafe` without any ergonomic wrapper, and the NVVM intrinsic block
is an unexplained alternative rather than a deliberate fallback. These issues will
compound when building the hostcall protocol and async executor on top of this crate.

---

## Issues Found

### Issue 1: Test kernel embedded in library crate

`kernel_sys_store_and_signal` is a `#[no_mangle] pub unsafe extern "ptx-kernel"` kernel
defined directly in `gpu-atomics/src/lib.rs`. A library crate that exports this symbol
will have it linked into every consumer. PTX kernels are not library functions — they are
entry points for GPU execution. Placing a kernel in a generic-purpose atomic primitives
crate conflates two unrelated concerns, pollutes the library's public namespace, and
prevents the crate from ever being used as a pure `no_std` library by non-PTX targets.

The kernel belongs in `gpu-kernel` or a dedicated integration-test crate. `gpu-atomics`
should export only the primitive operations.

### Issue 2: Extensive PTX duplication between gpu-kernel and gpu-atomics

`gpu-kernel/src/lib.rs` contains inline PTX for `membar.sys`, `st.release.sys.global.u32`,
`ld.acquire.sys.global.u32`, and `atom.cas.sys.global.b32` — all of which are also
implemented as functions in `gpu-atomics`. The `integration_sys_store` kernel in
`gpu-kernel` re-implements the entire store-then-flag protocol in raw inline asm rather
than calling `gpu_atomics::sys_store_release_u32` and `gpu_atomics::membar_sys`. This
defeats the purpose of having `gpu-atomics` as a separate crate and means that changes
to the atomic protocol must be made in two places.

`gpu-kernel` should depend on `gpu-atomics` and call the exported functions. The
duplicated PTX inline blocks in `gpu-kernel` should be deleted.

### Issue 3: No safe wrapper type (SysAtomic<T>)

All operations are bare `unsafe fn` accepting raw `*mut T` / `*const T`. There is no
encapsulating type that enforces the invariants documented in the safety comments
(pointer must be to pinned/mapped global memory; must be properly aligned). Without a
wrapper type, every call site must re-verify these conditions independently, and there
is no compile-time assistance for pairing acquire loads with release stores.

For the hostcall protocol, callers will frequently pair `sys_store_release_u32` with
`sys_load_acquire_u32` on the same location. A type like:

```rust
pub struct SysAtomic<T>(*mut T);  // invariant: points to pinned mapped global memory

impl SysAtomic<u32> {
    pub unsafe fn new(ptr: *mut u32) -> Self { ... }
    pub unsafe fn store_release(&self, val: u32) { ... }
    pub unsafe fn load_acquire(&self) -> u32 { ... }
    pub unsafe fn cas(&self, expected: u32, desired: u32) -> u32 { ... }
}
```

would make the protocol more legible and easier to audit. The `unsafe` is still
appropriate at construction time (the pointer invariant cannot be checked), but the
method calls themselves would be safe once the type is constructed, reducing the surface
area of `unsafe` code at the call site.

### Issue 4: NVVM intrinsic block is unexplained dead code

The `extern "C"` block exposing `nvvm_membar_sys` and `nvvm_atomic_add_sys_i32` is
marked as public and carries a doc comment saying "NVVM intrinsic fallbacks (for
comparison / older SM)". However, there is no conditional compilation (`#[cfg]`),
no version gating, no runtime selection, and no indication of when callers should
prefer the NVVM path over the inline-asm path. As exported `pub` items they form part
of the library's public API, but they duplicate functionality already provided by
`membar_sys()` and `sys_fetch_add_u32()`.

The NVVM block should either be gated behind a feature flag with clear documentation
on why it is an alternative, or removed and preserved only in the research notes.
Exporting two parallel sets of primitives without documented selection criteria will
confuse downstream users.

### Issue 5: Asymmetric type coverage — fetch_add has no u64 variant, CAS has no u64

`sys_store_release_u64` and `sys_load_acquire_u64` exist but `sys_cas_u32` has no u64
counterpart, and `sys_fetch_add_u32` has no u64 counterpart. The async executor will
require 64-bit atomics for pointer-sized state words (e.g., waker pointers). The
current asymmetry suggests the API was grown incrementally rather than designed for
completeness. A deliberate type matrix (u32/u64 × store/load/cas/fetch_add/exchange)
should be the explicit design target.

### Issue 6: No atomic exchange (xchg) primitive

The hostcall protocol will need an atomic exchange (`atom.exch.sys.global`) for
lock-free flag handoff patterns. CAS can substitute but at higher cost. The missing
`sys_exchange_u32` / `sys_exchange_u64` should be part of the initial API surface so
that protocol designers are not forced to use CAS loops where xchg suffices.

### Issue 7: Panic handler belongs in a higher-level crate

`gpu-atomics` defines its own `#[panic_handler]`. A panic handler is a program-global
symbol — only one may exist in the final link unit. Any crate that links against
`gpu-atomics` and also provides its own panic handler will get a duplicate symbol error.
The panic handler should live in the kernel binary crate (e.g., `gpu-kernel`) or in a
thin `gpu-runtime` crate, not in a utility library. The `#![no_std]` library crate
should omit `#[panic_handler]` entirely.

### Issue 8: gpu-host integration test bypasses gpu-atomics

`gpu-host/src/main.rs` launches the `integration_sys_store` kernel from `gpu-kernel`,
which does not use `gpu-atomics`. The host-side also uses `std::sync::atomic::AtomicU32`
directly for polling rather than any abstraction provided by this project. While this is
correct, it means the integration test does not actually exercise the `gpu-atomics`
library at all — it tests the raw inline PTX in `gpu-kernel`. A proper integration test
would use the kernel in `gpu-atomics` (`kernel_sys_store_and_signal`) and would document
which crate is under test.

---

## Positive Observations

**Correct PTX instruction selection.** The choice of `st.release.sys.global`,
`ld.acquire.sys.global`, and `atom.cas.sys.global.b32` is technically sound for SM70+
GPU-CPU communication. The safety comments on each function accurately explain the
contract. The decision to use `.sys` scope rather than `.gpu` or `.cta` scope is correct
for the hostcall use case.

**Crate separation is the right direction.** Extracting GPU atomic primitives into their
own crate is aligned with how VectorWare likely structures their implementation — a low-
level `sys_atomics` or `gpu_sync` layer that the hostcall runtime depends on. This crate
is the right abstraction boundary; the problems are in its implementation, not its
existence.

**The inline PTX approach over NVVM intrinsics is the right choice.** PTX inline
assembly gives exact control over instruction encoding, scope, and ordering qualifier.
The verified nightly/LLVM19 path using `asm_experimental_arch` is the appropriate
mechanism. The decision to document the exact PTX emitted for each function is valuable
for downstream debugging.

**`#[inline(always)]` usage is appropriate.** Since these are single-instruction
wrappers, forcing inlining avoids PTX call overhead on the GPU, which is correct.

**The producer/consumer flag protocol is correct.** The two-word pattern (data +
flag, both written with release stores, separated by `membar_sys`) is the standard
GPU-CPU signaling idiom. Its implementation in `kernel_sys_store_and_signal` is
semantically correct, even though it is in the wrong crate.

**Host-side use of `AtomicU32::load(Ordering::Acquire)` for polling is correct.**
The pairing of GPU `st.release.sys` with CPU `Ordering::Acquire` is the documented
correct pattern for GPU-CPU acquire/release pairs on mapped memory.

---

## Verdict Rationale

The verdict is **issues_found** rather than **needs_rework** because the core primitives
are correct and the crate boundary is sound. However, three issues require structural
changes before this API can serve as a stable foundation:

1. The panic handler in the library crate will block downstream crates from defining
   their own (Issue 7) — this is a hard blocker.
2. The duplicated PTX in `gpu-kernel` (Issue 2) and the misplaced kernel (Issue 1)
   mean the crate does not yet function as a proper reusable library.
3. The missing `SysAtomic<T>` wrapper (Issue 3) and incomplete type matrix (Issues 5, 6)
   should be addressed before building the hostcall protocol layer on top, to avoid
   designing that layer against an unstable API.

None of these require redesigning the PTX instruction strategy. The fixes are
organizational and additive. The crate can reach a solid state with targeted refactoring.
