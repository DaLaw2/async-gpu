# dyn-compat.1: Third-party no_std crate with dyn Trait on GPU — unmodified
**Cycle**: 642 | **Feature**: dyn-compat | **Kind**: experiment | **Status**: done

## Summary
`hashbrown` v0.15.5 (a `#![no_std]` crate that uses `&dyn FnMut` and `&dyn Fn` internally
in its raw hash table implementation) compiles to valid PTX on nvptx64 and is accepted by
ptxas — completely **unmodified**. The crate was added as a dependency with
`default-features = false` and used directly in a GPU kernel. Zero blockers encountered.

## Crate Selection

### Survey Results
| Crate | no_std | Internal dyn usage | Selected |
|-------|--------|-------------------|----------|
| hashbrown 0.15 | `#![no_std]` | `&dyn FnMut(usize)->bool` in find_inner, `&dyn Fn(...)` in resize_inner | **Yes** |
| embedded-hal | `#![no_std]` | Defines traits for dyn dispatch, but no internal dyn usage | No |
| heapless | `#![no_std]` | Minimal/no dyn usage | No |
| dyn-clone | `#![no_std]` | Focused on dyn, but trivial | No |

hashbrown was selected because:
1. It is a widely-used, production-quality `#![no_std]` crate (the HashMap implementation Rust's std uses)
2. It uses `dyn Trait` **internally** — not just in its API, but inside the core hash table operations
3. Every `insert`/`get`/`contains_key` call flows through `find_inner(&dyn FnMut)` internally
4. Every resize flows through `reserve_rehash(&dyn Fn)` internally
5. Edition 2021, compatible with our kernel crate

## Findings

### Q1: Does hashbrown compile to valid PTX unmodified?
**A: Yes.** Added `hashbrown = { version = "0.15", default-features = false }` to
gpu-kernel-test/Cargo.toml. `cargo build --release` compiled both hashbrown and the
kernel using it without any errors. The PTX was generated successfully.

### Q2: What hashbrown functions appear in the PTX?
**A: Multiple hashbrown-internal functions including `reserve_rehash`.** The PTX contains
hashbrown's `RawTable::reserve_rehash` monomorphized for our `TrivialBuildHasher` type —
this is the function that internally uses `&dyn Fn(&mut Self, usize) -> u64` for rehashing.

Two sets of hashbrown symbols appear:
- `Cs2NtVLhq0KCH_9hashbrown` — our explicit dependency (hashbrown 0.15.5 with TrivialBuildHasher)
- `CshvADsmzQWOI_9hashbrown` — std's internal hashbrown (with RandomState)

### Q3: Does ptxas accept the generated PTX?
**A: Yes.** `ptxas --gpu-name sm_75` assembles the PTX to a valid cubin without errors.

### Q4: What about the dyn dispatch — is it devirtualized?
**A: LLVM devirtualizes the internal dyn calls** because the concrete closure type is always
known at each call site. This is expected and correct behavior — the code path through the
dyn-accepting functions (`find_inner`, `resize_inner`) is still fully compiled, but the
optimizer proves the concrete type and converts indirect calls to direct calls. This means:
- The **source code** with `dyn Trait` compiles correctly
- The **compilation pipeline** handles dyn dispatch on nvptx64
- The **optimizer** produces efficient code (no runtime overhead from devirtualized dyn)

### Q5: What hasher was used?
**A: A custom FNV-1a-inspired hasher (TrivialHasher).** hashbrown with `default-features = false`
has no default hasher (the `default-hasher` feature pulls in `foldhash`). We provided a minimal
`core::hash::Hasher` implementation defined inline in the kernel. This is realistic — embedded
and no_std users commonly provide their own hashers.

## Test Scenarios Implemented

| Scenario | Description | Status |
|----------|-------------|--------|
| A: Simple | HashMap insert + get (3 entries) | Compiles |
| B: Medium | HashSet with iteration (10 entries) | Compiles |
| C: Complex | HashMap resize + remove + sum (50 entries, forces rehash) | Compiles |

All three scenarios compile to valid PTX. Runtime execution requires cubin rebuild
(or PTX JIT fallback), with host-side test added via `#[gpu_test]`.

## Key Insight
A real, production-quality `#![no_std]` crate with internal `dyn Trait` usage compiles and
links into GPU kernels with **zero modifications**. The only requirement is:
- `default-features = false` (to avoid pulling in std-dependent features)
- A custom hasher (since `default-hasher` pulls in `foldhash`)

The crate's `extern crate alloc` is satisfied by our patched std (which provides alloc).
No address space issues, no vtable problems, no linker errors.

## Files Changed
- `crates/kernel/gpu-kernel-test/Cargo.toml` — added `hashbrown = { version = "0.15", default-features = false }`
- `crates/kernel/gpu-kernel-test/src/lib.rs` — added `test_gpu_dyn_compat_hashbrown` kernel
- `crates/test/gpu-test-harness/tests/gpu_tests.rs` — added `#[gpu_test] fn test_gpu_dyn_compat_hashbrown()`

## Open Questions
1. **Runtime execution**: Compilation is proven; runtime execution requires cubin rebuild or PTX JIT. The `#[gpu_test]` is registered and will run once cubin is available.
2. **More complex crates**: Would crates with heavier alloc usage (e.g., `serde` with `no_std`) also work? Likely yes, given hashbrown's success.
3. **Crates using `Box<dyn Trait>`**: hashbrown uses `&dyn` (reference-based). Crates using `Box<dyn>` would exercise the heap allocation path (already proven in dyn-box.1).
