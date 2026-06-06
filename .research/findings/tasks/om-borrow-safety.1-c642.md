# om-borrow-safety.1 — Compile-fail tests for cross-scope memory access violations

**Kind**: experiment
**Status**: DONE
**Date**: 2026-06-06

## Summary

Wrote and verified 8 compile-fail tests proving the tiered memory type system (`SharedRef`/`GlobalRef`) catches memory hierarchy bugs at compile time. All 6 negative tests are correctly rejected; both positive tests compile as expected.

## Findings

### Compile-time enforcement mechanisms

Three independent type-system mechanisms prevent cross-scope memory misuse:

1. **`for<'scope>` HRTB on `block_scope()`** — The signature `F: for<'scope> FnOnce(&mut BlockScope<'scope>) -> R` prevents any `'scope`-parameterized type from escaping the closure. This catches:
   - Returning `SharedRef` from the closure (E0521: lifetime may not live long enough)
   - Storing `SharedRef` in outer variables (E0521: borrowed data escapes outside of closure)
   - Assigning to `Option<SharedRef>` outside the closure (same error)

2. **`!Send` for `SharedRef`** — `GpuRef<'_, T, Shared>` inherits `!Send` from `*mut T`, and no `Send` impl is provided (unlike `GpuRef<'_, T, Global>` which explicitly has `unsafe impl Send`). This catches:
   - Sending `SharedRef` across threads/warps (E0277: `GpuRef<'_, f32, Shared>` cannot be sent between threads safely)
   - Putting `SharedRef` inside a container that is then sent (E0277: within `CrossTierContainer`, the trait `Send` is not implemented)

3. **`!Sync` for `SharedRef`** — Same mechanism as `!Send`. Catches:
   - Sharing `SharedRef` across threads via `&SharedRef` (E0277: cannot be shared between threads safely)

4. **Invariant lifetime** — `PhantomData<&'scope mut &'scope ()>` makes the lifetime invariant, preventing covariant widening that could bypass the HRTB.

### Test cases (all verified)

| Test | Expected | Mechanism | Compiler Error |
|------|----------|-----------|---------------|
| `shared_ref_escape_scope` | FAIL | HRTB | E0521: borrowed data escapes outside of closure |
| `shared_ref_not_send` | FAIL | !Send | E0277: cannot be sent between threads safely |
| `shared_ref_not_sync` | FAIL | !Sync | E0277: cannot be shared between threads safely |
| `shared_ref_in_global_container` | FAIL | !Send (transitive) | E0277: within container, Send not implemented |
| `shared_ref_use_after_scope` | FAIL | HRTB | E0521: borrowed data escapes outside of closure |
| `shared_ref_return_from_scope` | FAIL | HRTB + invariance | lifetime may not live long enough |
| `valid_shared_ref_within_scope` | PASS | — | Compiles successfully |
| `valid_global_ref_is_send` | PASS | — | GlobalRef: Send+Sync confirmed |

### Positive tests (valid patterns)

- `SharedRef` used within its scope: alloc, read, write, sub_ref — all work
- `SharedRef` passed to functions within scope — works (lifetime subsumption)
- `GlobalRef` is `Send + Sync` — confirmed, asymmetry with `SharedRef` is deliberate

## Open Questions

1. **Trybuild integration**: Currently tests use a shell-script runner (`compile_fail_runner.sh`). Adding `trybuild` as a dev-dependency would enable `cargo test` integration, but gpu-runtime uses `no_std` and is not a workspace member, making trybuild integration non-trivial. The shell runner is sufficient for now.

2. **nvptx64-specific testing**: The compile_fail tests run on the host target (`x86_64`). The type-system constraints (lifetimes, Send/Sync) are target-independent, so host-target testing is sufficient. No PTX-specific compile errors need to be tested — the safety invariants are enforced by Rust's type system, not by backend code generation.

3. **`GridScope` escape tests**: The same HRTB pattern on `grid_scope()` should prevent `GlobalRef` escape. Not tested here because `grid_scope()` requires `unsafe` and raw pool pointers, making the test setup significantly more complex. The mechanism is identical to `block_scope()`.
