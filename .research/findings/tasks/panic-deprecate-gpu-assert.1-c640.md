# panic-deprecate-gpu-assert.1 — Migration plan: gpu_assert! → standard assert!

## Summary

Standard `assert!` is strictly superior to `gpu_assert!` on GPU. The patched std
panic hook already formats assertion failures with block/warp/lane coordinates,
source location, and the assertion message — with no 56-byte message limit and no
`buf` parameter required. `gpu_assert!` can be fully deprecated and removed.

## Findings

### 1. Can gpu_assert! be safely deprecated without breaking any kernel crate?

**Yes.** Only one live call site exists:
- `gpu-kernel-io/src/hostcall_kernels.rs:1279` — `trace_assert_test` kernel

The `test_gpu_assert_basic` kernel in `gpu-kernel-test` already uses standard
`assert!`/`assert_eq!` (the name is historical). No other kernel crate uses
`gpu_assert!`.

### 2. What should replace each usage of gpu_assert!?

| Location | Current | Replacement |
|---|---|---|
| `hostcall_kernels.rs:1279` | `gpu_assert!(buf, 1 + 1 == 2, "math works")` | `assert!(1 + 1 == 2, "math works")` |

The `buf` parameter is no longer needed because standard `assert!` routes
through `panic!` → patched std `default_hook` → `Stderr::write` →
`gpu_stdout_write` → `STDIO_HOSTCALL_BUF` (global). The hostcall buffer is
already initialized via `gpu_panic_init(buf)` at line 1272.

### 3. Should SERVICE_ASSERT protocol opcode be deprecated too?

**Yes, but in a later task.** SERVICE_ASSERT (opcode 14) and its entire
supporting stack should be deprecated:

- **Protocol**: `gpu-protocol/src/lib.rs` — `SERVICE_ASSERT` const, `ASSERT_MAX_MSG_LEN` const
- **Runtime**: `gpu-runtime/src/hostcall.rs` — `gpu_hostcall_assert()` function
- **Runtime**: `gpu-runtime/src/prelude.rs` — `SERVICE_ASSERT` re-export, `gpu_hostcall_assert` re-export
- **Host**: `gpu-host/src/hostcall.rs` — `handle_assert()` method, dispatch match arm

However, removing the opcode handler on the host side is safe to do
simultaneously — once no GPU code sends SERVICE_ASSERT, the handler is dead
code. The host match arm can be kept temporarily with a comment noting
deprecation, or removed outright since no GPU code will send it.

### 4. Should the deprecated gpu_assert! macro body be changed to just call standard assert!?

**No — remove it entirely.** A wrapper that delegates to `assert!` adds no
value (just a confusing indirection with an unused `buf` parameter). The macro
has only one call site, making a shim unnecessary. Direct replacement is
cleaner.

### 5. What about test_gpu_assert_basic — should it test standard assert! instead?

**It already does.** The kernel `test_gpu_assert_basic` in `gpu-kernel-test`
uses standard `assert_eq!`/`assert!`/`assert_ne!`. Its name refers to the
concept of "GPU assertions," not the `gpu_assert!` macro. No changes needed.

The `trace_assert_test` kernel in `gpu-kernel-io` is the one that actually
uses `gpu_assert!` and needs migration.

### 6. Migration steps (ordered)

**Phase 1: Replace call site** (one task)
1. In `hostcall_kernels.rs:1279`, replace `gpu_assert!(buf, 1 + 1 == 2, "math works")`
   with `assert!(1 + 1 == 2, "math works")`.
2. Run `trace_assert_test` to verify the kernel still passes.

**Phase 2: Remove macro** (one task)
3. Delete the `gpu_assert!` macro (both `cfg(feature = "gpu-trace")` and
   `cfg(not(feature = "gpu-trace"))` variants) from `gpu-runtime/src/lib.rs`
   (lines 396-451).
4. Build all kernel crates to confirm no remaining references.

**Phase 3: Remove SERVICE_ASSERT plumbing** (one task)
5. Remove `gpu_hostcall_assert()` from `gpu-runtime/src/hostcall.rs`.
6. Remove `gpu_hostcall_assert` from `gpu-runtime/src/prelude.rs` re-exports.
7. Remove `SERVICE_ASSERT` from `gpu-runtime/src/prelude.rs` re-exports.
8. Remove `handle_assert()` and its dispatch arm from `gpu-host/src/hostcall.rs`.
9. Add `#[deprecated]` or remove `SERVICE_ASSERT` and `ASSERT_MAX_MSG_LEN`
   from `gpu-protocol/src/lib.rs`. (Prefer removal — no external consumers.)

**Phase 4: Verify** (same task as Phase 3)
10. Run full CI (`scripts/ci-lint.sh`).
11. Run GPU tests to confirm assert behavior is preserved.

### Standard assert! is strictly superior

| Aspect | gpu_assert! | standard assert! |
|---|---|---|
| Buffer param | Required (`buf`) | Not needed (global) |
| Message limit | 56 bytes | Unlimited (chunked) |
| Source location | Not included | Included (file:line) |
| Thread ID | thread_idx, block_idx | block, warp, lane |
| Feature gating | Requires `gpu-trace` for diagnostics | Always available |
| Ergonomics | Custom macro, unfamiliar | Standard Rust, familiar |

## Open Questions

None — the design is straightforward. All questions from the task brief are
answered above. The migration is low-risk with a single call site.
