# panic-deprecate-gpu-assert — Feature Synthesis

## Verdict
Standard `assert!` fully replaces `gpu_assert!`. Only one call site exists
(`trace_assert_test` kernel). The macro, its hostcall function, SERVICE_ASSERT
opcode handler, and protocol constants are all dead code once that single
call site is migrated.

## Migration Plan (3 phases, all in one PR)
1. **Replace**: `gpu_assert!(buf, cond, msg)` → `assert!(cond, msg)` in
   `hostcall_kernels.rs:1279`. Run `trace_assert_test` to verify.
2. **Remove macro**: Delete `gpu_assert!` from `gpu-runtime/src/lib.rs`
   (both `cfg` variants, lines 396-451). Build all kernel crates.
3. **Remove plumbing**: Delete `gpu_hostcall_assert()`, `handle_assert()`,
   `SERVICE_ASSERT`/`ASSERT_MAX_MSG_LEN` constants, and all re-exports.
   Run full CI + GPU tests.

## Why standard assert! is superior
- No `buf` parameter needed (uses global `STDIO_HOSTCALL_BUF`)
- No 56-byte message limit (chunked output via `gpu_stdout_write`)
- Includes source file:line location automatically
- Reports block/warp/lane (not just thread_idx/block_idx)
- Works without `gpu-trace` feature flag
- Familiar Rust idiom — zero learning curve

## Risk: None
Single call site, `test_gpu_assert_basic` already uses standard `assert!`.
