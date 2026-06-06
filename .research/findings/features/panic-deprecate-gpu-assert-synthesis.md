# panic-deprecate-gpu-assert — Feature Synthesis

## Status: COMPLETE

`gpu_assert!` macro fully deprecated and removed.
Standard `assert!` now handles all GPU assertions via the panic handler.

## What was removed
- `gpu_assert!` macro (both trace and non-trace variants)
- `gpu_hostcall_assert()` in gpu-runtime
- `handle_assert()` + dispatch arm in gpu-host
- `SERVICE_ASSERT` (14), `ASSERT_MAX_MSG_LEN` in gpu-protocol
- All related re-exports from gpu-runtime prelude

## Why standard assert! is superior
- No `buf` parameter needed (uses global `STDIO_HOSTCALL_BUF`)
- No 56-byte message limit (chunked output via `gpu_stdout_write`)
- Includes source file:line location automatically
- Reports block/warp/lane (not just thread_idx/block_idx)
- Works without `gpu-trace` feature flag
- Familiar Rust idiom — zero learning curve

## Verification
- Workspace build: PASS
- CI lint (fmt + clippy + doc-tests + PTX): all PASS
- `test_gpu_assert_basic` kernel: unaffected (already uses standard assert)

## Risk: None
Single call site, `test_gpu_assert_basic` already uses standard `assert!`.
