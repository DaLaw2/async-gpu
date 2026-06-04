# lib-cleanup.3: Migrate warp.rs off warp-macro DSL

**Status**: done
**Date**: 2025-06-04

## Summary

Migrated all 8 `#[warp_async]` kernel functions in `warp.rs` to standard `async fn`
using `gpu_runtime::std_future` futures, then removed the `warp-macro` crate dependency
from both `gpu-kernel` and the workspace. All existing tests pass with identical behavior.

## Phase 1: DSL Understanding

The `#[warp_macro::warp_async]` proc macro transformed functions containing `warp_*!()`
DSL calls into `WarpFuture` state machine structs + kernel entry points. It supported:
- `warp_print!()`, `warp_open!()`, `warp_read!()`, `warp_write!()`, `warp_close!()`
- `if/else`, `match`, `loop/break` control flow
- `.await` on standard `impl Future<Output = bool>` types
- `?` operator for `Result<bool, u32>` returns

Each DSL macro became an INIT + WAIT state pair; control flow added DECISION states.

## Phase 2: Rewrite to Standard async fn

Each `#[warp_async]` function was rewritten as a standard `async fn` using:
- `GpuPrintFuture::new(buf, msg).await` for print
- `GpuOpenFuture::new(buf, path, flags).await` for file open
- `GpuReadFuture::new(buf, fd, &mut buf).await` for file read
- `GpuWriteFuture::new(buf, fd, data).await` for file write
- `GpuCloseFuture::new(buf, fd).await` for file close

Kernel entry points use `gpu_runtime::std_future::block_on()` to drive the async fn.
Only lane 0 (thread 0) executes the async logic; other lanes return early. This produces
identical observable behavior (same messages, same result values) as the WarpFuture approach.

### Functions migrated (8 total):
1. `warp_macro_print_test` -- 2 sequential prints
2. `warp_cfg_if_else_test` -- if/else branching with prints
3. `warp_cfg_loop_test` -- loop with break condition
4. `warp_cfg_match_test` -- match dispatch with 3 arms
5. `warp_cfg_nested_test` -- nested if/match control flow
6. `autonomous_pipeline` -- multi-step file I/O pipeline (3 modes)
7. `warp_try_open_test` -- `?` operator with Result return
8. `warp_await_test` -- two sequential `.await` calls
9. `warp_e2e_test` -- mixed `.await` + if/else + warp DSL

### Functions preserved (unchanged):
- `WarpPrintFuture` + `warp_future_print_test` -- hand-written WarpFuture PoC
- `WarpMultiPrintFuture` + `warp_future_multi_print_test` -- hand-written multi-hostcall
- `trivial_async`, `one_yield`, `rustc_async_baseline_test` -- rustc async baseline

## Phase 3: warp-macro Removal

- Removed `warp-macro` from `gpu-kernel/Cargo.toml` dependencies
- Removed `crates/macro/warp-macro` from workspace `Cargo.toml` members
- Updated `scripts/ci-lint.sh` to remove warp-macro from fmt, clippy, and doc checks
- The `crates/macro/warp-macro/` directory is preserved for git history

## Phase 4: Verification

### Build verification:
- `cargo +nightly-2026-06-03 build --release` (gpu-kernel, nvptx64) -- PASS, no warnings
- `AUTO_BUILD_KERNEL=0 cargo +stable check -p gpu-host` -- PASS

### Test verification (all pass):
- `ONLY_TEST=warp_e2e` -- PASS (3 messages: start, ok, mixed)
- `ONLY_TEST=warp_try` -- PASS (file opened, print succeeded)
- `ONLY_TEST=warp_await` -- PASS (2 messages: hello, done)
- Full test suite warp tests:
  - WarpFuture PoC (warp-future.4) -- PASS
  - WarpFuture Multi-Hostcall (warp-future.6) -- PASS
  - WarpFuture Proc Macro (warp-future.5) -- PASS
  - WarpFuture If/Else (warp-cfg.2) -- PASS (both branches)
  - WarpFuture Loop/Break (warp-cfg.3) -- PASS
  - WarpFuture Match (warp-cfg.4) -- PASS (all 3 arms)
  - WarpFuture Nested (warp-cfg.5) -- PASS (all 4 paths)
  - Hybrid Executor (hybrid-executor.1) -- PASS
  - Hybrid Stress (hybrid-executor.2) -- PASS
  - Autonomous Pipeline (gpu-compute.2) -- PASS (all 3 modes)

### CI lint:
- `scripts/ci-lint.sh` updated to remove warp-macro references
- Pre-existing formatting issue in `crates/core/gpu-host/src/gpu.rs` (not from this change)

## Design Decision: Lane 0 Only vs Warp-Cooperative

The original WarpFuture approach had all 32 lanes cooperatively executing one state machine.
The new approach runs the async fn on lane 0 only (`if thread_idx_x() != 0 { return; }`).

This is correct because:
1. The standard futures (GpuPrintFuture, etc.) are per-thread -- each thread would independently
   allocate packets and submit hostcalls if all 32 ran
2. The tests expect specific message counts (e.g., 2 messages, not 64)
3. The MIR pass provides warp cooperation at the coroutine state machine level, but the inner
   future poll is still per-thread
4. For compute-heavy workloads, the MIR pass + standard async fn with all 32 threads running
   is the correct approach (each thread does its own work, coordination via the MIR pass)

## Files Changed

- `crates/kernel/gpu-kernel/src/warp.rs` -- rewrote 8 functions from DSL to standard async
- `crates/kernel/gpu-kernel/Cargo.toml` -- removed warp-macro dependency
- `crates/kernel/gpu-kernel/Cargo.lock` -- updated (warp-macro + proc-macro2/quote/syn removed)
- `Cargo.toml` -- removed warp-macro from workspace members
- `Cargo.lock` -- updated (warp-macro removed)
- `scripts/ci-lint.sh` -- removed warp-macro from fmt/clippy/doc checks
