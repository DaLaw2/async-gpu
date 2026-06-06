# coro-impl.1: Generator compilation to PTX + warp-cooperative yield

## Summary

Implemented the full GPU coroutine generator API in `gpu-runtime`: `GpuGenerator` trait, `WarpCoroutineState` enum, `WarpBroadcast` trait with implementations for all scalar types, `for_each_yield` zero-buffered streaming combinator, `GeneratorTask` future adapter, and a `CounterGenerator` reference implementation. All code compiles cleanly for nvptx64 (sm_75) in both debug and release modes. No MIR pass changes were needed.

## Baseline

- `gpu-runtime` compiled cleanly for nvptx64 before changes (0 warnings, 0 errors)
- `cargo +stable fmt --check` and `cargo +stable clippy -- -D warnings` passed
- No existing generator types in the codebase

## Implementation

### New file: `crates/core/gpu-runtime/src/generator.rs`

**WarpCoroutineState<Y, R>** — Two-variant enum mirroring `core::ops::CoroutineState`:
- `Yielded(Y)` — generator produced a value, all lanes see the same Y
- `Complete(R)` — generator finished, all lanes see the same R

**WarpBroadcast trait** — `unsafe trait` for broadcasting values from lane 0 to all lanes:
- Unit type `()`: no-op (0 bytes)
- 8/16/32-bit types (u8, u16, u32, i8, i16, i32, f32, bool): single `shfl.sync.idx.b32`
- 64-bit types (u64, i64, f64): two `shfl.sync.idx.b32` calls (lo/hi halves)
- Tuple `(u32, u32)`: two `shfl.sync.idx.b32` calls
- `warp_broadcast_via_smem<T>` helper for types > 128 bits (shared memory fallback)

**GpuGenerator<R=()> trait** — `unsafe trait` for warp-cooperative generators:
- `type Yield: WarpBroadcast + Copy`
- `type Return: WarpBroadcast + Copy`
- `fn resume_warp(&mut self, arg: R, wcx: &mut WarpContext) -> WarpCoroutineState<Yield, Return>`
- Mirrors `WarpFuture::poll_warp` pattern: lane-0 execution + broadcast

**for_each_yield combinator** — zero-buffered streaming pipeline:
- Inline loop: resume generator, process yielded value, repeat
- At most ONE yielded value exists at any time
- `syncwarp` between iterations for convergence

**GeneratorTask<G, F>** — Future adapter for executor integration:
- Wraps generator + consumer closure as `Future<Output = ()>`
- Each poll drives one `resume_warp` + consumer call
- Yielded → `Poll::Pending`, Complete → `Poll::Ready`

**CounterGenerator** — Reference implementation:
- Yields 0, 1, 2, ..., count-1
- Returns sum of all yielded values
- Demonstrates the `GpuGenerator` implementation pattern with proper broadcast

### Modified files

- `crates/core/gpu-runtime/src/lib.rs` — added `pub mod generator;` with documentation
- `crates/core/gpu-runtime/src/prelude.rs` — added re-exports for all public generator types

## Verification

1. **nvptx64 check (debug)**: `cargo +nightly check` — PASS (0 errors, 0 warnings)
2. **nvptx64 check (release)**: `cargo +nightly check --release` — PASS
3. **nvptx64 full build (release)**: `cargo +nightly build --release` — PASS (actual codegen)
4. **Formatting**: `cargo +stable fmt --check` — PASS
5. **Clippy**: `cargo +stable clippy -- -D warnings` — PASS
6. **CI lint**: `bash scripts/ci-lint.sh` — PASS

## Open Questions

1. **`gen fn` vs `#[coroutine]` syntax** — The `GpuGenerator` trait is designed for manual implementation now. When Rust stabilizes `gen fn` or the coroutine trait, a blanket impl adapter can bridge native coroutines to `GpuGenerator`. No API changes needed.

2. **Executor `spawn_generator` helper** — The design doc shows `GpuExecutor::spawn_generator()`. This is not implemented yet because it requires modifying `executor.rs` (deferred to coro-impl.2 or the multi-generator theme). Users can manually wrap with `GeneratorTask` and call `spawn()`.

3. **Per-lane yield values** — The current design broadcasts lane 0's value to all lanes. A `DataParallelGenerator` with per-lane yields would be a T2+ extension.

## Impact on Downstream

- **coro-impl.2** (streaming pipeline demo): All types are ready. Demo can use `CounterGenerator` or implement a fibonacci generator, wrap with `for_each_yield`, produce visible output.
- **MIR pass**: No changes needed (confirmed by design investigation). The existing `WarpCooperativeTransform` handles generator coroutine bodies — discriminant broadcast + return barrier work unchanged.
- **Multi-generator theme**: `GeneratorTask` enables multiple generators in the same `GpuExecutor` — each is just another `Future<Output=()>`.
