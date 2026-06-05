# showcase-sc.1 — Structured Concurrency Example

## Status: DONE

## What was built

Created `examples/hostcall/structured-concurrency/` — a standalone host-side
example that showcases all five structured concurrency patterns on GPU.

### Files created

- `examples/hostcall/structured-concurrency/Cargo.toml` — standalone crate, depends on `gpu-host`
- `examples/hostcall/structured-concurrency/src/main.rs` — five demos with verification
- `examples/hostcall/structured-concurrency/run.sh` — launcher script

### Files modified

- `scripts/ci-lint.sh` — added `check structured-concurrency` to example host checks

## What the example demonstrates

1. **Producer-Consumer Pipeline** (`sc_producer_consumer`)
   - Two warps communicate via `BlockOneshotSlot` in shared memory
   - Producer fills buffer, signals via oneshot; consumer sums data
   - Expected: sum of 0..64 = 2016

2. **Cooperative Data-Parallel** (`sc_cooperative_parallel`)
   - `scope.spawn_all()` distributes work across all warps
   - Each warp doubles elements at stride offsets
   - Expected: sum of (i*2) for i in 0..128 = 16256

3. **Nested Scopes** (`sc_nested_scopes`)
   - Outer scope allocates persistent buffer, inner scope allocates scratch
   - Inner scope exits: scratch freed (watermark pop), outer buffer survives
   - Expected: inner_sum=280, outer_buf[0]=10, memory reclaimed

4. **Combined spawn + spawn_all** (`sc_combined_demo`)
   - Phase 1: producer/consumer via oneshot (spawn)
   - Phase 2: join_all ensures pipeline complete
   - Phase 3: cooperative reduction (spawn_all)
   - Expected: sum of (i*3) for i in 0..32 = 1488

5. **GridScope Multi-Block Reduce** (`sc_grid_reduce`)
   - Grid-level scope with global memory pool allocation
   - Worker warps act as virtual blocks with atomic completion counter
   - Expected: sum of 1..=128 = 8256

## Design decisions

- **Uses `gpu::custom()` builder** instead of `gpu::launch()` because the SC
  kernels require `shared_mem_bytes > 0` which `gpu::launch()` doesn't support.
- **No separate kernel crate** — the kernel code already exists in
  `crates/kernel/gpu-kernel-std/src/sc_demo.rs` and is embedded in the PTX.
  The example only needs a host-side launcher.
- **Follows the `thread-demo` pattern** — standalone crate with `[workspace]`,
  depends on `gpu-host`, no build.rs needed.

## Verification

- `cargo +stable check` passes
- `bash scripts/ci-lint.sh` — all checks pass including the new example
