# Structured Concurrency on GPU

Lifetime-bounded concurrency scopes on the GPU — `BlockScope`, `GridScope`,
shared-memory channels, `spawn_all`, and nested scopes with automatic memory
reclamation.

## What It Demonstrates

- **Producer-Consumer Pipeline** — two warps communicate via a
  `BlockOneshotSlot` inside a `BlockScope`. The producer fills a buffer,
  signals completion; the consumer waits, sums the data, returns the result.
- **Cooperative spawn_all** — all warps process data in parallel via
  `scope.spawn_all()`. Each warp handles a stride of elements. The scope
  ensures all warps finish before results are read.
- **Nested Scopes** — an outer scope allocates a persistent buffer; an inner
  scope allocates scratch space for a worker. When the inner scope exits, its
  memory is reclaimed (watermark pop) while the outer buffer survives.
- **Combined Primitives** — producer-consumer pipeline followed by cooperative
  reduction in a single scope, showing `spawn` + `join_all` + `spawn_all`
  composability.
- **GridScope Multi-Block Reduce** — grid-level structured concurrency with
  global memory allocation, atomic completion tracking, and parallel reduce
  across virtual blocks.

## Running

```bash
cd examples/hostcall/structured-concurrency
cargo run --release
```

## How It Works

### Kernel Side

- `gpu_runtime::scope::block_scope` — creates a lifetime-bounded scope
- `scope.alloc::<T>(n)` — bump-allocates from shared memory
- `scope.spawn(closure)` — dispatches a closure to an idle warp
- `scope.spawn_all(|wid, n_warps| { ... })` — data-parallel across all warps
- `block_oneshot(slot)` — shared-memory oneshot channel (~2 cycle latency)
- `gpu_runtime::scope::grid_scope` — grid-level scope with global memory pool

### Host Side

The host launches each kernel via `gpu::custom()`, allocates output buffers,
and verifies results against expected values.

## Expected Output

```
=== Structured Concurrency on GPU ===

--- Demo 1: Producer-Consumer Pipeline ---
  Producer filled 64-element shared buffer
  Consumer received signal, computed sum = 2016
  Verification: PASSED

--- Demo 2: Cooperative spawn_all ---
  All warps cooperatively doubled 128 elements
  Warp 0 summed output = 16256
  Verification: PASSED

--- Demo 3: Nested Scopes ---
  Outer scope: 64-element buffer (data[i] = i + 10)
  Inner scope: worker summed outer_buf[0..16] = 280
  After inner scope exit: outer_buf[0] = 10 (still valid)
  Memory reclaimed: yes
  Verification: PASSED

--- Demo 4: Combined spawn + spawn_all ---
  Phase 1: producer fills, consumer triples via oneshot signal
  Phase 2: join_all ensures pipeline complete
  Phase 3: spawn_all computes partial sums across warps
  Final reduced sum = 1488
  Verification: PASSED

--- Demo 5: GridScope Multi-Block Reduce ---
  GridScope allocated input (128 elements) from global pool
  Coordinator reduced to final sum = 8256
  Verification: PASSED

=== All demos passed! ===
```
