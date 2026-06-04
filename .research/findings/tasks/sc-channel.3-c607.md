# sc-channel.3 — Channel Throughput Benchmark

## Status: done
## Summary:
Added `sc_channel_bench` kernel to `sc_demo.rs` that measures channel throughput
across all four transport modes: block-scoped oneshot, block-scoped MPSC,
global-memory oneshot, and global-memory MPSC. Uses `clock_nanos()` (%globaltimer
PTX register) for nanosecond-precision timing. CI lint passes clean.

## Implementation

### Benchmark kernel: `sc_channel_bench`

Accepts a global memory pool (for global-scope channel tests) and an output buffer.
Runs four sequential benchmarks, each doing 1024 iterations of send/recv:

1. **Block oneshot** (`bench_block_oneshot`): Each iteration creates a fresh
   `BlockOneshotSlot<u32>` in shared memory via `block_scope`, spawns producer
   (warp 1) and consumer (warp 2). Measures full create-send-recv-join cycle.

2. **Block MPSC** (`bench_block_mpsc`): Single `block_scope` with a
   `BlockMpscChannel<u32, 8>` ring buffer in shared memory. Producer warp sends
   1024 messages, consumer warp receives all via `recv_spin()`.

3. **Global oneshot** (`bench_global_oneshot`): Uses `OneshotSlot<u32>` placed at
   the start of the pre-allocated global memory pool. Each iteration resets the
   slot, spawns producer (system-scope release store) and consumer (system-scope
   acquire spin-poll on state field).

4. **Global MPSC** (`bench_global_mpsc`): Uses `MpscChannel<u32, 8>` placed in
   the global pool. Producer sends 1024 messages via `try_send()` with
   system-scope atomics, consumer receives via `try_recv()` spin loop.

### Output layout (10 × u32)

| Slots     | Content                          |
|-----------|----------------------------------|
| [0..1]    | Block oneshot total_ns (u64 LE)  |
| [2..3]    | Block MPSC total_ns (u64 LE)     |
| [4..5]    | Global oneshot total_ns (u64 LE) |
| [6..7]    | Global MPSC total_ns (u64 LE)    |
| [8]       | N (iterations = 1024)            |
| [9]       | Success flag (1)                 |

### Launch config

- Grid: (1, 1, 1), Block: (128, 1, 1), Shared memory: 4096 bytes
- Global pool: >= 4096 bytes of device memory

### Expected results

- CTA-scope channels (block oneshot/MPSC) should be 10-50x faster than
  system-scope channels (global oneshot/MPSC) due to ~2-6 cycle CTA atomics
  vs ~100 cycle system-scope atomics.

## Files Changed:
- `crates/kernel/gpu-kernel-std/src/sc_demo.rs` — added `sc_channel_bench` kernel
  and four helper benchmark functions (`bench_block_oneshot`, `bench_block_mpsc`,
  `bench_global_oneshot`, `bench_global_mpsc`), added `BlockMpscChannel` import,
  added `write_u64_pair` helper, added `BENCH_ITERS` constant.
