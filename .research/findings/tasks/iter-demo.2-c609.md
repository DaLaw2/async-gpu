# iter-demo.2: GPU par_iter vs CPU Rayon benchmark

**Status**: done
**Kind**: experiment

## What was done

Added a benchmark comparing GPU `par_iter().map(|x| x * 2.0 + 1.0).collect()`
against CPU Rayon `par_iter().map(|x| x * 2.0 + 1.0).collect()` across six
data sizes (1K to 16M f32 elements).

The GPU kernel is `par_iter_map_collect` which uses 1 block x 128 threads
(4 warps) with warp-striped iteration. The CPU benchmark uses Rayon with
all available cores.

## Key finding: shared memory bug

Discovered that all existing par_iter tests crash with `CUDA_ERROR_ILLEGAL_ADDRESS`
because `launch_config()` uses `shared_mem_bytes: 0` but the kernels call
`init_shared_mem_allocator(512)` which accesses `.extern .shared dynamic_smem[]`.
Fixed the benchmark by using `shared_mem_bytes: 1024` in the launch config.
The existing tests in `run_all_par_iter_tests` and `run_par_iter_1m_test` still
have this bug (pre-existing, not introduced by this task).

## Benchmark results

| N      | CPU seq (ms) | Rayon (ms) | GPU e2e (ms) | GPU kernel (ms) | GPU/Rayon |
|--------|-------------|------------|-------------|-----------------|-----------|
| 1K     | 0.000       | 0.044      | 0.215       | 0.183           | 4.9x      |
| 10K    | 0.002       | 0.057      | 1.647       | 1.601           | 28.9x     |
| 100K   | 0.037       | 0.118      | 13.152      | 15.734          | 111.8x    |
| 1M     | 0.337       | 0.127      | 149.339     | 148.368         | 1178.3x   |
| 4M     | 1.449       | 1.009      | 608.313     | 602.687         | 602.7x    |
| 16M    | 45.028      | 10.664     | 3431.652    | 2506.865        | 321.8x    |

GPU end-to-end includes: host-to-device copy + kernel + device-to-host copy.

## Crossover point

**GPU never beats Rayon at any tested size.** The GPU is 5x to 1178x slower.

## Root cause analysis

1. **Single-block launch**: 1 of 22 SMs active = 4.5% GPU utilization
2. **Volatile memory access**: The warp-parallel iterator uses `read_volatile`
   and `write_volatile` (visible in PTX as `ld.volatile.global` / `st.volatile.global`),
   bypassing L1/L2 cache. At 16M elements, effective bandwidth is ~54 MB/s
   vs GTX 1660's 192 GB/s peak — 0.03% memory bandwidth utilization.
3. **spawn_all overhead**: The warp-parallel dispatch model writes function
   pointers and data pointers to global memory atomics for each warp's work,
   adding fixed overhead per launch.
4. **Transfer overhead**: For small sizes, PCIe htod+dtoh dominates (0.2ms at 1K).

## What would make GPU competitive

- **Multi-block launch** (22+ blocks): 22x improvement from full SM utilization
- **Non-volatile loads**: Use regular `ld.global` to benefit from L2 cache (6 MB on 1660)
- **Larger blocks**: 256 or 512 threads per block for better occupancy
- **Estimated crossover**: With multi-block + cached loads, GPU should beat
  Rayon at ~100K-1M elements (based on 192 GB/s peak vs Rayon's ~10 GB/s
  at 16M elements)

## Files changed

- `crates/test/gpu-test-harness/Cargo.toml` — added `rayon = "1"` dependency
- `crates/test/gpu-test-harness/src/tests_par_iter.rs` — added benchmark:
  `bench_launch_config`, `load_par_iter_module_fast`, `bench_gpu_par_iter`,
  `bench_rayon_par_iter`, `bench_cpu_sequential`, `run_par_iter_rayon_benchmark`
- `crates/test/gpu-test-harness/src/main.rs` — added `ONLY_TEST=par_iter_bench`
  entry point
