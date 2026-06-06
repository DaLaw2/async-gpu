# at-framework.1: Survey Tunable Kernel Parameters

**Status**: DONE
**Kind**: investigation

## Summary

Comprehensive survey of all tunable kernel launch parameters in async-gpu, their current
defaults, performance impact, and existing infrastructure that an auto-tuning framework
can leverage.

## Findings

### 1. Launch Parameters: Current State

Three parameters compose `cudarc::driver::LaunchConfig`:

| Parameter | Type | Description |
|-----------|------|-------------|
| `block_dim` | `(u32, u32, u32)` | Threads per block (x, y, z) |
| `grid_dim` | `(u32, u32, u32)` | Blocks per grid (x, y, z) |
| `shared_mem_bytes` | `u32` | Dynamic shared memory per block |

Additional tunable parameters not in LaunchConfig but affecting performance:

| Parameter | Description |
|-----------|-------------|
| Stream selection | Default stream vs dedicated `GpuStream` (ADR-20 two-tier model) |
| Kernel variant | V1/V2/V3/V4.1 dispatch (e.g., GEMM has 4 versions) |
| Tile dimensions | Compile-time constants baked into kernel variants |

### 2. Current Default Values

**Block size defaults by API layer:**

| API | Default block_dim.x | Where |
|-----|---------------------|-------|
| `gpu::run()` | 128 | `gpu.rs:107` — hardcoded |
| `gpu::run_with_output()` | 128 | `gpu.rs:147` — hardcoded |
| `gpu::launch()` | caller-specified | `gpu.rs:181` — parameter |
| `gpu::custom()` builder | 128 | `gpu.rs:665` — default, overridable via `.threads(n)` |
| `gpu::run_zero_param()` | 128 | `gpu.rs:256` — hardcoded |
| `KernelRegistry::config_1d()` | 256 | `nn/registry.rs:155` — hardcoded |
| `KernelRegistry::config_gemm()` | 128 | `nn/registry.rs:171` — hardcoded |
| `KernelRegistry::config_layernorm()` | 32 | `nn/registry.rs:182` — 1 warp |
| `KernelRegistry::config_attention()` | 32 | `nn/registry.rs:193` — 1 warp |
| `KernelRegistry::config_embedding()` | 256 | `nn/registry.rs:205` — hardcoded |
| `AutoScheduler::gpu_par_map()` | 256 | `scheduler.rs:300` — hardcoded |

**Block size in NN ops (actual kernel launches):**

| Kernel | block_dim | shared_mem | Grid formula |
|--------|-----------|------------|--------------|
| `layer_norm_v2/v3` | (256,1,1) | 2048 B | (num_rows, 1, 1) |
| `gemm_f32` | (128,1,1) | 3072 B | (M/32, N/16, 1) |
| `gemm_f32_v2` | (256,1,1) | 12544 B | (M/128, N/64, 1) |
| `gemm_f32_v3` | (256,1,1) | 16640 B | (M/128, N/128, 1) |
| `flash_attention` | (32,1,1) | 2*32*d*4 B | (1, ceil(S/32), 1) |
| `flash_attn_v3` | (128,1,1) | 16640 B | (n_heads, ceil(S/32), 1) |
| Elementwise ops | (256,1,1) | 0 | (ceil(N/256), 1, 1) |
| YOLO GEMM conv | (128,1,1) | 3072 B | from tile size |
| Hostcall kernels | (128,1,1) | 0 | (1, 1, 1) |
| Warp cooperative | (128,1,1) | 0 | (1, 1, 1) — 4 warps |
| Structured concurrency | (128,1,1) | 0-2048 B | (1, 1, 1) |

**Grid dimension defaults:**

- Hostcall kernels: always `(1, 1, 1)` — single block
- Elementwise: `(ceil(N / block_size), 1, 1)` — 1D coverage
- GEMM: 2D tiling `(ceil(M/tile_m), ceil(N/tile_n), 1)`
- Attention: `(n_heads, ceil(seq_len/tile), 1)` — 2D
- Always 1D block_dim (y=1, z=1), all parallelism in x

**Shared memory defaults:**

- Most ops: 0 (no dynamic shared memory)
- LayerNorm: 2048 B (reduction scratch)
- GEMM V1: 3072 B (32x16 + 16x16 tiles * 4 bytes)
- GEMM V2: 12544 B (128x64 + 64x64 tiles)
- GEMM V3: 16640 B (128x128 tiles + padding)
- Attention: 2 * tile_size * d_head * 4 bytes (K+V tiles)

### 3. Performance Impact Ranking

1. **Block size** — #1 impact. Controls occupancy, register pressure trade-off, warp scheduling.
   On sm_75 (GTX 1660): max 1024 threads/SM, 65536 regs/SM. Block size directly determines
   how many blocks fit per SM.

2. **Kernel variant selection** — #2 impact. V4.1 GEMM is ~90% cuBLAS vs V1 at ~30%.
   The dispatch logic in `matmul_auto()` already selects V2 vs V3 vs V4.1 based on matrix size.

3. **Shared memory allocation** — #3 impact. Determines tile size for GEMM/attention. More smem
   = larger tiles = fewer global memory accesses, but fewer blocks per SM. The sm_75 L1/smem
   partition is configurable (48KB max smem).

4. **Grid dimensions** — #4 impact. Must cover the problem size. Too few blocks = GPU underutilized.
   Too many blocks with too few elements = overhead dominates. 1D vs 2D tiling matters for
   memory coalescing in GEMM.

5. **Stream selection** — #5 impact. Only matters for multi-kernel pipelines. Current codebase
   mostly uses default stream. `GpuStream` exists but limited to pure compute (no hostcall).

### 4. Compile-Time Cost Model Integration

`resource_report.rs` provides:
- `parse_ptxas_output()` → per-kernel register count, spills, stack
- `KernelResources::occupancy(sm, block_size)` → theoretical occupancy %
- `SmConfig::sm_75()` → hardware limits (65536 regs, 1024 max threads, 16 max blocks, 48KB smem)
- `analyze_warnings()` → actionable recommendations (target register count for next occupancy tier)

**Key constraint from compile-time analysis:**
Register count per kernel sets an upper bound on block size. E.g., 255 regs → max 25% occupancy
at block=256. The occupancy calculator can determine the optimal block size given the register
count: `max_block = (regs_per_sm / (regs_per_thread * warp_size).ceil_to(256)) * warp_size`.

### 5. Auto-Tuning Loop Design

Proposed pipeline:

```
1. Candidate generation
   - Block sizes: [32, 64, 128, 256, 512, 1024] (filtered by register limit)
   - Grid dims: computed from problem size / block size
   - Shared mem: [0, kernel-specific tiers based on tile size]

2. Constraint filtering (static, from resource_report)
   - Reject block sizes that exceed max_threads_per_block for the kernel's register count
   - Reject shared mem that exceeds smem_per_sm
   - Reject configurations with < 12.5% occupancy

3. Warmup runs (N=3)
   - Each candidate config runs N times to warm caches and JIT

4. Timing runs (N=10)
   - Use CUDA events (cuEventCreate, cuEventRecord, cuEventElapsedTime) for GPU-side timing
   - OR host-side Instant::now() with synchronize (current benchmark approach)
   - Record median time (robust to outliers)

5. Selection
   - Pick config with lowest median time
   - Cache result keyed by (kernel_name, problem_size_bucket, device_ordinal)

6. Caching
   - In-memory HashMap for session lifetime
   - Optional disk persistence for cross-session reuse
```

### 6. Existing Reusable Infrastructure

| Component | Location | Reuse for auto-tuning |
|-----------|----------|----------------------|
| `LaunchConfig` | cudarc | Config struct — direct reuse |
| `CustomLaunchBuilder` | `gpu.rs:658-912` | Already has `.threads()`, `.grid()`, `.shared_mem()` |
| `resource_report` | `resource_report.rs` | Occupancy calculator, register limits, constraint filtering |
| `SmConfig::sm_75()` | `resource_report.rs:518` | Hardware constants for sm_75 |
| `KernelResources::occupancy()` | `resource_report.rs:590` | Predict occupancy without running |
| `kernel-resources.sh` | `scripts/` | External ptxas analysis (JSON output with `--json`) |
| Benchmark infra | `examples/std/benchmark/` | Warmup+timing pattern (Instant + synchronize) |
| `GpuStream` | `streams.rs` | Timing on separate streams |
| `GpuRuntime::launch_config()` | `runtime.rs:111` | Config construction helper |

**Missing infrastructure:**
- No CUDA event timing API (cuEventCreate/Record/ElapsedTime) — would need raw driver calls
- No tuning result cache
- No candidate generator
- No problem-size bucketing

### 7. How Other Frameworks Do This

- **CUDA Occupancy API** (`cuOccupancyMaxPotentialBlockSize`): Given a kernel function handle,
  returns the block size that maximizes occupancy accounting for register count and shared memory.
  This is the simplest approach — single API call, no benchmarking needed. Available via
  `cudarc::driver::sys::lib().cuOccupancyMaxPotentialBlockSize()`.

- **cuDNN**: Pre-selects algorithms via `cudnnFindConvolutionForwardAlgorithm()`. Runs all
  candidate algorithms and returns sorted by performance. Similar to our proposed tuning loop.

- **Triton**: JIT compiles kernels with different tile sizes and auto-tunes at first call.
  Caches results per (kernel, input shape). Config space defined by `@triton.autotune` decorator.

- **TVM/Ansor**: Full search over schedule space with cost models + measurement. Overkill for
  our use case.

**Recommended approach for async-gpu:**
1. Start with `cuOccupancyMaxPotentialBlockSize` for block size (free, no benchmarking)
2. Add empirical tuning for block size + shared mem for hot kernels (GEMM, attention)
3. Cache tuned configs per (kernel, problem_size_range, device)

## Open Questions

1. Should auto-tuning run at first kernel call (lazy) or at module load time (eager)?
   - Lazy is simpler but adds latency to first call
   - Eager is predictable but wastes time if some kernels never run

2. How to handle problem-size-dependent optimal configs?
   - GEMM optimal block size may differ for 512x512 vs 4096x4096
   - Bucket by powers of 2? Or just tune for the first size seen?

3. Should we expose `cuOccupancyMaxPotentialBlockSize` from cudarc's raw driver API?
   - It's available via `cudarc::driver::sys::lib()` but not wrapped
   - Would give us the occupancy-optimal block size with zero overhead

4. Where does the tuning cache live?
   - `GpuRuntime` field? Global static? Separate `TuningCache` struct?
   - Must handle multi-device scenarios (different GPUs have different optima)
