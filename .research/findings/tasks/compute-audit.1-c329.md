# compute-audit.1: Full scan of all compute functions in codebase
**Cycle**: 329 | **Theme**: compute-audit | **Kind**: investigation | **Status**: done

## Summary
Comprehensive audit of all compute-related functions across the codebase. Found 40+ compute
functions buried in kernel code, most reusable but none publicly exposed. Categorized into
6 groups: math intrinsics, warp primitives, block primitives, linear algebra (GEMM),
transformer/ML ops, and search/utility.

## Findings

### Source Files Scanned
- `crates/kernel/gpu-kernel/src/compute_gemm.rs` (~1428 lines)
- `crates/kernel/gpu-kernel/src/compute_transformer.rs` (~1091 lines)
- `crates/kernel/gpu-kernel/src/compute_math.rs` (55 lines — test only)
- `crates/kernel/gpu-kernel/src/compute_search.rs` (~500 lines)
- `crates/kernel/gpu-kernel/src/compute_mma.rs` (~200 lines)
- `crates/kernel/gpu-kernel/src/helpers.rs` (compute-relevant parts)
- `crates/core/gpu-atomics/src/lib.rs` (warp intrinsics)
- `crates/core/gpu-runtime/src/lib.rs` (executor, sync, channel)
- `examples/vector-math/kernel/src/lib.rs` (SAXPY, softmax)

### Q: What compute functions exist and are they reusable?

## Category 1: Math Intrinsics (EXTRACT → gpu_runtime::math)

| Function | Source | Reusable? | Notes |
|----------|--------|-----------|-------|
| `gpu_sqrtf(x: f32) -> f32` | helpers.rs:312 | YES | `sqrt.approx.f32` PTX |
| `gpu_exp_f32(x: f32) -> f32` | helpers.rs:290 | YES | `ex2.approx.f32` + log2(e) scaling |
| `gpu_exp(x: f32) -> f32` | vector-math kernel | YES | Same as above, duplicate |
| `gpu_instant_nanos() -> u64` | helpers.rs:146 | YES | `%globaltimer` PTX |

**Missing but needed:**
- `gpu_log_f32(x)` — `lg2.approx.f32`
- `gpu_sin_f32(x)` — `sin.approx.f32`
- `gpu_cos_f32(x)` — `cos.approx.f32`
- `gpu_rsqrt_f32(x)` — `rsqrt.approx.f32` (faster than 1/sqrt)
- `gpu_fma_f32(a, b, c)` — `fma.rn.f32`
- `gpu_abs_f32(x)` — `abs.f32`
- `gpu_min_f32(a, b)` / `gpu_max_f32(a, b)` — `min.f32` / `max.f32`
- `gpu_tanh_f32(x)` — `tanh.approx.f32` (SM75+)

## Category 2: Warp Primitives (EXTRACT → gpu_runtime::warp or gpu_atomics)

| Function | Source | Reusable? | Notes |
|----------|--------|-----------|-------|
| `warp_reduce_sum_f32(val) -> f32` | compute_transformer.rs:14 | YES | butterfly shfl.sync.bfly |
| `shfl_sync_idx_u32(mask, val, src_lane)` | gpu-atomics | YES (already public) | broadcast from lane |
| `syncwarp(mask)` | gpu-atomics | YES (already public) | bar.warp.sync |
| `lane_id() -> u32` | gpu-atomics | YES (already public) | %laneid |
| `activemask() -> u32` | gpu-atomics | YES (already public) | activemask.b32 |

**Missing but needed:**
- `warp_reduce_sum_u32(val) -> u32`
- `warp_reduce_max_f32(val) -> f32`
- `warp_reduce_min_f32(val) -> f32`
- `warp_prefix_sum_f32(val) -> f32` (inclusive scan)
- `warp_ballot(predicate) -> u32` — `vote.ballot.sync`
- `warp_all(predicate) -> bool` — `vote.all.sync`
- `warp_any(predicate) -> bool` — `vote.any.sync`
- `shfl_sync_up_u32(mask, val, offset)` — shuffle up
- `shfl_sync_down_u32(mask, val, offset)` — shuffle down
- `shfl_sync_bfly_u32(mask, val, offset)` — shuffle butterfly (used in reduce but not exposed)

## Category 3: Block Primitives (EXTRACT → gpu_runtime::block)

| Function | Source | Reusable? | Notes |
|----------|--------|-----------|-------|
| `bar_sync()` | helpers.rs:283 | YES | `bar.sync 0` |
| `get_dynamic_smem_ptr() -> *mut u8` | helpers.rs:271 | YES | `cvta.shared.u64` |

**Missing but needed:**
- `block_reduce_sum_f32(smem, val) -> f32` — uses shared memory + bar_sync
- `block_reduce_max_f32(smem, val) -> f32`
- `block_idx_x/y/z()` — wrapping nvptx intrinsics
- `block_dim_x/y/z()` — wrapping nvptx intrinsics
- `thread_idx_x/y/z()` — wrapping nvptx intrinsics
- `grid_dim_x/y/z()` — wrapping nvptx intrinsics

## Category 4: GEMM / Linear Algebra (EXTRACT → gpu_runtime::linalg)

| Function | Source | Reusable? | Notes |
|----------|--------|-----------|-------|
| `test_tiled_gemm` | compute_gemm.rs:21 | PARTIAL | MMA m16n8k16, tightly coupled to test |
| `full_gemm` | compute_gemm.rs:622+ | PARTIAL | f16 GEMM, kernel entry |
| `full_gemm_f32in` | compute_gemm.rs | PARTIAL | f32 input GEMM |
| `full_gemm_bf16` | compute_gemm.rs | PARTIAL | bfloat16 GEMM |
| `full_gemm_tf32` | compute_gemm.rs | PARTIAL | TensorFloat-32 GEMM |
| `gemm_f32` | compute_gemm.rs | PARTIAL | pure f32 GEMM (FMA) |
| `softmax_shared` | compute_gemm.rs:115 | YES | shared memory softmax |
| SAXPY | vector-math example | YES | `y[i] = a*x[i] + y[i]` |
| elementwise_mul | vector-math example | YES | `out[i] = x[i] * y[i]` |

**Note:** GEMM implementations are large kernel entry points with inline MMA PTX. Can be refactored
into: (a) MMA helper function, (b) tile loading helpers, (c) kernel that combines them.

## Category 5: Transformer / ML Ops (EXTRACT → gpu_runtime::nn)

| Function | Source | Reusable? | Notes |
|----------|--------|-----------|-------|
| `layer_norm` kernel | compute_transformer.rs:45 | YES | per-row norm with warp reduce |
| `gelu_activation` kernel | compute_transformer.rs:122 | YES | GELU(x) = x*0.5*(1+tanh(...)) |
| `attention_head` kernel | compute_transformer.rs:185 | PARTIAL | complex, multi-warp |
| `flash_attention` kernel | compute_transformer.rs:323 | PARTIAL | optimized attention |
| `flash_attention_kv` kernel | compute_transformer.rs:555 | PARTIAL | KV cache version |
| `embedding_lookup` kernel | compute_transformer.rs:774 | YES | simple table lookup |
| `bias_add` kernel | compute_transformer.rs:825 | YES | elementwise add bias |
| `elementwise_add` kernel | compute_transformer.rs:868 | YES | z[i] = x[i] + y[i] |
| `split_qkv` kernel | compute_transformer.rs:898 | PARTIAL | transformer-specific |
| `concat_heads` kernel | compute_transformer.rs:949 | PARTIAL | transformer-specific |
| `f32_to_f16x2_pack` kernel | compute_transformer.rs:993 | YES | format conversion |
| `kv_cache_append` kernel | compute_transformer.rs:1055 | PARTIAL | cache-specific |
| `softmax_exp` kernel | vector-math example | YES | exp(x - max) |
| `softmax_normalize` kernel | vector-math example | YES | x / sum |

## Category 6: Search / Utility

| Function | Source | Reusable? | Notes |
|----------|--------|-----------|-------|
| `grep_buffer` | helpers.rs:320 | PARTIAL | byte pattern search |
| vector similarity search | compute_search.rs | PARTIAL | complex WarpFuture |

## Recommended Public API Surface

### Must-have (high value, easy to extract):
1. **Math intrinsics**: sqrt, exp, log, sin, cos, rsqrt, fma, abs, min, max, tanh
2. **Warp reduce**: sum_f32, max_f32, min_f32 (butterfly pattern)
3. **Thread/block indexing**: thread_idx, block_idx, block_dim, grid_dim wrappers
4. **Block sync**: bar_sync, get_dynamic_smem_ptr
5. **Timer**: gpu_instant_nanos

### Should-have (medium effort):
6. **Warp vote**: ballot, all, any
7. **Warp shuffle variants**: up, down, butterfly (raw)
8. **Block-level reduce**: shared memory reduction
9. **Element-wise ops**: SAXPY, add, mul as generic GPU functions
10. **Activation functions**: GELU, softmax (as callable functions, not kernels)
11. **LayerNorm** as callable function

### Nice-to-have (significant refactoring):
12. **GEMM**: Extract MMA helper, tile loaders, composable GEMM
13. **Attention**: Too complex for simple extraction, keep as examples
14. **f16/bf16 conversion utilities**

## Open Questions
- Should compute utils go in `gpu-runtime` or a new `gpu-compute` crate?
  - Recommendation: `gpu-runtime::compute` submodules (math, warp, block, nn)
  - Keeps dependency graph simple, no new crate needed
- Should extracted functions be `unsafe fn` or safe wrappers?
  - Math intrinsics can be safe (no pointers)
  - Warp/block ops need `unsafe` (thread coordination assumptions)
  - Pointer-based ops (SAXPY etc.) definitely `unsafe`

## Impact on Downstream Tasks
- **compute-audit.2**: Use this inventory to design the API surface
- **compute-extract.1-3**: Extract functions per category
**Confidence**: high (direct source code analysis)
