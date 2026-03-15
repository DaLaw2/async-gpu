# compute-audit.2: API surface design for gpu_runtime::compute
**Cycle**: 329 | **Theme**: compute-audit | **Kind**: design | **Status**: done

## Summary
Defines the public API surface for GPU compute utilities. Functions are organized into
submodules of `gpu_runtime`: math, warp, block, nn. Each function is categorized by
safety, priority, and extraction difficulty.

## API Design

### Module Structure
```
gpu_runtime::
├── math       # Math intrinsics (safe functions)
├── warp       # Warp-level primitives (unsafe — thread coordination)
├── block      # Block-level primitives (unsafe — thread coordination)
├── nn         # Neural network / ML building blocks (unsafe — pointer-based)
└── index      # Thread/block/grid indexing helpers (safe wrappers)
```

### gpu_runtime::math (safe — no pointers, no coordination)

```rust
// Approximate math via PTX special function units
pub fn sqrt_f32(x: f32) -> f32;       // sqrt.approx.f32
pub fn rsqrt_f32(x: f32) -> f32;      // rsqrt.approx.f32
pub fn exp_f32(x: f32) -> f32;        // ex2.approx.f32 * log2(e)
pub fn log_f32(x: f32) -> f32;        // lg2.approx.f32 * ln(2)
pub fn sin_f32(x: f32) -> f32;        // sin.approx.f32
pub fn cos_f32(x: f32) -> f32;        // cos.approx.f32
pub fn abs_f32(x: f32) -> f32;        // abs.f32
pub fn min_f32(a: f32, b: f32) -> f32; // min.f32
pub fn max_f32(a: f32, b: f32) -> f32; // max.f32
pub fn fma_f32(a: f32, b: f32, c: f32) -> f32; // fma.rn.f32 (a*b+c)
pub fn tanh_f32(x: f32) -> f32;       // computed from exp
pub fn sigmoid_f32(x: f32) -> f32;    // 1 / (1 + exp(-x))
```

All safe because they operate on values, not pointers. Stubs on non-nvptx64 targets.

### gpu_runtime::warp (unsafe — requires warp-level coordination)

```rust
// Reductions (butterfly shuffle pattern, result in ALL lanes)
pub unsafe fn warp_reduce_sum_f32(val: f32) -> f32;
pub unsafe fn warp_reduce_sum_u32(val: u32) -> u32;
pub unsafe fn warp_reduce_max_f32(val: f32) -> f32;
pub unsafe fn warp_reduce_min_f32(val: f32) -> f32;

// Shuffle variants (expose raw shuffle ops)
pub unsafe fn shfl_sync_bfly_u32(mask: u32, val: u32, offset: u32) -> u32;
pub unsafe fn shfl_sync_up_u32(mask: u32, val: u32, delta: u32) -> u32;
pub unsafe fn shfl_sync_down_u32(mask: u32, val: u32, delta: u32) -> u32;

// Vote / ballot
pub unsafe fn warp_ballot(mask: u32, predicate: bool) -> u32;
pub unsafe fn warp_all(mask: u32, predicate: bool) -> bool;
pub unsafe fn warp_any(mask: u32, predicate: bool) -> bool;

// Already public in gpu-atomics: shfl_sync_idx_u32, syncwarp, lane_id, activemask
```

### gpu_runtime::block (unsafe — requires block-level coordination)

```rust
// Synchronization
pub unsafe fn bar_sync();                          // bar.sync 0
pub unsafe fn shared_mem_ptr() -> *mut u8;         // cvta.shared.u64

// Typed shared memory access
pub unsafe fn shared_mem_as<T>(offset: usize) -> *mut T;

// Block-level reductions (uses shared memory + bar_sync)
// Requires: shared memory allocated in launch config
pub unsafe fn block_reduce_sum_f32(smem: *mut f32, val: f32, tid: u32, block_size: u32) -> f32;
pub unsafe fn block_reduce_max_f32(smem: *mut f32, val: f32, tid: u32, block_size: u32) -> f32;
```

### gpu_runtime::index (safe — read-only hardware registers)

```rust
pub fn thread_idx_x() -> u32;
pub fn thread_idx_y() -> u32;
pub fn thread_idx_z() -> u32;
pub fn block_idx_x() -> u32;
pub fn block_idx_y() -> u32;
pub fn block_idx_z() -> u32;
pub fn block_dim_x() -> u32;
pub fn block_dim_y() -> u32;
pub fn block_dim_z() -> u32;
pub fn grid_dim_x() -> u32;
pub fn grid_dim_y() -> u32;
pub fn grid_dim_z() -> u32;

// Convenience: global thread index for 1D grids
pub fn global_thread_idx() -> u32;  // block_idx_x * block_dim_x + thread_idx_x
pub fn global_thread_count() -> u32; // grid_dim_x * block_dim_x

// Timer
pub fn clock_nanos() -> u64;  // %globaltimer
```

### gpu_runtime::nn (unsafe — pointer-based, GPU-parallel)

```rust
// Activation functions (element-wise, called per-thread)
pub unsafe fn gelu_f32(x: f32) -> f32;    // GELU activation
pub unsafe fn relu_f32(x: f32) -> f32;    // max(0, x)

// Warp-cooperative operations (must be called by all lanes)
pub unsafe fn warp_softmax_f32(val: f32) -> f32;  // softmax across warp lanes
pub unsafe fn warp_layer_norm_f32(val: f32, gamma: f32, beta: f32) -> f32;

// Kernel-level ops (full block participates)
// These are higher-level — may stay as example kernels rather than utils
```

## Decision: Where to Put What

| Location | What goes there | Rationale |
|----------|----------------|-----------|
| `gpu_runtime::math` | All scalar math intrinsics | Safe, no coordination needed |
| `gpu_runtime::warp` | Shuffle, reduce, vote | Needs all lanes active |
| `gpu_runtime::block` | bar_sync, shared mem, block reduce | Needs all threads in block |
| `gpu_runtime::index` | Thread/block/grid indexing | Safe hardware register reads |
| `gpu_runtime::nn` | GELU, ReLU, warp softmax | ML-specific building blocks |
| `gpu-atomics` | Keep existing: shfl_idx, syncwarp, lane_id, atomics | Already public, don't move |
| Example kernels | GEMM, attention, full pipelines | Too complex for utils, keep as demos |

## Extraction Priority

### Phase 1 — compute-extract.1 (warp primitives)
- `warp_reduce_sum_f32` — extract from compute_transformer.rs
- `shfl_sync_bfly_u32` — new, based on existing ASM pattern
- `warp_ballot`, `warp_all`, `warp_any` — new PTX intrinsics

### Phase 2 — compute-extract.2 (block primitives + indexing + math)
- All `gpu_runtime::math` functions
- All `gpu_runtime::index` functions
- `bar_sync`, `shared_mem_ptr`, `block_reduce_sum_f32`

### Phase 3 — compute-extract.3 (ML/NN ops)
- `gelu_f32`, `relu_f32`, `warp_softmax_f32`, `warp_layer_norm_f32`
- Refactor existing kernel code to USE these new utils (dogfooding)

## Open Questions
- None — design is straightforward extraction of proven code.

## Impact on Downstream Tasks
- **compute-extract.1**: Extract warp primitives per Phase 1
- **compute-extract.2**: Extract block/math/index per Phase 2
- **compute-extract.3**: Extract NN ops per Phase 3
- **demo-pipeline.1**: Can design demo once API surface is known
