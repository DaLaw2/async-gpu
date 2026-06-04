# coop-compute.1: Kernel-side matmul architecture design

**Status**: done
**Kind**: investigation
**Theme**: coop-compute — Kernel-side compute library

## Summary

Designed a kernel-side matmul callable from within `cooperative_map_with_params` context.
The naive MVP uses a scalar triple-loop partitioned by rows across warps, requires zero shared
memory, and fits naturally into the existing `CoopMapExtArgs` parameter passing. Estimated
performance is 0.05-0.15 GFLOPS — sufficient for the litmus test demo, with a clear upgrade
path to tiled GEMM for production use.

## Existing GEMM Landscape

The codebase has a rich set of HOST-LAUNCHED GEMM kernels in `compute_gemm.rs`:

| Kernel | Tile | Precision | Features |
|---|---|---|---|
| `gemm_f32` | 32x16 | f32 FMA | tiled, shared memory, 128 threads |
| `gemm_f32_v2` | 128x64 | f32 FMA | double-buffered, bank-conflict padding, 256 threads |
| `gemm_f32_v3` | 128x128 | f32 FMA | 8x8 register blocking, 256 threads |
| `full_gemm` | 32x16 | f16 MMA | Tensor Core, multi-block 2D tiling |
| `full_gemm_f32in` | 32x16 | f16 MMA | f32→f16 on-the-fly conversion |
| `full_gemm_bf16` | 32x16 | bf16 MMA | BF16 Tensor Cores (sm_80+) |
| `full_gemm_tf32` | 32x16 | tf32 MMA | TF32 Tensor Cores (sm_80+) |
| `full_gemm_splitk` | 32x16 | f16 MMA | K-dimension partitioning, atomicAdd |
| `int8_gemm_dp4a` | 1x1 | INT8 dp4a | per-element, INT32 accumulator |
| `int4_gemm_w4a16` | 1x1 | INT4/f32 | W4A16 dequant-on-the-fly |

**None** of these can be called from within a running kernel. They are all `extern "gpu-kernel"` entry
points that expect to own the entire block (thread indexing via `_thread_idx_x`, shared memory via
`get_dynamic_smem_ptr`, block indexing via `_block_idx_x`).

## The Constraint: cooperative_map Context

Inside `cooperative_map_with_params`, each warp has:
- **One lane executing**: Only lane 0 runs the user function (trampoline checks `lid == 0`)
- **No shared memory**: The cooperative framework doesn't allocate smem
- **No bar.sync**: Warps are independent; barrier-based algorithms don't work
- **No block indexing**: Single block, warps identified by `args.warp_id`
- **Parameters via `args.params: [u64; 4]`**: Up to 4 scalar values for M, K, N, etc.

This means the existing GEMM kernels (which rely on multi-thread cooperation within a warp,
shared memory, and bar.sync) cannot be directly reused.

## Design: Naive MVP (Option B)

### Function Signature

```rust
/// Kernel-side matmul: C[M,N] = A[M,K] x B[K,N], all row-major f32.
///
/// Called from cooperative_map_with_params with:
///   src = A pointer, dst = C pointer
///   params[0] = M, params[1] = K, params[2] = N, params[3] = B pointer
///
/// Each warp computes a subset of rows of C (striped by warp_id).
fn kernel_matmul(args: &CoopMapExtArgs) {
    let a = args.src as *const f32;
    let c = args.dst as *mut f32;
    let m = args.params[0] as usize;
    let k = args.params[1] as usize;
    let n = args.params[2] as usize;
    let b = args.params[3] as *const f32;

    // Each warp handles rows: warp_id, warp_id + n_warps, warp_id + 2*n_warps, ...
    let mut row = args.warp_id as usize;
    while row < m {
        let mut col = 0usize;
        while col < n {
            let mut sum = 0.0f32;
            let mut ki = 0usize;
            while ki < k {
                sum += unsafe {
                    core::ptr::read_volatile(a.add(row * k + ki))
                        * core::ptr::read_volatile(b.add(ki * n + col))
                };
                ki += 1;
            }
            unsafe {
                core::ptr::write_volatile(c.add(row * n + col), sum);
            }
            col += 1;
        }
        row += args.n_warps as usize;
    }
}
```

### Data Layout

- **A**: `[M, K]` row-major f32 — standard, matches most data sources
- **B**: `[K, N]` row-major f32 — standard row-major (not column-major)
- **C**: `[M, N]` row-major f32 — output

Row-major for all three matrices avoids the f16x2 packing and column-major layout
used by the existing MMA kernels. Since we're doing scalar FMA without Tensor Cores,
there's no hardware reason for special layouts.

### Caller Integration

```rust
// Inside unified_io_compute (the North Star demo):
let a_data: Vec<u8> = File::read("weights.bin");
let b_data: Vec<u8> = File::read("input.bin");
let mut c_data: Vec<u8> = vec![0u8; m * n * 4];

gpu_runtime::thread::cooperative_map_with_params(
    a_data.as_ptr() as *const u8,
    c_data.as_mut_ptr() as *mut u8,
    m * n,  // total output elements (used for documentation; actual partitioning is by row)
    [m as u64, k as u64, n as u64, b_data.as_ptr() as u64],
    kernel_matmul,
);

File::write("output.bin", &c_data);
```

### Why `len` = `m * n`

The `len` field in `CoopMapArgs` is typically "number of elements to process". For matmul,
this is M*N output elements. However, the actual partitioning is by ROW (not by element),
because each output element depends on an entire row of A and column of B — you can't
meaningfully split a single dot product across warps with this naive approach.

### Why B Pointer in params[3]

`cooperative_map_with_params` provides `(src, dst, len, params[4])`. The natural mapping is:
- `src` = A (the "input" matrix)
- `dst` = C (the "output" matrix)
- `params` = dimensions + extra pointers

B doesn't fit into the `src/dst` pair, so it goes into `params[3]` as a raw u64 pointer.
This works because B lives in heap-allocated global memory (Vec), accessible from all warps.

## Performance Estimate

### Naive Triple-Loop

For C[M,N] = A[M,K] x B[K,N]:
- FLOPs = 2 * M * K * N (one multiply + one add per element)
- Each warp (lane 0 only) does M/n_warps rows sequentially

With 4 warps on GTX 1660:
- Single lane = 1 thread at ~1.5 GHz
- Naive loop: ~1 FMA per 10-20 cycles (memory latency dominated for B column access)
- Effective: ~75-150 MFLOPS per lane
- 4 warps × 1 lane = ~300-600 MFLOPS total

For the demo (e.g., 64x64 matmul):
- FLOPs = 2 * 64 * 64 * 64 = 524,288
- At 300 MFLOPS: ~1.7 ms
- At 600 MFLOPS: ~0.9 ms

For 128x128: 2 * 128^3 = 4,194,304 FLOPs → ~7-14 ms.

### Comparison

| Implementation | GFLOPS | Notes |
|---|---|---|
| Naive kernel-side (this) | 0.3-0.6 | Single lane per warp, no smem |
| gemm_f32 (host-launched) | ~50-100 | 128 threads, tiled, shared memory |
| gemm_f32_v3 (host-launched) | ~200-400 | 256 threads, 8x8 blocking, double-buffered |
| cuBLAS | ~4,000-5,000 | Full GPU utilization |

### Is It Enough for the Demo?

**Yes.** The litmus test is "File::read → matmul → File::write in one kernel." The point is
the PROGRAMMING MODEL (I/O + compute unified), not peak performance. A 64x64 matmul at ~1ms
is instantaneous to a human observer.

## Upgrade Path (coop-compute.3)

For real workloads, the naive approach is ~100-1000x slower than host-launched GEMM.
The tiled upgrade (coop-compute.3) would:

1. **Use all 32 lanes**: Currently only lane 0 computes. The cooperative framework's trampoline
   checks `if lid == 0`, but we could add a variant that runs all lanes. Each lane would handle
   a different column of the output.

2. **Register blocking**: Each lane accumulates a small tile (e.g., 1x4 or 2x4) in registers.

3. **Sequential K-accumulation with prefetch**: Load B elements into registers ahead of use.

4. **Potential shared memory**: Would require extending the cooperative framework to allocate
   smem, or using a fixed shared memory region. This is a larger design change.

The critical insight is that the naive MVP and the tiled version share the SAME API and caller
code. The upgrade is purely internal to `kernel_matmul`.

## Design Decisions

1. **Row-major f32 for all matrices**: Simplest layout, no packing overhead, no precision loss.
   The existing MMA kernels use f16x2 packed layouts because Tensor Cores require it. Since
   we're on sm_75 (no tensor cores for this use case) and doing scalar FMA, f32 row-major
   is the right choice.

2. **Row-striped partitioning**: Each warp handles rows `[wid, wid+n_warps, wid+2*n_warps, ...]`.
   This gives good load balance when M >> n_warps. For small M (e.g., M=4, n_warps=4), each
   warp gets exactly one row — perfect.

3. **No shared memory for v1**: The cooperative framework runs each warp independently with
   only lane 0 active. Shared memory requires `bar.sync` for consistency, which is incompatible
   with the independent-warp model. The naive approach reads B from global memory (L1 cache
   helps for repeated accesses within a row).

4. **B pointer via params[3]**: The cooperative_map_with_params API naturally passes one input
   and one output pointer. Matmul has two inputs (A, B) and one output (C). Rather than
   creating a new API variant, we encode B as a u64 in the params array. This is idiomatic
   for the existing API.

5. **`read_volatile` / `write_volatile`**: Required because the pointers come from `Vec`
   allocations that the compiler doesn't know about at the function-pointer boundary.
   Without volatile, the compiler could optimize away the reads/writes.

## Files Changed

None (investigation only — design doc).
