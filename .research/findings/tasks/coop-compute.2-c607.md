# coop-compute.2: Naive kernel-side matmul MVP via cooperative_map_with_params

**Status**: done
**Kind**: experiment
**Theme**: coop-compute — Kernel-side compute library

## Summary

Implemented and verified a naive triple-loop matmul callable from `cooperative_map_with_params`.
C[8x6] = A[8x4] x B[4x6] with f32 row-major layout. All 48 output elements match CPU reference
within 1e-3 tolerance. Zero API changes required — the existing cooperative infrastructure is
sufficient for kernel-side compute.

## Implementation

### Kernel side (`crates/kernel/gpu-kernel/src/thread_test.rs`)

- Static arrays `MATMUL_A[32]` and `MATMUL_B[24]` as `AtomicU32` (f32 bit patterns via `to_bits()`)
- Helper `store_f32()` writes f32 into AtomicU32 slots
- `naive_matmul_kernel(args: &CoopMapExtArgs)` — triple-loop matmul:
  - `src` = A pointer, `dst` = C pointer
  - `params = [M, K, N, B_ptr]`
  - Row-striped partitioning: warp_id computes rows `i` where `i % n_warps == warp_id`
  - `read_volatile` / `write_volatile` for GPU memory access
- Entry point `cooperative_matmul_test(result: *mut f32)`:
  - Initializes A[i][j] = (i*K+j+1), B[i][j] = (i*N+j+1)*2
  - Calls `cooperative_map_with_params` with 4 warps

### Host side (`crates/test/gpu-test-harness/src/main.rs`)

- `ONLY_TEST=matmul` entry: launches `cooperative_matmul_test` with 48 f32 outputs, 128 threads
- CPU reference matmul with identical A/B initialization
- Element-wise comparison with 1e-3 tolerance

## Test Output

```
--- Cooperative Matmul Test (coop-compute.2) ---
  C[8x6] = A[8x4] x B[4x6], naive triple-loop via cooperative_map_with_params
  All 48 elements match CPU reference (tolerance 1e-3)
  Sample: C[0][0]=260.0, C[7][5]=3720.0
  Cooperative matmul: PASSED
```

## Key Observations

1. **No API changes needed**: `cooperative_map_with_params` with `[M, K, N, B_ptr]` params
   covers the matmul use case naturally. The `len` parameter is overloaded (output element
   count, not used for striding inside the callback).

2. **Static arrays work**: AtomicU32 statics for input data are visible to all warps in the
   no_std gpu-kernel crate. No heap allocation needed.

3. **f32 output via gpu::launch<f32>**: The generic launch API works seamlessly with f32
   output buffers, matching the kernel's `*mut f32` signature.

4. **Correctness at small scale**: With M=8, K=4, N=6 and integer-valued inputs, f32
   arithmetic introduces zero rounding error. The 1e-3 tolerance is conservative.

5. **Performance**: Not measured (not the goal). At 4 warps with lane-0-only execution,
   this is purely a correctness proof. The triple-loop inner kernel is O(M*K*N) per warp
   partition.

## Files Changed

- `crates/kernel/gpu-kernel/src/thread_test.rs` — added `cooperative_matmul_test` kernel
- `crates/test/gpu-test-harness/src/main.rs` — added `matmul` ONLY_TEST case with CPU verification
- `crates/core/gpu-host/kernel.ptx` — rebuilt with new kernel

## Next Steps

- coop-compute.3: Tiled matmul using all 32 lanes per warp (register blocking, shared memory)
- Larger matrix sizes to stress multi-warp partitioning
- Performance benchmarking against host-launched GEMM kernels
