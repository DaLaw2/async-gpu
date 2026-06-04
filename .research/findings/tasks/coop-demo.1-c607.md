# coop-demo.1: File::read → matmul → File::write in One Kernel

**Status**: done
**Kind**: experiment

## Summary

Implemented and verified the North Star litmus test: a single GPU kernel
reads two matrix files via `std::fs::File`, performs cooperative matmul
across all warps using `cooperative_map_with_params`, and writes the
result back via `std::fs::File`. All 48 output elements match the CPU
reference within 1e-3 tolerance. Kernel runs in 4.6ms on GTX 1660.

## What Was Built

### Kernel (`crates/kernel/gpu-kernel-std/src/lib.rs`)

- `matmul_callback` — naive matmul fn for `cooperative_map_with_params`:
  each warp computes rows i where i % n_warps == warp_id, triple-loop
  inner product per element.
- `matmul_io_inner` — the full pipeline function:
  1. Read dims (M, K, N) from device memory
  2. `File::open("matmul_a.bin")` + `read_to_end` -> Vec<u8> (M*K f32)
  3. `File::open("matmul_b.bin")` + `read_to_end` -> Vec<u8> (K*N f32)
  4. `cooperative_map_with_params(A, C, M*N, [M,K,N,B_ptr], matmul_callback)`
  5. `File::create("matmul_c.bin")` + `write_all` -> M*N f32 bytes
  6. Write success flag + element count to mapped result buffer
- `matmul_io_compute` — kernel entry: `stdio_init` + `gpu_libc_io_init` +
  `gpu_main_poll(|| matmul_io_inner(...))`.

### Host Test (`crates/test/gpu-test-harness/src/main.rs`)

- `run_matmul_io_compute`:
  1. Creates matmul_a.bin (8x4 f32) and matmul_b.bin (4x6 f32) with
     known values matching `cooperative_matmul_test` pattern.
  2. Loads kernel_std cubin (falls back to PTX JIT).
  3. Launches with block_dim=(128,1,1) = 4 warps, hostcall enabled.
  4. Reads matmul_c.bin, parses f32 values.
  5. Computes CPU reference C = A x B.
  6. Verifies all 48 elements within 1e-3 tolerance.
  7. Cleans up temp files.
- Wired into `ONLY_TEST=matmul_io` for quick iteration.

## Test Output

```
--- North Star: File::read -> matmul -> File::write (coop-demo.1) ---
  Created matmul_a.bin (8x4 = 128 bytes)
  Created matmul_b.bin (4x6 = 96 bytes)
  [GPU] [MATMUL] M=8 K=4 N=6
  [HOST] FILE OPEN: "matmul_a.bin" -> fd=1
  [HOST] FILE OPEN: "matmul_b.bin" -> fd=2
  [GPU] [MATMUL] Compute done: C[8x6]
  [HOST] FILE OPEN: "matmul_c.bin" -> fd=3
  [GPU] [MATMUL] DONE: File::read -> matmul -> File::write
  Kernel completed in 4.561853ms
  All 48 elements match CPU reference (tolerance 1e-3)
  Sample: C[0][0]=260.0, C[7][5]=3720.0
  NORTH STAR LITMUS TEST -- PASSED
```

## Key Observations

1. **File I/O works inside gpu_main_poll**: `std::fs::File` operations
   (open, read_to_end, create, write_all) work correctly when called from
   warp 0 lane 0 inside the cooperative thread pool context.

2. **Cooperative compute works with heap-allocated data**: The matmul
   operates on `Vec<u8>` data read from files (heap-allocated on GPU).
   `cooperative_map_with_params` correctly distributes rows across warps
   using the Vec's pointer passed via params.

3. **Build pipeline**: PTX compilation ~13s, ptxas cubin ~10min.
   CI lint passes on all host crates.

## Files Changed

- `crates/kernel/gpu-kernel-std/src/lib.rs` — added `matmul_callback`,
  `matmul_io_inner`, `matmul_io_compute`
- `crates/test/gpu-test-harness/src/main.rs` — added `run_matmul_io_compute`,
  wired into ONLY_TEST dispatch
- `crates/core/gpu-host/kernel_std.ptx` — rebuilt with new kernels
- `crates/core/gpu-host/kernel_std.cubin` — recompiled for sm_75
- `crates/test/gpu-test-harness/kernel_std.cubin` — copied for fast test loading
