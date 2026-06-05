# warp-cooperative — Cooperative Compute on GPU

Demonstrates warp-parallel compute patterns using the cooperative APIs
from `gpu_runtime::thread`. All warps execute in parallel, each handling
its partition of the data.

## What It Demonstrates

- `cooperative()` — all warps run the same closure, data partitioned by warp ID
- `cooperative_map()` — data-parallel transform with explicit (src, dst, len)
- `cooperative_reduce()` — multi-warp reduction to a single value
- `cooperative_map_with_params()` — parameterized map (scalar multiply, matmul)

## The Cooperative Model

```
gpu_main(|| {
    // Sequential: only warp 0 runs
    let data = prepare_data();

    // Cooperative: ALL warps participate
    cooperative(&|| {
        let wid = thread::current_id();
        let n = thread::available_parallelism() + 1;
        // Each warp processes elements at stride n
        for i in (wid..data.len()).step_by(n) {
            output[i] = transform(data[i]);
        }
    });

    // Sequential: back to warp 0 only
    verify_results(&output);
});
```

## Running

```bash
cd examples/hostcall/warp-cooperative
cargo run --release
```

## Expected Output

```
=== Cooperative Compute on GPU ===

--- Demo 1: cooperative() — basic parallel execution ---
  4 warps cooperatively filled 256 elements
  output[i] = i * 2 + 1 (verified all 256)
  Verification: PASSED

--- Demo 2: cooperative_map() — data-parallel transform ---
  4 warps cooperatively doubled 256 elements
  cooperative_map: no global atomics, explicit (src, dst, len)
  Verification: PASSED

--- Demo 3: cooperative_reduce() — multi-warp reduction ---
  4 warps reduced 256 elements to sum = 32640
  Each warp computed partial sum of its partition
  Verification: PASSED

--- Demo 4: cooperative_map_with_params() — scalar multiply ---
  4 warps multiplied 256 elements by scalar 7
  Scalar passed via params[0] — no closure captures needed
  Verification: PASSED

--- Demo 5: cooperative matmul — C = A x B ---
  C[8x6] = A[8x4] x B[4x6] — 4 warps, row-parallel
  Max error: 0.000000 (48 elements verified)
  Verification: PASSED

=== All demos passed! ===
```
