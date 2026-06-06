# Par-Iter

GPU parallel iterators with a Rayon-like API — map, filter, fold, collect on the GPU.

## What It Demonstrates

- Lazy iterator chains that fuse at compile time (zero intermediate buffers)
- `map` + `collect_into`: transform elements in parallel
- `map` + `sum` (fold): parallel reduction across warps
- `enumerate` + `map`: index-aware transforms
- `zip` + `map`: element-wise operations on two arrays
- `filter` + `map` + `sum`: conditional reduction (skip non-matching elements)
- Chained `.map().map()` calls proving monomorphization fusion

## Running

```bash
# Linux/macOS
bash run.sh

# Or directly
cd host && cargo run --release
```

## How It Works

### Kernel Side (gpu-runtime)

```rust
use gpu_runtime::par_iter::*;

// In a GPU kernel:
let data = unsafe { GpuSlice::from_raw_parts(input_ptr, len) };
let out = unsafe { GpuSliceMut::from_raw_parts(output_ptr, len) };

// Fused chain: compiles to a single loop per warp
data.par_iter()
    .map(|x| x * 2.0)
    .map(|x| x + 1.0)
    .collect_into(out);

// Reduction:
let sum: f32 = data.par_iter().map(|x| x * x).sum();

// Filter:
let filtered_sum: f32 = data.par_iter()
    .filter(|x| *x > threshold)
    .map(|x| x * x)
    .sum();
```

### Host Side

The host loads the pre-compiled kernel PTX and launches each demo kernel
via cudarc, then downloads and verifies results against a CPU reference.

## Key Source Files

- Kernel API: `crates/core/gpu-runtime/src/par_iter.rs`
- Kernel demos: `crates/kernel/gpu-kernel-compute/src/par_iter_demo.rs`
- Host tests: `crates/test/gpu-test-harness/src/tests_par_iter.rs`

## Expected Output

```
=== Par-Iter Example: GPU Parallel Iterators ===

--- Demo 1: map + collect ---
  PASSED

--- Demo 2: map + sum (reduction) ---
  PASSED

--- Demo 3: enumerate + map + collect ---
  PASSED

--- Demo 4: zip + map + collect ---
  PASSED

--- Demo 5: filter + map + sum ---
  PASSED

--- Demo 6: chained map + collect (fusion proof) ---
  Two separate .map() calls — fused at compile time
  Zero intermediate buffers (register-to-register)
  PASSED

=== All demos complete! ===
```
