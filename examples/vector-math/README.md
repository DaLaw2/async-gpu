# Vector Math

Pure GPU compute kernels for common numerical operations, demonstrating CPU-GPU cooperative algorithms with no hostcall overhead.

## What It Demonstrates

- SAXPY (`y = a*x + y`), the classic GPU compute benchmark
- Dot product via GPU element-wise multiply with CPU-side reduction
- Numerically stable softmax using a multi-pass GPU-CPU cooperative strategy
- Approximate `exp()` via PTX `ex2.approx` intrinsic and log2(e) scaling
- Launching kernels with multi-block grid configurations (256 threads per block)

## How It Works

1. **SAXPY** -- The `saxpy` kernel computes `y[i] = a * x[i] + y[i]` for 1024 elements. Each thread handles one element, identified by `blockIdx.x * blockDim.x + threadIdx.x`. The host uploads vectors, launches the kernel, and verifies every element matches the expected value.
2. **Dot Product** -- The `elementwise_mul` kernel computes `out[i] = x[i] * y[i]` on the GPU for 1024 elements. The host then sums the products on the CPU to get the final dot product, demonstrating a split GPU-CPU reduction pattern.
3. **Softmax (Pass 1)** -- The CPU scans the input to find `max_val` for numerical stability. The `softmax_exp` kernel computes `exp(input[i] - max_val)` for each of 256 elements using an approximate `exp()` implemented as `2^(x * log2(e))` via inline PTX.
4. **Softmax (Pass 2)** -- The host sums the exponentiated values on the CPU. The `softmax_normalize` kernel divides each element by this sum, producing a valid probability distribution.
5. The host verifies the softmax output sums to 1.0 and matches a CPU reference implementation within tolerance.

## Running

```bash
# Linux/macOS
bash run.sh

# Windows
run.bat
```

## Expected Output

```
=== Vector Math Example ===

[host] CUDA device initialized.
[host] PTX module loaded.

--- Demo 1: SAXPY (y = 2.0 * x + y) ---
[host] SAXPY (1024 elements): PASSED

--- Demo 2: Dot Product (GPU multiply, CPU reduce) ---
[host] dot(x,y) GPU = 9270.00, CPU = 9270.00
[host] Dot Product (1024 elements): PASSED

--- Demo 3: Softmax (GPU-CPU cooperative) ---
[host] softmax sum = 1.000000 (expected 1.0)
[host] max |GPU - CPU| = 0.000000
[host] Softmax (256 elements): PASSED

=== All demos complete! ===
```

## Key PTX to Inspect

- **`ex2.approx.ftz.f32`**: The `softmax_exp` kernel uses this PTX instruction for fast approximate 2^x, which is the core of the `gpu_exp()` helper. Look for the `mul.f32` (by log2(e)) immediately before it.
- **`fma.rn.f32`**: The SAXPY kernel may compile the multiply-add into a fused multiply-add instruction, depending on optimization level.
- **Thread indexing pattern**: All four kernels share the same `mad.lo.s32` pattern for computing the global thread index from `%ctaid.x`, `%ntid.x`, and `%tid.x`.
- **Bounds checking**: Each kernel has an early `setp.ge` + `@%p bra` exit guard comparing the thread index against `n`.
