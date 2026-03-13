# repo-cleanup.3: Extract compute example from search/GEMM tests
**Cycle**: 174 | **Theme**: repo-cleanup | **Kind**: experiment | **Status**: done

## Summary
Created `examples/vector-math/` — a standalone pure-compute example with no
hostcall dependency. Three kernels: SAXPY (element-wise), elementwise_mul (for
dot product), and softmax (GPU exp + normalize with CPU max/sum). All pass
validation against CPU reference. Demonstrates CPU-GPU cooperative compute.

## Findings

### Q: Can vector search or GEMM become a standalone example?
A: GEMM is too complex for an intro example (requires shared memory, tiling).
Instead created simpler compute kernels (SAXPY, elementwise_mul, softmax)
that demonstrate key patterns without shared memory complications.
**Confidence**: high

### Q: What is the simplest compute-focused example that demonstrates GPU kernels?
A: SAXPY (element-wise) + dot product (GPU mul + CPU sum) + softmax (multi-pass
GPU-CPU cooperation). This covers element-wise ops, data transfer, and multi-kernel
pipelines without needing shared memory or hostcalls.
**Confidence**: high

## Unexpected Discoveries
Static shared memory (`global_asm!(".shared ...")`) + `cvta.shared.u64` causes
CUDA_ERROR_ILLEGAL_ADDRESS in standalone kernels without gpu-runtime. Dynamic
shared memory via `.extern .shared` works but the `cvta.shared.u64` path hangs.
Workaround: avoid shared memory in simple examples, use CPU-GPU cooperative
approach instead (GPU does element-wise, CPU does reduction).
