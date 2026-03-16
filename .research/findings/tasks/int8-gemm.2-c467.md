# int8-gemm.2: INT8 GEMM kernel with dp4a + benchmark vs f32
**Cycle**: 467 | **Theme**: int8-gemm | **Kind**: experiment | **Status**: done

## Summary
INT8 dp4a GEMM kernel implemented and verified. Correctness is good (1.3-1.7% relative
error), but performance is 7-10x SLOWER than f32 due to: (1) CPU-side quantization + packing
overhead, (2) naive 1-thread-per-output kernel without shared memory tiling.

## Findings
### Q: Does INT8 dp4a produce correct results?
A: Yes. Per-tensor quantization for A, per-column for B. Relative error ~1.3% for GPT-2
typical dimensions (768×768, 768×3072). This is within expected INT8 quantization tolerance.
**Confidence**: high

### Q: Is INT8 faster than f32?
A: NO. Naive dp4a kernel is 7-10x slower than tiled f32 GEMM:
- 768×768: f32 0.44ms vs INT8 4.69ms (0.09x)
- 768×3072: f32 1.65ms vs INT8 24.70ms (0.07x)

Root causes:
1. CPU roundtrip for quantization + packing (~90% of INT8 time)
2. No shared memory tiling in dp4a kernel
3. f32 GEMM is already well-optimized with tiling
**Confidence**: high

## Baseline
f32 GEMM: 0.44ms (768×768), 1.65ms (768×3072)

## Unexpected Discoveries
- The dp4a instruction itself is fast; the bottleneck is entirely host-side quantization.
- Pre-quantized weights (done once at load time) + GPU-side activation quantization would
  eliminate the CPU overhead, but requires a tiled dp4a kernel to beat f32.

## Impact on Downstream Tasks
- int8-gpt2 can proceed with pre-quantized weights for functional correctness demo
- Performance improvement requires tiled dp4a kernel (future optimization)
