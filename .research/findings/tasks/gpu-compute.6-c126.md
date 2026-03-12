# gpu-compute.6: Element-wise GPU compute kernels (softmax)
**Cycle**: 126 | **Theme**: gpu-compute | **Kind**: experiment | **Status**: done

## Summary

Implemented softmax with shared memory parallel reduction on GPU via Rust inline PTX. Uses `ex2.approx.f32` for fast exponential, tree reduction in shared memory for max and sum, and per-thread normalization. Verified with 16-element input: sum=1.0, monotonicity preserved, correct argmax.

## Findings

### Q: Can softmax with shared memory reduction work in Rust GPU code?
A: Yes. The implementation follows the standard GPU softmax pattern:
1. **Max reduction**: Tree reduction in shared memory (log2(N) steps)
2. **Exp computation**: `exp(x - max)` via `ex2.approx.f32` PTX instruction (hardware exponential)
3. **Sum reduction**: Tree reduction of exp values in shared memory
4. **Normalization**: `exp_val / sum` per thread

Results for input [0, 1, 2, ..., 15]:
- Sum of outputs: 1.000000 (exact to 6 decimal places)
- Monotonicity preserved: ✓
- Last element (input=15) has largest softmax value (0.632121): ✓
- Distribution: softmax[0] ≈ 0.000000, softmax[15] ≈ 0.632121

The `ex2.approx.f32` instruction computes 2^x using hardware approximation. We compute exp(x) = 2^(x × log2(e)) where log2(e) ≈ 1.442695. The approximation is accurate enough for softmax (errors are dominated by the normalization).
**Confidence**: high

### Q: Can RoPE and LayerNorm be implemented as pure Rust GPU functions?
A: Not tested in this experiment (focused on softmax as the most complex case requiring reduction). However, the building blocks are now proven:
- Shared memory parallel reduction: ✓ (used for max and sum)
- Fast math intrinsics (`ex2.approx.f32`): ✓
- `bar.sync` for block synchronization: ✓
- Per-thread computation with shared data: ✓

RoPE (position encoding) would be simpler — pure per-element trig computation without reduction.
LayerNorm would use the same reduction pattern as softmax (mean + variance).
**Confidence**: high (for feasibility, not tested)

## Unexpected Discoveries
- `ex2.approx.f32` is a single PTX instruction that runs on the GPU's Special Function Unit (SFU). Combined with the `log2(e)` multiply, it provides a fast exp() without needing libm or software emulation.
- The shared memory reduction pattern works cleanly with `bar_sync()` — no explicit `__syncthreads()` wrapper needed, just raw `bar.sync 0;` inline PTX.

## Changes Made
- **crates/gpu-kernel/src/lib.rs**: Added `gpu_exp_f32()` helper and `test_softmax` kernel
- **crates/gpu-host/src/main.rs**: Added `run_softmax_test()` with sum/monotonicity/argmax verification

## Open Questions
1. Accuracy of `ex2.approx.f32` vs full-precision exp — is it sufficient for inference?
2. Performance scaling: 16 elements fits in 1 warp, but real softmax needs 1000+ elements
3. Vectorized loads (`ld.global.v4`) for better memory throughput

## Impact on Downstream Tasks
- Proves GPU-side ML compute kernels are feasible in Rust
- Combined with MMA (gpu-compute.3/5), shared memory (gpu-compute.4), and hostcall I/O, the path to GPU-autonomous inference is open
