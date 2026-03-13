# transformer-layer.2: GELU Activation Kernel
**Cycle**: 1 | **Theme**: transformer-layer | **Kind**: experiment | **Status**: done

## Summary
Implemented GELU activation kernel using the tanh approximation: GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))). tanh computed via exp: tanh(z) = (exp(2z)-1)/(exp(2z)+1) using `ex2.approx.f32`. Verified with 1024 elements across [-5, 5] range, max error 4.8e-7.

## Findings

### Q: Can GELU be accurately approximated using ex2.approx.f32 on GPU?
A: Yes. Using the standard tanh approximation formula with `gpu_exp_f32()` (which uses `ex2.approx.f32` internally with ln(2) scaling), the max error is 4.8e-7 — essentially at f32 machine epsilon.
**Confidence**: high

### Q: What is the max error vs PyTorch GELU across the typical activation range?
A: 4.8e-7 across [-5, 5]. PyTorch uses the same tanh approximation formula, so the primary error source is the `ex2.approx.f32` instruction, which is negligible.
**Confidence**: high

## Design Notes
- Element-parallel: grid_dim = (ceil(n/256), 1, 1), block_dim = (256, 1, 1).
- Each thread processes one element — maximum parallelism.
- No shared memory needed.

## Impact on Downstream Tasks
- **transformer-layer.4** (FFN block): GELU component ready.
