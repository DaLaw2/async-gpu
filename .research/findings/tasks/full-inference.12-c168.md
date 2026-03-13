# full-inference.12: Fix NaN at step 22 — numerically stable GELU
**Cycle**: 168 | **Theme**: full-inference | **Kind**: experiment | **Status**: done

## Summary
Root cause identified and fixed: GELU kernel's tanh computation overflows when
`exp(2z)` exceeds f32 range, producing `Inf/Inf = NaN`. Fixed by clamping the
tanh input to ±10 (tanh(10) = 1.0 in f32 precision). Generation now produces
50 tokens without NaN on all test prompts.

## Findings

### Q: Is the NaN f32-specific?
A: **Yes.** CPU f64 generation confirmed in full-inference.11 to produce 50+ tokens
without NaN. GPU f32 NaN at step 22 was caused by f32 exp overflow in GELU.
**Confidence**: high

### Q: Does online softmax prevent the NaN?
A: **Not applicable.** The NaN was NOT in softmax (flash_attention already uses
online softmax). The NaN originated in the GELU activation function. Sub-operation
diagnostics pinpointed: post-FFN-up+bias had max=45.7 (no NaN), but post-GELU
had nan=4 (4 NaN values). These 4 NaN values propagated through FFN-down GEMM
to corrupt the entire 768-dim prediction row.
**Confidence**: high

### Q: What is the maximum generation length after the fix?
A: At least 50 tokens (the test limit with SEQ=128). No NaN observed at any step.
With SEQ=128 and prompt_len=5, max generation is 123 tokens. Could potentially
generate longer with larger SEQ buffer.
**Confidence**: high

## Root Cause Analysis

The GELU kernel computes:
```
inner = sqrt(2/pi) * (x + 0.044715 * x^3)
tanh(inner) = (exp(2*inner) - 1) / (exp(2*inner) + 1)
```

When `x = 45.7` (from FFN up projection):
- `inner = 0.798 * (45.7 + 0.0447 * 95551) = 0.798 * 4313.5 ≈ 3442`
- `exp(2 * 3442) = exp(6884)` → f32 overflow → Inf
- `(Inf - 1) / (Inf + 1) = Inf / Inf = NaN`

**Fix**: Clamp inner to ±10 before computing tanh. Since `tanh(10) ≈ 1.0` in
f32 precision (difference < 1e-9), this is mathematically exact for f32.

## Diagnostic Methodology

1. Per-layer check at step 22: L0-L10 clean, L11 = 768/768 NaN
2. Sub-operation diagnostics at step 22 L11:
   - pre-LN1 hidden: max=71.8 ✓
   - post-LN1: max=2.5 ✓
   - post-QKV+bias: max=9.1 ✓
   - post-flash_attn: max=1.9 ✓
   - post-residual1: max=77.7 ✓
   - post-LN2: max=3.4 ✓
   - post-FFN-up+bias: max=45.7 ✓
   - **post-GELU: nan=4** ← NaN source

## Impact on Downstream Tasks
- full-inference.13 (50+ token validation) can proceed — NaN is fixed
- GELU fix is general — prevents NaN for any input magnitude
