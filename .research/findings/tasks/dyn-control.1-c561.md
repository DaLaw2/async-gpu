# dyn-control.1: Variable-length GPT-2 generation with per-sample EOS stopping
**Cycle**: 561 | **Theme**: dyn-control | **Kind**: experiment | **Status**: done

## Summary
Added top-k sampling with temperature to GPT-2 and created `dynamic-control` example
demonstrating variable-length generation. Different prompts and seeds produce different
token counts (60-100 range), with some hitting EOS early — impossible with CUDA graphs.

## Findings
### Q: Can we demonstrate variable-length generation as dynamic control flow?
A: Yes. Added `generate_cached_sampling()` with top-k + temperature to `Gpt2Model`.
The `dynamic-control` example shows three demos:
1. Variable-length generation: 8 prompts produce 63-100 tokens (1 hits EOS early)
2. Same prompt, 5 seeds: produces 60, 91, 100 tokens (length varies by seed)
3. Temperature sweep: same prompt + seed at different temperatures → different text

Each generation runs a different number of GPU kernel launches. CUDA graphs cannot
express this because the loop iteration count depends on model output.

**Confidence**: high

## Code Changes
- `crates/core/gpu-host/src/nn/models/gpt2.rs`:
  - Added `SimpleRng` (xorshift64) for reproducible sampling
  - Added `top_k_sample()` function with temperature + softmax over top-k
  - Added `generate_cached_sampling()` method to `Gpt2Model`
- Created `examples/std/dynamic-control/` example with 3 demos

## Performance
- ~97ms/token with KV cache (GPT-2 Small, single sample)
- EOS early stopping saves proportional compute (63/100 = 37% savings)

## Open Questions
- Batch generation (multiple samples in parallel) would be more impressive but
  requires batch dimension throughout the model — significant refactor
- Early-exit inference (dyn-control.2) is the next step

## Impact on Downstream Tasks
- dyn-control.2 (early-exit) can build on this example
- dyn-control.3 (design doc) can reference these results
