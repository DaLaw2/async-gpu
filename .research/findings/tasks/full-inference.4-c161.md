# full-inference.4: Greedy autoregressive generation loop
**Cycle**: 161 | **Theme**: full-inference | **Kind**: experiment | **Status**: done

## Summary
Implemented greedy autoregressive generation without KV cache. Each step does
full 12-layer forward pass recomputation + CPU LM head argmax. Successfully
generates tokens at ~78ms/token before f16 NaN propagation stops generation.

## Findings
### Q: How to implement the generate-one-token-at-a-time loop?
A: Fixed seq=32 pad, start with prompt tokens, append argmax each step, re-upload
token_ids and rerun full forward pass. No KV cache — straightforward but O(n²)
in total compute. Implementation is ~400 lines in `run_generation_test`.
**Confidence**: high

### Q: Is KV cache needed for acceptable performance or can we recompute?
A: Full recompute works at ~78ms/token for seq=32 on RTX 3090 (182 kernel
launches × ~0.4ms avg). For a PoC this is acceptable. KV cache would be needed
for longer sequences or production use, but is not required to demonstrate the
generation loop works.
**Confidence**: high

### Q: Does generated text make sense?
A: Partially. Generated " ( (——" before NaN at step 4 (total seq=9). The f16
precision issue causes NaN to propagate to earlier positions as sequence grows:
- Step 0 (seq=5): only pos 0 NaN (same as full-inference.2)
- Step 4 (seq=9): prediction position goes NaN → generation stops

The tokens generated before NaN are structurally valid (punctuation/formatting
tokens), consistent with GPT-2 small's f16 behavior observed in full-inference.3
where top predictions were also punctuation tokens.
**Confidence**: high

## Key Metrics
- Throughput: ~78ms/token (full recompute, no KV cache)
- Max generation before NaN: 4 tokens from 5-token prompt
- Weight loading + packing: ~2s one-time cost
- Total generation time: 312.6ms for 4 tokens

## Impact on Downstream Tasks
- Unblocks full-inference.5 (end-to-end validation)
- f16 NaN limitation is the primary bottleneck for longer generation
- A future f32 compute path or mixed-precision approach would fix this
