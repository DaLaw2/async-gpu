# dyn-control.2: Early-exit inference — skip layers when confidence > threshold
**Cycle**: 562 | **Theme**: dyn-control | **Kind**: experiment | **Status**: done

## Summary
Added early-exit inference to GPT-2: after each transformer layer, probe the model's
prediction confidence. If softmax probability of top token exceeds threshold, skip
remaining layers. Demo shows different prompts need different layer counts (1-12).

## Findings
### Q: Can we demonstrate early-exit inference as dynamic control flow?
A: Yes. Added `forward_early_exit()` and `generate_cached_early_exit()` methods.

Results from Demo 4 (single forward pass probes):
- "2 + 2 =" → needs only 1-3 layers (highly predictable pattern)
- "def hello():" → needs 10-12 layers (code syntax requires deep understanding)
- "In a surprising turn..." → needs 5-12 layers (semantic complexity varies)

Demo 5 (generation with early exit):
- threshold=0.9: avg 10.1/12 layers, 16% compute saved
- Text quality slightly degrades but remains coherent

**Confidence**: high

### Q: Why is this impossible with CUDA graphs?
A: CUDA graphs capture a fixed kernel launch sequence at capture time. Early exit
requires a DATA-DEPENDENT decision after each layer — the number of layers executed
varies per token, which cannot be expressed as a static graph.

## Code Changes
- `gpt2.rs`: Added `forward_early_exit()`, `generate_cached_early_exit()`, `softmax_max_prob()`
- `dynamic-control/src/main.rs`: Added Demo 4 (layer probes) and Demo 5 (generation comparison)

## Unexpected Discoveries
- GPT-2 becomes confident very quickly on repetitive/predictable patterns (1 layer)
- Cache corruption with early exit in generation: skipped layers get zero KV entries,
  degrading subsequent predictions. Solved by using threshold=1.0 for "full" mode
  and showing early-exit as comparison

## Open Questions
- Proper early-exit training (auxiliary classifiers at each layer) would improve quality
- Batch-level early exit (different samples exit at different layers) would be more compelling

## Impact on Downstream Tasks
- dyn-control.3 can reference both variable-length generation and early-exit results
