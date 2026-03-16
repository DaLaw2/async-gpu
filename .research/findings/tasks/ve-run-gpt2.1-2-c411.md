# ve-run-gpt2.1+2: GPT-2 inference end-to-end verification
**Cycle**: 411 | **Theme**: ve-run-gpt2 | **Kind**: experiment | **Status**: done

## Summary
Fixed 3 bugs in nn API pipeline. GPT-2 Small now generates coherent text via gpt2-inference example.
Cached and non-cached generation produce identical output.

## Bugs Fixed

### 1. flash_attention launch config (cycle 409)
- Grid was `(seq_len, 1, 1)` → should be `(1, n_q_tiles, 1)` per head
- shared_mem_bytes was 0 → needs `2 * 32 * d_head * 4` = 16KB
- Caused CUDA_ERROR_ILLEGAL_ADDRESS

### 2. embedding_lookup grid (cycle 411)
- Grid was `(seq_len, 1, 1)` with 256 threads → only covered seq_len×256 elements
- For seq_len=5, d_model=768: covered 1280 of 3840 elements → positions 2-4 had ZERO embeddings
- Fixed to `(ceil(total/256), 1, 1)` matching the kernel's 1D indexing

### 3. LM head weight layout (cycle 411)
- `transpose_2d(wte, vocab, embd)` before `Linear::new` → double transposition
- wte is `[vocab, embd]` which already matches Linear's `[out_features, in_features]`
- Extra transpose corrupted the logit projection → garbage token predictions
- Fix: pass wte directly to Linear::new without transpose

## Verification
- 3 prompts generate coherent English text:
  - "The capital of France is the capital of the French Republic..."
  - "In a world where AI is a big problem, it's important to understand how it works..."
  - "Once upon a time, the world was a place of great beauty and great danger..."
- Cached and non-cached: MATCH on all 3 prompts
- 15 nn unit tests pass
- CI passes

## Performance
- Model load: ~230ms (497.8 MB safetensors)
- Model build: ~620ms (upload weights to GPU)
- Generation: ~790ms/token (greedy, non-cached), ~760ms/token (KV-cached)
