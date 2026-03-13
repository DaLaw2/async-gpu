# BS41 Proposer: NaN Persistence + PyTorch Reference Strategy

## 1. Active Epics Assessment

### gpu-inference (HIGH priority)

| # | Criterion | Status | Blocker |
|---|-----------|--------|---------|
| 1 | Real GPT-2 weights from safetensors | DONE | — |
| 2 | BPE tokenizer correct token IDs | DONE | — |
| 3 | Single-layer matches PyTorch (atol=0.01, rtol=0.01) | NOT VALIDATED | No PyTorch reference exists |
| 4 | 12-layer correct top-1 prediction | UNKNOWN | f16: "-", f32: " a" — neither verified against reference |
| 5 | Greedy decoding produces "Paris" | FAILED | Generates garbage; NaN at step 20 |
| 6 | Token-for-token match for 3+ prompts | NOT TESTED | Blocked by #5 |

**Critical path**: Criteria 3-6 are ALL blocked by the same root issue: we have
no ground truth. We cannot distinguish "implementation bug" from "model limitation"
from "unrealistic criterion" without a PyTorch reference.

**Progress since bs40**: full-inference.5.1 found the NaN was from padding
contamination (not f16 overflow). full-inference.6 proved f32 GEMM also produces
" a" (not "Paris"). The zero_pad fix eliminated NaN at SEQ=32, but NaN recurs
at step 20 when generating with SEQ=128.

### public-api (MEDIUM priority)

| # | Criterion | Status |
|---|-----------|--------|
| 1 | gpu-host is library crate with public API | DONE |
| 2 | Standalone example program | PENDING (host-sdk.3) |
| 3 | Automated build system | PARKED |
| 4 | 3+ working examples | PARKED |

**Assessment**: host-sdk.3 is actionable but lower priority than fixing inference.
Should be tackled opportunistically.

### Gaps

1. **No PyTorch reference at all** — this has been the #1 gap since bs38 (6 cycles
   ago) and remains unresolved. Every analysis keeps concluding "we need a reference"
   but none has been obtained.
2. **NaN recurs at step 20 with extended generation** — the zero_pad fix only works
   for the initial forward pass, not for the autoregressive loop at longer sequences.
3. **No layer-by-layer comparison** — even if top-1 differs from PyTorch, we have
   no data on WHERE the divergence begins (layer 1? layer 6? layer 12?).

---

## 2. Systems Analysis: Why Is Inference Producing Garbage?

### Hypothesis A: GPT-2 Small Genuinely Cannot Predict "Paris"

**Evidence for**: f32 GEMM (full precision, no Tensor Cores) produces " a" as
top-1 with top-5 being common function words. If the precision were the issue,
f32 should fix it.

**Evidence against**: GPT-2 small (124M params) is a language model trained on
WebText. "The capital of France is" is a trivial factual completion. Multiple
online demos show GPT-2 completing this correctly. GPT-2 117M was shown to
produce coherent factual text in the original paper.

**Probability**: 30%. The model *should* know this, but temperature=0 greedy
decoding from a 124M model is not guaranteed to produce any specific factual answer.

### Hypothesis B: Weight Loading or Layout Bug

**Evidence for**: The top-1 being " a" (a generic function word) for f32 suggests
the model is behaving like an untrained or mis-loaded model — it is defaulting to
high-frequency tokens rather than producing contextually relevant output.

**Evidence against**: model-loading was validated; the embedding test passes.

**Probability**: 25%. Weight loading was tested at the layer level, but there are
subtle layout issues that could manifest:
- Row-major vs column-major confusion in GEMM weight packing (`pack_weight`)
- Transposition mismatch: HuggingFace GPT-2 stores `c_attn.weight` as [768, 2304]
  but some implementations expect [2304, 768]
- Bias indexing: c_attn_bias is [2304] — if Q/K/V split is wrong, attention computes
  on wrong subspaces

### Hypothesis C: Attention or LayerNorm Bug Corrupting Representations

**Evidence for**: The generated text is not random — it is high-frequency English
tokens. This suggests the model processes input but loses semantic content
somewhere in the pipeline. Attention is the most complex kernel.

**Specific concerns**:
1. **FlashAttention causal masking**: If the causal mask is off-by-one (masking
   position i from attending to itself), the model loses autoregressive structure.
2. **Head dimension scaling**: `1/sqrt(d_head)` = `1/sqrt(64)` = 0.125. If this
   is applied as `1/sqrt(768)` instead, attention scores are 3.5x too small,
   producing near-uniform attention and washing out content.
3. **Concat heads ordering**: If heads are concatenated in the wrong order,
   the projection matrix multiplies wrong head outputs into wrong channels.

**Probability**: 30%. These are subtle bugs that would not cause NaN but would
produce semantically wrong output.

### Hypothesis D: NaN at Step 20 Is a Separate Bug

The zero_pad fix eliminated NaN for the initial forward pass (all 12 layers clean).
But NaN recurs at step 20 during generation. This suggests:

1. **Activation magnitude growth**: After 20 autoregressive steps, activations may
   grow beyond f16 range even for valid (non-padded) positions. Each step feeds the
   previous output back through 12 layers — compound growth across 20×12 = 240
   layer applications.

2. **Padding re-contamination**: At step 20 with 25 tokens (5 prompt + 20 generated),
   rows 25-127 are padded. The zero_pad kernel runs after each residual add, but if
   any kernel BETWEEN zero_pad calls reads padding rows (e.g., GEMM tile loading),
   NaN can still leak in.

3. **f32 GEMM avoids NaN**: The fact that f32 GEMM did not report NaN (in the
   SEQ=32 test) but f16 GEMM does suggests the NaN IS precision-related, just
   manifesting differently than originally hypothesized.

**Probability**: High (this is observable, not a hypothesis). The question is
whether it is f16 overflow in multiplied values (not accumulator), or tile
contamination from a timing issue in zero_pad application.

### Hypothesis E: The Entire Forward Pass Is Correct But LM Head Is Wrong

The LM head computes logits = hidden[last_pos] @ wte.T on CPU. If `last_pos`
indexing is wrong (e.g., off-by-one), the model reads the wrong position's hidden
state, producing unrelated predictions.

**Probability**: 15%. Simple to verify.

---

## 3. Compiler/GPU Architecture Analysis

### GEMM Precision Landscape

| Mode | Multiply | Accumulate | Precision | Status |
|------|----------|------------|-----------|--------|
| f16 MMA | f16×f16 (10-bit mantissa) | f32 | ~3 decimal digits | Implemented, used by default |
| f32 FMA | f32×f32 (23-bit mantissa) | f32 | ~7 decimal digits | Implemented (full-inference.6) |
| TF32 MMA | tf32×tf32 (10-bit mantissa, 8-bit exp) | f32 | ~3 digits, wider range | Not implemented |

**Key insight from full-inference.6**: f32 GEMM is only 13% slower than f16 TC at
this scale (memory-bound, not compute-bound). This eliminates the performance
argument for f16 — we should use f32 GEMM as the default until inference is
correct, then optimize.

### f16 Pack/Unpack Correctness

The `pack_f16x2` function converts f32 weights to f16 pairs for the MMA path.
The `full_gemm_f32in` kernel still uses `f16x2` packed weights (column-major with
f16 pair packing) but accumulates in f32. This means even the "f32 input" GEMM
still reads weights as f16 — it only keeps activations in f32.

**Wait — is `full_gemm_f32in` truly f32?** The name says "f32 input" meaning
activations are f32, but weights are still packed as f16x2 and the MMA instruction
uses f16 operands. Only `gemm_f32` (the pure FMA kernel from full-inference.6)
does true f32×f32 multiplication.

This distinction matters: `full_gemm_f32in` still loses weight precision to f16
truncation. If a weight value is 0.123456789, it becomes 0.12346 in f16. Over
millions of multiplications, this could shift outputs.

### zero_pad Kernel Placement

For the zero_pad fix to work, it must run after EVERY operation that could
produce non-zero values in padded positions. The current placement is:
- After embedding lookup (1 call)
- After each residual add (2 per layer × 12 layers = 24 calls)

But what about:
- After bias_add? (bias is added to ALL rows including padded ones)
- After layer_norm? (LN of zeros = beta, which is non-zero for most layers)

If zero_pad runs after residual_add but LN runs on the result (including non-zero
beta in padded rows), those padded rows carry LN beta values into GEMM, which
then tile-contaminates valid rows. This would explain why NaN recurs at step 20:
the padded-row values accumulate through the beta→GEMM→bias→residual→zero_pad
cycle, growing each step until they overflow.

---

## 4. Skeptic Challenges

### Challenge 1: "f32 GEMM also predicts ' a'" Does Not Prove the Model Can't Predict "Paris"

The f32 GEMM test (full-inference.6) ran with the SAME codebase — same embedding,
same attention, same LayerNorm, same concat, same LM head. If ANY of these has a
bug, f32 GEMM would also produce wrong output. The f32 test only eliminates
GEMM precision as a variable — it does not validate the rest of the pipeline.

**Untested**: attention correctness with real weights, weight transpose correctness,
head split/concat ordering, LM head position indexing.

### Challenge 2: "GPT-2 Small Is Too Weak" Is an Untested Assumption

Full-inference.6 concluded "GPT-2 small likely lacks factual accuracy for this
prompt" with "high confidence." But this confidence is based on zero external
validation. The only evidence is that OUR implementation doesn't produce "Paris" —
which could equally mean our implementation is buggy.

**This is circular reasoning**: "Our model doesn't predict Paris → GPT-2 can't
predict Paris → our model is correct." The conclusion assumes its own premise.

### Challenge 3: The NaN "Fix" May Be Masking a Deeper Bug

The zero_pad fix treats symptoms (NaN in padded rows) rather than root cause
(why do padded rows develop pathological values?). In a correct implementation,
padded rows with zero embeddings should produce small, well-behaved values
through the transformer — zeros through LN produce beta, beta through GEMM
produces small values, etc.

If padded rows develop NaN after only 2-3 layers, something is wrong with the
numerical stability of the pipeline even for small inputs. The zero_pad fix
papers over this, and the NaN returning at step 20 suggests the fix is incomplete.

### Challenge 4: No Test Validates End-to-End Correctness at the Layer Level

The project validated individual kernels (GEMM, LN, attention, GELU) in isolation
with synthetic data. But no test runs a COMPLETE transformer layer with REAL
weights and compares against a reference. The integration could have bugs that
unit tests miss:
- Weight matrix applied to wrong dimension
- Output of kernel A fed to kernel B with wrong layout
- Off-by-one in buffer reuse between layers

### Challenge 5: The Generation Loop May Have a Token Feeding Bug

In autoregressive generation, the model should attend to ALL previous tokens
including the newly generated one. If the position encoding or causal mask
treats the new token differently (e.g., wrong position ID), the model will
produce degrading output as the generated sequence grows — exactly what we see.

---

## 5. Concrete Recommendations

### PRIORITY 1 (CRITICAL): Get PyTorch Reference Output

**Approach**: Write a Python script and ask the user to run it.

This is not optional. This has been the #1 recommendation for 3 consecutive
brainstorm sessions and remains unacted upon. Without it, all further inference
work is shooting in the dark.

**Proposed task: `full-inference.7` — Create PyTorch reference script**

The script should output:
1. Top-10 predictions for "The capital of France is" (with probabilities)
2. 20-token greedy generation from that prompt
3. Layer-by-layer hidden state norms (for each of 12 layers: mean, std, max|val|)
4. Final logits for the last position (full 50257-dim vector, saved as binary)

Save output as `models/pytorch_reference.json` (human-readable) and
`models/reference_logits.bin` (binary for comparison).

**Why a script, not web API**: The HuggingFace Inference API runs their hosted
model which may differ from `gpt2` (could be quantized, different version). A
local PyTorch run with `GPT2LMHeadModel.from_pretrained("gpt2")` is definitive.

### PRIORITY 2: Layer-by-Layer Diagnostic with f32 GEMM

**Proposed task: `full-inference.8` — Per-layer hidden state statistics**

Using the f32 GEMM path (eliminates precision as a variable), dump after each
layer:
- `mean(hidden[last_pos])`, `std(hidden[last_pos])`, `max(abs(hidden[last_pos]))`
- Top-5 token predictions if we applied LM head at this layer's output

Compare these against the PyTorch reference layer-by-layer statistics. The first
layer where statistics diverge significantly identifies the buggy component.

**Depends on**: Priority 1 (needs reference to compare against).

### PRIORITY 3: Weight Transpose Audit

**Proposed task: `full-inference.9` — Verify weight layout matches HuggingFace**

HuggingFace GPT-2 convention:
- `transformer.h.{i}.attn.c_attn.weight` is shape [768, 2304] (input × output)
- This is TRANSPOSED relative to standard linear layer convention [output, input]
- GPT-2 uses Conv1D internally, which stores weights as [in, out]

Our `pack_weight(w, k, n)` function packs with k=768, n=2304. It iterates
`col in 0..n, kp in 0..k/2`. If the weight is [768, 2304] (row-major, already
[in, out]), then `w[k0 * n + col]` reads row k0, column col — this is correct
for column-major packing of [768, 2304].

BUT: if `model.rs` loads the weight without transposing, and the HuggingFace
weight is already [in, out], and the GEMM kernel expects column-major [in, out],
then it should be correct. **This needs explicit verification** by:
1. Loading one weight, printing its shape from safetensors metadata
2. Computing `input @ weight` on CPU for a known input
3. Comparing against GPU GEMM output for the same input

### PRIORITY 4: Extended Generation NaN Root Cause

**Proposed task: `full-inference.10` — NaN root cause in extended generation**

Run generation with f32 GEMM (not f16) at SEQ=128 and check if NaN still appears
at step 20. If f32 GEMM also produces NaN at step 20, the bug is NOT precision-related
and is likely:
- zero_pad placement incomplete (missing after bias_add or LN)
- Position encoding overflow at positions > 32
- Buffer reuse contamination between steps

If f32 GEMM does NOT produce NaN at step 20, the bug IS precision-related
(f16 weight pack losing range), and the fix is to use f32 GEMM for generation.

### PRIORITY 5: Host-SDK Standalone Example

**Proposed task: `host-sdk.3` — Standalone example program**

Lower priority than inference correctness, but actionable and independent. Can be
done in parallel if a session has spare capacity.

### Execution Order

```
Session N (immediate):
  1. Write PyTorch reference script → ask user to run it
  2. While waiting: run f32 GEMM generation at SEQ=128 to check NaN
  3. While waiting: audit weight transpose in model.rs vs safetensors

Session N+1 (after reference obtained):
  4. Compare layer-by-layer statistics against PyTorch reference
  5. Identify divergence point → targeted fix
  6. Re-run generation → validate against reference

Session N+2 (cleanup):
  7. host-sdk.3 standalone example
  8. Revise epic criteria if PyTorch confirms model limitations
```

### Epic Criteria Revision (Conditional)

IF PyTorch reference confirms GPT-2 small does not predict "Paris":
- Criterion 5 → "Greedy decoding output matches PyTorch GPT-2 token-for-token"
- Criterion 6 → unchanged (already requires token-for-token match)
- This reframes success from "factual correctness" to "implementation correctness"

IF PyTorch reference confirms GPT-2 small DOES predict "Paris":
- We have a bug. The layer-by-layer diagnostic (Priority 2) will find it.
- Most likely candidates: weight transpose, attention head ordering, LM head indexing.

---

## Summary of Key Insights

1. **The project is stuck in a loop**: 3 consecutive brainstorms have concluded
   "get PyTorch reference" as #1 priority. This MUST happen before any further
   inference debugging. All other work is speculative without ground truth.

2. **f32 GEMM producing " a" does NOT prove model limitation** — it only proves
   GEMM precision is not the sole issue. Other bugs could produce wrong output
   regardless of GEMM precision. The circular reasoning ("our model doesn't
   predict Paris → GPT-2 can't") must be broken with external reference.

3. **NaN at step 20 in extended generation is a separate bug** from the original
   padding contamination. It likely stems from incomplete zero_pad placement or
   activation magnitude growth. Testing with f32 GEMM at SEQ=128 would isolate
   the cause.

4. **Weight layout is the highest-risk untested assumption**. GPT-2's Conv1D
   convention stores weights transposed relative to standard nn.Linear. If
   `model.rs` loads without appropriate handling, every GEMM computes the wrong
   linear transformation — producing plausible but semantically wrong output.

5. **The f32 GEMM path should be the default** until inference is validated.
   At only 13% slower than f16 TC at this scale, there is no performance reason
   to use f16 for debugging.
