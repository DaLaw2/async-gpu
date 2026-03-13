# BS38 Proposer: Public API Epic + GPU Inference Re-evaluation

## Active Epics Assessment

### gpu-inference: What Has Actually Been Achieved

The `gpu-inference` epic was declared with these success criteria:

1. "Multi-head self-attention runs correctly on GPU with real weight values, verified against CPU reference"
2. "A complete transformer layer (attention + FFN + LayerNorm) produces output matching PyTorch reference within f16 tolerance (atol=1e-2, rtol=1e-2)"
3. "GPU autonomously orchestrates computation — host only provides weights and input"

**Honest assessment of what was delivered:**

- Criterion 1: Partially met. MHA runs on GPU, but with *synthetic* weights (`(col + k0 * 3 + seed) % 7 + 1) * scale`), not real HuggingFace weights. Verified against CPU reference, but never against PyTorch.
- Criterion 2: **Failed by own criteria.** The stated tolerance was (atol=1e-2, rtol=1e-2). The actual test relaxed this to (atol=0.05, rtol=0.10) — a 5x and 10x relaxation respectively. With the original tolerance, ~60% of elements fail. The findings document (transformer-layer.7-c1.md) admits this openly: "With strict PyTorch-style tolerances (atol=0.01, rtol=0.01), ~60% of elements exceed threshold." This should have been flagged as a FAIL with a note about needing f32 intermediate precision.
- Criterion 3: Not met. The host orchestrates all 13 steps — each kernel launch is a separate `dev.synchronize()` call from the host. The GPU is not autonomous; it executes individual kernels dispatched by the host.

**What genuinely works:**
- Individual kernels: LayerNorm, GELU, attention_head, bias_add, split_qkv, concat_heads, elementwise_add, full_gemm, f32_to_f16x2_pack
- These are correct and useful building blocks
- The tiled GEMM (768x768) works well with f16 inputs/f32 accumulation
- The overall pipeline structure is sound

### Gap Analysis: Every Missing Component for Real GPT-2 Text Generation

GPT-2 (small, 117M parameters) requires the following end-to-end:

| Component | Status | Notes |
|-----------|--------|-------|
| BPE Tokenizer | Missing | GPT-2 uses byte-level BPE with ~50,257 token vocabulary |
| Token embedding lookup | Missing | Embedding matrix: [50257, 768] = 154MB f32 |
| Positional embedding | Missing | Learned positional: [1024, 768] = 3MB f32 |
| Causal attention mask | Missing | Upper-triangular mask preventing attending to future tokens |
| Single transformer layer | Partial | Works with synthetic weights, wrong tolerance, no causal mask |
| 12-layer stacking | Missing | Need to iterate the layer 12 times with separate weights per layer |
| Final LayerNorm | Missing | One more LN after all 12 layers |
| LM head (linear projection) | Missing | [768, 50257] projection to vocabulary logits |
| Softmax over vocabulary | Missing | Temperature-scaled softmax over 50,257 logits |
| Token sampling / argmax | Missing | Greedy or top-k/top-p sampling |
| Autoregressive loop | Missing | Feed predicted token back as input, repeat |
| Weight loading (safetensors/HF) | Missing | Parse safetensors format, map to kernel parameter layout |
| KV cache | Missing | For efficient autoregressive generation |
| Sequence length > 32 | Missing | Current attention kernel hardcodes seq_len <= 32 (1 thread per position) |

**Scale of the problem:**
- GPT-2 small has 117M parameters = ~234MB in f16
- GPT-2's context window is 1024 tokens; current attention supports only 32
- Vocabulary is 50,257 tokens; the LM head GEMM is [seq, 768] x [768, 50257] — massive
- Autoregressive generation requires KV caching to avoid O(n^2) recomputation

## Public API Epic Analysis

### What Users Need

Currently, the entire project is internal test infrastructure. A user who wants to use async_gpu would need to:
1. Clone the repo
2. Read `main.rs` (239 lines of test calls) to understand what's available
3. Copy-paste test code into their own project
4. Manually build the GPU kernel PTX
5. Manually copy the PTX file
6. Manually manage hostcall buffers, CUDA device init, kernel loading

There is no library crate, no documentation, no examples, no build automation.

### Host-side SDK Design

The core reusable components in `gpu-host` are:
- `HostcallBuffer` — pinned memory allocation, listener loop, service dispatch
- `error::GpuHostError` — error types
- `mapped_mem` — helper for mapped memory allocation

These are already in a `lib.rs`, but `lib.rs` only exports `error` and `hostcall`. The `mapped_mem` module is `pub(crate)`. The `tests_*` modules contain all the kernel launch logic that should be extracted into reusable APIs.

**What a host-side SDK should provide:**
1. `GpuDevice` — high-level wrapper around `CudaDevice` + PTX loading
2. `KernelBuilder` — configure grid/block dims, shared memory, launch kernels
3. `HostcallRuntime` — manages buffer allocation + listener lifecycle
4. Memory management — typed wrappers for device/mapped memory
5. Tensor abstraction — at minimum, a `GpuTensor<T>` for data on device

**Complexity: Moderate.** Most of the infrastructure exists but is tangled into test functions. Extraction requires careful API design but no new research.

### GPU Kernel Library

The `gpu-kernel` crate is a monolithic kernel binary. Users cannot selectively use individual kernels. The compute kernels (GEMM, LayerNorm, GELU, attention, softmax) are research prototypes with hardcoded assumptions.

**What a kernel library should provide:**
1. Standalone, documented kernel functions with clear parameter contracts
2. Helper functions (warp reduce, shared memory management) as reusable building blocks
3. A way to compose kernels without modifying `gpu-kernel/src/lib.rs`

**Complexity: Hard.** The `nvptx64` target does not have a linker — all code must be in one compilation unit. Users cannot just `use gpu_kernel::gemm` from a separate kernel crate. This is a fundamental architectural constraint. Possible approaches:
- Macro-based: users `include!()` kernel modules
- Source distribution: users copy kernel source and compile together
- Monolithic: ship one big PTX with all kernels, users pick what they launch

### Build System

Current build process (from MEMORY.md):
```bash
cargo +nightly rustc --manifest-path crates/gpu-kernel/Cargo.toml --release \
  --target nvptx64-nvidia-cuda -Zbuild-std=core \
  -- --emit=asm -C linker=echo -C target-cpu=sm_86
cp crates/gpu-kernel/target/.../gpu_kernel.s crates/gpu-host/kernel.ptx
```

This is a manual, error-prone, undocumented process. Users need:
1. Correct nightly Rust toolchain
2. `nvptx64-nvidia-cuda` target
3. Knowledge of the exact rustc flags
4. Manual PTX file copy

**What build tooling should provide:**
- `build.rs` script that compiles kernel code to PTX automatically
- Cargo workspace integration so `cargo build` just works
- Target GPU architecture selection (currently hardcoded `sm_86`)

**Complexity: Moderate-Hard.** `build.rs` calling `cargo rustc` for a different target is tricky (recursive cargo invocation). Alternative: use `xtask` pattern or a Makefile. The PTX compilation itself works; the challenge is automation and cross-platform support.

### Documentation Strategy

- API documentation (rustdoc) for all public types and functions
- Getting Started guide: prerequisites, first kernel, run hello-gpu
- Architecture overview: hostcall protocol, kernel structure, async model
- Examples: hello-gpu, vector-add, async-file-io, compute-pipeline

**Complexity: Moderate.** Time-consuming but straightforward.

### Proposed Themes and Tasks

#### Theme: host-sdk
**Goal:** Extract a reusable library crate from gpu-host that external users can depend on.
**Success criteria:**
- `gpu-host` is a library crate (not just binary) with public, documented API
- A standalone example program can use `gpu-host` as a dependency to run a GPU kernel
- All existing tests continue to pass

**Tasks:**
- `host-sdk.1` (design): API surface design — what types/traits/functions are public
- `host-sdk.2` (experiment): Extract `GpuDevice` + `KernelBuilder` + `HostcallRuntime` from test code
- `host-sdk.3` (experiment): Create `GpuTensor<T>` abstraction over CudaSlice + mapped memory
- `host-sdk.4` (experiment): Standalone example that uses the SDK as a library dependency

#### Theme: kernel-lib
**Goal:** Make GPU kernel functions reusable and composable for external users.
**Success criteria:**
- Users can write their own kernel that calls provided helper functions
- Compute kernels have documented parameter contracts and usage examples
- Kernel compilation workflow supports user-written kernels

**Tasks:**
- `kernel-lib.1` (investigation): Evaluate approaches for kernel code reuse given no nvptx linker
- `kernel-lib.2` (experiment): Extract helper functions (warp_reduce, smem helpers) into a reusable module
- `kernel-lib.3` (experiment): Create a template/example user kernel that imports helpers

#### Theme: build-tooling
**Goal:** Automate kernel compilation so `cargo build` produces runnable output.
**Success criteria:**
- Single `cargo build` command compiles both host and kernel code
- Target GPU architecture is configurable
- Works on a clean clone with correct toolchain installed

**Tasks:**
- `build-tooling.1` (investigation): Evaluate build.rs vs xtask vs Makefile approaches
- `build-tooling.2` (experiment): Implement chosen approach with PTX auto-generation
- `build-tooling.3` (experiment): Validate on a fresh clone

#### Theme: examples-docs
**Goal:** Provide documentation and examples sufficient for a new user to get started.
**Success criteria:**
- `cargo doc` generates useful API documentation
- At least 3 working examples (hello-gpu, async-io, compute)
- Getting started guide covers prerequisites and first run

**Tasks:**
- `examples-docs.1` (experiment): Write hello-gpu minimal example
- `examples-docs.2` (experiment): Write async file I/O example
- `examples-docs.3` (experiment): Write compute pipeline example
- `examples-docs.4` (design): Getting started guide and architecture overview

## GPU Inference Re-evaluation

### Honest Gap Analysis

| Component | Complexity | Reusable from Current Code? | Notes |
|-----------|-----------|---------------------------|-------|
| BPE tokenizer | **Moderate** | No | ~500 lines Rust. Merge rules + byte-to-token mapping. Could use existing `tokenizers` crate on host. |
| Token embedding | **Trivial** | No | Simple table lookup. One kernel: output[i] = embedding_table[token_id[i]]. |
| Positional embedding | **Trivial** | No | output[pos][d] = token_emb[pos][d] + pos_emb[pos][d]. One elementwise_add call. |
| Causal attention mask | **Moderate** | Partially (attention kernel needs modification) | Must modify `attention_head` to apply -inf mask where j > i. Current kernel has no mask support. |
| Attention for seq > 32 | **Hard** | No, must rewrite | Current kernel uses 1 thread per sequence position, max 32. For seq=1024, need tiled attention (FlashAttention-style or multi-block). This is a significant rewrite. |
| 12-layer stacking | **Moderate** | Partially | Host-side orchestration of 13 steps x 12 layers = 156 kernel launches. Manageable but needs clean weight management. |
| Final LayerNorm | **Trivial** | Yes | Same kernel, different weights. |
| LM head projection | **Hard** | Partially | [seq, 768] x [768, 50257] GEMM. 50257 is not a nice power-of-2. Current GEMM requires dimensions to be multiples of 16. Need edge-tile handling or padding. |
| Softmax over 50,257 | **Moderate** | No | Current softmax works for N<=32. Need a general-purpose softmax for large vectors with warp-level reduction. |
| Token sampling | **Trivial** | No | argmax or top-k on host side. ~20 lines. |
| Autoregressive loop | **Moderate** | No | Generate one token at a time, append to input, re-run. Need KV cache for efficiency. |
| KV cache | **Hard** | No | Cache K/V for previous positions, only compute attention for new position. Significant memory management. Without it, generation is O(n^2 * L) which is impractical for long sequences. |
| Weight loading (safetensors) | **Moderate** | No | Parse safetensors binary format. ~200 lines Rust on host. Map weight names to kernel params. GPT-2 small has ~148 weight tensors. |
| f16 quantization accuracy | **Research risk** | N/A | Current pipeline has ~10% relative error after 3 GEMM stages. 12 layers = ~36 GEMM stages. Error may compound catastrophically. May need f32 intermediates between layers. |

### Critical Risks the Previous Assessment Ignored

1. **Compound precision loss.** One layer already has 9.6% max relative error. With 12 layers, each feeding into the next, this compounds. After 12 layers the output may be garbage. This is the single biggest risk. The fix (keeping f32 between GEMM stages, only quantizing to f16 for MMA inputs) changes the GEMM kernel and doubles intermediate memory.

2. **Attention scaling.** The attention kernel (`attention_head`) is hardcoded for seq_len <= 32. GPT-2 needs 1024. This is not a parameter change — it requires a fundamentally different algorithm. FlashAttention-style tiling is a research project in itself.

3. **LM head dimension.** 50,257 is prime-ish (50257 = 50257). The current GEMM requires N to be a multiple of 16. Padding to 50,272 wastes memory and compute. Or the kernel needs edge-tile handling.

4. **Memory budget.** GPT-2 small weights: ~234MB f16. Plus KV cache: 12 layers x 2 x 1024 x 768 x 4 bytes = ~75MB. Plus activations during forward: ~50MB. Total: ~360MB. Fits on most GPUs, but must be carefully managed.

5. **No actual PyTorch validation.** The success criterion says "matching PyTorch reference" but no PyTorch code was ever run. The CPU reference in `tests_compute.rs` is hand-written and may itself have bugs. Real validation requires running `transformers` library and comparing tensor outputs.

### Proposed Revised Themes and Tasks

#### Theme: model-loading
**Goal:** Load real GPT-2 weights from HuggingFace safetensors format.
**Success criteria:**
- Can parse GPT-2 small safetensors file and extract all 148 weight tensors
- Weights uploaded to GPU memory with correct layout for existing kernels
- Weight shapes validated against known GPT-2 architecture

**Tasks:**
- `model-loading.1` (investigation): Analyze GPT-2 safetensors format — tensor names, shapes, dtypes
- `model-loading.2` (experiment): Implement safetensors parser on host (or use `safetensors` crate)
- `model-loading.3` (experiment): Upload weights to GPU, validate shapes against expected architecture
- `model-loading.4` (experiment): Single-layer forward pass with real weights, compare against PyTorch

#### Theme: tokenizer
**Goal:** Implement GPT-2 BPE tokenizer for text-to-tokens and tokens-to-text.
**Success criteria:**
- Tokenize arbitrary English text into GPT-2 token IDs
- Detokenize token IDs back to text
- Output matches HuggingFace tokenizers library for test sentences

**Tasks:**
- `tokenizer.1` (investigation): Analyze GPT-2 tokenizer files (vocab.json, merges.txt)
- `tokenizer.2` (experiment): Implement BPE encoder/decoder on host side
- `tokenizer.3` (experiment): Validate against HuggingFace tokenizers for 10+ test sentences

#### Theme: attention-scale
**Goal:** Scale attention kernel to support seq_len up to 1024.
**Success criteria:**
- Attention kernel handles arbitrary seq_len (not just <= 32)
- Causal mask applied correctly (no attending to future tokens)
- Output matches PyTorch attention for seq=128, seq=512

**Tasks:**
- `attention-scale.1` (investigation): Survey approaches — tiled attention, FlashAttention, multi-block
- `attention-scale.2` (experiment): Implement causal mask in current 32-seq attention kernel (minimal change)
- `attention-scale.3` (experiment): Implement multi-block attention for seq > 32
- `attention-scale.4` (experiment): Validate at seq=128, seq=256 against PyTorch

#### Theme: precision-fix
**Goal:** Fix compound f16 quantization error that fails the original tolerance.
**Success criteria:**
- Single layer matches PyTorch within (atol=0.01, rtol=0.01) — the ORIGINAL stated tolerance
- 12-layer stacked output does not diverge catastrophically from PyTorch

**Tasks:**
- `precision-fix.1` (investigation): Profile per-stage error to identify worst quantization points
- `precision-fix.2` (experiment): Keep f32 activations between GEMM stages (skip f16 roundtrip)
- `precision-fix.3` (experiment): Re-validate single layer with f32 intermediates against PyTorch
- `precision-fix.4` (experiment): Stack 12 layers and measure error accumulation

#### Theme: full-inference
**Goal:** Run complete GPT-2 inference: text in, text out.
**Success criteria:**
- Given a text prompt, produce coherent continuation text
- Token embedding + 12 layers + LM head + sampling all working
- Output matches HuggingFace GPT-2 for greedy decoding (same input = same output)

**Tasks:**
- `full-inference.1` (experiment): Implement token + positional embedding kernels
- `full-inference.2` (experiment): Implement 12-layer stacked forward pass
- `full-inference.3` (experiment): Implement LM head + softmax-over-vocabulary
- `full-inference.4` (experiment): Implement greedy autoregressive generation loop
- `full-inference.5` (experiment): End-to-end validation: same prompt → same output as HuggingFace

#### Theme: inference-validation
**Goal:** Rigorous validation of GPU inference against HuggingFace reference.
**Success criteria:**
- Per-layer intermediate outputs match PyTorch within tight tolerance
- Generated text is identical to HuggingFace for greedy decoding
- At least 3 different prompts produce coherent, matching output

**Tasks:**
- `inference-validation.1` (experiment): Write PyTorch reference script that dumps per-layer intermediates
- `inference-validation.2` (experiment): Compare each GPU layer output against corresponding PyTorch dump
- `inference-validation.3` (experiment): End-to-end text generation comparison (3+ prompts)

### Revised Success Criteria for gpu-inference Epic

The old criteria were:
1. MHA with "real weight values" — actually used synthetic
2. Match PyTorch "within f16 tolerance (atol=1e-2, rtol=1e-2)" — actually relaxed to 5x/10x
3. GPU "autonomously orchestrates" — actually host-orchestrated

**Proposed new criteria:**
1. Real GPT-2 small weights loaded from HuggingFace safetensors
2. BPE tokenizer produces correct token IDs matching HuggingFace
3. Single-layer output matches PyTorch within (atol=0.01, rtol=0.01) — no relaxation
4. 12-layer stacked output matches PyTorch within (atol=0.05, rtol=0.05)
5. Given the prompt "The capital of France is", the model generates text containing "Paris"
6. Greedy-decoded output matches HuggingFace GPT-2 token-for-token for at least 3 test prompts

## Cross-Epic Dependencies

```
Public API                          gpu-inference (revised)
---------                          ----------------------
host-sdk ─────────────────────────► model-loading (needs SDK to load weights)
kernel-lib ───────────────────────► attention-scale (needs kernel helpers)
build-tooling                       precision-fix
examples-docs ◄───────────────────── full-inference (best example IS inference)
                                    tokenizer (independent)
                                    inference-validation
```

**Does Public API block gpu-inference?** Partially. The current test infrastructure works for inference development, but:
- `model-loading` would benefit from a clean `GpuDevice` + `GpuTensor` API
- Without a library crate, inference code must live inside `gpu-host/main.rs` — ugly but functional

**Does gpu-inference block Public API?** No, but inference is the best showcase example for the API.

**Recommended execution strategy:**
1. Start `host-sdk` and `precision-fix` in parallel (both are prerequisites)
2. `tokenizer` and `model-loading` can run in parallel (independent)
3. `attention-scale` blocks `full-inference`
4. `full-inference` depends on everything above
5. `examples-docs` should come last (document what exists)
6. `build-tooling` is independent, can run anytime

## Risk Assessment

### Hardest Unsolved Problems

1. **Attention at seq=1024 (Hard, Research Risk)**
   The current attention kernel processes one sequence position per thread (max 32). For 1024 positions, this needs a fundamentally different approach. FlashAttention requires careful shared memory tiling, online softmax computation, and multi-pass accumulation. This is the hardest single kernel to implement correctly.

2. **Compound f16 Precision (Hard, Research Risk)**
   12 layers of f16→f32→f16 roundtrips may cause catastrophic precision loss. If f32 intermediates don't fix this, we may need to abandon f16 GEMM entirely and use f32 GEMM (losing Tensor Core speedup). This would require rewriting the GEMM kernel.

3. **LM Head Dimension Mismatch (Moderate)**
   50,257 is not divisible by 16. The GEMM kernel tile size is 16. Options: pad to 50,272 (wastes compute), implement edge-tile handling (complicates kernel), or quantize vocabulary (changes model semantics).

4. **No nvptx Linker for Kernel Composition (Architecture Issue)**
   Users cannot link separate kernel crates. All GPU code must be in one compilation unit. This fundamentally limits kernel-lib's usability. There is no clean solution — only workarounds (include!, source copy, monolithic PTX).

### Where We Might Need to Change Approach

- **GEMM kernel**: If f16 precision proves insufficient after 12 layers, may need f32-only GEMM path (no Tensor Cores). Performance drops but correctness is non-negotiable.
- **Attention kernel**: If FlashAttention-style tiling proves too complex in inline PTX, may need to limit seq_len and accept a performance penalty with naive multi-block attention.
- **Kernel distribution**: If the no-linker constraint makes kernel-lib impractical, may need to shift to a "template project" model where users fork the whole kernel crate.

### Complexity Genuinely Ignored in Previous Assessment

1. **"PASSED" at wrong tolerance.** transformer-layer.7 declared PASSED at (0.05, 0.10) when the criterion was (0.01, 0.01). This should have been FAILED with a note about needing precision improvements.
2. **No causal mask.** The attention test computes full bidirectional attention. GPT-2 is unidirectional (causal). Without the mask, the model produces wrong outputs regardless of precision.
3. **seq=32 limitation not flagged.** GPT-2's minimum useful context is at least 128 tokens. The 32-token limit was never identified as a blocker.
4. **No real weight loading.** "With real weight values" was a success criterion. Synthetic weights were used and the criterion was declared met.
5. **Host orchestration vs GPU autonomy.** The pipeline is 13 separate host-dispatched kernel launches with `dev.synchronize()` between each. This is standard CUDA programming, not GPU-autonomous execution.

## Concrete Recommendations

### Priority Ordering of Themes

**Phase 1 (parallel, unblock everything):**
1. `host-sdk` — extract library crate (enables clean development for everything else)
2. `precision-fix` — fix f16 compound error (blocks all inference work)
3. `tokenizer` — independent, can run immediately

**Phase 2 (parallel, core inference):**
4. `model-loading` — real weights (depends on host-sdk for clean API)
5. `attention-scale` — seq > 32 with causal mask (independent, hard)

**Phase 3 (sequential):**
6. `full-inference` — assemble everything (depends on 1-5)
7. `inference-validation` — rigorous validation (depends on 6)

**Phase 4 (polish, parallel with Phase 3):**
8. `kernel-lib` — kernel reuse story
9. `build-tooling` — automated build
10. `examples-docs` — documentation

### Which Epic to Tackle First

**Both epics should start simultaneously**, with different themes:
- Public API: start with `host-sdk` (Phase 1)
- gpu-inference: start with `precision-fix` and `tokenizer` (Phase 1)

The `host-sdk` work directly benefits inference development, so there is no conflict.

### Key Decision Gates

1. **After precision-fix.3**: Does single-layer match PyTorch at (atol=0.01, rtol=0.01) with f32 intermediates? If NO → evaluate f32-only GEMM.
2. **After precision-fix.4**: Does 12-layer output stay within (atol=0.05, rtol=0.05)? If NO → fundamental precision approach must change.
3. **After attention-scale.3**: Does multi-block attention work for seq=128+? If NO → evaluate reduced sequence length or different algorithm.
4. **After model-loading.4**: Does real-weight single-layer match PyTorch? If NO → investigate weight layout / transposition bugs.
5. **After full-inference.5**: Does greedy output match HuggingFace? This is the final gate — either it works or it does not.
