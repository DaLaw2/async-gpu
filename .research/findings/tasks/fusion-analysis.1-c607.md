# fusion-analysis.1 — Fusable MIR Patterns Investigation

## Status: done
## Summary

The GPT-2 forward pass launches approximately 135 kernels, of which ~60% are memory-bound elementwise ops (bias_add, GELU, elementwise_add, LayerNorm) that are prime fusion targets. The existing codebase already demonstrates two fusion strategies: NVRTC-compiled fused kernels (layer_norm_residual_dual) and V1-GEMM-level fused kernels (gemm_bias_gelu/relu). The recommended approach for auto-fusion is a **tape-level fusion pass** that pattern-matches the autograd tape before execution, generating fused NVRTC kernels at runtime — this fits the existing architecture, avoids MIR-pass complexity, and delivers the North Star of "users write idiomatic Rust, compiler fuses the ops."

## 1. Fusable Patterns Taxonomy

### 1.1 Elementwise Chains (highest priority)

**Pattern**: Any sequence of element-wise ops on tensors of the same shape where each output feeds exactly one consumer.

Concrete examples from the GPT-2 forward pass:

| Chain | Ops today | Kernels | Fused kernels |
|-------|-----------|---------|---------------|
| `matmul + bias_add + gelu` | matmul_v2, bias_add, gelu_forward_v2 | 3 | 1 (gemm_bias_gelu already exists) |
| `matmul + bias_add` | matmul_v2, bias_add | 2 | 1 (epilogue fusion) |
| `elementwise_add + layer_norm` | elementwise_add_v3, layer_norm_v3 | 2 | 1 (layer_norm_residual_dual exists) |
| `x * scale + bias` (arbitrary) | mul, add | 2 | 1 (fused elementwise) |

**Fusability rules**:
- All operands must have the same element count (no broadcast mismatch)
- Each intermediate result must have exactly one consumer (no fan-out)
- The chain must be memory-bound (arithmetic intensity < HW threshold)

### 1.2 Activation Fusion (post-GEMM epilogue)

**Pattern**: `activation(matmul(A, B) + bias)` — the matmul itself is compute-bound and cannot fuse with other matmuls, but the bias-add and activation are cheap enough to fold into the GEMM epilogue.

Already implemented: `matmul_fused()` with `FusedActivation::Gelu` and `FusedActivation::Relu`. Currently only works with the V1 GEMM kernel. Extending to cuBLAS/V2/V4 requires epilogue callback support.

**Missing fusions**:
- cuBLAS GEMM with epilogue (cuBLAS 11+ supports `cublasLtMatmulDescSetAttribute` for bias+activation epilogues)
- V2/V3/V4 custom GEMM kernels with inline epilogue (straightforward: add bias+activation after writing C[i][j])

### 1.3 Broadcast Operations

**Pattern**: `scalar * tensor`, `tensor + scalar`, `tensor + [N]_broadcast_to_[M,N]`

Current status: `bias_add` is already a broadcast fusion (adds a [N] vector to each row of [M, N]). No generic broadcast fusion exists.

Fusability: These are trivially fusable into any elementwise chain — the scalar/vector just becomes an additional parameter to the fused kernel.

### 1.4 Reduction Chains

**Pattern**: `sum(x * y)` (dot product), `mean(x^2)` (variance), `max(softmax(x))`

These are **partially fusable**: the elementwise part (x*y) can fuse, but the reduction itself requires a different execution pattern (parallel reduction with shared memory). The fusion boundary is: elementwise ops fuse freely with each other, but a reduction terminates the chain.

LayerNorm is a special case: it does reduction (mean, variance) internally, so it cannot fuse with prior elementwise ops in the general case, but CAN fuse with a preceding elementwise_add (as demonstrated by `layer_norm_residual`) because the reduction happens after the add.

### 1.5 Non-Fusable Boundaries

Operations that **cannot** participate in elementwise fusion:
- **GEMM/matmul**: Compute-bound, already using register blocking and shared memory. Fusing an elementwise op INTO the GEMM epilogue is possible (and desirable), but fusing two GEMMs is not.
- **Attention**: Flash attention uses a specialized tiling/online-softmax algorithm. It is its own fusion island.
- **Conv2d**: Same as GEMM — the im2col + GEMM pattern is compute-bound.
- **Reshape/transpose**: These change memory layout, not values. They break contiguity assumptions.
- **Embedding lookup**: Gather operation with irregular access pattern.

## 2. Compilation Pipeline Options

### 2.1 MIR-Level Fusion (rejected for now)

**How it would work**: Add a new MIR pass (like WarpCooperativeTransform) that detects sequences of `ops::gelu(ops::bias_add(ops::matmul(...)))` calls in the MIR, merges them into a single fused call.

**Pros**:
- After inlining, the compiler can see through Module::forward() boundaries
- Pattern matching on MIR is well-understood (the project already does it for warp-cooperative transform)
- Fusion happens at compile time — zero runtime overhead

**Cons**:
- The nn ops layer operates on `GpuTensor` (opaque GPU buffers) — the MIR pass would need to understand that `ops::gelu` followed by `ops::bias_add` on the same tensor is fusable, which requires domain-specific knowledge that doesn't exist in MIR
- MIR patterns are fragile: any refactoring of the ops API breaks the pass
- The WarpCooperativeTransform is targeted (coroutine state machines); a fusion pass would need to match arbitrary op sequences
- Maintenance burden: patching rustc is already a significant cost center

**Verdict**: MIR-level fusion is architecturally possible but not the right abstraction level. The MIR pass doesn't know about tensor shapes, memory layout, or GPU kernel launch semantics. These are runtime properties.

### 2.2 LLVM IR-Level Fusion (not applicable)

The nn ops launch GPU kernels via the CUDA driver API — from LLVM's perspective, these are opaque function calls to `cuLaunchKernel`. LLVM cannot reason about kernel fusion. This approach is a non-starter.

### 2.3 Runtime/Tape-Level Fusion (recommended)

**How it would work**: The autograd tape already records every op (Matmul, BiasAdd, Gelu, ElemAdd, LayerNorm, etc.) with full metadata (shapes, parameters). A fusion pass would:

1. Walk the tape entries
2. Identify fusable subsequences (e.g., BiasAdd → Gelu on same-shape tensors)
3. Generate a fused CUDA kernel via NVRTC (already used for layer_norm_residual_dual)
4. Replace the N separate kernel launches with 1 fused launch
5. Cache the compiled kernel for reuse

**Pros**:
- Sees actual tensor shapes at runtime — can make optimal fusion decisions
- NVRTC compilation is already proven in the codebase (fused LN+residual, GEMM V4/V4.1)
- No rustc patches needed — pure library-level implementation
- The tape provides a natural "trace" of the computation graph
- First-call compile cost is amortized over subsequent calls (same shapes = same kernel)
- Fits the North Star: users write `gelu(linear(x))`, the runtime fuses automatically

**Cons**:
- First-call latency for NVRTC compilation (~50-200ms per fused kernel)
- Must handle the cache correctly (key: op-sequence + shapes)
- Tape-level fusion only works during training (when the tape is recording) — for inference, need a separate "trace and fuse" mode

**Architecture sketch**:
```
// Before fusion: tape has [Matmul, BiasAdd, Gelu]
// After fusion: tape has [Matmul, FusedBiasGelu]
// The FusedBiasGelu entry points to an NVRTC-compiled kernel
```

### 2.4 Library-Level Explicit Fusion (already exists, extend)

**How it would work**: Provide explicit fused APIs like `matmul_fused()`, `layer_norm_residual_dual()`.

**Pros**: Zero overhead, maximum control, already proven.
**Cons**: Doesn't scale — every new fusion pattern needs a hand-written kernel. Violates the North Star (user must know about fused APIs).

**Verdict**: Keep as the performance baseline. Auto-fusion should generate equivalent kernels.

### 2.5 Hybrid: Tape Replay with Codegen (best fit)

Combine 2.3 and 2.4:
- Hand-written fused kernels for the highest-impact patterns (already done)
- Tape-level pattern matching to AUTOMATICALLY select fused kernels when the pattern matches
- NVRTC codegen for patterns that don't have hand-written kernels
- Progressive: start with pattern-matching to existing fused kernels, then add codegen

## 3. Existing NN Pipeline Analysis

### 3.1 GPT-2 Forward Pass Kernel Count

Per `TransformerBlock::forward()` (with cublas feature, fused LN+residual):

| Step | Operation | Kernel launches |
|------|-----------|----------------|
| LN1 | `ln_1.forward(input)` → `layer_norm_v3` | 1 |
| QKV proj | `qkv_proj.forward(ln1)` → `matmul_v2` (or cuBLAS) + `bias_add` | 2 |
| Split QKV | `split_qkv` | 1 |
| Attention | `multi_head_flash_attention` | 1 |
| Concat heads | `concat_heads` | 1 |
| Out proj | `out_proj.forward(concat)` → matmul + bias_add | 2 |
| Fused LN+res | `layer_norm_residual_dual(input, attn_out, ...)` | 1 |
| FFN up | `ffn_up.forward(ln2)` → matmul + bias_add | 2 |
| GELU | `gelu.forward(ffn_hidden)` → `gelu_forward_v2` | 1 |
| FFN down | `ffn_down.forward(ffn_act)` → matmul + bias_add | 2 |
| Residual add | `elementwise_add(residual, ffn_out)` | 1 |
| **Block total** | | **~15** |

For GPT-2 Small (12 layers): 12 * 15 = 180 block kernels + embedding (1) + final LN (1) + LM head matmul+bias (2) = **~184 total**. The previously measured ~135 was likely with a different kernel counting methodology or without all bias_adds counted separately.

Note: cuBLAS matmul path counts as 1 kernel (cuBLAS internally may use multiple, but from the driver perspective it's one launch).

### 3.2 Fusion Opportunities by Frequency

Per transformer block, the fusable patterns are:

1. **matmul + bias_add** (occurs 4x per block = 48x total): The most frequent pattern. bias_add is a trivial elementwise op that reads the entire output tensor just to add a small vector. GEMM epilogue fusion eliminates this entirely.

2. **bias_add + gelu** (occurs 1x per block = 12x total): After FFN up projection. Already has a hand-written fused kernel (`gemm_bias_gelu`) but it's only used via `matmul_fused()` (V1 GEMM path). The V2/cuBLAS path doesn't use it.

3. **elementwise_add + layer_norm** (occurs 1x per block = 12x total, already fused with cublas): `layer_norm_residual_dual` saves 1 kernel launch per block.

4. **elementwise_add alone** (occurs 1x per block = 12x total): The final residual add. Could potentially fuse with the next block's LN1 in a cross-block fusion.

### 3.3 Time Breakdown Estimate

For memory-bound ops at GPT-2 Small dims (128 tokens, 768 hidden):
- Each elementwise op on [128, 768] = 98K floats = 393 KB. At 300 GB/s bandwidth, the data transfer takes ~1.3 us, but kernel launch overhead is ~5-10 us. **Launch overhead dominates.**
- Each matmul is compute-bound and takes ~0.1-1 ms depending on dims.
- Fusing 2 elementwise ops saves ~5-10 us per fusion (one fewer launch). Over 12 blocks with 3-4 fusable pairs each, that's ~200-500 us savings.
- For inference (small batch), this is significant: total forward is ~10-30 ms, so ~2-5% speedup.

### 3.4 Existing Fused Kernels

| Kernel | Fuses | Location |
|--------|-------|----------|
| `layer_norm_residual` | elementwise_add + layer_norm | `norm.rs` (NVRTC) |
| `layer_norm_residual_dual` | elementwise_add + layer_norm (dual output) | `norm.rs` (NVRTC) |
| `gemm_bias_gelu` | matmul + bias_add + GELU | `gemm.rs` (PTX cubin) |
| `gemm_bias_relu` | matmul + bias_add + ReLU | `gemm.rs` (PTX cubin) |
| `batchnorm_silu` | batch_norm + SiLU | `norm.rs` (PTX cubin) |

## 4. Academic/Industry Precedents

### 4.1 XLA (TensorFlow)

- Operates on HLO (High-Level Optimizer) graph — a dataflow IR of tensor operations
- Fusion strategy: greedily fuses elementwise ops into "fusion instructions"
- Fusion boundaries: reductions, scatters, transposes
- Generates fused LLVM IR → PTX at compile time
- **Relevance**: HLO is analogous to our autograd tape — a sequence of typed ops with shape metadata. XLA's greedy elementwise fusion is directly applicable.

### 4.2 TorchInductor (PyTorch torch.compile)

- Traces the computation graph via torch.fx
- Groups fusable ops into "fusion groups" using a graph-partitioning algorithm
- Generates Triton kernels for fused groups
- Uses a cache (TorchInductor cache) to avoid recompilation
- **Relevance**: Most directly analogous to our proposed approach. The trace = our tape. The Triton codegen = our NVRTC codegen. The cache = our kernel cache. Key insight: they do NOT modify the compiler (Python/C++ boundary) — fusion is entirely at the framework level.

### 4.3 TVM

- Schedule-based: user (or AutoTVM/Ansor) specifies how to fuse and tile
- Fusion rules: elementwise→elementwise (always), reduce→elementwise (inject), complex→elementwise (output)
- Generates optimized CUDA/OpenCL/Metal code
- **Relevance**: TVM's fusion rule taxonomy (injective, reduction, complex) maps well to our op categories. But TVM requires explicit scheduling — our goal is automatic fusion.

### 4.4 MLIR

- Dialect-based: ops live in dialects (linalg, tensor, memref, gpu)
- Fusion happens via "tiling and fusing" passes that transform linalg ops
- Recent work on "producer-consumer fusion" in linalg dialect
- **Relevance**: MLIR's approach is too heavyweight for our use case. We don't need a general-purpose compiler IR — we need to fuse a known set of ~10 op types.

### 4.5 cuDNN/cuBLAS Fusion

- cuBLAS 11+ supports GEMM epilogue fusion via `cublasLtMatmul` — can fuse bias+activation into the GEMM output write
- cuDNN 8+ has a "fusion engine" that can fuse conv+BN+activation
- **Relevance**: For GEMM epilogue fusion specifically, using cuBLAS epilogue API is the path of least resistance. Avoids writing custom fused GEMM kernels for every activation.

### 4.6 Key Insight from Literature

All successful fusion systems share a common architecture:
1. **Trace/record** the computation graph (HLO, fx trace, TVM relay, our tape)
2. **Pattern match** to identify fusable groups (greedy elementwise grouping)
3. **Codegen** the fused kernel (LLVM, Triton, TVM, NVRTC)
4. **Cache** the compiled kernel (keyed on op-sequence + shapes)

None of them do fusion at the source-language compiler level (rustc/clang). This validates our approach of doing fusion at the tape/runtime level rather than in the MIR pass.

## 5. Constraints and Tradeoffs

### 5.1 Architecture Constraints

- **Patched rustc is for async/await only**: The WarpCooperativeTransform is narrowly scoped to coroutine state machines. Adding tensor fusion to rustc would be a fundamentally different kind of transformation and would conflate two unrelated concerns.
- **NVRTC is available and proven**: The codebase already compiles CUDA C at runtime for 3 fused kernels. This is the natural codegen backend.
- **Autograd tape records ops**: The tape is the computation graph. It has shape metadata. It is the right place to look for fusion patterns.
- **cuBLAS is used for small-M matmuls**: cuBLAS epilogue fusion (via cublasLt) should be used when available, rather than writing custom epilogue kernels.

### 5.2 Implementation Effort

| Approach | Effort | Impact | Risk |
|----------|--------|--------|------|
| Pattern-match tape → call existing fused kernels | Low (1-2 weeks) | Medium — picks low-hanging fruit | Low |
| Extend matmul_v2/v4 with epilogue fusion | Medium (1 week) | High — eliminates 48 bias_add launches | Low |
| cuBLAS epilogue fusion via cublasLt | Medium (1 week) | High — same as above, cuBLAS path | Medium (API complexity) |
| NVRTC codegen for arbitrary elementwise chains | High (3-4 weeks) | High — general solution | Medium (correctness) |
| MIR-pass fusion | Very high (months) | Low — wrong abstraction level | High |

### 5.3 Inference vs Training

- **Inference**: No tape recording by default. Need a "tracing" mode that records one forward pass, fuses, then replays fused. This is exactly `torch.compile`'s approach.
- **Training**: Tape is already recording. Fusion can happen on tape replay (backward). But backward fusion is harder — the fused kernel must also produce correct gradients.
- **Recommendation**: Start with inference-only fusion (forward pass tracing). Training fusion is a separate, harder problem.

## 6. Recommended Approach

### Phase 1: GEMM Epilogue Fusion (highest ROI, lowest risk)

Extend `matmul_v2`, `matmul_v4_1`, and the cuBLAS path to support bias+activation epilogues:
- For custom GEMM kernels: add `bias` and `activation` parameters to the output write loop
- For cuBLAS: use `cublasLtMatmul` with epilogue attributes
- This eliminates 48 kernel launches per GPT-2 forward pass (4 bias_adds + 1 GELU per block * 12 blocks, minus already-fused ones)

**How the user sees it**: `Linear::forward()` automatically fuses bias_add into the matmul. When `GELU` follows `Linear`, the layer detects this and uses the fused path. No API change needed.

### Phase 2: Tape-Level Pattern Matching

Build a `FusionOptimizer` that walks the autograd tape and replaces known patterns:
- `[Matmul, BiasAdd, Gelu]` → `[FusedMatmulBiasGelu]`
- `[ElemAdd, LayerNorm]` → `[FusedLNResidual]`
- Use the existing hand-written fused kernels as the implementation

This is a pure library change. The tape already has all the information needed.

### Phase 3: NVRTC Codegen for Arbitrary Elementwise Chains

Build a code generator that takes a sequence of elementwise ops and produces a fused CUDA kernel:
- Input: `[(BiasAdd, shape=[128,768]), (Gelu, shape=[128,768])]`
- Output: NVRTC-compiled kernel that does `gelu(x + bias)` in one pass
- Cache keyed on `(op_sequence, shapes)`

This is the general solution that delivers the North Star. It subsumes Phase 2.

### Phase 4 (future): Cross-Block Fusion

The residual add at the end of block N could fuse with the LayerNorm at the start of block N+1. This requires the fusion pass to see across Module::forward() boundaries, which is straightforward at the tape level (the tape is flat, not hierarchical).

## Files Changed: none
