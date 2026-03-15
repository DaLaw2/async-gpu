# BS-102 Skeptic Analysis: Roadmap Challenges

**Date**: 2026-03-15
**Role**: Skeptic
**Roadmap under review**: verify-examples → testing-infra → autograd → gpu-safety

---

## 1. Scope Creep Risk

### 1.1 Autograd is an iceberg, not a feature

The proposer likely frames autograd as "add backward kernels for existing ops." In reality:

- **Forward ops**: 13 ops currently (matmul, conv2d, layer_norm, batch_norm, batch_norm_silu, gelu, silu, sigmoid, relu, max_pool2d, upsample_nearest_2x, embedding_lookup, scaled_dot_product_attention). Each one needs a backward kernel.
- **Conv2d backward** is *three* separate operations: gradient w.r.t. input (col2im of transposed weight × grad), gradient w.r.t. weight (input_col^T × grad), gradient w.r.t. bias (reduce over spatial dims). The current conv2d already does im2col on CPU host-side (`col_host = dev.dtoh_sync_copy(&col_dev)`) — the backward needs the *same* im2col matrix saved from forward. That is a storage and API problem, not just a kernel problem.
- **Attention backward** through flash attention is notoriously hard. The whole point of flash attention is *not* materializing the S = Q·K^T matrix. But the backward *requires* recomputing it tile-by-tile. This is a research-grade kernel. FlashAttention-2's backward kernel is ~500 lines of CUDA. In hand-written inline PTX? That is months of work.
- **Realistic task count**: I estimate 30-40 tasks minimum for a useful autograd (tape + 13 backward ops + optimizer + tests + memory management). The entire ml-framework epic was 50 tasks. Autograd could be equally large.

**Mitigation**: Scope autograd to *linear-only models first* (Linear + LayerNorm + GELU + embedding). Skip conv2d and attention backward entirely in v1. That cuts the task count to ~15 and covers GPT-2 fine-tuning (the higher-value target).

### 1.2 gpu-safety: solving a theoretical problem?

I searched the codebase for evidence of GPU memory safety bugs. The `GpuTensor` struct owns its `CudaSlice<f32>` and drops it when the tensor is dropped. `cudarc` already provides:
- RAII memory management via `CudaSlice<T>` (freed on Drop)
- `Arc<CudaDevice>` prevents device from outliving allocations
- Type-safe kernel launch via `LaunchAsync`

**Question**: What actual bug would `GpuRef<'a, T>` prevent that the current design does not? The only scenario I can construct is someone calling `GpuTensor::as_ptr()` and holding the raw `CUdeviceptr` past the tensor's lifetime — but that requires `unsafe` code the user would have to write intentionally.

**Mitigation**: Before creating a gpu-safety epic, document 3 concrete bug scenarios that GpuRef would prevent. If those scenarios require `unsafe` user code, then the safety boundary is already correct and GpuRef adds compile-time overhead for no real benefit.

### 1.3 Testing-infra: property-based testing is overkill for 13 ops

Property-based testing (e.g., proptest/quickcheck) shines when the input space is large and edge cases are non-obvious. For GPU tensor ops:
- Input shapes are constrained (must match kernel tile sizes: M%32=0, K%16=0, N%16=0 after padding)
- Numerical precision is deterministic for given inputs (f32 FMA is IEEE-754 compliant)
- The interesting bugs are not in random shapes — they are in specific padding boundary conditions (e.g., M=33, K=17) which should be *explicit* test cases, not randomly discovered

Property-based testing would generate thousands of random shapes, most of which test the same padding code path. Meanwhile, you burn GPU time on each test.

**Mitigation**: Golden-file tests with carefully chosen boundary shapes (exact tile multiples, tile+1, tile-1, 1×1, very large) provide better coverage with 10x less infrastructure. Save the property testing for the autograd tape logic (which is purely CPU-side and benefits from random graph structures).

---

## 2. Ordering Challenges

### 2.1 Why autograd before gpu-safety?

If GpuRef is genuinely needed, then building autograd *without* it means the autograd code will need to be rewritten. Autograd tapes hold references to saved tensors — exactly the kind of code where lifetime tracking matters most. Building autograd first creates technical debt that gpu-safety would then have to clean up.

**Counter-argument**: If gpu-safety turns out to be unnecessary (see 1.2), then waiting for it delays autograd pointlessly.

**Recommendation**: Resolve the gpu-safety question *first* (1-2 investigation tasks, not a full epic) before committing to autograd. If GpuRef is needed, design it before autograd consumes it.

### 2.2 testing-infra before verify-examples?

The roadmap puts verify-examples first. But verify-examples is about running existing code and fixing paths — it does not create any testing infrastructure. If testing-infra came first, verify-examples would benefit from the harness.

**Counter-argument**: verify-examples is mostly about "does it run at all?" — you do not need a fancy harness to run `cargo run --example gpt2-inference`. The current verify-examples epic is correctly scoped as a smoke test.

**Verdict**: Current ordering is fine. verify-examples is a prerequisite sanity check; testing-infra builds on top of it.

### 2.3 Can any of these run in parallel?

- **verify-examples** and **testing-infra** are sequential (testing-infra needs working examples to test).
- **gpu-safety investigation** (not the full epic) could run in parallel with verify-examples — it is a design investigation, not code.
- **autograd** depends on testing-infra (you need numerical accuracy tests to validate backward passes).

**Recommendation**: Run the gpu-safety investigation as a 2-task spike during verify-examples. This resolves the "do we even need GpuRef?" question before committing resources.

---

## 3. Technical Feasibility Challenges

### 3.1 Autograd tape must live on host — tension with GPU execution

The forward pass runs on GPU via kernel launches orchestrated from the host. The autograd tape records *which ops were called* and *what their saved tensors are*. This is inherently a host-side data structure.

**This actually works fine with the current architecture** because:
- Every op in `gpu-host/src/nn/ops/` is already a host-side Rust function
- The tape would wrap these host-side functions, recording inputs/outputs
- Backward passes would be additional host-side functions that launch backward kernels

The real problem is **saved activations**. The forward pass for conv2d currently does:
```
let col_host = dev.dtoh_sync_copy(&col_dev)?;
```
The im2col result is downloaded to host, used, and dropped. For autograd, you need to *save* the im2col result for the backward pass. This doubles VRAM usage for conv layers.

**Mitigation**: Implement activation checkpointing from day 1. For conv2d, re-run im2col in the backward pass instead of saving it. This trades compute for memory — a well-understood tradeoff.

### 3.2 GpuRef vs cudarc's existing RAII

Looking at the actual tensor code:

```rust
pub struct GpuTensor {
    data: CudaSlice<f32>,     // RAII — freed on Drop
    shape: SmallVec<[usize; 4]>,
    strides: SmallVec<[usize; 4]>,
    device: Arc<CudaDevice>,  // prevents device outliving allocations
}
```

`CudaSlice<f32>` is already `!Copy` and `!Clone` (manual clone requires `clone_tensor()`). The ownership model is already sound. `GpuRef<'a, T>` would add a *borrowing* layer — a non-owning reference to device memory with a lifetime tied to the owning tensor.

**When would you need this?** Only when you want to pass a *view* of a tensor without cloning. Currently, `reshape()` and `transpose()` both *copy* the data. Zero-copy views would require GpuRef. But zero-copy views are a performance optimization, not a safety feature.

**Recommendation**: Rename this epic from "gpu-safety" to "zero-copy tensor views" — that is what it actually is. Frame it as a performance feature, not a safety feature. And consider whether the current copy-on-reshape is actually a bottleneck before building it.

### 3.3 Backward kernels in inline PTX: conv2d is the hard case

- **matmul backward**: `dA = dC · B^T`, `dB = A^T · dC`. Both are just matmuls with transposed inputs. The existing `gemm_f32` kernel handles this. **Feasible.**
- **layer_norm backward**: analytically straightforward, similar complexity to forward. **Feasible.**
- **activation backward**: element-wise (e.g., GELU backward = x·Φ'(x) + Φ(x)). Trivial kernel. **Feasible.**
- **conv2d backward w.r.t. input**: requires col2im — the inverse of im2col. The im2col kernel already exists; col2im is structurally similar but scatters instead of gathers. **Feasible but needs a new kernel.**
- **conv2d backward w.r.t. weight**: reshape grad to [C_out, H_out*W_out], multiply by im2col^T. This is a matmul + im2col. **Feasible with existing kernels.**
- **attention backward**: Requires recomputing Q·K^T per tile, applying softmax backward, then matmul backward for V projection. The flash_attention kernel is already the most complex kernel in the codebase. Writing a backward variant in inline PTX that is *correct and numerically stable* is extremely ambitious. **High risk.**
- **max_pool2d backward**: Requires saving argmax indices from forward. Current max_pool2d does not save these. Needs kernel modification. **Medium effort.**

**Recommendation**: Explicitly classify backward kernels into tiers:
- **Tier 1** (reuse existing kernels): matmul, bias_add, elementwise_add, embedding
- **Tier 2** (new but simple kernels): activations, layer_norm, batch_norm
- **Tier 3** (significant new kernels): conv2d (col2im), max_pool2d (argmax)
- **Tier 4** (research-grade): attention backward

Scope autograd v1 to Tier 1 + Tier 2 only. Tier 3-4 are separate epics.

### 3.4 VRAM pressure with autograd

GPT-2 Small has 124M parameters = ~500MB in f32. Forward pass activations for a 1024-token sequence:
- Embedding output: 1024 × 768 × 4 bytes = 3MB
- Per-block (12 blocks): attention QKV + attention output + FFN hidden + FFN output ≈ 4 × 1024 × 768 × 4 ≈ 12MB per block, 144MB total
- Total forward: ~650MB

With autograd saving activations: double the activation memory = ~800MB total. This fits on an 8GB GPU but leaves little room for larger models or batch sizes > 1.

YOLOv8-nano is smaller (~3.2M params) so less of a concern.

**Mitigation**: Gradient checkpointing is not optional — it is a requirement for autograd to be usable. Budget 3-4 tasks for it within the autograd epic.

---

## 4. Alternative Approaches

### 4.1 Instead of a full autograd tape: manual backward functions

Many frameworks started with manually defined backward functions per module (no tape). The `Module` trait could gain:
```rust
fn backward(&self, grad_output: &GpuTensor, saved: &SavedTensors) -> Result<GpuTensor>;
```
Each module saves what it needs during forward and uses it during backward. No tape, no graph tracing. This is simpler, more predictable, and sufficient for fine-tuning known architectures.

**Tradeoff**: You cannot do `loss.backward()` on arbitrary computation graphs. You must manually call backward through the model. For training known architectures (GPT-2, YOLO), this is perfectly fine.

**Recommendation**: Start with manual backward, upgrade to tape-based autograd only if users need dynamic graphs.

### 4.2 Instead of GpuRef: arena allocator

An arena allocator that owns all tensors for a forward pass, then frees them all at once:
```rust
let arena = GpuArena::new(device, capacity);
let x = arena.alloc_tensor(&[batch, features])?;
// ... use x ...
drop(arena); // frees everything
```
This is simpler than lifetime tracking, avoids fragmentation, and fits the "forward pass = one scope" pattern naturally. It also naturally supports autograd (the backward arena holds saved tensors until backward completes).

**Recommendation**: Evaluate arena-based memory management as an alternative to GpuRef. It solves the same "who owns this memory?" question with less type-system complexity.

### 4.3 Instead of property-based testing: differential testing against PyTorch

Rather than random input generation, run the same inputs through PyTorch and async_gpu, then compare outputs. This catches real numerical divergence, not hypothetical edge cases. A simple Python script can generate golden files:
```python
import torch
x = torch.randn(2, 3, 224, 224)
conv = torch.nn.Conv2d(3, 16, 3, padding=1)
y = conv(x)
torch.save({'input': x, 'weight': conv.weight, 'bias': conv.bias, 'output': y}, 'conv2d_golden.pt')
```

**Recommendation**: Golden files from PyTorch are cheaper to create, easier to maintain, and more trustworthy than property-based tests. Use them for ops validation. Reserve property testing for tape correctness (CPU-only, fast).

---

## 5. What is Missing from the Roadmap

### 5.1 No documentation plan

Autograd introduces fundamentally new APIs (tape, backward, optimizers). Users need:
- A tutorial showing forward → loss → backward → step
- API docs explaining what gets saved during forward
- Error messages explaining common mistakes (e.g., "cannot backward through a non-differentiable op")

**Without documentation, autograd is unusable.** Budget 2-3 documentation tasks within the autograd epic.

### 5.2 No error message quality plan

Current `NnError` has 4 variants. Autograd would need:
- `GraphError::NoGradient { op_name }` — op does not support backward
- `GraphError::DetachedTensor` — tensor was not tracked
- `GraphError::BackwardTwice` — tried to backward through a consumed tape
- `GraphError::ShapeMismatch` in backward (different from forward mismatch)

Bad error messages in autograd are *the* number one complaint in every ML framework. Plan for it.

### 5.3 No gradient checkpointing plan

As noted in 3.4, VRAM pressure makes gradient checkpointing a requirement, not an optimization. The roadmap does not mention it. This should be a theme within autograd, not an afterthought.

### 5.4 No mixed-precision training plan

The project already has f16 MMA kernels (tensor-core-gemm epic). Training in mixed precision (f16 forward, f32 gradients, loss scaling) is standard practice. If autograd does not consider this from the start, retrofitting it will be painful.

**Recommendation**: At minimum, design the autograd tape to be precision-aware (track dtype per tensor) even if mixed-precision training is deferred.

### 5.5 No optimizer epic

Autograd without optimizers is useless. SGD, Adam, and AdamW are each 1-2 tasks, but they need:
- Parameter groups (different learning rates for different layers)
- State (Adam momentum/variance stored on GPU)
- Gradient clipping
- Learning rate scheduling

This is another 8-10 tasks not accounted for in the roadmap.

---

## 6. Summary Verdict

| Epic | Feasibility | Risk | Recommendation |
|------|------------|------|----------------|
| verify-examples | High | Low | Proceed as planned — essential prerequisite |
| testing-infra | Medium | Medium | Downscope: golden-file tests only, skip property testing |
| autograd | Low-Medium | **High** | Scope to Tier 1+2 ops only, manual backward (no tape), add optimizer + checkpointing themes |
| gpu-safety | Low | Low (because unnecessary) | Spike investigation first (2 tasks), likely reframe as zero-copy views |

**Biggest risk**: Autograd scope explosion. The proposer should present a task breakdown proving it is achievable in <20 tasks. If the breakdown exceeds 25 tasks, split into autograd-v1 (linear models) and autograd-v2 (conv + attention).

**Recommended ordering**: verify-examples → gpu-safety spike (parallel) → testing-infra (golden files) → autograd-v1 (linear backward + SGD/Adam) → autograd-v2 (conv/attention backward + checkpointing)
