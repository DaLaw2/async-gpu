# BS-104 Skeptic: ONNX Loader, Larger Training, GPU Memory Pool

Challenging the proposer's analysis on all three directions.

---

## 1. Challenges to SafeTensors Generic Loader

### Who is the user?

The proposer assumes someone wants to "load any HuggingFace model." But async-gpu
is a research project demonstrating async/await on GPU with a patched Rust std. The
target audience is systems programmers curious about GPU programming in Rust, not ML
engineers who already have PyTorch/ONNX Runtime.

**Question the proposer didn't answer:** Has a single person ever asked "how do I
load my custom model into async-gpu"? The existing GPT-2 and YOLO loaders work. They
are hardcoded, yes, but they run. A "generic loader" is a solution in search of a
problem.

### The ModelDef abstraction is just moving the hardcoding

The proposer's `ModelDef` schema:
```rust
enum LayerDef {
    Linear { weight: String, bias: Option<String>, in_f: usize, out_f: usize },
    Conv2d { weight: String, bias: Option<String>, ... },
    ...
}
```

This is not "generic" — it is the same hardcoded knowledge (tensor names, shapes,
transposition rules) wrapped in a struct. The user still needs to know the exact
tensor names from the safetensors file, the exact shapes, and whether transposition is
needed. The only thing saved is ~50 lines of boilerplate per model.

For GPT-2, the user would write something like:
```rust
ModelDef { layers: vec![
    LayerDef::Linear { weight: "h.0.attn.c_attn.weight".into(), ... },
    LayerDef::Linear { weight: "h.0.attn.c_proj.weight".into(), ... },
    // 50 more lines of this
]}
```

Compared to the current approach of just writing Rust code that loads weights directly,
the ModelDef adds indirection without real simplification. The user trades "write Rust
code" for "write a ModelDef spec" — same effort, worse debugging.

### 16 hours for marginal code dedup

The proposer estimates 16h (6 tasks). The payoff is deduplicating GPT-2 and YOLO
loaders that share no code because they have fundamentally different architectures.
GPT-2 uses Conv1D-style transposed weights; YOLO uses fused ConvBnSilu blocks with
batch norm folding. A "generic" loader that handles both will inevitably grow
model-specific branches, defeating the purpose.

### The "models unlocked" list is aspirational

"ResNet, VGG, BERT, DistilBERT, MobileNet, EfficientNet (partial)" — but:
- ResNet/VGG: need working Conv2d backward to train them (currently BLOCKED)
- BERT/DistilBERT: need tokenizer integration, segment embeddings, masked LM head
- MobileNet: needs depthwise separable conv (not implemented)
- EfficientNet: needs SE blocks, swish activation, compound scaling

The loader doesn't "unlock" these models. The missing ops and training infrastructure
does. The loader is the easy part; the hard part is everything else.

### Verdict: DEFER

The generic loader is a nice-to-have but doesn't solve the actual bottleneck. The
hardcoded loaders work. Spend the 16h on something that changes what the system can do,
not on refactoring how it loads weights.

---

## 2. Challenges to GPT-2 LoRA Training

### The optimizer CPU roundtrip is the elephant in the room

The proposer correctly identifies that SGD/Adam do `to_host()` -> CPU update ->
`from_host()` for EVERY parameter, EVERY step. But then buries this as subtask
`lora.3` (4h estimate) and moves on to the exciting LoRA stuff.

Let's be honest: without GPU-side optimizer kernels, any training benchmark will show
the GPU being *slower* than CPU for small models, because every step requires:
1. Forward pass (GPU) - fast
2. Backward pass (GPU) - fast
3. Optimizer step (GPU->CPU->GPU per param) - catastrophically slow

For LoRA with rank-8 on Q,V projections across 12 layers, that's 48 parameter tensors
(24 A matrices + 24 B matrices). Each one does a roundtrip. At ~0.5ms per roundtrip
(alloc + memcpy both ways), that's 24ms of pure overhead per step — potentially more
than the actual compute for small LoRA ranks.

**The proposer should have made GPU optimizer kernels the FIRST task, not the third.**

### LoRA implementation complexity is underestimated

LoRA is not "just inject A*B into Linear":
1. Need to freeze base weights (no gradient computation for them)
2. Need to scale the LoRA output by alpha/rank
3. Need to handle LoRA for different projection types (Q, K, V, O, FFN)
4. Need to merge LoRA weights for inference after training
5. Need to save/load LoRA adapter weights separately

The proposer estimates 3h for the LoRA struct and 3h for wiring — 6h total. Real LoRA
implementations (e.g., HuggingFace PEFT) are ~2000 lines. Even a minimal version is
more like 500 lines with proper gradient handling.

### CrossEntropy backward is not wired into the tape

The proposer mentions this in passing under "What Infrastructure is Needed" but doesn't
create a task for it. Without gradient flow through the loss function, the entire
training loop produces zero gradients. This is not a nice-to-have; it is a
**showstopper**.

Looking at the actual backward.rs code, `cross_entropy_backward()` exists as a function
but is not registered in the tape-based autograd system. This means `loss.backward()`
won't produce any gradients for any layer. The proposer's task list assumes this works.

### Text data pipeline is hand-waved

`lora.4` — "Text data pipeline: tokenize WikiText-2, batch iterator (3h)"

This requires:
1. Downloading WikiText-2 (or shipping it with the repo — 11MB)
2. Tokenizing with tiktoken (is tiktoken-rs actually a dependency? Let me check...)
3. Building a batched sequence iterator with padding/truncation
4. Handling variable-length sequences (or fixed-length chunking)

For someone who hasn't built a text pipeline in Rust before, 3h is aggressive. More
realistic: 6-8h including debugging tokenizer edge cases.

### The 18h estimate is probably 30-40h

Summing up the underestimates:
- LoRA struct + wiring: 6h -> 10h (complexity underestimated)
- GPU optimizer: 4h -> 6h (need two kernels + integration + testing)
- Text pipeline: 3h -> 7h (tokenizer + batching + debugging)
- CrossEntropy backward wiring: 0h -> 3h (not even in the task list!)
- End-to-end integration + debugging: 3h -> 8h (always takes 2-3x longer)
- Benchmarking: 2h -> 3h (needs careful methodology)

Revised total: ~37h — more than double the estimate.

### Verdict: PROCEED, but restructure

LoRA training IS the highest-impact demo. But the task ordering is wrong:
1. **First**: Wire CrossEntropy backward into the tape (showstopper)
2. **Second**: GPU optimizer kernels (showstopper for performance)
3. **Third**: LoRA struct + integration
4. **Fourth**: Data pipeline + training loop
5. **Last**: Benchmarking

And be honest about the timeline: this is 4-5 weeks, not 2-3 days.

---

## 3. Challenges to GPU Memory Pool

### The proposer's own numbers kill the proposal

Let's quote the proposer directly:

> "cuMemAlloc overhead is typically 1-10us per call. At 370 calls: ~0.4-3.7ms overhead.
> For GPT-2 at 164ms/tok, this is 0.2-2.3% — measurable but not dominant."

**0.2-2.3% improvement.** The proposer then spends 13h of estimated effort to capture
this. That is a terrible ROI.

For comparison, the gs-spike.2 analysis found that `weight.reshape()` in conv2d
**copies the entire weight tensor** every forward call. The reshape/transpose overhead
in GPT-2 is 72 unnecessary copies per forward pass (6 reshapes x 12 layers). Each
reshape copies up to 3MB. That's ~216MB of unnecessary memcpy per forward pass.

At ~12GB/s for device-to-device copy, 216MB takes ~18ms — **11% of the 164ms total**.
Fixing zero-copy reshape would give 5-50x more speedup than a memory pool.

### The explicit-return API is a footgun

The proposer's Option 3 design requires callers to manually `pool.recycle()` temporary
buffers. If any code path forgets to recycle (error returns, early exits, panics), the
buffer is dropped normally via cuMemFree — silently degrading to non-pooled behavior.

Worse, every op function signature changes to accept `&mut GpuMemPool`. This threads
pool state through the entire ops/ module. If a function calls another function that
allocates (e.g., matmul calls a padding helper), the pool must be passed through
every layer. This is a pervasive API change.

### Arc<CudaSlice> makes pooling fundamentally awkward

The proposer acknowledges this but then proposes to only pool "scratch" allocations,
not output tensors. This means:
- Scratch buffers in gemm (padding, status): pooled (saves maybe 5 allocs per matmul)
- The actual output tensor: NOT pooled (this is the biggest allocation!)

The vast majority of allocation bytes are output tensors, not scratch space. Pooling
only scratch is like optimizing the wrong 20%.

### cudarc doesn't support custom allocators

The proposer confirms: "cudarc 0.12 does NOT support custom allocators." This means
we cannot intercept `CudaSlice::drop()`. Any pooling must work *around* cudarc, not
*with* it. This is fighting the framework, which always produces fragile code.

### Measure before building

The proposer orders the tasks as: build pool (3h) -> benchmark (2h). This is
backwards. **Benchmark first** (2h) to determine if the problem exists. If cuMemAlloc
overhead is <1ms (which the proposer's own estimate suggests), skip the pool entirely.

### Verdict: SKIP (measure only)

Do `mempool.2` (benchmark measurement) as a standalone task. If allocation overhead
is >5ms per GPT-2 forward pass, reconsider. Based on the proposer's own analysis,
it will be <2ms. The 13h would be far better spent on zero-copy reshape, which the
gs-spike.2 analysis identified as an 11% overhead.

---

## 4. What the Proposer is Missing

### The 2.4x gap (164ms vs 68ms) is NOT allocation overhead

The proposer's memory pool section tries to explain part of this gap, but 0.2-2.3%
doesn't account for it. The real culprits, identified in previous research:

1. **Weight reshaping copies** (gs-spike.2): 72 unnecessary memcpys per forward pass.
   Each `tensor.reshape()` allocates a new buffer and copies. ~18ms (11% of 164ms).

2. **Per-op kernel launch overhead**: The nn API launches separate kernels for each
   operation. The raw kernel path fuses operations. The nn API does ~60+ kernel
   launches per forward pass vs ~12 for raw kernels.

3. **No weight pre-padding**: GEMM requires 16-byte alignment. The nn API pads weights
   on every forward call. The raw kernel path pads once at load time.

4. **Transpose via CPU roundtrip**: `tensor.transpose()` goes GPU->CPU->CPU transpose
   ->GPU. This is insane for a GPU compute library.

Fixing items 1, 3, and 4 would likely close most of the 2.4x gap. None of the three
proposed directions address these.

### The project's unique value is being neglected

async-gpu's differentiator is:
- Async/await on GPU with patched Rust std
- Real `println!`, `File::open`, `Result<T,E>` on GPU
- Cooperative warp scheduling for I/O

None of the three proposals leverage this. ONNX loading, LoRA training, and memory
pools are things that cuDNN, PyTorch, and every other GPU framework already do better.
Building these is playing catch-up on ground where we'll always lose.

**What WOULD leverage the unique value?**
- A demo where GPU kernels do async I/O during training (e.g., streaming data from
  disk using GPU-side `File::read`)
- A demo where error handling on GPU produces meaningful Result types that propagate
  to the host
- Cooperative multi-kernel scheduling where warps yield during memory-bound phases

### The "quick win" MLP suggestion is the right instinct, wrong scope

The proposer suggests a 4-layer MLP [4096, 2048, 1024, 10] as a quick win. This is
reasonable but doesn't need to be a separate task — it's a parameter change to the
existing MNIST training test. Change hidden dims from [128, 64] to [4096, 2048, 1024]
and you have it. This is 15 minutes, not 2 hours.

---

## 5. My Recommendations

### For each direction:

| Direction | Verdict | Rationale |
|-----------|---------|-----------|
| SafeTensors Generic Loader | **DEFER** | Code dedup, not capability expansion. The hardcoded loaders work. |
| GPT-2 LoRA Training | **PROCEED** (restructured) | Highest showcase value, but fix showstoppers first (CrossEntropy backward, GPU optimizer) and double the time estimate. |
| GPU Memory Pool | **SKIP** (measure only) | Proposer's own numbers show <2.3% impact. Spend 2h benchmarking to confirm, then move on. |

### What I would prioritize instead:

**Priority 1: Close the nn API performance gap (the 2.4x problem)**
- Zero-copy reshape (tensor views already designed in gs-spike.2)
- Weight pre-padding at load time (one-time cost, not per-forward)
- GPU-native transpose (eliminate CPU roundtrip)
- Estimated impact: 30-50% latency reduction for nn API
- Estimated effort: 15-20h

**Priority 2: GPU optimizer kernels (standalone, not buried in LoRA)**
- `sgd_update` and `adam_update` PTX kernels (elementwise, simple)
- Eliminate the to_host/from_host roundtrip in optim.rs
- This unblocks ALL training workloads, not just LoRA
- Estimated effort: 6-8h

**Priority 3: Wire CrossEntropy backward into the tape**
- Without this, no classification training works properly
- Estimated effort: 3-4h

**Priority 4: THEN do LoRA training**
- With priorities 1-3 done, LoRA training becomes straightforward and will actually
  show good numbers
- Estimated effort: 15-20h (realistic)

### The meta-point

The proposer is thinking about "what new features to add." The skeptic says: **make
what you have work well first.** The nn API is 2.4x slower than raw kernels. The
optimizer does CPU roundtrips. CrossEntropy doesn't flow gradients. Fix these
foundational issues before building new capabilities on top of a shaky base.

Building LoRA on top of a broken optimizer and non-flowing gradients will produce a
demo that is slow, buggy, and embarrassing. Fix the foundation, THEN build the showcase.
