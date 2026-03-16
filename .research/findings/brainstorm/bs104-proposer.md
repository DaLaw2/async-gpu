# BS-104 Proposer: ONNX Loader, Larger Training, GPU Memory Pool

Three directions analyzed against the current codebase state (115 PTX kernels, 52 nn tests,
GPT-2 164ms/tok nn API, MNIST 91.2% training, CIFAR-10 CNN training).

---

## 1. ONNX / SafeTensors Generic Model Loader

### Current State

SafeTensors loading is **model-specific and hardcoded**:

- `model.rs` — GPT-2 loader: knows exact tensor names (`h.0.attn.c_attn.weight`), fixed shapes
  (768, 2304), Conv1D→Linear transposition logic.
- `model_yolo.rs` — YOLO loader: knows layer indices (`model.0.conv.weight`), ConvBnSilu
  weight groupings, detect head structure.
- `nn/models/gpt2.rs::from_weights()` — manually wires GPT-2 weights into Embedding, Linear,
  LayerNorm, TransformerBlock layers with hardcoded dimension extraction.

There is NO abstraction between "safetensors file" and "nn layer graph."

### What a Generic Loader Would Look Like

**SafeTensors Generic Loader** (feasible, high value):

```
SafeTensorsFile → TensorMap<String, (Vec<f32>, Vec<usize>)>
                       ↓
              ModelDefinition (declarative)
                       ↓
              nn Module graph (Linear, Conv2d, LayerNorm, ...)
```

A `ModelDefinition` would be a declarative spec:
```rust
struct ModelDef {
    layers: Vec<LayerDef>,  // ordered
}
enum LayerDef {
    Linear { weight: String, bias: Option<String>, in_f: usize, out_f: usize },
    Conv2d { weight: String, bias: Option<String>, c_out: usize, c_in: usize, ... },
    LayerNorm { weight: String, bias: String, eps: f32 },
    Sequential(Vec<LayerDef>),
    // ...
}
```

The loader reads safetensors, matches tensor names to LayerDef weight keys, handles
transpositions (configurable per-layer), and constructs the nn Module graph.

**ONNX Loader** (much harder, questionable value):

ONNX defines a computation graph with ~180 op types. Current nn ops cover:

| ONNX Op | async-gpu equivalent | Status |
|---------|---------------------|--------|
| MatMul | ops::matmul (gemm_f32) | OK |
| Gemm | ops::matmul + bias_add | OK |
| Conv | ops::conv2d (im2col+GEMM) | OK |
| Relu | ops::relu | OK |
| Sigmoid | ops::sigmoid | OK |
| LayerNormalization | ops::layer_norm | OK |
| Gelu | ops::gelu | OK |
| Softmax | CPU in loss.rs | Partial |
| Add | ops::elementwise_add | OK |
| Reshape | tensor.reshape() | OK |
| Transpose | tensor.transpose() | OK |
| MaxPool | ops::max_pool2d | OK |
| BatchNormalization | ops::batch_norm | OK |
| Concat | ops::concat_channels | Partial (CHW only) |
| Gather (embedding) | ops::embedding_lookup | Partial |
| Split | Not implemented | MISSING |
| Slice | Not implemented | MISSING |
| Resize (upsample) | ops::upsample_nearest_2x | Fixed 2x only |
| ReduceMean | Not implemented | MISSING |
| Mul (elementwise) | Not implemented | MISSING |
| Div (elementwise) | Not implemented | MISSING |
| Pow | Not implemented | MISSING |
| Sqrt | Not implemented | MISSING |
| Attention (fused) | ops::multi_head_flash_attention | Custom |

**Missing for basic ONNX coverage**: ~10-15 ops (Split, Slice, ReduceMean, elementwise
Mul/Div/Pow/Sqrt, Clip, Flatten, Squeeze/Unsqueeze, Cast, Pad).

### Dependency Analysis

- **safetensors crate** (v0.7): Already a dependency. Generic loader is free.
- **ONNX**: Would need `prost` (protobuf) to parse `.onnx` files. The `onnx-rs` crate
  exists but is poorly maintained. Better to use `prost` directly with the ONNX proto schema.
  Adds ~2 new deps. Alternatively, parse ONNX via Python export to a simpler JSON format.

### Concrete Proposal

**Build a generic SafeTensors loader first (skip ONNX for now).**

Rationale: ONNX op coverage explosion is a real risk. We would need to implement 10-15 new
ops to handle even simple models, each needing PTX kernels, tests, and autograd integration.
That is 3-4 weeks of work for uncertain payoff. SafeTensors is already our format and covers
all practical use cases (HuggingFace models all ship as safetensors).

**Tasks:**

1. `loader.1` — Define `ModelDef` schema and `TensorMap` abstraction (2h)
2. `loader.2` — Implement `load_safetensors_generic()` that reads any .safetensors file
   into `TensorMap` with dtype conversion (f16→f32, bf16→f32) (3h)
3. `loader.3` — Implement `build_model()` that takes `ModelDef` + `TensorMap` → `Vec<Box<dyn Module>>` (4h)
4. `loader.4` — Refactor GPT-2 loading to use generic loader (validate correctness) (2h)
5. `loader.5` — Refactor YOLO loading to use generic loader (2h)
6. `loader.6` — Add ResNet-18 as third model using generic loader (demonstrates generality) (3h)

**Total: ~16h (1 theme, 6 tasks)**

**Models unlocked**: Any HuggingFace model that uses only our supported layers — ResNet,
VGG, BERT (with some work), DistilBERT, MobileNet, EfficientNet (partial).

### Risk Mitigation for ONNX

If ONNX is needed later, the path is:
1. Add `prost` + ONNX proto (1 task)
2. Map ONNX graph to `ModelDef` (1 task)
3. Implement missing ops one-at-a-time as specific models demand them

This defers the op coverage explosion until a concrete model needs it.

---

## 2. Larger Training Examples

### Current Autograd Capabilities

Backward pass supports:
- **Full**: Matmul, ElemAdd, BiasAdd, Gelu, Silu, Sigmoid, Relu, LayerNorm, Attention
- **Partial (dInput only, no dWeight)**: Conv2d, BatchNorm, MaxPool2d, UpsampleNearest
- **Not implemented**: CrossEntropy backward (gradient not flowing), Embedding backward

Critical limitation: **Conv2d backward computes dInput but NOT dWeight**. This means CNN
training cannot update convolutional filters — only fully-connected layers get updated.
The `conv2d_backward_cpu()` in backward.rs explicitly says "weight gradient not yet needed."

Second limitation: **Optimizers (SGD, Adam) do CPU roundtrips for every parameter update**.
`optim.rs` does `to_host()` → update on CPU → `from_host()` for each parameter tensor.
For large models this is a massive bottleneck.

Third limitation: **Loss backward uses passthrough** for MseLoss and doesn't flow for
CrossEntropy. The `cross_entropy_backward()` function exists but isn't wired into the
tape-based backward system.

### Candidate Analysis

**A) ResNet-18 on CIFAR-100**
- Layers: Conv2d (20 conv layers), BatchNorm, ReLU, MaxPool, Linear (1 FC), skip connections
- Problem: Conv2d backward doesn't compute dWeight → cannot train conv layers
- Feasibility: **BLOCKED** until Conv2d weight gradient is implemented
- Matrix sizes: Conv filters are small (3x3, 64-512 channels), FC layer is 512×100
- GPU advantage: Moderate (conv is im2col+GEMM, larger batch sizes help)

**B) GPT-2 Fine-tuning (LoRA)**
- Layers: Linear (all have backward), LayerNorm (has backward), Attention (has backward)
- LoRA only trains low-rank A,B matrices injected into Linear layers — all Matmul backward
- Feasibility: **FEASIBLE** with current autograd (all ops have backward)
- Matrix sizes: 768×768 and 768×3072 — GEMM-heavy, good GPU utilization
- Problem: Need to load pretrained GPT-2 weights + add LoRA adapters + text data pipeline
- Data: WikiText-2 or similar (tokenizer already available via tiktoken-rs)
- GPU advantage: **HIGH** — dominated by large matmuls where GPU excels

**C) Simple Language Model from Scratch**
- Layers: Embedding, Linear, LayerNorm, Attention
- Problem: Embedding backward not implemented
- Feasibility: **PARTIAL** — need embedding backward (gradient scatter)
- Matrix sizes: Comparable to GPT-2 LoRA but from random init
- GPU advantage: HIGH (same GEMM-heavy workload)

**D) MLP on Larger Dataset (Fashion-MNIST, SVHN, tabular)**
- Layers: Linear + activations only
- Feasibility: **FEASIBLE** right now
- Problem: Not very impressive — "just a bigger MNIST"
- GPU advantage: Moderate (larger hidden dims help, but still just GEMMs)

### Data Pipeline

Current approach loads everything into host memory, then uploads per-batch:
```rust
// Pseudo-code from test_utils.rs training tests
for epoch in 0..n_epochs {
    for batch in data.chunks(batch_size) {
        let input = GpuTensor::from_host(batch, shape, &dev)?;
        // forward, backward, update
    }
}
```

Each `from_host()` does `htod_sync_copy` which calls `cuMemcpyHtoD`. For large datasets,
this is fine if data fits in RAM. For training, the bottleneck is:
1. `to_host()` in optimizer (GPU→CPU for each param each step) — **this is the real problem**
2. Batch upload (CPU→GPU, amortized across many forward ops)

### Concrete Proposal

**Build GPT-2 LoRA fine-tuning as the flagship training example.**

Rationale:
- All required ops have backward (Matmul, BiasAdd, LayerNorm, Attention, activations)
- Large matrices showcase GPU advantage over CPU
- Demonstrates practical transfer learning use case
- Tokenizer already available (tiktoken-rs)
- Pre-trained weights already loadable
- LoRA reduces trainable params from 124M to ~0.3M (rank-8 on Q,V projections)

**Tasks:**

1. `lora.1` — Implement LoRA adapter struct (A: [d, r], B: [r, d], alpha) (3h)
2. `lora.2` — Wire LoRA into GPT-2 Linear layers (inject into forward path) (3h)
3. `lora.3` — Move SGD/Adam parameter updates to GPU (eliminate CPU roundtrip) (4h)
   - New PTX kernels: `sgd_update`, `adam_update` (elementwise)
4. `lora.4` — Text data pipeline: tokenize WikiText-2, batch iterator (3h)
5. `lora.5` — End-to-end training loop with loss logging (3h)
6. `lora.6` — Benchmark: GPU vs CPU training time per epoch, loss convergence (2h)

**Total: ~18h (1 theme, 6 tasks)**

**Secondary example (much simpler, could come first):**

`mlp-large.1` — Larger MLP: 3-layer [2048, 1024, 512, 10] on Fashion-MNIST (4h)
- Demonstrates GPU speedup scaling with hidden dimension
- No new infrastructure needed
- Quick win to show before LoRA

### What Infrastructure is Needed

1. **GPU-side optimizer kernels** — Critical. Current CPU roundtrip for every param update
   is the #1 bottleneck for any serious training. Without this, even LoRA training will be
   slower on GPU than CPU for small rank.
2. **Proper CrossEntropy backward in tape** — Need to wire `cross_entropy_backward()` into
   the tape-based backward system so gradients flow through loss.
3. **LoRA module** — New layer type that wraps Linear with low-rank adaptation.

---

## 3. GPU Memory Pool

### Current Problem

Every tensor allocation goes through cudarc's `CudaDevice`:
- `device.alloc_zeros::<f32>(numel)` — calls `cuMemAlloc` + `cuMemsetD8`
- `device.htod_sync_copy(data)` — calls `cuMemAlloc` + `cuMemcpyHtoD`
- Drop of `CudaSlice` — calls `cuMemFree`

In a single forward pass of GPT-2 (12 layers), counting allocations in ops/:
- `gemm.rs`: 7 alloc calls (output buffer, pad buffers, status) per matmul
- `conv.rs`: 9 alloc calls per conv2d (im2col buffer, output, padding)
- `activation.rs`: 2 alloc calls per activation (output + status)
- `norm.rs`: 3 alloc calls per layer_norm
- `attention.rs`: 3 alloc calls per attention op
- `reshape.rs`: 4 alloc calls per reshape/bias_add

**Rough estimate for GPT-2 forward (12 layers):**
- Per layer: ~2 matmul + 2 layernorm + 1 attention + 2 activation + 3 bias_add ≈ 30 allocs
- 12 layers + embedding + lm_head ≈ **370+ cuMemAlloc/cuMemFree pairs per token**

`cuMemAlloc` overhead is typically 1-10μs per call. At 370 calls: ~0.4-3.7ms overhead.
For GPT-2 at 164ms/tok, this is 0.2-2.3% — **measurable but not dominant**.

For training (forward + backward + optimizer), the count roughly triples: ~1000+ alloc/free
pairs per step. At that point, ~1-10ms per step is worth optimizing.

### Design Options

**A) Slab Allocator (size-class buckets)**

```
Pool {
    buckets: HashMap<usize, Vec<CudaSlice<f32>>>,  // size → free list
}
alloc(n) → round up to next power-of-2, check bucket, fallback to cuMemAlloc
free(slice) → return to bucket instead of cuMemFree
```

- Pros: Simple, good cache hit rate for repeated tensor sizes (common in NN)
- Cons: Internal fragmentation (power-of-2 rounding), memory not returned to system
- Integration: Replace `device.alloc_zeros()` calls with `pool.alloc(n)`

**B) Arena Allocator (per-batch reset)**

```
Arena {
    buffer: CudaSlice<u8>,  // one big allocation
    offset: usize,          // bump pointer
}
alloc(n) → bump offset, return sub-slice
reset() → offset = 0 (between batches)
```

- Pros: Zero fragmentation, zero alloc overhead (just pointer bump)
- Cons: Cannot free individual tensors (must reset whole arena). Does NOT work with
  `Arc<CudaSlice>` tensor views — cudarc CudaSlice owns its allocation and frees on drop.
- Integration: **INCOMPATIBLE with current design**. Would need to replace CudaSlice with
  a custom type that references into the arena. This is a massive refactor.

**C) Buddy Allocator**

- Pros: Low fragmentation, O(log n) alloc/free
- Cons: Complex implementation, overkill for NN workloads where tensor sizes are repetitive

### Integration with Arc<CudaSlice> Tensor Views

This is the key challenge. `GpuTensor.data` is `Arc<CudaSlice<f32>>`. When Arc refcount
drops to 0, `CudaSlice::drop()` calls `cuMemFree`. A memory pool needs to intercept this.

Options:
1. **Wrapper type**: `PooledSlice` that returns to pool on drop instead of freeing.
   Requires changing `GpuTensor.data` from `Arc<CudaSlice<f32>>` to `Arc<PooledSlice>`.
2. **cudarc custom allocator**: cudarc 0.12 does NOT support custom allocators. The
   `CudaDevice` struct directly calls `cuMemAlloc`/`cuMemFree`. No hook point.
3. **Pool above cudarc**: Don't intercept drop. Instead, provide `pool.get(size)` that
   returns a pre-allocated `CudaSlice` and `pool.return(slice)` that stores it. The caller
   must explicitly return slices instead of dropping them. This is error-prone but doesn't
   require changing GpuTensor.

### cudarc API Constraints

cudarc 0.12.1 (`CudaDevice`) API:
- `alloc_zeros<T>(len)` → `CudaSlice<T>` — allocates via `cuMemAlloc`
- `htod_sync_copy(&[T])` → `CudaSlice<T>` — allocates + copies
- No custom allocator support
- `CudaSlice` implements `Drop` which calls `cuMemFree`

To use raw CUDA: could use `cudarc::driver::sys::cuMemAlloc_v2` directly, but then we lose
cudarc's safety guarantees and device tracking.

### Concrete Proposal

**Build a slab allocator with explicit return (Option 3), measure impact.**

This is the simplest approach that gives measurable results without touching GpuTensor internals:

```rust
struct GpuMemPool {
    buckets: HashMap<usize, Vec<CudaSlice<f32>>>,
    device: Arc<CudaDevice>,
}
impl GpuMemPool {
    fn alloc(&mut self, n: usize) -> CudaSlice<f32> { ... }
    fn recycle(&mut self, slice: CudaSlice<f32>) { ... }
}
```

Change ops to accept `&mut GpuMemPool` and use `pool.alloc()` instead of `device.alloc_zeros()`.
After use, explicitly `pool.recycle()` temporary buffers (padding, im2col scratch, status words).

**This does NOT change GpuTensor** — only temporary/scratch allocations in ops get pooled.
Output tensors still use normal allocation (they live beyond the op).

**Tasks:**

1. `mempool.1` — Implement `GpuMemPool` with slab buckets + power-of-2 rounding (3h)
2. `mempool.2` — Benchmark: measure cuMemAlloc overhead in GPT-2 forward (instrument with timers) (2h)
3. `mempool.3` — Integrate pool into `ops::gemm` (highest alloc count) (2h)
4. `mempool.4` — Integrate pool into `ops::conv` and `ops::norm` (2h)
5. `mempool.5` — Before/after benchmark: GPT-2 inference latency with pool (2h)
6. `mempool.6` — Training benchmark: MNIST training loop with/without pool (2h)

**Total: ~13h (1 theme, 6 tasks)**

**Benchmark workload**: GPT-2 inference is ideal because it has the most ops per forward
pass (12 transformer layers). The 370+ alloc/free pairs make the overhead measurable.
Training would benefit more in absolute terms (3x more allocs per step).

---

## 4. Priority Recommendation

### Scoring: Impact × Feasibility / Effort

| Direction | Impact | Feasibility | Effort | Score | Notes |
|-----------|--------|-------------|--------|-------|-------|
| SafeTensors generic loader | 7/10 | 9/10 | 16h | 3.9 | Unlocks many models, low risk |
| GPT-2 LoRA training | 9/10 | 7/10 | 18h | 3.5 | High showcase value, needs optimizer fix |
| GPU Memory Pool | 5/10 | 8/10 | 13h | 3.1 | Measurable but not dramatic improvement |

### Recommended Order

**1st: GPU Memory Pool (mempool)** — Start here despite lower impact score because:
- Shortest effort (13h)
- Provides infrastructure that benefits ALL subsequent work
- The benchmark task (mempool.2) gives us hard numbers on actual allocation overhead
- If overhead is <1%, we can deprioritize and move on quickly
- If overhead is >3%, the pool pays for itself in every future benchmark

**2nd: SafeTensors Generic Loader (loader)** — Build this next because:
- Unlocks ResNet-18 and other models for the training showcase
- Reduces code duplication (GPT-2 + YOLO loaders share no code)
- Required foundation for loading LoRA base weights cleanly

**3rd: GPT-2 LoRA Training (lora)** — Build last because:
- Highest impact but needs optimizer GPU kernels (lora.3) as prerequisite
- Benefits from memory pool (training = 3x more allocs)
- Benefits from generic loader (cleaner weight loading)

### Parallelization Opportunities

- `mempool.1-2` (pool + benchmark) can run in parallel with `loader.1-2` (schema + generic load)
- `lora.1-2` (LoRA struct + wiring) can start once loader is partially done
- `mempool.3-5` (integration) should complete before `lora.5-6` (training benchmarks)

### Dependencies

```
mempool.2 (benchmark) → decides if mempool.3-6 are worth doing
loader.4 (GPT-2 refactor) → enables lora.2 (LoRA into GPT-2)
lora.3 (GPU optimizer) → required before lora.5-6 (training loop + benchmark)
mempool.5 (pool done) → improves lora.6 (training benchmark numbers)
```

### Quick Win First

Before any of the three themes, do one small task:

`mlp-large.0` — Build a 4-layer MLP [4096, 2048, 1024, 10] training demo on synthetic data.
All ops have backward, no new infrastructure needed, takes ~2h. This immediately demonstrates
GPU training speedup scaling with matrix size and sets a baseline for future comparisons.
