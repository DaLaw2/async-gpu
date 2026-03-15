# BS-102 Proposer: Roadmap Deep Analysis

**Direction**: verify-examples → testing-infra → autograd → gpu-safety
**Date**: 2026-03-15
**Context**: 22 completed epics, 462 tasks, nn module with 13 ops / 11 layers / 23 kernels

---

## Active Epics Assessment

### verify-examples (HIGH priority, in progress)

Current state: epic created with 5 themes, 12 tasks. Task `ve-model-paths.1` is the
active task (model path audit). No tasks completed yet.

**Success criteria progress**:
1. All model files from `/models/` — NOT MET. Current code uses `env!("CARGO_MANIFEST_DIR")` with relative `../../../models/` chains.
2. GPT-2 example runs end-to-end — NOT VERIFIED. `nn::models::gpt2::Gpt2Model` exists but no standalone run confirmed.
3. YOLOv8 example runs end-to-end — NOT VERIFIED. `nn::models::yolov8::YoloV8Nano` exists.
4. Existing tests pass — NOT VERIFIED.
5. nn layer unit tests — NOT MET. Only tensor unit tests (8) exist; no layer-level numerical tests.

**Key blocker**: Model files (~500MB GPT-2, ~6MB YOLO) must exist on disk for any
E2E verification. The path centralization is a prerequisite for everything else.

### How the 4 Epics Chain Together

```
verify-examples ──────────────────────────────┐
  (validate what exists, fix paths, add tests) │
                                                ▼
testing-infra ─────────────────────────────────┐
  (golden files, numerical harness, property)  │
  NEEDS: working examples to capture goldens   │
                                                ▼
autograd ──────────────────────────────────────┐
  (tape, backward kernels, optimizers)         │
  NEEDS: testing harness for gradient checks   │
                                                ▼
gpu-safety ────────────────────────────────────┘
  (GpuRef<'a,T>, lifetime tracking)
  NEEDS: autograd tape to motivate lifetimes
```

**Dependencies are strictly sequential for the core path**, but within each epic
there are parallelizable tasks. The critical path is:

1. **verify-examples** unlocks golden file capture for testing-infra
2. **testing-infra** provides the numerical comparison harness needed to validate backward passes
3. **autograd** creates the memory management complexity that motivates gpu-safety
4. **gpu-safety** is the capstone — it can only be designed once we know what lifetime patterns autograd introduces

However, some **cross-cutting work** can start early:
- gpu-safety *design* (trait sketches, RFC) can start during autograd implementation
- testing-infra *framework code* (harness, tolerance utilities) can start during verify-examples

---

## Systems Analysis

### Memory Model for Autograd Tape

The autograd tape records the computational graph so we can traverse it backward.
Each tape entry must capture:

```rust
/// A single operation recorded on the autograd tape.
struct TapeEntry {
    /// Unique ID for this operation (monotonic counter).
    op_id: u64,
    /// Which operation was performed (for dispatching backward).
    op_kind: OpKind,
    /// IDs of input tensors (edges in the graph).
    inputs: SmallVec<[TensorId; 4]>,
    /// ID of the output tensor.
    output: TensorId,
    /// Saved tensors needed for backward (e.g., activations for matmul).
    /// These are REFERENCES into a tensor storage — NOT copies.
    saved_tensors: SmallVec<[TensorId; 4]>,
    /// Metadata: shapes, hyperparams needed for backward (stride, padding, etc.)
    saved_meta: OpMeta,
}

/// The full autograd tape — append-only during forward, consumed during backward.
pub struct Tape {
    entries: Vec<TapeEntry>,
    /// Pool of tensors referenced by the tape. Uses generation-counting
    /// to avoid dangling references.
    tensors: TensorPool,
    /// Whether recording is active.
    recording: bool,
}
```

**Key design decisions**:

1. **Tape entries store TensorIds, not Arc<GpuTensor>**. This avoids reference cycles
   and lets the tape own the lifetime of intermediate tensors. The TensorPool is a
   slab allocator mapping TensorId → GpuTensor.

2. **saved_tensors vs recomputation**: For memory-constrained cases, we can implement
   gradient checkpointing later. For v1, save all intermediates.

3. **OpMeta is an enum**: Each op stores only the metadata it needs for backward.

```rust
enum OpMeta {
    Matmul,  // no extra metadata — shapes from saved tensors
    Conv2d { stride: usize, padding: usize, kh: usize, kw: usize },
    LayerNorm { eps: f32, d_model: usize },
    Attention { n_heads: usize, d_head: usize },
    // Activations: no metadata (element-wise, shape preserved)
    Activation(ActivationKind),
    BatchNorm { eps: f32, c: usize, hw: usize },
    MaxPool2d { kernel_size: usize, stride: usize },
    Linear,  // just matmul + bias_add, decomposed
    EmbeddingLookup { vocab_size: usize },
    ElementwiseAdd,
    BiasAdd,
}
```

4. **Tape lifetime**: Created at the start of a training step, consumed by `.backward()`,
   then dropped. Not persistent across steps. This means the tape has a natural
   scope — it lives on the stack of the training loop.

### GpuRef Design

The goal: prevent use-after-free of GPU memory without runtime overhead.

```rust
/// A borrowed view into GPU memory. Cannot outlive the owning GpuTensor.
/// This is the "read-only reference" type for kernel arguments.
pub struct GpuRef<'a, T: DeviceRepr> {
    ptr: CUdeviceptr,
    len: usize,
    _marker: PhantomData<&'a T>,
}

/// A mutable borrowed view into GPU memory.
pub struct GpuRefMut<'a, T: DeviceRepr> {
    ptr: CUdeviceptr,
    len: usize,
    _marker: PhantomData<&'a mut T>,
}
```

**How to add without breaking existing API**:

The existing API uses `&CudaSlice<f32>` and `&mut CudaSlice<f32>` for kernel arguments.
We can introduce GpuRef as an *additional* API layer that gates access:

```rust
impl GpuTensor {
    /// Borrow the tensor's device memory as a read-only reference.
    pub fn as_ref(&self) -> GpuRef<'_, f32> {
        GpuRef {
            ptr: *self.data.device_ptr(),
            len: self.numel(),
            _marker: PhantomData,
        }
    }

    /// Borrow the tensor's device memory as a mutable reference.
    pub fn as_mut(&mut self) -> GpuRefMut<'_, f32> {
        GpuRefMut {
            ptr: *self.data.device_ptr(),
            len: self.numel(),
            _marker: PhantomData,
        }
    }
}
```

**Migration strategy**: Phase 1 adds GpuRef as opt-in. Phase 2 makes ops accept
`impl Into<GpuRef<'_, f32>>` via a trait. Phase 3 deprecates raw pointer access.

This is **non-breaking** because:
- Existing code continues to use `tensor.data()` → `&CudaSlice<f32>`
- New code can use `tensor.as_ref()` → `GpuRef<'_, f32>`
- Kernel launch wrappers accept both via a trait

**What GpuRef prevents**:
- Holding a device pointer after the tensor is dropped
- Two mutable references to the same device memory (Rust's borrow checker)
- Accidentally passing a freed buffer to a kernel launch

**What GpuRef cannot prevent** (without runtime checks):
- Aliasing through raw `CUdeviceptr` values (unsafe code can bypass)
- Cross-device pointer confusion (device 0 ptr used on device 1)

### Testing Infrastructure Architecture

```
tests/
├── golden/                    # Known-good outputs (committed to git)
│   ├── gpt2_forward_32tok.json    # {"logits_top5": [...], "shape": [...]}
│   ├── yolo_bus_detections.json   # [{"class": "bus", "conf": 0.87, "bbox": [...]}]
│   ├── linear_3x4_ref.json       # {"input": [...], "weight": [...], "output": [...]}
│   └── layernorm_768_ref.json
├── harness/
│   ├── mod.rs                 # Test harness utilities
│   ├── numerical.rs           # Tolerance comparison functions
│   ├── golden.rs              # Golden file load/save/compare
│   └── property.rs            # Property test generators
└── integration/
    ├── test_linear.rs         # Per-layer integration tests
    ├── test_conv2d.rs
    ├── test_attention.rs
    ├── test_gpt2_e2e.rs
    └── test_yolo_e2e.rs
```

**Golden file format**: JSON with these fields:
```json
{
    "op": "layer_norm",
    "input_shape": [4, 768],
    "params": {"eps": 1e-5},
    "input_data": [1.0, 2.0, ...],
    "output_data": [0.123, -0.456, ...],
    "tolerance": {"rtol": 1e-5, "atol": 1e-6}
}
```

**Numerical comparison functions**:
```rust
/// Compare two f32 slices with configurable tolerance.
pub fn assert_close(actual: &[f32], expected: &[f32], rtol: f32, atol: f32) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        let tol = atol + rtol * e.abs();
        let diff = (a - e).abs();
        assert!(
            diff <= tol,
            "element {i}: actual={a}, expected={e}, diff={diff}, tol={tol}"
        );
    }
}

/// Compare with per-element max absolute error reporting.
pub fn max_abs_error(actual: &[f32], expected: &[f32]) -> f32 {
    actual.iter().zip(expected).map(|(a, e)| (a - e).abs()).fold(0.0f32, f32::max)
}

/// Compare with mean squared error.
pub fn mse(actual: &[f32], expected: &[f32]) -> f32 {
    let n = actual.len() as f32;
    actual.iter().zip(expected).map(|(a, e)| (a - e).powi(2)).sum::<f32>() / n
}
```

---

## Compiler Analysis

### Backward Kernels Needed

For full autograd, each forward op needs a corresponding backward kernel:

| Forward Op | Backward Kernel(s) | Complexity | Notes |
|---|---|---|---|
| `matmul(A,B)→C` | `dA = dC * B^T`, `dB = A^T * dC` | **Low** — reuse `gemm_f32` | Just transposed launches |
| `conv2d(X,W)→Y` | `dX = col2im(W^T * dY)`, `dW = dY * col(X)^T` | **Medium** — need `col2im` kernel | im2col exists, col2im is new |
| `layer_norm(X,γ,β)→Y` | `dX = f(dY,X,γ,μ,σ)`, `dγ`, `dβ` | **High** — reductions + chain rule | Fused kernel for efficiency |
| `attention(Q,K,V)→O` | `dQ,dK,dV` from `dO` | **Very High** — flash-attention bwd | Multi-pass algorithm |
| `gelu(X)→Y` | `dX = dY * gelu'(X)` | **Low** — element-wise | gelu'(x) formula known |
| `silu(X)→Y` | `dX = dY * (σ(X) + X*σ(X)*(1-σ(X)))` | **Low** — element-wise | |
| `sigmoid(X)→Y` | `dX = dY * Y * (1-Y)` | **Low** — element-wise | Uses saved output |
| `relu(X)→Y` | `dX = dY * (X > 0)` | **Low** — element-wise | Currently CPU, need GPU |
| `batch_norm(X)→Y` | `dX,dγ,dβ` | **High** — reductions | Similar to layernorm_bwd |
| `max_pool2d(X)→Y` | `dX` via index mask | **Medium** — need argmax saved | Forward must save indices |
| `embedding(ids)→E` | `dE` scatter-add into weight grad | **Medium** — atomic scatter | Needs atomicAdd |
| `elementwise_add` | Identity (pass-through) | **Trivial** | dA=dC, dB=dC |
| `bias_add` | Sum reduction for bias grad | **Low** | |

**New PTX kernels required**: ~8-10 new kernels
- `gelu_backward`, `silu_backward`, `sigmoid_backward`, `relu_forward` (GPU version)
- `col2im` (reverse of im2col)
- `layer_norm_backward` (fused reduction kernel)
- `batch_norm_backward` (fused reduction kernel)
- `embedding_backward` (atomicAdd scatter)
- `max_pool2d_forward_with_indices` (modified forward saving argmax)
- `max_pool2d_backward` (scatter using saved indices)

Flash attention backward is **not needed for v1** — we can decompose attention into
matmul + softmax + matmul and use those individual backward passes. This trades
memory for implementation simplicity.

### Can Inline PTX Handle Backward Pass Complexity?

**Yes, but with caveats.**

The existing forward kernels are all inline PTX in Rust (`core::arch::asm!`). The
backward kernels follow the same patterns:

- **Element-wise backward** (gelu, silu, sigmoid, relu): Identical pattern to
  forward activations. ~20 lines of PTX each. No concern.

- **GEMM backward**: Zero new kernel code. `dA = matmul(dC, B^T)` and `dB = matmul(A^T, dC)`
  just reuse `gemm_f32` with transposed arguments. The transpose is handled by the
  host-side `ops::matmul` padding/layout logic.

- **Reduction backward** (layer_norm, batch_norm): These are the hardest. The forward
  `layer_norm` kernel is 90 lines of PTX. The backward is ~150 lines because it
  requires two reduction passes (dγ, dβ) plus a per-element pass (dX). This is
  within the range of what inline PTX handles — the existing flash attention kernel
  is ~200 lines.

- **col2im**: Reversal of im2col. Same complexity as im2col (~40 lines). Each output
  position atomically accumulates contributions from overlapping patches.

**Conclusion**: Inline PTX is sufficient. The most complex backward kernel
(layer_norm_backward) is comparable to existing flash_attention complexity.

### Impact on Compilation Time

Adding ~10 backward kernels to `compute_transformer.rs` and `compute_cnn.rs` will:
- Increase PTX compilation by ~15-20% (from ~23 to ~33 kernels)
- Add ~200-300 lines of PTX asm blocks
- No impact on host compilation (backward kernels compile to nvptx64 only)

This is manageable. The current build already takes several minutes for the
nvptx64 target; adding 10 more kernels won't meaningfully change this.

---

## GPU Architecture Analysis

### Memory Pressure from Autograd

**Forward pass memory** (GPT-2 Small as reference):
- Parameters: 124M × 4 bytes = 496 MB
- Activations per layer: seq×768 × 4 bytes × ~6 tensors = ~1.1 MB/layer @ seq=32
- Total forward activations (12 layers): ~13 MB @ seq=32, ~416 MB @ seq=1024

**Autograd overhead**:
- Saved activations (for backward): same as forward activations = 1x
- Gradients: same size as parameters + activations = ~1x
- Tape metadata: negligible (~100 bytes per op × ~100 ops = 10KB)

**Total memory for training** (seq=32, GPT-2 Small):
```
Parameters:     496 MB
Saved activations: ~13 MB  (tape)
Gradients:       496 MB   (parameter gradients)
Act. gradients:   ~13 MB  (backprop intermediates, freed as we go)
Optimizer state: 992 MB   (Adam: 2× parameter size for m,v)
─────────────────────────
Total:          ~2.0 GB   (fits in 4GB VRAM easily @ seq=32)
```

For seq=1024 (full GPT-2 context):
```
Parameters:     496 MB
Saved activations: ~416 MB
Gradients:       496 MB
Optimizer state: 992 MB
─────────────────────────
Total:          ~2.4 GB   (still fits in 4GB, tight in 3GB)
```

**Mitigation strategies** (implement later, not v1):
1. Gradient checkpointing: trade compute for memory by recomputing activations
2. Mixed precision: f16 activations + f32 gradients halves activation memory
3. Gradient accumulation: process mini-batches to reduce peak memory

### Backward Pass Parallelism

Within a single backward pass through a transformer layer:

```
Forward:  embed → [LN → MHA → add → LN → FFN → add] × 12 → LN → logits

Backward traverses in reverse:
  dLogits → dLN_final → 12×[dAdd → dFFN → dLN → dAdd → dMHA → dLN] → dEmbed
```

**What can overlap**:
- dγ and dβ computation (two independent reductions) within layer_norm_backward
- dW_q, dW_k, dW_v computations in MHA backward (three independent GEMMs)
- dW and db computation in Linear backward (GEMM + reduction)

**What CANNOT overlap** (data dependencies):
- dX must complete before the next layer backward starts
- Within attention: dV needs dO, dK needs dQ×K result, dQ needs dK×V result

**Practical overlap strategy**: Use CUDA streams for independent GEMM launches.
The existing `streams.rs` module supports this. In v1, keep it simple with
synchronous backward — optimize with streams in v2.

### Safety Checking Overhead

**Compile-time (GpuRef)**: Zero runtime overhead. The lifetime annotations are
erased at compilation. The borrow checker prevents the misuse at compile time.

**Runtime checks we should add**:
1. Shape validation in backward ops: ~10ns per op (negligible vs. kernel time)
2. NaN/Inf detection: Optional, via a `debug_assert!`-style kernel that scans
   tensor for non-finite values. ~1µs for a 768-element tensor.
3. Gradient magnitude monitoring: print warning if gradient norm exceeds threshold.

**None of these are on the critical path.** A GEMM kernel launch takes ~1ms; shape
validation takes ~10ns. The overhead is < 0.001%.

---

## Concrete Recommendations

### Epic 1: verify-examples (already created)

**Themes**: 5 themes, 12 tasks (as in state.toml)
**Estimated completion**: ~15 tasks total (12 existing + 3 potential rework)
**Priority**: Highest — blocks everything else

No changes to existing plan. Focus on:
1. `ve-model-paths` (3 tasks) — centralize paths
2. `ve-run-gpt2` (2 tasks) — run + verify GPT-2
3. `ve-run-yolo` (2 tasks) — run + verify YOLO
4. `ve-existing-tests` (2 tasks) — run + fix test suites
5. `ve-nn-unit-tests` (3+ tasks) — add layer numerical tests

### Epic 2: testing-infra

**Themes and tasks**:

#### Theme: `ti-harness` — Numerical comparison harness
- `ti-harness.1`: Design test harness API (assert_close, max_abs_error, mse)  [investigation]
- `ti-harness.2`: Implement numerical comparison utilities in `nn/test_utils.rs`  [experiment]
- `ti-harness.3`: Add configurable tolerance presets (f32_loose, f32_strict, gradient) [experiment]
- **4 tasks**

#### Theme: `ti-golden` — Golden file infrastructure
- `ti-golden.1`: Design golden file JSON schema (input, output, params, tolerance)  [design]
- `ti-golden.2`: Implement golden file load/save/compare functions  [experiment]
- `ti-golden.3`: Capture golden outputs from verified GPT-2 run  [experiment]
- `ti-golden.4`: Capture golden outputs from verified YOLO run  [experiment]
- `ti-golden.5`: Golden regression test: assert current output matches golden  [experiment]
- **5 tasks**

#### Theme: `ti-property` — Property-based testing
- `ti-property.1`: Investigate proptest/quickcheck for GPU tensor operations  [investigation]
- `ti-property.2`: Property: matmul associativity `(A*B)*C ≈ A*(B*C)`  [experiment]
- `ti-property.3`: Property: layer_norm idempotence (normalize already-normalized input) [experiment]
- `ti-property.4`: Property: conv2d with identity kernel = input  [experiment]
- **4 tasks**

#### Theme: `ti-cpu-ref` — CPU reference implementations
- `ti-cpu-ref.1`: CPU f64 matmul reference for gradient checking  [experiment]
- `ti-cpu-ref.2`: CPU f64 layer_norm reference  [experiment]
- `ti-cpu-ref.3`: CPU f64 conv2d reference (im2col + matmul)  [experiment]
- `ti-cpu-ref.4`: CPU f64 attention reference (softmax + matmul)  [experiment]
- **4 tasks**

**Total tasks**: ~17
**Depends on**: verify-examples (need working examples to capture goldens)

### Epic 3: autograd

**Themes and tasks**:

#### Theme: `ag-tape` — Autograd tape and tensor graph
- `ag-tape.1`: Design TapeEntry, Tape, TensorPool structs  [design]
- `ag-tape.2`: Implement TensorPool (slab allocator with generation counting)  [experiment]
- `ag-tape.3`: Implement Tape with recording/playback  [experiment]
- `ag-tape.4`: Add `requires_grad` flag to GpuTensor  [experiment]
- `ag-tape.5`: Modify forward ops to record on tape when recording=true  [experiment]
- `ag-tape.6`: Implement `backward()` traversal (reverse topological order)  [experiment]
- `ag-tape.7`: Test: simple chain `a → linear → relu → loss → backward`  [experiment]
- **7 tasks**

#### Theme: `ag-elemwise-bwd` — Element-wise backward kernels
- `ag-elemwise-bwd.1`: `gelu_backward` PTX kernel + host wrapper  [experiment]
- `ag-elemwise-bwd.2`: `silu_backward` PTX kernel + host wrapper  [experiment]
- `ag-elemwise-bwd.3`: `sigmoid_backward` PTX kernel + host wrapper  [experiment]
- `ag-elemwise-bwd.4`: `relu_forward` GPU kernel (replace CPU fallback) + `relu_backward`  [experiment]
- `ag-elemwise-bwd.5`: Numerical gradient check for all activations (finite differences)  [experiment]
- **5 tasks**

#### Theme: `ag-gemm-bwd` — GEMM/Linear backward
- `ag-gemm-bwd.1`: `matmul_backward`: dA = dC * B^T, dB = A^T * dC (reuse gemm_f32)  [experiment]
- `ag-gemm-bwd.2`: `linear_backward`: decompose into matmul_bwd + bias_add_bwd  [experiment]
- `ag-gemm-bwd.3`: `bias_add_backward`: sum reduction kernel for bias gradient  [experiment]
- `ag-gemm-bwd.4`: Numerical gradient check for Linear layer  [experiment]
- **4 tasks**

#### Theme: `ag-norm-bwd` — Normalization backward kernels
- `ag-norm-bwd.1`: Design `layer_norm_backward` algorithm (Jacobian decomposition)  [investigation]
- `ag-norm-bwd.2`: Implement `layer_norm_backward` PTX kernel  [experiment]
- `ag-norm-bwd.3`: Implement `batch_norm_backward` PTX kernel  [experiment]
- `ag-norm-bwd.4`: Numerical gradient check for LayerNorm and BatchNorm  [experiment]
- **4 tasks**

#### Theme: `ag-conv-bwd` — Convolution backward
- `ag-conv-bwd.1`: Implement `col2im` PTX kernel  [experiment]
- `ag-conv-bwd.2`: `conv2d_backward_data`: dX via col2im(W^T × dY)  [experiment]
- `ag-conv-bwd.3`: `conv2d_backward_weight`: dW via dY × col(X)^T  [experiment]
- `ag-conv-bwd.4`: Numerical gradient check for Conv2d  [experiment]
- **4 tasks**

#### Theme: `ag-attn-bwd` — Attention backward (decomposed)
- `ag-attn-bwd.1`: Decompose attention into matmul+softmax+matmul for backward  [design]
- `ag-attn-bwd.2`: Implement `softmax_backward` PTX kernel  [experiment]
- `ag-attn-bwd.3`: Full MHA backward via decomposed ops  [experiment]
- `ag-attn-bwd.4`: Numerical gradient check for MultiHeadAttention  [experiment]
- **4 tasks**

#### Theme: `ag-misc-bwd` — Remaining backward ops
- `ag-misc-bwd.1`: `embedding_backward` with atomicAdd scatter  [experiment]
- `ag-misc-bwd.2`: `max_pool2d_forward_with_indices` + `max_pool2d_backward`  [experiment]
- `ag-misc-bwd.3`: `elementwise_add_backward` (trivial: identity)  [experiment]
- `ag-misc-bwd.4`: `upsample_backward` (average pooling pattern)  [experiment]
- **4 tasks**

#### Theme: `ag-optim` — Optimizers
- `ag-optim.1`: Design Optimizer trait + SGD implementation  [experiment]
- `ag-optim.2`: Implement Adam optimizer (with m,v state tensors)  [experiment]
- `ag-optim.3`: `sgd_step` and `adam_step` PTX kernels (fused update)  [experiment]
- `ag-optim.4`: End-to-end training loop: overfit a 2-layer MLP on XOR  [experiment]
- **4 tasks**

#### Theme: `ag-loss` — Loss functions
- `ag-loss.1`: `cross_entropy_loss` forward + backward  [experiment]
- `ag-loss.2`: `mse_loss` forward + backward  [experiment]
- `ag-loss.3`: Numerical gradient check for losses  [experiment]
- **3 tasks**

**Total tasks**: ~35
**Depends on**: testing-infra (numerical gradient checking is essential)

### Epic 4: gpu-safety

**Themes and tasks**:

#### Theme: `gs-gpuref` — GpuRef<'a, T> core type
- `gs-gpuref.1`: Design GpuRef<'a, T> and GpuRefMut<'a, T> types  [design]
- `gs-gpuref.2`: Implement GpuRef with From<&GpuTensor> conversion  [experiment]
- `gs-gpuref.3`: Add GpuRef/GpuRefMut accessors to GpuTensor  [experiment]
- `gs-gpuref.4`: Compile-time test: verify borrow checker rejects use-after-drop  [experiment]
- **4 tasks**

#### Theme: `gs-kernel-safety` — Safe kernel launch wrappers
- `gs-kernel-safety.1`: Design SafeKernelArg trait (GpuRef implements, raw ptr does not)  [design]
- `gs-kernel-safety.2`: Implement SafeKernelArg for GpuRef<'_, f32>, &CudaSlice<f32>  [experiment]
- `gs-kernel-safety.3`: Add safe launch wrappers to KernelRegistry  [experiment]
- `gs-kernel-safety.4`: Migrate one op (matmul) to safe launch as proof of concept  [experiment]
- **4 tasks**

#### Theme: `gs-tape-safety` — Autograd tape lifetime safety
- `gs-tape-safety.1`: Audit tape TensorPool for dangling reference risk  [investigation]
- `gs-tape-safety.2`: Add generation counting to TensorPool for dangling detection  [experiment]
- `gs-tape-safety.3`: Ensure backward() consumes tape (move semantics, not borrow)  [experiment]
- **3 tasks**

#### Theme: `gs-device-safety` — Cross-device pointer safety
- `gs-device-safety.1`: Add device_id to GpuRef, validate at kernel launch  [experiment]
- `gs-device-safety.2`: Test: compile error when passing device-0 tensor to device-1 kernel  [experiment]
- **2 tasks**

#### Theme: `gs-migration` — Migrate existing code to safe API
- `gs-migration.1`: Migrate all activation ops to GpuRef  [experiment]
- `gs-migration.2`: Migrate norm ops  [experiment]
- `gs-migration.3`: Migrate gemm/conv ops  [experiment]
- `gs-migration.4`: Migrate all layers  [experiment]
- `gs-migration.5`: Deprecate raw pointer access in public API  [experiment]
- **5 tasks**

**Total tasks**: ~18
**Depends on**: autograd (tape lifetimes are the primary motivation)

### Summary

| Epic | Themes | Tasks | Depends On | Parallel Work |
|---|---|---|---|---|
| verify-examples | 5 | ~15 | — | None (must go first) |
| testing-infra | 4 | ~17 | verify-examples (goldens) | ti-harness + ti-cpu-ref can start during VE |
| autograd | 9 | ~35 | testing-infra (grad checks) | ag-elemwise-bwd in parallel with ag-tape |
| gpu-safety | 5 | ~18 | autograd (tape patterns) | gs-gpuref design can start during autograd |
| **Total** | **23** | **~85** | | |

### Risk Areas and Mitigations

1. **layer_norm_backward complexity**: The Jacobian of layernorm has 3 terms that
   must be accumulated in a single kernel pass for performance. Risk: numerical
   instability with f32.
   - **Mitigation**: Start with a CPU reference implementation, validate with finite
     differences, then port to PTX.

2. **Memory pressure for training large models**: GPT-2 Small training needs ~2GB;
   Medium needs ~6GB. Users with 4GB VRAM will be constrained.
   - **Mitigation**: Start with GPT-2 Small. Add gradient checkpointing in a
     follow-up epic.

3. **GpuRef adoption friction**: Changing the kernel launch API may break user code.
   - **Mitigation**: Phase the rollout. GpuRef is opt-in first, with deprecated but
     functional old API.

4. **Flash attention backward**: The efficient backward of flash attention is a
   research-level algorithm (Dao et al. 2022). Implementing it in inline PTX is
   extremely complex.
   - **Mitigation**: Decompose attention into matmul+softmax+matmul for backward.
     This uses 2× more memory but is much simpler to implement. Flash attention
     backward can be a separate epic later.

5. **Autograd tape thread safety**: If training uses CUDA streams for parallelism,
   the tape must handle concurrent recording.
   - **Mitigation**: v1 tape is single-threaded. Add `Send + Sync` bounds and
     interior mutability only when needed.

6. **col2im kernel correctness**: col2im requires atomic accumulation for overlapping
   patches, and atomicAdd for f32 has determinism issues on GPU.
   - **Mitigation**: Use `atomicAdd` (which is deterministic for f32 on SM 7.0+),
     or serialize the accumulation per output element.
