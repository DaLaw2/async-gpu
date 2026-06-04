# fusion-analysis.2 — Design: Tape-Level Pattern Matching Rules for Fusion Candidates

## Status: done
## Summary

This document defines the complete design for tape-level kernel fusion: formal fusion pattern rules, the `FusionOptimizer` data structures, a pattern matching algorithm, NVRTC codegen templates for fused elementwise kernels, and integration points with the existing autograd tape. The design covers all four fusion phases from fusion-analysis.1: GEMM epilogue, tape pattern match, NVRTC codegen, and cross-block fusion.

## 1. Fusion Pattern Rules — Formal Definition

### 1.1 Op Classification

Every `OpKind` falls into one of four categories that determine fusion behavior:

```rust
/// Classification of ops for fusion eligibility.
enum OpClass {
    /// Element-wise: output[i] = f(input[i], ...)
    /// Always fusable with other Elementwise ops on same-shape tensors.
    Elementwise,
    /// Compute-bound op that cannot fuse with peers, but can absorb
    /// elementwise ops into its epilogue (output write phase).
    ComputeBound,
    /// Reduction: consumes elements and produces fewer elements.
    /// Terminates elementwise chains but may absorb a prefix.
    Reduction,
    /// Structural: changes memory layout, not values. Fusion barrier.
    Structural,
}
```

Classification table:

| OpKind       | OpClass      | Fusable as producer? | Fusable as consumer? |
|--------------|-------------|---------------------|---------------------|
| Gelu         | Elementwise | Yes                 | Yes                 |
| Silu         | Elementwise | Yes                 | Yes                 |
| Sigmoid      | Elementwise | Yes                 | Yes                 |
| Relu         | Elementwise | Yes                 | Yes                 |
| BiasAdd      | Elementwise | Yes                 | Yes                 |
| ElemAdd      | Elementwise | Yes                 | Yes                 |
| Matmul       | ComputeBound| Epilogue only       | No                  |
| Conv2d       | ComputeBound| Epilogue only       | No                  |
| Attention    | ComputeBound| No (opaque)         | No                  |
| LayerNorm    | Reduction   | No                  | Can absorb prefix   |
| BatchNorm    | Reduction   | No                  | Can absorb prefix   |
| CrossEntropy | Reduction   | No                  | No                  |
| MseLoss      | Reduction   | No                  | No                  |
| Embedding    | Structural  | No                  | No                  |
| MaxPool2d    | Structural  | No                  | No                  |
| UpsampleNearest | Structural | No               | No                  |

### 1.2 Fusability Predicates

Three predicates must ALL hold for two consecutive tape entries `A` (producer) and `B` (consumer) to be fusable:

```rust
/// Can tape entry `producer` fuse with `consumer`?
fn can_fuse(producer: &TapeEntry, consumer: &TapeEntry, ref_counts: &HashMap<TensorId, usize>) -> bool {
    // P1: Producer's output feeds exactly one consumer (no fan-out).
    //     If the intermediate tensor is read by multiple ops, we cannot
    //     eliminate it — both consumers need the value.
    let single_consumer = ref_counts[&producer.output] == 1;

    // P2: Shape compatibility — both ops operate on the same element count.
    //     For elementwise fusion, shapes must be identical.
    //     For GEMM epilogue, the consumer shape == GEMM output shape.
    let shape_compatible = shapes_match(producer, consumer);

    // P3: Op class compatibility — see the fusion matrix below.
    let class_compatible = fusion_matrix(classify(producer.op), classify(consumer.op));

    single_consumer && shape_compatible && class_compatible
}
```

### 1.3 Fusion Matrix

The matrix defines which (producer_class, consumer_class) pairs can fuse:

| Producer ↓ \ Consumer → | Elementwise | ComputeBound | Reduction | Structural |
|--------------------------|------------|-------------|-----------|-----------|
| **Elementwise**          | YES        | No          | PARTIAL*  | No        |
| **ComputeBound**         | EPILOGUE** | No          | No        | No        |
| **Reduction**            | No         | No          | No        | No        |
| **Structural**           | No         | No          | No        | No        |

- `*PARTIAL`: Elementwise ops can fuse INTO certain reductions (e.g., `elementwise_add → layer_norm` is a known pattern with a hand-written kernel). This is handled as a special-case pattern, not general fusion.
- `**EPILOGUE`: ComputeBound (GEMM) can absorb the next elementwise op(s) into its output write loop. Limited to a small set of known epilogue patterns.

### 1.4 Concrete Fusion Patterns

Each pattern is defined as a sequence of `OpKind` values with constraints:

```rust
/// A fusion pattern: a sequence of ops that can be replaced by a single fused op.
struct FusionPattern {
    /// Ordered sequence of OpKind values to match.
    ops: Vec<OpKind>,
    /// The fused replacement op kind.
    replacement: FusedOpKind,
    /// Additional constraints beyond the standard fusability predicates.
    constraints: Vec<PatternConstraint>,
    /// Priority (higher = try first). Longer patterns have higher priority.
    priority: u32,
}
```

**Pattern catalog (ordered by priority):**

| ID | Pattern | Replacement | Constraint | Priority | Kernel launches saved |
|----|---------|-------------|------------|----------|----------------------|
| P1 | `Matmul → BiasAdd → Gelu` | `FusedMatmulBiasGelu` | V1 GEMM path only (dims ≤ padded) | 100 | 2 |
| P2 | `Matmul → BiasAdd → Relu` | `FusedMatmulBiasRelu` | V1 GEMM path only | 100 | 2 |
| P3 | `ElemAdd → LayerNorm` | `FusedElemAddLayerNorm` | cublas feature, d_model % 4 == 0 | 90 | 1 |
| P4 | `Matmul → BiasAdd` | `FusedMatmulBias` | Any GEMM path (V2/V4.1/cuBLAS epilogue) | 80 | 1 |
| P5 | `BiasAdd → Gelu` | `FusedBiasGelu` | Same shape, NVRTC codegen | 70 | 1 |
| P6 | `BiasAdd → Silu` | `FusedBiasSilu` | Same shape, NVRTC codegen | 70 | 1 |
| P7 | `BiasAdd → Relu` | `FusedBiasRelu` | Same shape, NVRTC codegen | 70 | 1 |
| P8 | `BiasAdd → Sigmoid` | `FusedBiasSigmoid` | Same shape, NVRTC codegen | 70 | 1 |
| P9 | `ElemAdd → Gelu` | `FusedAddGelu` | Same shape, NVRTC codegen | 60 | 1 |
| P10 | Arbitrary elementwise chain (2-5 ops) | `FusedElementwiseChain` | All same shape, NVRTC codegen | 50 | N-1 |

**What CANNOT fuse (explicit exclusions):**

1. **GEMM → GEMM**: Both are compute-bound; fusing them would require a fundamentally different algorithm. They share no data locality.
2. **Attention → anything**: Flash attention is a self-contained fusion island with its own tiling strategy. Fusing into or out of it would break its numerics.
3. **Different tensor shapes**: `bias_add([128,768]) → gelu([128,3072])` — shapes differ, so they operate on different data. Cannot fuse.
4. **Fan-out (ref_count > 1)**: If tensor T is consumed by ops A and B, we cannot eliminate T by fusing its producer with A — B still needs it.
5. **Ops with saved tensors needed by backward**: If the intermediate tensor is in `saved` for some other entry's backward, eliminating it would break gradient computation. (However, during inference-only mode this constraint is relaxed.)
6. **In-place ops with aliased inputs**: `elementwise_add` is in-place (`a += b`). If `a` is also used elsewhere, fusing must preserve the write.

## 2. Tape Representation for Fusion

### 2.1 Current Tape Structure (as-is)

From `crates/core/gpu-host/src/nn/autograd/tape.rs`:

```rust
pub struct TapeEntry {
    pub op: OpKind,           // What operation
    pub inputs: Vec<TensorId>,  // Input tensor IDs
    pub output: TensorId,     // Output tensor ID
    pub saved: Vec<TensorId>,  // Tensors saved for backward
    pub meta: OpMeta,         // Op-specific metadata (shapes, etc.)
}
```

**Available metadata per op:**

| OpKind | OpMeta variant | Shape info available |
|--------|---------------|---------------------|
| Matmul | `Matmul { m, k, n }` | Full: output is [m, n] |
| BiasAdd | `BiasAdd { n_cols }` | Partial: n_cols, but not rows. Need to infer from input. |
| LayerNorm | `LayerNorm { rows, d, eps }` | Full: input/output is [rows, d] |
| Gelu/Silu/Sigmoid/Relu | `None` | **Missing**: no shape recorded |
| ElemAdd | `None` | **Missing**: no shape recorded |
| Embedding | `Embedding { vocab_size }` | Partial |
| Attention | `Attention { seq, d, causal }` | Full |
| BatchNorm | `BatchNorm { channels, hw, ... }` | Full |

### 2.2 Required Tape Extensions

The fusion optimizer needs shape information for every op. Currently, activations and element-wise ops record `OpMeta::None`. Two options:

**Option A: Add shape to TapeEntry directly (recommended)**

```rust
pub struct TapeEntry {
    pub op: OpKind,
    pub inputs: Vec<TensorId>,
    pub output: TensorId,
    pub saved: Vec<TensorId>,
    pub meta: OpMeta,
    /// Output tensor shape — needed by the fusion optimizer.
    /// Added as a lightweight field (SmallVec avoids heap alloc for <= 4 dims).
    pub output_shape: SmallVec<[usize; 4]>,
}
```

This is backward-compatible — existing code just needs to populate the new field at each `record_op` call site. The shape is always known at recording time (it's the shape of the output tensor). Cost: 32 bytes per tape entry (SmallVec inline storage).

**Option B: Build a side table from TensorPool**

The `TensorPool` already stores tensors by ID. The fusion optimizer could look up shapes from the pool. But the pool may not be available during inference (no tape recording), so Option A is more robust.

**Recommendation**: Option A. The 32 bytes per entry is trivial compared to the GPU memory for the actual tensors.

### 2.3 Reference Count Computation

The fusion optimizer needs to know how many consumers each intermediate tensor has. This is computed in a single pass over the tape:

```rust
fn compute_ref_counts(tape: &[TapeEntry]) -> HashMap<TensorId, usize> {
    let mut counts: HashMap<TensorId, usize> = HashMap::new();
    for entry in tape {
        for &input_id in &entry.inputs {
            *counts.entry(input_id).or_insert(0) += 1;
        }
    }
    counts
}
```

## 3. Pattern Matching Algorithm

### 3.1 Algorithm: Greedy Longest-Match Forward Scan

The algorithm scans the tape from front to back, greedily matching the longest applicable pattern at each position. This is optimal for non-overlapping patterns (which ours are, since each op can participate in at most one fusion group).

```rust
/// Result of the fusion optimization pass.
struct FusionPlan {
    /// Ranges of tape entries to fuse, with their replacement.
    fusions: Vec<FusionGroup>,
}

struct FusionGroup {
    /// Inclusive start index in the tape.
    start: usize,
    /// Exclusive end index in the tape.
    end: usize,
    /// The fused op to replace entries[start..end].
    fused_op: FusedOpKind,
    /// Precompiled CUDA kernel (or reference to cached kernel).
    kernel: FusedKernel,
}

fn find_fusion_candidates(tape: &[TapeEntry]) -> FusionPlan {
    let ref_counts = compute_ref_counts(tape);
    let patterns = get_pattern_catalog(); // sorted by priority desc, then length desc
    let mut fusions = Vec::new();
    let mut i = 0;

    while i < tape.len() {
        let mut best_match: Option<(usize, &FusionPattern)> = None;

        // Try each pattern starting at position i
        for pattern in &patterns {
            let len = pattern.ops.len();
            if i + len > tape.len() {
                continue;
            }

            // Check if ops match
            let slice = &tape[i..i + len];
            if !ops_match(slice, pattern) {
                continue;
            }

            // Check fusability predicates for each consecutive pair
            let mut all_fusable = true;
            for j in 0..len - 1 {
                if !can_fuse(&slice[j], &slice[j + 1], &ref_counts) {
                    all_fusable = false;
                    break;
                }
                // Check data flow: producer's output == consumer's input
                if !slice[j + 1].inputs.contains(&slice[j].output) {
                    all_fusable = false;
                    break;
                }
            }

            // Check pattern-specific constraints
            if all_fusable && pattern.constraints_satisfied(slice) {
                match &best_match {
                    None => best_match = Some((len, pattern)),
                    Some((best_len, _)) if len > *best_len => {
                        best_match = Some((len, pattern));
                    }
                    _ => {} // keep longer match
                }
            }
        }

        if let Some((len, pattern)) = best_match {
            fusions.push(FusionGroup {
                start: i,
                end: i + len,
                fused_op: pattern.replacement,
                kernel: compile_or_lookup(pattern, &tape[i..i + len]),
            });
            i += len; // skip fused entries
        } else {
            i += 1;
        }
    }

    FusionPlan { fusions }
}
```

### 3.2 Why Greedy Over Optimal

- **Greedy**: O(N * P) where N = tape length, P = number of patterns. For GPT-2 Small: N ≈ 184, P ≈ 10, so ~1840 comparisons per forward pass. Negligible.
- **Optimal** (e.g., dynamic programming to minimize total kernel launches): Would be O(N^2 * P) and add complexity for minimal benefit. The patterns don't overlap in practice — a matmul is never part of two fusion groups.
- **Graph analysis** (finding connected components of fusable ops): Unnecessary because the tape is already topologically sorted. A forward scan naturally respects data dependencies.

### 3.3 Handling the Arbitrary Elementwise Chain (P10)

Pattern P10 is special: it matches any sequence of 2-5 consecutive elementwise ops on the same shape. After all specific patterns (P1-P9) have been tried, the algorithm falls back to P10:

```rust
fn try_elementwise_chain(tape: &[TapeEntry], start: usize, ref_counts: &HashMap<TensorId, usize>) -> Option<FusionGroup> {
    let mut end = start;
    let first_shape = &tape[start].output_shape;

    while end + 1 < tape.len() && end - start < 5 {
        let next = &tape[end + 1];
        if classify(next.op) != OpClass::Elementwise {
            break;
        }
        if &next.output_shape != first_shape {
            break;
        }
        if !next.inputs.contains(&tape[end].output) {
            break;
        }
        if ref_counts[&tape[end].output] > 1 {
            break;
        }
        end += 1;
    }

    let chain_len = end - start + 1;
    if chain_len >= 2 {
        Some(FusionGroup {
            start,
            end: end + 1,
            fused_op: FusedOpKind::ElementwiseChain,
            kernel: codegen_elementwise_chain(&tape[start..end + 1]),
        })
    } else {
        None
    }
}
```

## 4. NVRTC Codegen Templates

### 4.1 Elementwise Chain Template

The core codegen template for fusing N elementwise ops into a single kernel:

```cuda
// Template: fused elementwise chain kernel
// Generated by FusionOptimizer::codegen_elementwise
// Ops: {op_list}
// Shape: [{rows}, {cols}]

extern "C" __global__ void fused_elementwise_{hash}(
    const float* __restrict__ input,    // first op's input
    float* __restrict__ output,         // last op's output
    {extra_params}                      // bias vectors, etc.
    unsigned int n                      // total element count
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idx = tid * 4;

    if (idx + 3 < n) {
        float4 v = *reinterpret_cast<const float4*>(&input[idx]);

        // --- Begin fused op chain ---
        {fused_ops_float4}
        // --- End fused op chain ---

        *reinterpret_cast<float4*>(&output[idx]) = v;
    } else {
        // Scalar tail
        for (unsigned int i = idx; i < n && i < idx + 4; i++) {
            float val = input[i];
            {fused_ops_scalar}
            output[i] = val;
        }
    }
}
```

### 4.2 Per-Op Code Fragments

Each elementwise op maps to a code fragment that operates on either a `float4 v` or a scalar `float val`:

```rust
/// Generate CUDA C code fragment for one elementwise op.
fn op_codegen(op: OpKind, param_idx: &mut usize) -> (String, String, Vec<String>) {
    // Returns: (float4_code, scalar_code, extra_params)
    match op {
        OpKind::Gelu => (
            // Approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            r#"
        {
            const float SQRT_2_OVER_PI = 0.7978845608f;
            const float COEFF = 0.044715f;
            float4 tmp;
            tmp.x = SQRT_2_OVER_PI * (v.x + COEFF * v.x * v.x * v.x);
            tmp.y = SQRT_2_OVER_PI * (v.y + COEFF * v.y * v.y * v.y);
            tmp.z = SQRT_2_OVER_PI * (v.z + COEFF * v.z * v.z * v.z);
            tmp.w = SQRT_2_OVER_PI * (v.w + COEFF * v.w * v.w * v.w);
            v.x = 0.5f * v.x * (1.0f + tanhf(tmp.x));
            v.y = 0.5f * v.y * (1.0f + tanhf(tmp.y));
            v.z = 0.5f * v.z * (1.0f + tanhf(tmp.z));
            v.w = 0.5f * v.w * (1.0f + tanhf(tmp.w));
        }
            "#.into(),
            r#"
        {
            const float SQRT_2_OVER_PI = 0.7978845608f;
            const float COEFF = 0.044715f;
            float tmp = SQRT_2_OVER_PI * (val + COEFF * val * val * val);
            val = 0.5f * val * (1.0f + tanhf(tmp));
        }
            "#.into(),
            vec![],
        ),

        OpKind::Relu => (
            r#"
        v.x = fmaxf(v.x, 0.0f);
        v.y = fmaxf(v.y, 0.0f);
        v.z = fmaxf(v.z, 0.0f);
        v.w = fmaxf(v.w, 0.0f);
            "#.into(),
            "val = fmaxf(val, 0.0f);".into(),
            vec![],
        ),

        OpKind::Silu => (
            r#"
        v.x = v.x / (1.0f + expf(-v.x));
        v.y = v.y / (1.0f + expf(-v.y));
        v.z = v.z / (1.0f + expf(-v.z));
        v.w = v.w / (1.0f + expf(-v.w));
            "#.into(),
            "val = val / (1.0f + expf(-val));".into(),
            vec![],
        ),

        OpKind::Sigmoid => (
            r#"
        v.x = 1.0f / (1.0f + expf(-v.x));
        v.y = 1.0f / (1.0f + expf(-v.y));
        v.z = 1.0f / (1.0f + expf(-v.z));
        v.w = 1.0f / (1.0f + expf(-v.w));
            "#.into(),
            "val = 1.0f / (1.0f + expf(-val));".into(),
            vec![],
        ),

        OpKind::BiasAdd => {
            let p = *param_idx;
            *param_idx += 1;
            (
                format!(r#"
        {{
            unsigned int col = idx % n_cols_{p};
            float4 bv = *reinterpret_cast<const float4*>(&bias_{p}[col]);
            v.x += bv.x;
            v.y += bv.y;
            v.z += bv.z;
            v.w += bv.w;
        }}
                "#),
                format!(r#"
        {{
            unsigned int col = i % n_cols_{p};
            val += bias_{p}[col];
        }}
                "#),
                vec![
                    format!("const float* __restrict__ bias_{p}"),
                    format!("unsigned int n_cols_{p}"),
                ],
            )
        },

        OpKind::ElemAdd => {
            let p = *param_idx;
            *param_idx += 1;
            (
                format!(r#"
        {{
            float4 bv = *reinterpret_cast<const float4*>(&addend_{p}[idx]);
            v.x += bv.x;
            v.y += bv.y;
            v.z += bv.z;
            v.w += bv.w;
        }}
                "#),
                format!("val += addend_{p}[i];"),
                vec![format!("const float* __restrict__ addend_{p}")],
            )
        },

        _ => panic!("op_codegen called on non-elementwise op: {:?}", op),
    }
}
```

### 4.3 Full Codegen Example

For the chain `BiasAdd → Gelu` on shape `[128, 3072]`:

**Generated kernel:**

```cuda
extern "C" __global__ void fused_elementwise_a7f3b2c1(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ bias_0,
    unsigned int n_cols_0,
    unsigned int n
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idx = tid * 4;

    if (idx + 3 < n) {
        float4 v = *reinterpret_cast<const float4*>(&input[idx]);

        // Op 1: BiasAdd
        {
            unsigned int col = idx % n_cols_0;
            float4 bv = *reinterpret_cast<const float4*>(&bias_0[col]);
            v.x += bv.x; v.y += bv.y; v.z += bv.z; v.w += bv.w;
        }

        // Op 2: Gelu
        {
            const float SQRT_2_OVER_PI = 0.7978845608f;
            const float COEFF = 0.044715f;
            float4 tmp;
            tmp.x = SQRT_2_OVER_PI * (v.x + COEFF * v.x * v.x * v.x);
            tmp.y = SQRT_2_OVER_PI * (v.y + COEFF * v.y * v.y * v.y);
            tmp.z = SQRT_2_OVER_PI * (v.z + COEFF * v.z * v.z * v.z);
            tmp.w = SQRT_2_OVER_PI * (v.w + COEFF * v.w * v.w * v.w);
            v.x = 0.5f * v.x * (1.0f + tanhf(tmp.x));
            v.y = 0.5f * v.y * (1.0f + tanhf(tmp.y));
            v.z = 0.5f * v.z * (1.0f + tanhf(tmp.z));
            v.w = 0.5f * v.w * (1.0f + tanhf(tmp.w));
        }

        *reinterpret_cast<float4*>(&output[idx]) = v;
    } else {
        for (unsigned int i = idx; i < n && i < idx + 4; i++) {
            float val = input[i];
            { unsigned int col = i % n_cols_0; val += bias_0[col]; }
            {
                const float SQRT_2_OVER_PI = 0.7978845608f;
                const float COEFF = 0.044715f;
                float tmp = SQRT_2_OVER_PI * (val + COEFF * val * val * val);
                val = 0.5f * val * (1.0f + tanhf(tmp));
            }
            output[i] = val;
        }
    }
}
```

**Launch config:** `grid = ceil(n / 1024)`, `block = (256, 1, 1)`, where `n = 128 * 3072 = 393216`.

### 4.4 BiasAdd float4 Alignment Note

The BiasAdd float4 codegen above assumes `n_cols % 4 == 0` for vectorized bias reads. When this is not the case (rare for transformer dims), the codegen falls back to scalar bias loads:

```cuda
// Fallback for n_cols not divisible by 4
v.x += bias_0[(idx    ) % n_cols_0];
v.y += bias_0[(idx + 1) % n_cols_0];
v.z += bias_0[(idx + 2) % n_cols_0];
v.w += bias_0[(idx + 3) % n_cols_0];
```

## 5. Kernel Cache

### 5.1 Cache Key

```rust
/// Unique key for a fused kernel in the cache.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct FusionCacheKey {
    /// Ordered list of ops in the chain.
    ops: Vec<OpKind>,
    /// Output shape (all ops in the chain share this shape for elementwise fusion).
    shape: Vec<usize>,
    /// Op-specific parameters that affect codegen (e.g., n_cols for BiasAdd).
    params: Vec<u64>,
}
```

### 5.2 Cache Structure

```rust
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Thread-safe cache of compiled fused kernels.
struct FusionCache {
    /// Map from cache key to compiled kernel module name + function name.
    cache: Mutex<HashMap<FusionCacheKey, CachedKernel>>,
}

struct CachedKernel {
    /// NVRTC module name (for `dev.get_func(module, func)`)
    module_name: String,
    /// Function name within the module
    func_name: String,
    /// Launch configuration
    grid_dim: (u32, u32, u32),
    block_dim: (u32, u32, u32),
}
```

### 5.3 Cache Behavior

- **First call**: NVRTC compilation (~50-200ms), result cached.
- **Subsequent calls**: Hash lookup (~100ns), direct kernel launch.
- **Cache key includes shape**: Different batch sizes or hidden dims produce different keys. This is correct: the kernel launch config depends on element count.
- **Cache lifetime**: Process-scoped (static). Cleared on process exit. For long-running training, the cache stabilizes after the first forward pass (since shapes are constant).

For GPT-2 Small with constant shapes:
- Forward pass: at most ~5-6 unique fused kernels (bias_add+gelu, bias_add alone at different dims, elementwise_add+layer_norm, etc.)
- Total first-call overhead: ~300-1200ms
- Amortized over subsequent iterations: negligible

## 6. FusionOptimizer Data Structure

### 6.1 Core Type

```rust
/// Tape-level fusion optimizer.
///
/// Analyzes an autograd tape to find fusable op sequences,
/// generates fused CUDA kernels via NVRTC, and caches them.
pub struct FusionOptimizer {
    /// Compiled kernel cache (shared across calls).
    cache: Arc<FusionCache>,
    /// Whether to enable GEMM epilogue fusion (requires specific GEMM versions).
    enable_gemm_epilogue: bool,
    /// Whether to enable NVRTC codegen for arbitrary chains.
    enable_nvrtc_codegen: bool,
    /// Maximum chain length for arbitrary elementwise fusion.
    max_chain_length: usize,
}

impl FusionOptimizer {
    pub fn new() -> Self {
        Self {
            cache: Arc::new(FusionCache::new()),
            enable_gemm_epilogue: true,
            enable_nvrtc_codegen: true,
            max_chain_length: 5,
        }
    }

    /// Analyze a tape and produce a fusion plan.
    ///
    /// Does NOT modify the tape. Returns a plan that can be executed.
    pub fn analyze(&self, tape: &[TapeEntry]) -> FusionPlan {
        find_fusion_candidates(tape)
    }

    /// Execute a fusion plan: for each FusionGroup, compile (or look up)
    /// the fused kernel and return launch descriptors.
    pub fn prepare(
        &self,
        plan: &FusionPlan,
        dev: &Arc<CudaDevice>,
    ) -> Vec<PreparedFusion> {
        plan.fusions.iter().map(|group| {
            let key = group.cache_key();
            let kernel = self.cache.get_or_compile(&key, || {
                group.codegen_and_compile(dev)
            });
            PreparedFusion { group: group.clone(), kernel }
        }).collect()
    }
}
```

### 6.2 FusedOpKind Enum

```rust
/// Kinds of fused operations that the optimizer can produce.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum FusedOpKind {
    // --- Hand-written fused kernels (pattern match → existing kernel) ---

    /// matmul + bias_add + gelu → existing `gemm_bias_gelu` kernel
    MatmulBiasGelu,
    /// matmul + bias_add + relu → existing `gemm_bias_relu` kernel
    MatmulBiasRelu,
    /// elementwise_add + layer_norm → existing `layer_norm_residual` / `_dual` kernel
    ElemAddLayerNorm,

    // --- GEMM epilogue fusion (modify GEMM output write) ---

    /// matmul + bias_add → cuBLAS epilogue or modified GEMM kernel
    MatmulBias,

    // --- NVRTC-generated fused kernels ---

    /// Generic fused elementwise chain: N ops → 1 kernel via NVRTC codegen
    ElementwiseChain,
}
```

## 7. Integration with Existing Tape/Autograd

### 7.1 Where Fusion Happens

Fusion must happen AFTER the tape is recorded and BEFORE execution. In the current architecture, ops execute eagerly (the tape records what already happened). To enable fusion, we need a **deferred execution** mode:

**Option A: Trace-then-replay (inference only)**

```
1. First forward pass: execute eagerly, record tape as normal
2. FusionOptimizer.analyze(tape) → FusionPlan
3. FusionOptimizer.prepare(plan) → compiled fused kernels
4. Second forward pass: use fused kernels for matched patterns
```

This is the `torch.compile` / `torch.jit.trace` approach. The first pass is un-optimized; subsequent passes with the same shapes use fused kernels.

**Option B: Lazy execution with fusion window**

```
1. Ops are recorded but NOT executed (deferred)
2. When a synchronization point is hit (or N ops accumulated),
   the fusion optimizer scans the pending ops
3. Fused groups are executed as single kernels
4. Unfused ops are executed normally
```

This is more complex but avoids the "first pass is slow" problem.

**Recommendation**: Option A for v1. It maps directly onto the existing architecture:

```rust
/// Traced forward pass wrapper for inference with fusion.
pub struct TracedForward {
    /// The fusion plan from the first (tracing) run.
    plan: FusionPlan,
    /// Compiled fused kernels.
    prepared: Vec<PreparedFusion>,
}

impl TracedForward {
    /// Trace: run the model once to capture the tape.
    pub fn trace<F: FnOnce() -> R, R>(f: F) -> (R, Tape) {
        let tape = Tape::new();
        autograd::with_tape(tape, f)
    }

    /// Compile: analyze the tape and prepare fused kernels.
    pub fn compile(tape: &Tape, dev: &Arc<CudaDevice>) -> Self {
        let optimizer = FusionOptimizer::new();
        let plan = optimizer.analyze(tape.entries());
        let prepared = optimizer.prepare(&plan, dev);
        Self { plan, prepared }
    }

    // The replay step would need integration with a modified forward pass
    // that dispatches to fused kernels when the pattern matches.
    // This is the implementation challenge for a future task.
}
```

### 7.2 Integration Points in the Existing Code

**Recording side (already works, needs minor extension):**

Each op's `record_op` call site needs to include `output_shape`:

- `activation.rs`: Add `output_shape: SmallVec::from_slice(input.shape())` — output shape == input shape for elementwise.
- `reshape.rs` (`bias_add`): Add `output_shape: SmallVec::from_slice(input.shape())`.
- `reshape.rs` (`elementwise_add`): Add `output_shape: SmallVec::from_slice(a.shape())`.
- `gemm.rs` (`matmul`): Add `output_shape: SmallVec::from_slice(&[m, n])`.
- `norm.rs` (`layer_norm`): Add `output_shape: SmallVec::from_slice(input.shape())`.

**Execution side (new, needed for fusion):**

The `FusionOptimizer` produces a `FusionPlan`. The plan maps tape entry ranges to fused kernels. During a fused forward pass:

1. Walk the tape sequentially.
2. For entries covered by a `FusionGroup`, launch the fused kernel with the appropriate inputs (first entry's input, last entry's output, plus any extra parameters like bias vectors).
3. For entries NOT covered by a fusion group, execute normally.

**Backward compatibility:**

Fusion is entirely opt-in. The existing `Tape`, `TapeEntry`, `backward()` all remain unchanged. The `FusionOptimizer` is a new module that reads the tape but does not modify it. Backward pass continues to use the original (unfused) tape entries for gradient computation.

### 7.3 Backward Pass Considerations

For training (not just inference), fused kernels need fused backward passes. This is significantly more complex:

- `fused_bias_gelu_backward` needs to compute both `d_bias` and `d_gelu` in one kernel.
- The fused forward kernel must save any intermediates needed for backward (e.g., pre-activation values for GELU derivative).

**Recommendation**: Defer training fusion to a later phase. Start with inference-only fusion (forward pass tracing + replay). The existing hand-written fused kernels (gemm_bias_gelu, layer_norm_residual_dual) already handle the most impactful training cases.

## 8. GPT-2 Fusion Analysis — Concrete Impact

### 8.1 Per-Block Fusion Opportunities

For a single `TransformerBlock::forward()`:

```
Op sequence (current):
  1. layer_norm_v3           (LN1)            → standalone
  2. matmul_v2               (QKV proj)       → [2,3] fusable
  3. bias_add                (QKV bias)       → [2,3] fusable
  4. split_qkv               (structural)     → barrier
  5. multi_head_flash_attn   (attention)      → barrier
  6. concat_heads            (structural)     → barrier
  7. matmul_v2               (out proj)       → [7,8] fusable
  8. bias_add                (out bias)       → [7,8] fusable
  9. layer_norm_residual_dual (fused LN+res)  → already fused
  10. matmul_v2              (FFN up)         → [10,11,12] fusable
  11. bias_add               (FFN up bias)    → [10,11,12] fusable
  12. gelu_forward_v2        (GELU)           → [10,11,12] fusable
  13. matmul_v2              (FFN down)       → [13,14] fusable
  14. bias_add               (FFN down bias)  → [13,14] fusable
  15. elementwise_add_v3     (residual add)   → standalone (but see cross-block)
```

**Fusion plan per block:**

| Group | Entries | Pattern | Fused kernel | Launches saved |
|-------|---------|---------|-------------|----------------|
| F1 | [2,3] | Matmul→BiasAdd | GEMM epilogue (P4) | 1 |
| F2 | [7,8] | Matmul→BiasAdd | GEMM epilogue (P4) | 1 |
| F3 | [10,11,12] | Matmul→BiasAdd→Gelu | gemm_bias_gelu (P1) | 2 |
| F4 | [13,14] | Matmul→BiasAdd | GEMM epilogue (P4) | 1 |

**Per-block savings: 5 kernel launches** (15 → 10).

### 8.2 Total GPT-2 Small Impact

| Component | Current launches | After fusion | Saved |
|-----------|-----------------|-------------|-------|
| 12 blocks | 12 * 15 = 180 | 12 * 10 = 120 | 60 |
| Embedding | 1 | 1 | 0 |
| Final LN | 1 | 1 | 0 |
| LM head (matmul) | 1 | 1 | 0 |
| LM head (bias_add) | 1 | 1* | 0* |
| **Total** | **184** | **124** | **60** |

*LM head has no bias (tied weights), so no fusion opportunity there.

### 8.3 Cross-Block Fusion (Phase 4, Future)

The residual add at the end of block N and the layer_norm at the start of block N+1 are candidates:

```
Block N:   ... → elementwise_add (residual)
Block N+1: layer_norm → ...
```

This is pattern P3 (`ElemAdd → LayerNorm`), already matched by `layer_norm_residual_dual`. The tape sees the flat sequence — it does not know about block boundaries. So cross-block fusion is free once the pattern matcher is in place.

This would save an additional **11 launches** (one per block boundary, 12 blocks = 11 boundaries), bringing the total from 124 to **113** (61% of original).

## 9. Implementation Phasing

### Phase 1: Shape in TapeEntry + FusionOptimizer skeleton (prerequisite)
- Add `output_shape` field to `TapeEntry`.
- Update all `record_op` call sites to populate shape.
- Implement `FusionOptimizer::analyze()` with the greedy scan algorithm.
- Unit test: verify correct pattern detection on a hand-built tape.

### Phase 2: Pattern match → existing fused kernels
- Implement P1 (Matmul→BiasAdd→Gelu → `gemm_bias_gelu`).
- Implement P3 (ElemAdd→LayerNorm → `layer_norm_residual_dual`).
- These are pure routing: no codegen, just dispatching to existing kernels.
- Integration test: run GPT-2 forward with FusionOptimizer, verify output matches unfused.

### Phase 3: NVRTC codegen for elementwise chains
- Implement the codegen template from Section 4.
- Implement FusionCache with hash-based lookup.
- Implement P5-P10 (arbitrary elementwise chains).
- Benchmark: measure kernel launch reduction and wall-clock speedup.

### Phase 4: GEMM epilogue fusion (P4)
- Extend `matmul_v2` / cuBLAS path to accept optional bias parameter in the output write.
- This is a kernel modification, not just routing. More invasive but highest per-launch impact.

## Files Changed: none
