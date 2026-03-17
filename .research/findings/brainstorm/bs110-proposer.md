# Brainstorm 110 — Proposer: Strategic Direction for Breadth
**Cycle**: 560 | **Date**: 2026-03-17 | **Level**: Deep (Proposer)

## Project State Assessment

The project stands at **560 cycles, 670 completed tasks, 36 completed epics** with only the
evergreen `codebase-health` epic remaining active. The framework has proven its thesis
comprehensively: async/await on GPU, warp-cooperative scheduling, real Rust std, full NN
inference pipeline (GPT-2, YOLO, ResNet, MobileNet), NN training (MNIST, CIFAR, LoRA),
ONNX runtime (43 ops), differentiable physics, GPU-autonomous RAG, formal verification
(TLA+, 367M states), graph compiler, persistent kernel, quantization (INT8 + INT4), and
Tensor Core GEMM.

The strategic question now is: **how do we prove the framework handles ALL GPU scenarios,
not just neural networks?** The user has explicitly identified classical ML (tree models,
random forest, XGBoost-style) as a priority. The existing coverage is deep in NN territory
but thin in other GPU compute domains. Breadth is the path forward.

---

## Active Epics Assessment

### codebase-health (evergreen)
**Status**: Effectively satisfied. No actionable criteria remain unfilled. This epic serves
as a housekeeping umbrella and does not drive new feature development.

**Verdict**: No active epics are producing work. The project needs new epics to resume
forward progress.

---

## Deferred Criteria from Completed Epics

| Epic | Deferred Item | Root Cause |
|------|--------------|------------|
| `int8-inference` | GPT-2 full INT8 inference | Needs tiled dp4a for competitive perf |
| `int8-inference` | Per-token latency improvement | Naive dp4a is 7-10x slower than tiled f32 |
| `int4-quantization` | GPT-2 INT4 inference (coherent text) | INT4 Linear layer not integrated with GPT-2 |
| `kernel-fusion` | GPT-2 nn API integration | Needs Linear layer to detect + dispatch fused path |
| `kernel-fusion-v2` | 15% GPT-2 latency reduction | Needs fused_prepadded_b variant |
| `persistent-kernel` | Dispatch latency < 10us | 16.1us achieved; kernel re-launch is faster at 9.5us |
| `gpu-rag` | Hostcall round-trips | Pipeline uses nn API orchestration instead |

---

## NEW Epic Proposals

### 1. `classical-ml` — Classical ML on GPU: Decision Trees, Random Forest, k-Means
**Priority**: P1 (Immediate)

**WHY**: This is the single highest-impact gap in the project's coverage. Every GPU
framework demo shows matrix multiplication and neural networks. Decision trees, random
forests, and k-means represent fundamentally different compute patterns: irregular branching
(trees), ensemble aggregation (forests), iterative clustering with convergence (k-means).
The user explicitly requested tree models. Proving async_gpu handles these non-NN workloads
is the strongest possible breadth argument.

**Success Criteria**:
1. GPU decision tree inference: load a trained tree, classify 10K+ samples in parallel
   (each thread walks the tree independently)
2. GPU random forest: ensemble of 100+ trees, majority-vote aggregation, accuracy matches
   CPU reference on a standard dataset (e.g., iris or wine)
3. GPU k-means clustering: Lloyd's algorithm, K=10, 100K points, iterative until
   convergence, GPU speedup >= 5x over CPU reference
4. GPU k-nearest neighbors (KNN): brute-force distance computation, K=5, 10K query points
   against 50K reference points
5. Standalone examples in examples/std/classical-ml/

**Estimated Complexity**: Medium. Decision tree traversal is simple per-thread code.
k-means requires GPU reduction (already proven in ONNX ReduceMean). Random forest is
embarrassingly parallel. KNN is distance matrix + partial sort.

**Dependencies**: None beyond existing framework. GpuTensor for data, existing GPU kernels
for reductions. No new PTX intrinsics needed.

**Technical Notes**:
- Tree inference: each thread gets a sample, walks the tree stored in global memory
  (node array with left/right child indices + split feature/threshold). Divergent warps
  are expected — this is a stress test for the framework's handling of irregular control flow.
- k-means: requires GPU-side atomic add for centroid accumulation, or a reduction kernel.
  Both patterns already exist in the codebase.
- Random forest: embarrassingly parallel over trees AND over samples. Two-level parallelism
  is a good showcase for the async model.

---

### 2. `fft-signal` — FFT and Signal Processing on GPU
**Priority**: P2 (Near-term)

**WHY**: FFT is the second most important GPU workload after GEMM. It appears in audio
processing, image filtering, spectral analysis, scientific simulation, and as a subroutine
in convolution (conv via FFT). Implementing Cooley-Tukey FFT on GPU exercises shared memory
butterfly operations, bit-reversal permutation, and synchronization barriers — all patterns
that stress-test the async framework differently from matrix operations.

**Success Criteria**:
1. GPU radix-2 FFT: complex-to-complex, power-of-2 lengths up to 2^20 (1M points)
2. Correctness: max error < 1e-5 vs CPU reference (numpy/manual DFT)
3. 1D convolution via FFT: signal * kernel in frequency domain, matches direct convolution
4. Spectral analysis demo: compute power spectrum of a synthetic signal, identify frequency
   peaks
5. GPU speedup >= 10x over single-threaded CPU FFT for N >= 2^16

**Estimated Complexity**: Medium-Large. The Cooley-Tukey butterfly is well-understood but
GPU shared memory staging requires careful implementation. Bit-reversal permutation and
twiddle factor computation are straightforward.

**Dependencies**: None. Pure compute, no framework extensions needed.

**Technical Notes**:
- Shared memory butterfly: each block processes a sub-FFT in shared memory, then global
  memory for cross-block stages. This exercises `__syncthreads()` heavily.
- The framework already has complex number support? If not, a simple `Complex { re: f32, im: f32 }`
  struct suffices.
- Stretch goal: inverse FFT, 2D FFT (for image processing).

---

### 3. `graph-algorithms` — Graph Algorithms on GPU: BFS, PageRank, SSSP
**Priority**: P2 (Near-term)

**WHY**: Graph algorithms represent the extreme end of irregular memory access — scatter/
gather patterns, poor locality, data-dependent parallelism. If async_gpu handles graph
traversal well, it proves the framework is not limited to regular data-parallel workloads.
BFS and PageRank are canonical GPU graph benchmarks (Graph500, Gunrock).

**Success Criteria**:
1. CSR (Compressed Sparse Row) graph representation on GPU — load from edge list
2. GPU BFS: level-synchronous BFS on a graph with 100K+ vertices, correctness matches
   CPU reference
3. GPU PageRank: iterative until convergence (delta < 1e-6), matches CPU reference within
   1e-4
4. GPU speedup >= 3x over CPU reference for graphs with 1M+ edges
5. Standalone example with synthetic graph generation (RMAT or Erdos-Renyi)

**Estimated Complexity**: Medium. CSR construction is CPU-side. BFS frontier expansion is
a well-studied GPU kernel. PageRank is iterative SpMV (sparse matrix-vector multiply),
which connects to the sparse operations theme.

**Dependencies**: None. CSR is a simple data structure (two arrays: row_offsets, col_indices).

**Technical Notes**:
- BFS frontier: each iteration, threads scan the frontier and discover new vertices. Atomic
  operations mark visited vertices. This exercises GPU atomics heavily — a strength of the
  framework (gpu-atomics crate).
- PageRank: essentially iterative SpMV (y = A*x) with normalization. The sparse matrix
  pattern is shared with proposed sparse operations epic.
- These algorithms are memory-bound, not compute-bound. This is a different performance
  profile from GEMM/CNN and tests the framework's memory subsystem handling.

---

### 4. `sparse-ops` — Sparse Matrix Operations: SpMV, SpGEMM, Sparse Solvers
**Priority**: P3 (Near-term)

**WHY**: Sparse operations are fundamental to scientific computing, graph analytics, and
real-world ML (where data matrices are often sparse). SpMV (sparse matrix-vector multiply)
is the #1 kernel in scientific HPC. Implementing CSR SpMV on GPU proves the framework
handles indirect memory access patterns and irregular parallelism.

**Success Criteria**:
1. CSR SpMV kernel: y = A*x where A is sparse in CSR format, correctness < 1e-5 vs dense
2. CSR SpGEMM (sparse-sparse multiply): C = A*B, both sparse, output in CSR format
3. Conjugate gradient solver: solve Ax=b for sparse symmetric positive-definite A,
   convergence within 1000 iterations
4. Performance: SpMV throughput >= 5 GFLOP/s for large sparse matrices (1M rows, ~10 nnz/row)
5. Reuse CSR format from graph-algorithms epic if both are implemented

**Estimated Complexity**: Medium. CSR SpMV is a well-studied GPU kernel (one thread per row,
or one warp per row for load balancing). SpGEMM is harder (dynamic output size). CG solver
is iterative composition of SpMV + dot product + axpy.

**Dependencies**: Benefits from graph-algorithms CSR format if done together.

---

### 5. `gpu-sort-db` — Database Primitives: GPU Sort, Hash Join, Aggregation
**Priority**: P3 (Near-term)

**WHY**: GPU-accelerated databases are an emerging industry trend (RAPIDS cuDF, BlazingSQL,
HeavyDB). Sort, hash join, and group-by aggregation are the three most important database
operators. Implementing these proves the framework handles data-intensive operations with
complex data movement patterns (partitioning, radix scatter, hash table probing).

**Success Criteria**:
1. GPU radix sort: sort 10M 32-bit keys in < 50ms, correctness matches std::sort
2. GPU hash join: join two tables (1M rows x 100K rows) on integer key, output matching
   pairs
3. GPU group-by aggregation: SUM/COUNT/AVG by group key, 1M rows with 1K groups
4. GPU speedup >= 5x over single-threaded CPU for radix sort (10M keys)
5. Standalone example demonstrating a simple SQL-like query pipeline

**Estimated Complexity**: Large. Radix sort requires multi-pass prefix sum + scatter.
Hash join requires hash table construction + probing. Both are well-documented but have
many implementation details.

**Dependencies**: None. Pure compute kernels operating on arrays.

**Technical Notes**:
- Radix sort: this is the canonical GPU sort algorithm. Requires prefix sum (scan) as a
  building block — scan is also useful for many other algorithms.
- Hash join: build phase creates a hash table (open addressing or chaining), probe phase
  looks up keys. The hash table pattern is similar to GPU HashMap concepts explored in
  previous brainstorms.
- Group-by: essentially a hash-based aggregation — each thread hashes its key, atomically
  updates the accumulator. Exercises GPU atomics.

---

### 6. `image-processing` — Image Processing: Blur, Edge Detection, Histogram
**Priority**: P4 (Long-term)

**WHY**: Image processing is a classic GPU workload that exercises 2D stencil patterns
(convolution with small kernels), histogram computation (atomic scatter), and pixel-parallel
operations. It is highly visual and produces immediately compelling demos. It also connects
to the existing YOLO/ResNet vision pipeline — a user might want to preprocess images on GPU
before inference.

**Success Criteria**:
1. GPU Gaussian blur (separable, 2-pass): process 1080p image in < 5ms
2. GPU Sobel edge detection: gradient magnitude computation, correct output
3. GPU histogram (256 bins): atomic-based histogram of 1080p grayscale image
4. GPU image resize (bilinear interpolation): arbitrary scale factor
5. End-to-end demo: load PPM image, blur + edge detect, save result (uses existing file I/O!)

**Estimated Complexity**: Small-Medium. All operations are well-known stencil or
element-wise patterns. The PPM I/O is already in the codebase (bus.ppm exists in models/).

**Dependencies**: Existing GPU file I/O for loading/saving images. Connects to YOLO
inference pipeline.

**Technical Notes**:
- The bus.ppm file already exists in models/ — immediate test data available.
- Gaussian blur with separable kernels is a 1D convolution in two passes — connects to
  FFT convolution if both epics are done.
- Histogram demonstrates atomic scatter pattern, different from reduction.
- This epic produces the most visually compelling demo output of any proposal.

---

### 7. `int4-gpt2` — Complete INT4 GPT-2 Inference (Reopen)
**Priority**: P3 (Near-term)

**WHY**: The INT4 kernel and quantizer already work (4.7x compression). The missing piece
is integrating Int4Linear into the GPT-2 model and generating coherent text. This is a
small amount of work to close a gap that has been explicitly tracked as deferred. It also
provides a dramatic demo: GPT-2 running at 1/4 the memory footprint with coherent output.

**Success Criteria**:
1. Int4Linear layer integrated into GPT-2 model (replace Linear layers with Int4Linear)
2. GPT-2 INT4 inference produces coherent text (top-5 match f32 for >= 2/3 prompts)
3. Model memory reduced >= 3x from f32 baseline
4. Per-token latency measured and documented

**Estimated Complexity**: Small. The Int4Linear layer and quantized weights already exist.
The work is integration: loading INT4 weights into the GPT-2 model and routing forward()
through Int4Linear.

**Dependencies**: Existing int4-quantization work (kernel, quantizer, model_int4.safetensors).

---

### 8. `tiled-dp4a` — Tiled INT8 GEMM for Competitive Quantized Inference (Reopen)
**Priority**: P4 (Long-term)

**WHY**: The naive dp4a kernel works but is 7-10x slower than tiled f32 GEMM. Shared memory
tiling for INT8 would make quantized inference practically useful, not just a correctness
demonstration. This also unblocks full INT8 GPT-2 inference.

**Success Criteria**:
1. Tiled dp4a INT8 GEMM with shared memory blocking (tile size >= 32x32)
2. Performance within 2x of f32 tiled GEMM for same dimensions
3. GPT-2 full INT8 inference: load quantized weights, generate coherent text
4. Per-token latency competitive with f32 (< 2x overhead)

**Estimated Complexity**: Medium. Shared memory tiling pattern is identical to existing f32
tiled GEMM — the difference is packing INT8 values into u32 for dp4a and handling INT32
accumulation.

**Dependencies**: Existing int8-inference dp4a kernel as starting point.

---

### 9. `gpu-crypto` — Cryptographic Primitives on GPU: Hashing, AES, Merkle Trees
**Priority**: P5 (Long-term / Speculative)

**WHY**: GPU-accelerated cryptography is used in cryptocurrency mining, password cracking,
zero-knowledge proofs, and secure data processing. SHA-256 and AES are compute-intensive
and embarrassingly parallel per-block. Merkle trees combine hashing with tree reduction.
This is a niche but high-impact GPU use case that no Rust GPU framework covers.

**Success Criteria**:
1. GPU SHA-256: hash 1M messages in parallel, correctness matches CPU reference
2. GPU AES-128 encryption: ECB mode, 1M blocks in parallel
3. GPU Merkle tree: construct tree from 1M leaf hashes, verify root
4. GPU throughput >= 1 GH/s (giga-hashes/second) for SHA-256

**Estimated Complexity**: Medium. SHA-256 and AES are fixed algorithms with well-known GPU
implementations. The challenge is bit manipulation and register pressure, not algorithmic.

**Dependencies**: None. Pure compute.

---

### 10. `monte-carlo` — Monte Carlo Simulations: Option Pricing, Pi Estimation, Ray Tracing
**Priority**: P4 (Long-term)

**WHY**: Monte Carlo methods are a major GPU workload in finance (option pricing), physics
(particle transport), and graphics (ray tracing). They exercise GPU random number generation,
independent sampling, and reduction for aggregation. This category is distinct from both NN
and classical ML — it proves the framework handles stochastic simulation.

**Success Criteria**:
1. GPU PRNG: xoshiro256++ or similar, 1M independent streams, passes basic statistical tests
2. Black-Scholes Monte Carlo: price a European call option with 10M paths, result within
   1% of analytical Black-Scholes price
3. Pi estimation: 100M random points, result accurate to 4+ decimal places
4. GPU speedup >= 20x over CPU for 10M-path Monte Carlo

**Estimated Complexity**: Small-Medium. The core is per-thread random sampling and reduction.
PRNG state management per thread is the main implementation detail.

**Dependencies**: None. Pure compute.

---

## Epics Worth Reopening

### High ROI: `int4-gpt2` (INT4 GPT-2 inference)
**Effort**: Small (integration only, kernel + weights exist)
**Impact**: High — closes a tracked gap, provides dramatic memory reduction demo
**Recommendation**: Include as P3, can be done in 1-2 sessions

### Medium ROI: `tiled-dp4a` (Tiled INT8 GEMM)
**Effort**: Medium (shared memory tiling, new kernel)
**Impact**: Medium — makes INT8 inference practical, not just correct
**Recommendation**: P4, do after higher-priority breadth work

### Low ROI: `kernel-fusion` / `kernel-fusion-v2` (Fused GEMM + activation in GPT-2)
**Effort**: Medium (needs fused_prepadded_b variant)
**Impact**: Low — the pre-padded B path is already faster than unfused, and 15% speedup
is marginal compared to the 2-3x gains from quantization or Tensor Core
**Recommendation**: Skip. The ROI is poor and the problem is that pre-padding is already
a form of fusion.

### Low ROI: `persistent-kernel` (Dispatch latency < 10us)
**Effort**: High (architectural change needed)
**Impact**: Low — kernel re-launch at 9.5us already beats the persistent kernel's 16.1us.
The experiment demonstrated that CUDA's launch overhead is lower than mapped-memory
polling overhead.
**Recommendation**: Skip. The result is a valid negative finding.

### Low ROI: `gpu-rag` (Hostcall round-trips)
**Effort**: Medium (rewrite pipeline to use raw hostcall kernel instead of nn API)
**Impact**: Low — the nn API orchestration approach works well and is more practical
**Recommendation**: Skip. The current architecture is actually better than the original
vision.

---

## Priority Ordering

### Tier 1 — Immediate (Next Session)
| Rank | Epic | Rationale |
|------|------|-----------|
| **1** | `classical-ml` | User explicitly requested tree models. Highest vision alignment (non-NN GPU compute). Medium complexity, no dependencies. |

### Tier 2 — Near-term (Next Week)
| Rank | Epic | Rationale |
|------|------|-----------|
| **2** | `fft-signal` | Fundamental GPU workload, exercises entirely different compute pattern (butterfly, shared memory stages). |
| **3** | `graph-algorithms` | Irregular memory access — the hardest test for any GPU framework. Proves generality. |
| **4** | `int4-gpt2` (reopen) | Small effort to close a tracked gap. Dramatic demo value (4-bit GPT-2). |
| **5** | `sparse-ops` | Scientific computing foundation. Shares CSR format with graph-algorithms. |

### Tier 3 — Long-term (Next 2 Weeks)
| Rank | Epic | Rationale |
|------|------|-----------|
| **6** | `gpu-sort-db` | Emerging GPU use case, exercises complex data movement patterns. |
| **7** | `image-processing` | Visually compelling, connects to existing YOLO pipeline, uses existing bus.ppm. |
| **8** | `monte-carlo` | Stochastic simulation, new compute category (PRNG + reduction). |
| **9** | `tiled-dp4a` (reopen) | Makes INT8 practical. Medium effort. |
| **10** | `gpu-crypto` | Niche but unique — no other Rust GPU framework covers this. |

---

## Key Insight

The project has saturated the neural network axis: inference, training, LoRA, ONNX,
quantization, autograd, differentiable physics. Each additional NN feature yields
diminishing returns for proving framework generality. The maximum impact comes from
**breadth across GPU compute domains**: classical ML, signal processing, graph algorithms,
database operations, scientific computing.

`classical-ml` is the clear #1 priority because:
1. The user explicitly requested it (tree models, XGBoost-style)
2. It exercises fundamentally different compute patterns (irregular branching vs. regular GEMM)
3. It proves the framework is not an NN-only tool
4. The implementation complexity is tractable (decision trees are simple per-thread code)

The combination of `classical-ml` + `fft-signal` + `graph-algorithms` would cover three
entirely new GPU compute categories in a single week, transforming the project's breadth
story from "deep in NN" to "covers ML, signal processing, graph analytics, NN, physics,
and RAG."
