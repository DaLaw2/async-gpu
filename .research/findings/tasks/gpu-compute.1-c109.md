# gpu-compute.1: Survey — GPU autonomous complex compute capabilities
**Cycle**: 109 | **Theme**: gpu-compute | **Kind**: investigation | **Status**: done

## Summary

GPU autonomous compute — where the GPU drives multi-step computation without per-step host orchestration — requires four capabilities beyond the current async_gpu hostcall I/O stack: (1) device-side memory management (bump/slab/pool allocators in kernel code), (2) persistent kernel scheduling with work queues, (3) GPU-local data pipeline orchestration (load → compute → store without round-tripping to CPU), and (4) large working-set management for model weights and intermediate buffers. The current async_gpu stack provides excellent foundations (hostcall protocol, warp-cooperative async, sideband bulk transfer) but the gap to autonomous inference centers on compute throughput primitives (matrix multiply, attention kernels) and GPU-side memory lifecycle management.

## Findings

### Q: What does 'autonomous compute' require beyond current hostcall I/O?

A: The current async_gpu model is "GPU requests services from CPU" — the GPU is a client that asks the host to open files, read data, print messages, etc. True autonomous compute inverts this: the GPU is the controller that decides what to compute next, manages its own data, and only contacts the host for external I/O that physically requires it (disk, network).

Concretely, autonomous compute requires:

1. **Persistent kernel execution**: The GPU kernel does not exit between steps. It runs a main loop that pulls work from queues, executes it, and repeats. This eliminates kernel launch overhead (5-15 μs per launch) and lets the GPU maintain state across operations. The literature calls these "persistent threads" or "megakernels" — thread blocks launched to fill all SMs, executing a work-stealing loop.

2. **Device-side control flow**: The GPU decides what to do next based on results of previous steps. async_gpu's WarpFuture state machine already provides this — a 16-state or 20-state WarpFuture is exactly a persistent kernel with GPU-driven control flow. This is a strength of the current design.

3. **Compute kernels**: For inference, the GPU needs GEMM (matrix multiply), softmax, layer normalization, RoPE, and attention computation. These are pure GPU operations that do not need hostcall — they are standard CUDA/PTX math. The gap is that async_gpu has no math kernel library yet.

4. **Data staging**: The GPU needs to load model weights and input data into GPU memory, keep them resident, and access them across multiple compute steps. Currently, the sideband buffer provides 1 MB of bulk transfer, but inference models require GB-scale weight storage and dynamic KV cache management.

5. **Minimal host dependency**: The host should only be involved for (a) initial model weight loading from disk to GPU VRAM, (b) receiving inference requests, and (c) returning results. All intermediate computation (dozens of transformer layers, sampling, etc.) stays GPU-side.

**Confidence**: high

### Q: What memory management is needed on GPU?

A: GPU-side memory management has three tiers of complexity:

**Tier 1 — Static pre-allocation (current async_gpu)**:
The host pre-allocates all buffers (hostcall buffer, sideband, kernel parameters) and passes pointers to the kernel. The GPU uses fixed-layout memory. This is what async_gpu does today and it works for I/O-centric workloads.

**Tier 2 — Bump allocator + arena (needed for compute pipelines)**:
For multi-step compute, intermediate buffers (e.g., attention scores, layer outputs) have known sizes at pipeline construction time. A simple bump allocator on a pre-allocated arena (similar to the existing sideband `sideband_alloc()`) can manage these. The arena is allocated by the host, but the GPU manages sub-allocation. This is the immediate next step for async_gpu.

**Tier 3 — Dynamic allocation with reuse (needed for inference)**:
LLM inference requires dynamic KV cache management where memory blocks are allocated per-request and freed when requests complete. vLLM's PagedAttention eliminates 60-80% of KV cache memory waste by treating GPU memory like OS virtual memory — fixed-size pages mapped on demand. A GPU-side slab allocator (fixed-size blocks from a pool) would enable this. CUDA's built-in `malloc()` on device is available but slow (global memory heap, only 8 MB default, poor performance due to serialized access). Custom allocators like halloc or a Rust-native slab allocator are needed for production throughput.

**Key sizes for inference**:
- Model weights: 1-70+ GB (static, loaded once)
- KV cache per token per layer: ~256 bytes (FP16, 128-dim heads) × 2 (K+V) = 512 bytes
- For 32 layers, 4K context: 32 × 4096 × 512 = 64 MB per request
- For 100 concurrent requests at 4K context: ~6.4 GB KV cache

**What async_gpu needs**: Extend the sideband model to a general GPU memory arena with bump allocation (Tier 2), then add slab allocation for KV cache blocks (Tier 3). The host allocates the arena via `cuMemAlloc`; the GPU manages sub-allocation via atomic bump pointer or free-list.

**Confidence**: high

### Q: What scheduling/control flow is needed?

A: Three scheduling patterns emerge from the literature and from how inference engines work:

**Pattern 1 — Sequential pipeline (current WarpFuture)**:
A single warp executes a linear sequence of steps: load → compute → store. async_gpu's WarpFuture state machine already implements this with up to 20 states. For a single inference request through a single transformer layer, this pattern works directly. The gap is scaling to 32+ layers sequentially (would need a state machine with ~200 states or a loop construct in WarpFuture).

**Pattern 2 — Persistent work queue**:
Multiple warps sit in a loop pulling tasks from a shared work queue. This is the "persistent threads" or "megakernel" pattern. A coordinator (could be warp 0) dispatches tasks — e.g., "compute layer 5 attention for batch slot 3". Workers pull from the queue, execute, and signal completion. This requires:
- A lock-free work queue (async_gpu's CAS-based stacks could be adapted)
- Task descriptors containing: operation type, input/output buffer pointers, dimensions
- Completion signaling (atomic counter or flag per task)

**Pattern 3 — Continuous batching scheduler (for serving)**:
vLLM and TensorRT-LLM run a scheduler that, every forward pass, decides which requests to process. In a GPU-autonomous model, this scheduler would run on the GPU itself. It needs to:
- Track which requests are in prefill vs decode phase
- Allocate/free KV cache blocks per request
- Decide batch composition (which requests to run together for maximum SM utilization)
- Handle request arrival and completion

For async_gpu, the immediate path is **Pattern 1 extended with loops** — allow WarpFuture state machines to loop over transformer layers. Pattern 2 (work queues) would be the next step for multi-request serving. Pattern 3 is the long-term goal.

**Control flow gap**: The current WarpFuture proc macro (`#[warp_async]`) is being extended with if/else/loop/match in the warp-cooperative-async epic. Once loops work, a WarpFuture can iterate over N transformer layers without needing N×states — this directly enables Pattern 1 for inference.

**Confidence**: high

### Q: What are the gaps between current async_gpu and autonomous inference?

A: Mapping current capabilities to inference requirements:

| Requirement | Current Status | Gap |
|---|---|---|
| Load model weights from disk to GPU | Bulk read via sideband (1 MB limit) | Need larger sideband or host-side `cuMemAlloc` + `cuMemcpy` for multi-GB transfers |
| Store weights in GPU VRAM | Host pre-allocates, passes pointer | Works — host allocates weight buffer, GPU accesses via pointer |
| Matrix multiplication (GEMM) | Not implemented | **Critical gap** — need PTX WMMA/MMA instructions or inline PTX for Tensor Cores |
| Softmax / LayerNorm / RoPE | Not implemented | Medium gap — element-wise ops, straightforward in Rust + PTX |
| Attention computation | Not implemented | **Critical gap** — need FlashAttention-style tiled implementation |
| KV cache management | Not implemented | Need GPU-side slab allocator on pre-allocated arena |
| Multi-step pipeline control | WarpFuture state machine (up to 20 states) | Need loop support (warp-cfg epic) to iterate over layers |
| Batch scheduling | Not applicable yet | Long-term — GPU-side scheduler for concurrent requests |
| Token sampling | Not implemented | Small gap — argmax/top-k on GPU is well-understood |
| Tokenizer | Not applicable on GPU | Use hostcall to CPU-side tokenizer (reasonable) |
| Result return to host | Hostcall protocol works | Works — GPU signals result via hostcall, host reads it |

**Critical path items**:
1. **Tensor Core GEMM**: This is ~90% of inference compute. Without matrix multiply, nothing else matters. Requires Tensor Core instructions (WMMA/MMA via inline PTX). The Rust `nvptx64` target can emit these via `asm!()`.
2. **FlashAttention-equivalent**: Tiled attention with shared memory to avoid O(N²) memory. Complex but well-documented algorithm.
3. **WarpFuture loop support**: To iterate over transformer layers without state explosion.
4. **Large buffer management**: Host allocates multi-GB buffers, GPU manages sub-regions.

**What's feasible with current stack**:
- Single transformer layer forward pass (with new GEMM + attention kernels)
- GPU-autonomous file processing pipelines (already demonstrated)
- Simple compute pipelines with hostcall I/O at boundaries

**What needs new infrastructure**:
- Tensor Core integration (WMMA/MMA inline PTX)
- Shared memory management for tiled algorithms
- GPU-side memory arena with slab allocator
- Multi-warp coordination for parallel layer computation

**Confidence**: high

### Q: How do existing GPU inference engines structure their compute?

A: Inference engines use a fundamentally different architecture from async_gpu's current model. Here is how the major engines work:

**TensorRT / TensorRT-LLM**:
- **Compilation approach**: Takes a neural network graph (ONNX or Python model definition), applies graph-level optimizations (layer fusion, quantization, kernel selection), and produces an optimized engine.
- **Layer fusion**: Combines convolution + bias + ReLU into a single kernel, eliminating intermediate memory writes. For transformers, fuses QKV projection, attention, and output projection where possible.
- **Kernel auto-tuning**: Benchmarks multiple kernel implementations per operation for the target GPU and specific tensor dimensions, selecting the fastest.
- **Execution flow**: Host-driven — the CPU runtime calls `enqueue()` which launches a sequence of pre-compiled CUDA kernels. Each transformer layer is ~3-5 kernel launches (fused). The GPU executes kernels, CPU handles scheduling.
- **Overlap scheduler**: TensorRT-LLM's overlap scheduler launches GPU work for step N+1 before CPU finishes processing step N's results, achieving up to 22% throughput improvement. The CPU and GPU run in parallel.
- **Memory**: Static tensor allocation at compile time. KV cache uses paged blocks managed by host-side KVCacheManager.
- **CUDA Graphs**: Captures sequences of kernel launches into a graph that can be replayed with minimal CPU overhead, using padding to maximize graph cache hit rate.

**vLLM**:
- **PagedAttention**: Core innovation — KV cache stored in fixed-size blocks (default 16 tokens/block) mapped like OS virtual memory pages. Eliminates 60-80% memory waste from fragmentation.
- **Continuous batching**: Scheduler runs every iteration (not every batch). Finished sequences are immediately evicted, new requests start without waiting for the batch to complete.
- **Hierarchical KV cache**: GPU memory → CPU memory → external storage. Cache misses cascade through the hierarchy.
- **Execution**: Host-driven. Python scheduler decides batch composition, then launches CUDA kernels (FlashAttention/FlashInfer, GEMM via cuBLAS or custom kernels).
- **FlashInfer integration**: JIT-compiled attention kernels specialized for current model config and sequence length. Block-sparse KV cache formats. Load-balanced scheduling across GPU threads. Reduces inter-token latency by 29-69%.

**Common pattern across all engines**:
1. Host receives requests and tokenizes input
2. Host scheduler decides which requests to batch together
3. Host launches GPU kernels for one forward pass (prefill or decode)
4. GPU executes: embedding → N × (attention + FFN) → output projection → logits
5. Host reads logits, runs sampling (or GPU-side sampling in some engines)
6. Host updates KV cache metadata, decides next batch
7. Repeat until all requests complete

**Key insight for async_gpu**: All current inference engines are host-orchestrated. The GPU is a compute accelerator, not an autonomous agent. The CPU decides what to compute; the GPU does the math. async_gpu's model (GPU as agent, host as service provider) is fundamentally different and potentially more efficient for certain workloads because:
- Zero kernel launch overhead (persistent kernel)
- No CPU-GPU synchronization between layers (GPU manages its own pipeline)
- GPU-driven control flow can adapt in real-time (e.g., early exit, speculative decoding decisions on-GPU)

However, this model faces challenges:
- Tensor Core utilization may be lower without TensorRT's auto-tuning
- Memory management complexity moves to GPU (limited tooling)
- Debugging GPU-side schedulers is much harder than CPU-side ones

**Confidence**: high

## Unexpected Discoveries

1. **CUDA Cluster Launch Control (Blackwell)**: NVIDIA's newest architecture provides hardware-level work stealing across thread block clusters. This is exactly the primitive needed for GPU-autonomous scheduling. Thread blocks can cancel and redistribute work without host intervention. This is a hardware validation of the persistent kernel approach async_gpu is building in software.

2. **Device-side task graph execution**: Recent research (ICS 2025) demonstrates persistent megakernels that execute entire task graphs on-GPU, with warps as workers drawing tasks from queues. This is architecturally similar to what WarpFuture + work queues would provide. The key finding is that GPU-to-GPU communication via NVLink can happen entirely without CPU involvement, and iteration loops inside persistent kernels ("moving the loop inside the kernel") enable full application execution on-device.

3. **GPU-initiated resource allocation**: GPUs can now request additional resources (memory, even other GPUs) via device-side callbacks during kernel execution. This aligns with async_gpu's hostcall model — the GPU requesting `SERVICE_MALLOC` from the host is exactly this pattern, just implemented via mapped memory instead of hardware callbacks.

4. **vLLM's cost impact**: Stripe achieved 73% inference cost reduction by migrating to vLLM, handling 50M daily API calls on 1/3 of their previous GPU fleet. This demonstrates the value of efficient scheduling and memory management — areas where async_gpu's zero-overhead persistent kernel model could potentially do even better.

5. **CUDA `alloca` (SM 3.0+)**: CUDA 11.3+ supports `alloca` for stack-based dynamic allocation in device code. This is faster than device `malloc` and could be useful for small temporary buffers in compute kernels. Available via inline PTX in Rust.

## Open Questions

1. **Tensor Core access from Rust PTX**: Can `asm!()` blocks emit `wmma.load`, `wmma.mma`, `wmma.store` instructions correctly on the `nvptx64` target? This is the single most important technical question for inference feasibility. Need an experiment.

2. **Shared memory from Rust**: Tiled algorithms (FlashAttention, optimized GEMM) require shared memory (`__shared__` in CUDA C). In PTX this is `.shared` address space. Can Rust's `nvptx64` target allocate and use shared memory? Need investigation.

3. **Register pressure**: Persistent kernels with complex state machines may use many registers, reducing occupancy (fewer warps per SM). What is the register usage of current WarpFuture kernels? Is this a concern for compute-heavy extensions?

4. **Multi-warp coordination for GEMM**: A single matrix multiply for inference (e.g., 4096×4096 weight matrix) requires cooperation across hundreds of warps. How does this compose with the WarpFuture model where each warp has its own state machine?

5. **Weight loading strategy**: Should model weights be loaded via the existing hostcall bulk transfer (many round-trips) or should the host pre-load weights into GPU memory before launching the persistent kernel? The latter is simpler and more efficient but requires the host to know the memory layout in advance.

## Impact on Downstream Tasks

- **gpu-compute.2** (proposed): Experiment — Tensor Core GEMM via inline PTX `wmma.*` instructions. This is the critical path item. If WMMA works from Rust `asm!()`, inference becomes feasible.
- **gpu-compute.3** (proposed): Experiment — Shared memory allocation and usage from Rust `nvptx64`. Required for FlashAttention and tiled GEMM.
- **gpu-compute.4** (proposed): Design — GPU memory arena with bump allocator, extending sideband model to general-purpose allocation.
- **gpu-compute.5** (proposed): Design — Persistent kernel work queue protocol, reusing hostcall CAS-based stack primitives for GPU-internal task distribution.
- **warp-cfg tasks**: Loop support in `#[warp_async]` is a prerequisite for iterating over transformer layers. This epic's completion directly unblocks inference pipeline construction.
- **Decision gate**: After gpu-compute.2 and gpu-compute.3 results are in, a decision is needed: is GPU-autonomous inference via async_gpu's model viable, or should the project focus on GPU-autonomous I/O pipelines (which are already demonstrated) and leave inference to existing engines?
