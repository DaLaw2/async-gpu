# docs-examples.1: Audit — post-607 features lacking standalone examples

## Summary

Audited all 24 standalone examples (14 in `examples/std/`, 10 in `examples/hostcall/`).
Cross-referenced against 8 post-607 features. Found **5 features completely missing
standalone examples**, 2 existing examples missing README.md, and a dependency hygiene
issue (11 examples import `gpu_host` directly instead of the `async_gpu` facade crate).

## Findings

### 1. Existing examples inventory

#### examples/std/ (14 examples)
| Example            | Demos                                      | README | Dep          |
|--------------------|-------------------------------------------|--------|--------------|
| benchmark          | SGEMM/cuBLAS, memory-bound, conv2d, flash attn | Yes | gpu-host     |
| cifar-train        | Conv2d + FC training, autograd             | Yes    | gpu-host     |
| diff-physics       | Differentiable simulation, Euler, gradient opt | Yes | gpu-host     |
| dynamic-control    | GPT-2 variable-length gen, data-dep branching | Yes | gpu-host     |
| gpt2-inference     | GPT-2 124M inference, nn module            | Yes    | gpu-host     |
| gpt2-lora          | LoRA fine-tuning, autograd tape            | Yes    | gpu-host     |
| gpu-rag            | RAG pipeline (embed+search+gen), bench-fused | Yes  | gpu-host     |
| graph-algorithms   | BFS, PageRank (cudarc raw)                 | Yes    | cudarc only  |
| mnist-cnn          | 2-layer CNN training, GPU conv2d           | Yes    | gpu-host     |
| mnist-train        | MLP training, autograd                     | Yes    | gpu-host     |
| monte-carlo        | Xoshiro256++, Pi, Black-Scholes (cudarc raw) | Yes | cudarc only  |
| resnet-cifar       | ResNet-18 inference + mini-resnet training | Yes    | gpu-host     |
| thread-demo        | thread::spawn, JoinHandle, warp reuse      | Yes    | gpu-host     |
| yolo-detect        | YOLOv8-nano detection                      | Yes    | gpu-host     |

#### examples/hostcall/ (10 examples)
| Example              | Demos                                    | README | Dep        |
|----------------------|------------------------------------------|--------|------------|
| async-io             | File I/O from GPU, pipelined compute     | Yes    | gpu-host   |
| async-pipeline       | WarpFuture branching + pipelined I/O     | Yes    | gpu-host   |
| gpu-channels         | Oneshot, MPSC, GpuExecutor               | **NO** | async-gpu  |
| hello-gpu            | println, file I/O, thread::spawn         | Yes    | async-gpu  |
| parallel-search      | 32-lane warp-cooperative grep            | Yes    | async-gpu  |
| structured-concurrency | BlockScope, spawn_all, nested scopes  | **NO** | async-gpu  |
| tcp-echo             | GPU-initiated TCP networking             | Yes    | gpu-host   |
| tokio-offload        | GpuTask async launch + event streaming   | Yes    | async-gpu  |
| vector-math          | SAXPY, dot product, softmax              | Yes    | async-gpu  |
| warp-cooperative     | cooperative(), map(), reduce(), matmul   | Yes    | async-gpu  |

### 2. Post-607 features — coverage gap analysis

| Feature              | Standalone Example? | In-crate Example? | Test Coverage? | Notes |
|----------------------|--------------------|--------------------|----------------|-------|
| **transparent-data** (GpuArray<T>) | **MISSING** | No | Yes (gpu_array.rs unit tests) | GpuArray is exported from facade crate; no example shows zero-explicit-transfer usage |
| **auto-fusion**      | **MISSING**        | No                 | Yes (fusion.rs unit tests, tests_cnn.rs) | FusionPlan detection exists; gpu-rag has `--bench-fused` but that's buried, not standalone |
| **dyn-dispatch**     | **MISSING**        | No                 | Yes (kernel tests: test_gpu_dyn_trait, Box<dyn Trait>, dyn-perf benchmark) | Thorough kernel tests but no user-facing example |
| **auto-tuning**      | **MISSING**        | No                 | Yes (auto_tune_bench.rs) | TuningCache + AutoTuner API exists; no example shows warmup-based parameter search |
| **par_iter**         | **MISSING**        | No                 | Yes (par_iter_demo.rs kernels, tests_par_iter.rs) | 6 kernel demos exist but no standalone example with host driver |
| **gpu_test macro**   | **MISSING (sort of)** | N/A              | Yes (used in gpu-test-harness) | The macro itself is well-documented; could use a "how to write GPU tests" example |
| **structured-concurrency** | **EXISTS** (hostcall/structured-concurrency) | No | Yes | Example is current (BlockScope, spawn_all, GridScope). Missing README.md |
| **tiered-memory** (SharedRef/GlobalRef) | **MISSING** | No | Yes (compile_fail tests) | Used internally in scope.rs; no standalone showcase of SharedRef/GlobalRef typed pointers |

### 3. Dependency hygiene issues

11 of 24 examples import `gpu_host` directly instead of the `async_gpu` facade crate:
- std/: benchmark, cifar-train, gpt2-lora, gpu-rag, mnist-cnn, mnist-train, resnet-cifar, thread-demo
- hostcall/: async-io, async-pipeline, tcp-echo

This is a code-level issue (the nn features require `gpu-host` with feature flags, and the
facade crate may not re-export everything). Whether to migrate is a design decision.

### 4. Stale patterns

- No examples use `GpuVec` (it's only used in crate-internal examples `gpuvec_pipeline.rs`
  and `unified_pipeline.rs`). No staleness issue in standalone examples.
- No examples use `GpuArray` at all — this is the gap, not staleness.
- `graph-algorithms` and `monte-carlo` use raw `cudarc` without any async-gpu API — these
  are "raw CUDA" examples, not async-gpu showcases.

### 5. Missing README.md files

Two examples lack README.md:
1. `examples/hostcall/gpu-channels/` — no README
2. `examples/hostcall/structured-concurrency/` — no README

All 22 other examples have README.md.

## Open Questions

1. Should `transparent-data` example use `GpuArray` exclusively, or show the progression
   from `GpuVec` (explicit) to `GpuArray` (transparent)?
2. Should `auto-fusion` example be nn-centric (tape fusion detection) or show a simpler
   elementwise chain auto-fused into a single kernel?
3. Should `par_iter` example be a standalone binary in `examples/hostcall/` (since par_iter
   runs on GPU kernel side), or in `examples/std/` (since it could be driven from host)?
4. Should the `gpu_test` example be a test crate showing how to use `#[gpu_test]`, or a
   standalone binary? The macro generates `#[test]` functions — a test crate seems more natural.
5. For `dyn-dispatch`, should the example show `&dyn Trait` only, or also `Box<dyn Trait>`
   and the performance comparison?
