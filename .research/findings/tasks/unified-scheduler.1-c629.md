# unified-scheduler.1 — Scheduler Type System Design

Investigation: CpuScheduler / GpuScheduler / AutoScheduler architecture for unified-runtime.

## Executive Summary

The scheduler is NOT a magic "run this closure on GPU" system. GPU kernels require
pre-compiled PTX — you cannot send an arbitrary Rust closure to the GPU at runtime.
Instead, the scheduler is a **work-routing abstraction** that decides WHERE pre-registered
work units execute (CPU thread pool vs GPU kernel launch), with a unified async interface
that hides the routing decision from the user.

The key insight: the project already has two execution models that produce `impl Future`:
1. **CPU work**: `tokio::task::spawn_blocking(|| ...)` → `JoinHandle<T>`
2. **GPU work**: `GpuTask::launch(func, config, args)` → `Result<()>` (async)

The scheduler trait unifies these behind a single `submit()` interface.

## Proposed Scheduler Trait Hierarchy

### Core Trait

```rust
/// Describes a unit of work that can be scheduled on CPU or GPU.
///
/// Work items are NOT closures — they are typed descriptors that the
/// scheduler interprets. This is critical because GPU work must reference
/// pre-compiled kernel functions, not arbitrary closures.
pub trait WorkItem: Send + 'static {
    /// The result type produced by this work item.
    type Output: Send + 'static;
}

/// A CPU-executable work item (just a closure).
pub struct CpuWork<F, T> {
    f: F,
    _phantom: PhantomData<T>,
}

/// A GPU-executable work item (kernel name + config + data buffers).
pub struct GpuWork {
    kernel: &'static str,
    ptx_module: &'static str,  // which PTX module contains the kernel
    grid: (u32, u32, u32),
    threads: u32,
    // Data is passed via GpuBuffer handles, not embedded in the work item
}

/// The scheduler trait — routes work to an execution target.
///
/// Schedulers are async-first: `submit()` returns a future that resolves
/// when the work completes.
pub trait Scheduler: Send + Sync + 'static {
    /// Submit a CPU work item (closure).
    fn submit_cpu<F, T>(&self, work: F) -> impl Future<Output = Result<T>>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static;

    /// Submit a GPU work item (kernel dispatch).
    fn submit_gpu(&self, work: GpuWork, buffers: &[&GpuBuffer]) -> impl Future<Output = Result<()>>;
}
```

### Why NOT a single generic `submit<W: WorkItem>(w: W)`

A unified `submit()` that accepts both CPU and GPU work sounds elegant but
breaks down because:

1. **GPU work is fundamentally different from CPU work**: GPU work is a kernel
   name + launch config + buffer handles. CPU work is a closure. These have
   different type signatures and different execution mechanics.

2. **The AutoScheduler needs to inspect work type**: To route work, the scheduler
   must know whether it CAN run on GPU (has a kernel) or MUST run on CPU (I/O).
   A single `WorkItem` trait hides this information.

3. **The user already knows the work type**: When writing `par_iter().map(|x| x * 2.0)`,
   the user knows this is data-parallel compute. When writing `File::read()`, they
   know it's I/O. Forcing both through one trait adds ceremony for no benefit.

### Recommended: Two-method trait with unified result type

```rust
pub trait Scheduler: Send + Sync {
    /// Execute a CPU-bound function on the thread pool.
    async fn cpu<F, T>(&self, f: F) -> Result<T, SchedulerError>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static;

    /// Launch a GPU kernel with the given configuration and buffers.
    async fn gpu(
        &self,
        kernel: KernelSpec,
        buffers: BufferBindings,
    ) -> Result<(), SchedulerError>;
}
```

Where `KernelSpec` captures everything needed to launch:

```rust
pub struct KernelSpec {
    pub name: &'static str,
    pub module: PtxModuleRef,  // reference to loaded PTX module
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared_mem: u32,
    pub hostcall: bool,
}
```

## CpuScheduler

**Trivial wrapper around tokio's blocking pool.**

```rust
pub struct CpuScheduler {
    // No state needed — delegates to tokio runtime
}

impl Scheduler for CpuScheduler {
    async fn cpu<F, T>(&self, f: F) -> Result<T, SchedulerError>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        tokio::task::spawn_blocking(f)
            .await
            .map_err(SchedulerError::JoinError)
    }

    async fn gpu(
        &self,
        _kernel: KernelSpec,
        _buffers: BufferBindings,
    ) -> Result<(), SchedulerError> {
        Err(SchedulerError::NoGpu)
    }
}
```

CpuScheduler.gpu() returns an error because this scheduler has no GPU access.
This is useful for testing, CI environments without GPUs, and fallback paths.

**Integration with existing code**: CpuScheduler is essentially what
`AsyncGpuRuntime::synchronize()` already does — offloads blocking work to
`tokio::task::spawn_blocking`. The difference is CpuScheduler is the ONLY
scheduler that can't do GPU work.

## GpuScheduler

**Wraps `GpuContext` / `GpuStdModule` for kernel dispatch.**

```rust
pub struct GpuScheduler {
    runtime: Arc<GpuRuntime>,
    /// Pre-loaded PTX modules (keyed by module name)
    modules: HashMap<&'static str, LoadedModule>,
}

impl Scheduler for GpuScheduler {
    async fn cpu<F, T>(&self, f: F) -> Result<T, SchedulerError>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        // GpuScheduler CAN run CPU work — just delegates to blocking pool
        tokio::task::spawn_blocking(f)
            .await
            .map_err(SchedulerError::JoinError)
    }

    async fn gpu(
        &self,
        kernel: KernelSpec,
        buffers: BufferBindings,
    ) -> Result<(), SchedulerError> {
        let rt = Arc::clone(&self.runtime);
        let func = rt.require_func(kernel.module.name(), kernel.name)?;
        let config = LaunchConfig {
            grid_dim: kernel.grid,
            block_dim: kernel.block,
            shared_mem_bytes: kernel.shared_mem,
        };

        // Launch + sync on blocking thread (same pattern as GpuTask::launch)
        tokio::task::spawn_blocking(move || {
            unsafe { func.launch(config, buffers.as_params())? };
            rt.synchronize()
        })
        .await
        .map_err(SchedulerError::JoinError)?
        .map_err(SchedulerError::Gpu)
    }
}
```

**Key design decision**: GpuScheduler.cpu() ALSO works (delegates to blocking pool).
This means GpuScheduler is strictly more capable than CpuScheduler. The reason
to keep CpuScheduler separate: GPU-less environments, testing, and explicit
documentation of "this code path never touches the GPU."

**Integration with existing APIs**:
- `gpu::custom("kernel")...prepare()...launch()` → becomes `scheduler.gpu(KernelSpec, buffers)`
- `gpu::run("kernel")` → becomes `scheduler.gpu(KernelSpec::simple("kernel"), buffers)` with hostcall
- `GpuStdModule::load()` → module pre-loading at scheduler construction time

## AutoScheduler

**Routes work based on work descriptor metadata.**

```rust
pub struct AutoScheduler {
    gpu: GpuScheduler,
    /// Minimum element count before GPU dispatch is worthwhile.
    /// Below this threshold, CPU is faster due to launch overhead.
    gpu_threshold: usize,
}

impl AutoScheduler {
    pub fn new(device_ordinal: usize) -> Result<Self> {
        let gpu = GpuScheduler::new(device_ordinal)?;
        Ok(Self {
            gpu,
            gpu_threshold: 4096, // ~4K elements: below this, CPU wins
        })
    }
}

impl Scheduler for AutoScheduler {
    async fn cpu<F, T>(&self, f: F) -> Result<T, SchedulerError>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        self.gpu.cpu(f).await
    }

    async fn gpu(
        &self,
        kernel: KernelSpec,
        buffers: BufferBindings,
    ) -> Result<(), SchedulerError> {
        self.gpu.gpu(kernel, buffers).await
    }
}
```

**But the real value of AutoScheduler is at a higher level**, not in the
`Scheduler` trait itself. The trait has explicit `cpu()` and `gpu()` methods
— the user already chose. AutoScheduler's value comes from **higher-level
combinators** that use it:

```rust
impl AutoScheduler {
    /// Execute a data-parallel operation on the best target.
    ///
    /// - Small data (< threshold): runs on CPU via Rayon/sequential
    /// - Large data (>= threshold): routes to GPU par_iter kernel
    pub async fn par_map<T, U, F>(
        &self,
        data: &[T],
        f: F,
    ) -> Result<Vec<U>, SchedulerError>
    where
        T: Copy + Send + Sync + 'static,
        U: Copy + Send + Sync + 'static,
        F: Fn(T) -> U + Copy + Send + Sync + 'static,
    {
        if data.len() < self.gpu_threshold {
            // CPU path: sequential or rayon
            self.cpu(move || data.iter().map(f).collect()).await
        } else {
            // GPU path: upload → par_iter kernel → download
            // Uses GpuBuffer for automatic transfer
            let input = GpuBuffer::from_slice(data)?;
            let output = GpuBuffer::alloc_zeros(data.len())?;
            self.gpu(
                KernelSpec::par_iter_map(data.len()),
                BufferBindings::new(&[&input, &output]),
            ).await?;
            Ok(output.download()?)
        }
    }
}
```

### AutoScheduler Heuristics

| Signal | Route | Rationale |
|--------|-------|-----------|
| `data.len() < 4096` | CPU | GPU launch overhead (~20us) dominates for small data |
| `data.len() >= 4096` and bulk compute | GPU | GPU wins at scale |
| I/O operation (File, Network) | CPU always | I/O goes through hostcall anyway; direct CPU is faster |
| Hostcall-heavy kernel | CPU (consider) | Too many hostcalls = GPU blocked on host round-trips |
| Single scalar computation | CPU | No parallelism to exploit |

The 4096 threshold is derived from project benchmarks: GPU par_iter crossover
vs CPU is around 1K-10K elements depending on operation complexity. 4096 is
a conservative middle ground.

## Integration with Existing APIs

### How schedulers compose with gpu::run / gpu::launch

The existing API surface does NOT change. Schedulers are an ADDITIONAL layer:

```
User-facing (new):       scheduler.gpu(spec, buffers)
                                ↓
Existing (unchanged):     gpu::custom("kernel")
                              .threads(256)
                              .elements(n)
                              .prepare()?
                              .launch(args)?
```

The scheduler calls into the existing `gpu::custom()` / `GpuContext` / `GpuStdModule`
APIs internally. No refactoring of those APIs needed.

### How schedulers compose with par_iter

par_iter currently runs entirely on GPU within a kernel. The scheduler doesn't
change this — par_iter chains compile to GPU code via monomorphization. What the
scheduler adds:

1. **Host-side orchestration**: Upload data, launch the par_iter kernel, download results
2. **CPU fallback**: For small data, run the same operation on CPU instead
3. **Automatic transfer**: `GpuBuffer` handles host↔device copies (see unified-transfer.1)

### How schedulers compose with async/await

The project has two async layers:
- **Host-side async** (tokio): `AsyncGpuRuntime`, `GpuTask` — async kernel launch
- **GPU-side async** (warp-cooperative): BlockScope spawn, channel send/recv

The scheduler operates at the HOST-SIDE async layer only. It is a tokio-compatible
abstraction that wraps the existing `AsyncGpuRuntime` / `GpuTask` patterns.

GPU-side async (warp-cooperative scheduling within a kernel) is orthogonal and
unchanged. The scheduler decides WHETHER to launch a kernel; once launched,
the kernel's internal warp scheduling is its own business.

## Practical Feasibility Analysis

### What the scheduler CAN do
- Route work to CPU or GPU based on data size and operation type
- Provide a unified async interface (`scheduler.cpu()`, `scheduler.gpu()`)
- Abstract away CUDA device init, PTX loading, and memory management
- Enable graceful fallback when GPU is unavailable

### What the scheduler CANNOT do
- Magically run arbitrary CPU closures on GPU (PTX must be pre-compiled)
- Transparently convert `|x| x * 2.0` to GPU code at runtime
- Avoid the fundamental GPU compilation boundary

### The "one async fn" vision

The North Star says "users write one async fn mixing I/O and compute, the runtime
decides what runs where." This is achievable IF:

1. Compute operations use pre-compiled kernels exposed via high-level combinators
   (`par_map`, `par_reduce`, `par_filter`). These are already compiled to PTX.
2. I/O operations use the scheduler's CPU path (which is just tokio).
3. Data transfer is automatic (GpuBuffer tracks location).

The user writes:
```rust
async fn pipeline(scheduler: &AutoScheduler) -> Vec<f32> {
    let data = scheduler.cpu(|| std::fs::read("input.bin")).await?;  // I/O → CPU
    let floats: Vec<f32> = parse_floats(&data);
    let result = scheduler.par_map(&floats, |x| x * 2.0 + 1.0).await?;  // compute → GPU
    scheduler.cpu(|| std::fs::write("output.bin", &result)).await?;  // I/O → CPU
    result
}
```

This is NOT "transparent GPU" — the user explicitly calls `par_map` for data-parallel
work. But it IS "zero GPU concepts" — no kernel names, no launch configs, no
memory management, no CUDA device initialization.

## Limitations and Open Questions

### L1: The closure problem
GPU work is kernel-name-based, not closure-based. `scheduler.par_map(data, |x| x * 2.0)`
must map to a PRE-COMPILED kernel that implements "apply user function to each element."
This works for par_iter (via monomorphization at compile time) but does NOT work for
arbitrary runtime closures.

**Mitigation**: The par_iter kernel uses Rust monomorphization — the closure IS compiled
to PTX at build time. The `par_map` combinator on AutoScheduler instantiates a generic
par_iter kernel. This works because the closure is known at compile time.

### L2: Kernel registration
GpuScheduler needs pre-loaded PTX modules. Who loads them and when?

**Recommendation**: At scheduler construction time, load all PTX modules from `ptx::ALL`.
This is a one-time cost. New modules can be registered via `scheduler.register_module()`.

### L3: Buffer lifetime management
`BufferBindings` must reference live GPU buffers. Who owns the buffers?

**Recommendation**: `GpuBuffer<T>` (from unified-transfer theme) is the answer.
It tracks location (host/device/both) and handles transfers automatically.
The scheduler doesn't own buffers — it borrows them.

### L4: Error propagation across CPU/GPU boundary
CPU errors are `std::io::Error`, GPU errors are `GpuHostError`. The scheduler
needs a unified error type.

**Recommendation**: `SchedulerError` enum wrapping both, convertible via `From`.

### L5: Multi-GPU scheduling
AutoScheduler currently assumes device 0. Multi-GPU support is out of scope
for unified-scheduler.2 but the trait design should not prevent it.

**Recommendation**: `GpuScheduler::new(device_ordinal)` already parameterizes
the device. Multi-GPU AutoScheduler is a future extension.

## Recommendations for unified-scheduler.2 (Implementation)

### Phase 1: Foundation types (1 task)
1. Create `crates/core/gpu-host/src/scheduler.rs` module
2. Define `Scheduler` trait with `cpu()` and `gpu()` methods
3. Define `KernelSpec` and `BufferBindings` types
4. Define `SchedulerError` error type
5. Implement `CpuScheduler` (trivial — tokio delegation)

### Phase 2: GpuScheduler (1 task)
1. Implement `GpuScheduler` wrapping `Arc<GpuRuntime>`
2. Auto-load `ptx::ALL` modules at construction
3. Wire `gpu()` method through existing `GpuContext::launch()` path
4. Add `GpuScheduler::new(ordinal)` constructor

### Phase 3: AutoScheduler heuristics (unified-scheduler.3)
1. Implement `AutoScheduler` with configurable threshold
2. Add `par_map`, `par_reduce`, `par_filter` high-level combinators
3. CPU fallback path for small data
4. Benchmark to calibrate threshold (use existing par_iter vs Rayon data)

### Where to put the code
- `crates/core/gpu-host/src/scheduler.rs` — trait + CpuScheduler + GpuScheduler
- Feature-gated behind `async` feature (same as `async_rt` module)
- AutoScheduler in same module or split to `scheduler/auto.rs` if large

### Dependencies
- unified-transfer.1 results needed for `GpuBuffer` / `BufferBindings` design
- Existing `async_rt` module provides the `spawn_blocking` pattern
- Existing `gpu.rs` module provides `GpuContext` / `CustomLaunchBuilder`
