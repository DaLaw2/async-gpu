# unified-scheduler.2 — CpuScheduler + GpuScheduler Foundation Types

Implementation of the scheduler trait and two concrete schedulers for the
unified-runtime theme.

## What was implemented

### New file: `crates/core/gpu-host/src/scheduler.rs`

**Scheduler trait** with two methods:
- `cpu<F, T>(f: F) -> Result<T>` — run a CPU-bound closure
- `gpu_launch<T>(kernel, output_len, threads) -> Result<Vec<T>>` — launch a GPU kernel

**CpuScheduler** (zero-field unit struct):
- `cpu()` runs the closure directly and returns `Ok(result)`
- `gpu_launch()` returns `Err(GpuHostError::NoGpu)`

**GpuScheduler** (zero-field unit struct):
- `cpu()` runs the closure directly (same as CpuScheduler)
- `gpu_launch()` delegates to `crate::gpu::launch(kernel, output_len, threads)`

### Modified: `crates/core/gpu-host/src/error.rs`

Added `GpuHostError::NoGpu` variant with Display impl:
`"no GPU available (CPU-only scheduler)"`

### Modified: `crates/core/gpu-host/src/lib.rs`

Added `pub mod scheduler;` to the module tree (unconditional, not feature-gated).

## Design decisions

1. **Synchronous, not async**: The trait methods are synchronous. The design doc
   proposed async methods, but the existing `gpu::launch()` API is synchronous,
   and adding async here would require tokio as a non-optional dependency.
   Async wrappers can be added in a future task if needed.

2. **Unit structs, not stateful**: Both schedulers are zero-sized unit structs.
   GpuScheduler delegates to `gpu::launch()` which creates a fresh CUDA device
   on each call (matching the existing API pattern). Stateful schedulers with
   pre-loaded modules are deferred to AutoScheduler (unified-scheduler.3).

3. **Not feature-gated**: The scheduler module is always available (no feature
   gate). It only depends on cudarc types in the trait signature, which are
   already unconditional dependencies of gpu-host.

4. **Two-method trait**: Following the design doc's recommendation — explicit
   `cpu()` and `gpu_launch()` rather than a single polymorphic `submit()`.
   GPU work (kernel name + config) is fundamentally different from CPU work
   (closures), so a unified submit adds complexity without benefit.

## Verification

- `cargo check -p gpu-host` — compiles clean
- `cargo clippy -p gpu-host` — no new warnings (pre-existing `unused_mut` in memory.rs)
- `cargo test -p gpu-host --lib scheduler` — 4/4 tests pass:
  - `cpu_scheduler_runs_closure`
  - `cpu_scheduler_rejects_gpu`
  - `gpu_scheduler_runs_closure`
  - `scheduler_trait_is_object_safe_ish`

## What comes next

- **unified-scheduler.3**: AutoScheduler with data-size heuristics and
  high-level combinators (`par_map`, `par_reduce`, `par_filter`)
- Async wrappers (if the unified-runtime theme needs them)
- Stateful GpuScheduler with pre-loaded PTX modules
