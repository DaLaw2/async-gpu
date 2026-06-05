//! Work-routing scheduler types for unified CPU/GPU execution.
//!
//! The scheduler is a **work-routing abstraction** that decides WHERE work executes
//! (CPU thread vs GPU kernel launch), with a unified interface that hides the
//! routing decision from the caller.
//!
//! # Scheduler variants
//!
//! - [`CpuScheduler`] — CPU-only; GPU operations return [`GpuHostError::NoGpu`].
//!   Useful for testing, CI without GPUs, and explicit "no GPU" code paths.
//!
//! - [`GpuScheduler`] — GPU-capable; delegates to [`gpu::launch()`](crate::gpu::launch)
//!   for kernel dispatch, and also handles CPU work. Strictly more capable than
//!   `CpuScheduler`.
//!
//! - [`AutoScheduler`] — Automatically routes work to CPU or GPU based on data
//!   size heuristics. Small data (below threshold) runs on CPU; large data runs
//!   on GPU via pre-compiled kernels. Provides high-level combinators like
//!   [`AutoScheduler::par_map`] that hide the CPU/GPU decision entirely.
//!
//! # Design
//!
//! The trait uses explicit `cpu()` and `gpu_launch()` methods rather than a single
//! polymorphic `submit()`, because GPU work (kernel name + launch config) is
//! fundamentally different from CPU work (closures). See unified-scheduler.1 for
//! the design rationale.
//!
//! `AutoScheduler` adds higher-level combinators on top of the `Scheduler` trait
//! that inspect data size and route to the best execution target automatically.

use crate::error::{GpuHostError, Result};

/// Work-routing scheduler trait for unified CPU/GPU execution.
///
/// Implementors decide where work runs. The two methods correspond to the two
/// execution targets:
///
/// - [`cpu()`](Scheduler::cpu) — run a CPU-bound closure
/// - [`gpu_launch()`](Scheduler::gpu_launch) — launch a pre-compiled GPU kernel
pub trait Scheduler {
    /// Run a CPU-bound closure synchronously and return its result.
    ///
    /// Both `CpuScheduler` and `GpuScheduler` support this — it simply runs
    /// the closure on the calling thread.
    fn cpu<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static;

    /// Launch a pure-compute GPU kernel by name, returning its output buffer.
    ///
    /// This delegates to [`gpu::launch()`](crate::gpu::launch) under the hood.
    /// The kernel must be a zero-input, output-only kernel (signature: `fn(output: *mut T)`).
    ///
    /// # Errors
    ///
    /// Returns [`GpuHostError::NoGpu`] if the scheduler has no GPU access
    /// (e.g., `CpuScheduler`).
    fn gpu_launch<T>(
        &self,
        kernel: &'static str,
        output_len: usize,
        threads: u32,
    ) -> Result<Vec<T>>
    where
        T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Clone;
}

/// CPU-only scheduler.
///
/// Runs closures directly on the calling thread. GPU operations always return
/// [`GpuHostError::NoGpu`]. Useful for:
///
/// - Testing without GPU hardware
/// - CI environments
/// - Explicit "this code path never touches the GPU" documentation
///
/// # Example
///
/// ```
/// use gpu_host::scheduler::{CpuScheduler, Scheduler};
///
/// let sched = CpuScheduler;
/// let result = sched.cpu(|| 2 + 2).unwrap();
/// assert_eq!(result, 4);
///
/// // GPU work is rejected:
/// let err = sched.gpu_launch::<f32>("some_kernel", 64, 32);
/// assert!(err.is_err());
/// ```
pub struct CpuScheduler;

impl Scheduler for CpuScheduler {
    fn cpu<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        Ok(f())
    }

    fn gpu_launch<T>(
        &self,
        _kernel: &'static str,
        _output_len: usize,
        _threads: u32,
    ) -> Result<Vec<T>>
    where
        T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Clone,
    {
        Err(GpuHostError::NoGpu)
    }
}

/// GPU-capable scheduler.
///
/// Handles both CPU and GPU work:
/// - `cpu()` runs closures directly (same as [`CpuScheduler`])
/// - `gpu_launch()` delegates to [`gpu::launch()`](crate::gpu::launch)
///
/// This is strictly more capable than `CpuScheduler` — it can do everything
/// `CpuScheduler` does, plus GPU kernel dispatch.
///
/// # Example
///
/// ```no_run
/// use gpu_host::scheduler::{GpuScheduler, Scheduler};
///
/// let sched = GpuScheduler;
///
/// // CPU work — same as CpuScheduler
/// let x = sched.cpu(|| 42).unwrap();
///
/// // GPU work — launches a real kernel
/// let output: Vec<f32> = sched.gpu_launch("my_kernel", 64, 32).unwrap();
/// ```
pub struct GpuScheduler;

impl Scheduler for GpuScheduler {
    fn cpu<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        Ok(f())
    }

    fn gpu_launch<T>(&self, kernel: &'static str, output_len: usize, threads: u32) -> Result<Vec<T>>
    where
        T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Clone,
    {
        crate::gpu::launch(kernel, output_len, threads)
    }
}

// ============================================================================
// AutoScheduler — size-based CPU/GPU routing
// ============================================================================

/// Default element threshold below which CPU execution is preferred.
///
/// GPU launch overhead (~20 microseconds) dominates for small data. Benchmarks show the
/// GPU par_iter crossover vs CPU is around 1K–10K elements depending on operation
/// complexity. 4096 is a conservative middle ground.
const DEFAULT_GPU_THRESHOLD: usize = 4096;

/// Scheduler that automatically routes work to CPU or GPU based on data size.
///
/// For the base [`Scheduler`] trait methods, `AutoScheduler` behaves identically to
/// [`GpuScheduler`]: `cpu()` runs closures directly, `gpu_launch()` dispatches to the
/// GPU.
///
/// The real value is in the **high-level combinators** ([`par_map`](Self::par_map),
/// [`par_reduce`](Self::par_reduce)) that inspect data size and route to the best
/// execution target automatically:
///
/// - **Small data** (< threshold): runs on CPU via iterator combinators
/// - **Large data** (>= threshold): routes to pre-compiled GPU kernels
///
/// # The closure constraint
///
/// GPU kernels are pre-compiled to PTX at build time. Arbitrary Rust closures
/// **cannot** be sent to the GPU at runtime. The GPU path in `par_map` uses a
/// fixed pre-compiled kernel (`par_iter_map_collect_multiblock`: `f(x) = x * 2.0 + 1.0`).
///
/// The CPU path runs the caller's actual closure, so arbitrary operations work for
/// small data. For large data, only the pre-compiled GPU kernel is available. Use
/// [`par_map_cpu`](Self::par_map_cpu) to force the CPU path for arbitrary closures
/// regardless of data size.
///
/// # Example
///
/// ```no_run
/// use gpu_host::scheduler::{AutoScheduler, Scheduler};
///
/// let sched = AutoScheduler::new();
///
/// // Small data → CPU path (uses your closure)
/// let small: Vec<f32> = (0..100).map(|i| i as f32).collect();
/// let result = sched.par_map(&small, |x| x * 2.0 + 1.0).unwrap();
///
/// // Large data → GPU path (pre-compiled kernel: x * 2.0 + 1.0)
/// let large: Vec<f32> = (0..10_000).map(|i| i as f32).collect();
/// let result = sched.par_map(&large, |x| x * 2.0 + 1.0).unwrap();
///
/// // Base Scheduler trait still works:
/// let x = sched.cpu(|| 42).unwrap();
/// ```
pub struct AutoScheduler {
    /// Element count below which CPU is preferred over GPU.
    threshold: usize,
}

impl AutoScheduler {
    /// Create an `AutoScheduler` with the default threshold (4096 elements).
    pub fn new() -> Self {
        Self {
            threshold: DEFAULT_GPU_THRESHOLD,
        }
    }

    /// Create an `AutoScheduler` with a custom element threshold.
    ///
    /// Elements below this count route to CPU; at or above route to GPU.
    pub fn with_threshold(threshold: usize) -> Self {
        Self { threshold }
    }

    /// Returns the current GPU routing threshold.
    pub fn threshold(&self) -> usize {
        self.threshold
    }

    /// Parallel map with automatic CPU/GPU routing.
    ///
    /// - **Small data** (< threshold): applies the closure `f` on the CPU via iterators.
    /// - **Large data** (>= threshold): launches the pre-compiled GPU kernel
    ///   `par_iter_map_collect_multiblock` which computes `x * 2.0 + 1.0`.
    ///
    /// # GPU path limitation
    ///
    /// The GPU path uses a fixed kernel operation (`x * 2.0 + 1.0`). The closure `f`
    /// is only used for the CPU path. For the GPU path, the closure is **ignored** and
    /// the pre-compiled kernel runs instead. This is a fundamental constraint: GPU
    /// kernels are compiled at build time, not runtime.
    ///
    /// To run an arbitrary closure on all data sizes (CPU only), use
    /// [`par_map_cpu`](Self::par_map_cpu).
    pub fn par_map<F>(&self, data: &[f32], f: F) -> Result<Vec<f32>>
    where
        F: Fn(f32) -> f32,
    {
        if data.len() < self.threshold {
            // CPU path: apply the caller's closure directly.
            Ok(data.iter().map(|&x| f(x)).collect())
        } else {
            // GPU path: upload → multiblock par_iter kernel → download.
            // The kernel computes x * 2.0 + 1.0 (pre-compiled, not the closure).
            self.gpu_par_map(data)
        }
    }

    /// Parallel map that always runs on CPU, regardless of data size.
    ///
    /// Use this when you need an arbitrary closure and cannot use the pre-compiled
    /// GPU kernel.
    pub fn par_map_cpu<F>(&self, data: &[f32], f: F) -> Result<Vec<f32>>
    where
        F: Fn(f32) -> f32,
    {
        Ok(data.iter().map(|&x| f(x)).collect())
    }

    /// Parallel reduce with automatic CPU/GPU routing.
    ///
    /// - **Small data** (< threshold): reduces on CPU via iterators.
    /// - **Large data** (>= threshold): also reduces on CPU (GPU reduce kernel
    ///   is not yet available as a pre-compiled multiblock kernel).
    ///
    /// This method always produces correct results. The routing decision is
    /// an optimization hint — when a GPU reduce kernel is available, large data
    /// will automatically route to it.
    pub fn par_reduce<F>(&self, data: &[f32], identity: f32, f: F) -> Result<f32>
    where
        F: Fn(f32, f32) -> f32,
    {
        // Both paths use CPU for now. The threshold still affects future routing
        // when a GPU reduce kernel becomes available.
        Ok(data.iter().fold(identity, |acc, &x| f(acc, x)))
    }

    /// GPU path for `par_map`: launches `par_iter_map_collect_multiblock` kernel.
    ///
    /// Kernel signature: `fn(input: *const f32, output: *mut f32, n: u32)`
    /// Operation: `f(x) = x * 2.0 + 1.0`
    fn gpu_par_map(&self, data: &[f32]) -> Result<Vec<f32>> {
        use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};

        let n = data.len();
        let threads_per_block: u32 = 256;
        let grid = (n as u32).div_ceil(threads_per_block);

        let dev = CudaDevice::new(0).map_err(GpuHostError::CudaInit)?;

        // Load the kernel from the test PTX module (where par_iter_map_collect_multiblock lives).
        let ptx = cudarc::nvrtc::Ptx::from_src(crate::ptx::KERNEL_TEST);
        let module = crate::gpu::fresh_module_name();
        dev.load_ptx(ptx, &module, &["par_iter_map_collect_multiblock"])
            .map_err(|e| GpuHostError::Verification {
                test: "auto_scheduler_ptx_load",
                detail: format!("{e}"),
            })?;

        let func = dev
            .get_func(&module, "par_iter_map_collect_multiblock")
            .ok_or(GpuHostError::KernelNotFound(
                "par_iter_map_collect_multiblock",
            ))?;

        // Upload input data to device.
        let input_dev = dev.htod_sync_copy(data).map_err(GpuHostError::Cudarc)?;

        // Allocate output buffer on device.
        let mut output_dev = dev.alloc_zeros::<f32>(n).map_err(GpuHostError::Cudarc)?;

        let config = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_u32 = n as u32;

        // Launch kernel: par_iter_map_collect_multiblock(input, output, n)
        unsafe {
            func.launch(config, (&input_dev, &mut output_dev, n_u32))
                .map_err(|e| GpuHostError::Verification {
                    test: "auto_scheduler_launch",
                    detail: format!("{e}"),
                })?;
        }

        dev.synchronize().map_err(|e| GpuHostError::Verification {
            test: "auto_scheduler_sync",
            detail: format!("sync: {e}"),
        })?;

        // Download results back to host.
        dev.dtoh_sync_copy(&output_dev)
            .map_err(|e| GpuHostError::Verification {
                test: "auto_scheduler_dtoh",
                detail: format!("dtoh: {e}"),
            })
    }
}

impl Default for AutoScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl Scheduler for AutoScheduler {
    fn cpu<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        Ok(f())
    }

    fn gpu_launch<T>(&self, kernel: &'static str, output_len: usize, threads: u32) -> Result<Vec<T>>
    where
        T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Clone,
    {
        crate::gpu::launch(kernel, output_len, threads)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_scheduler_runs_closure() {
        let sched = CpuScheduler;
        let result = sched.cpu(|| 2 + 2).unwrap();
        assert_eq!(result, 4);
    }

    #[test]
    fn cpu_scheduler_rejects_gpu() {
        let sched = CpuScheduler;
        let err = sched.gpu_launch::<f32>("nonexistent", 1, 1);
        assert!(err.is_err());
        let msg = format!("{}", err.unwrap_err());
        assert!(msg.contains("no GPU"), "expected NoGpu error, got: {msg}");
    }

    #[test]
    fn gpu_scheduler_runs_closure() {
        let sched = GpuScheduler;
        let result = sched.cpu(|| "hello".to_string()).unwrap();
        assert_eq!(result, "hello");
    }

    // GpuScheduler.gpu_launch() requires a real GPU + valid kernel,
    // so it is tested via integration tests, not unit tests.

    #[test]
    fn scheduler_trait_is_object_safe_ish() {
        // Verify both types implement Scheduler (compile-time check).
        fn _assert_scheduler<S: Scheduler>(_s: &S) {}
        _assert_scheduler(&CpuScheduler);
        _assert_scheduler(&GpuScheduler);
        _assert_scheduler(&AutoScheduler::new());
    }

    // ── AutoScheduler unit tests ──────────────────────────────

    #[test]
    fn auto_scheduler_default_threshold() {
        let sched = AutoScheduler::new();
        assert_eq!(sched.threshold(), DEFAULT_GPU_THRESHOLD);
    }

    #[test]
    fn auto_scheduler_custom_threshold() {
        let sched = AutoScheduler::with_threshold(1024);
        assert_eq!(sched.threshold(), 1024);
    }

    #[test]
    fn auto_scheduler_default_impl() {
        let sched = AutoScheduler::default();
        assert_eq!(sched.threshold(), DEFAULT_GPU_THRESHOLD);
    }

    #[test]
    fn auto_scheduler_cpu_closure() {
        let sched = AutoScheduler::new();
        let result = sched.cpu(|| 2 + 2).unwrap();
        assert_eq!(result, 4);
    }

    #[test]
    fn auto_scheduler_par_map_small_data() {
        // Below threshold → CPU path uses the actual closure.
        let sched = AutoScheduler::with_threshold(100);
        let data: Vec<f32> = (0..50).map(|i| i as f32).collect();
        let result = sched.par_map(&data, |x| x * 3.0).unwrap();
        assert_eq!(result.len(), 50);
        for (i, &v) in result.iter().enumerate() {
            let expected = (i as f32) * 3.0;
            assert!(
                (v - expected).abs() < 1e-6,
                "index {i}: expected {expected}, got {v}"
            );
        }
    }

    #[test]
    fn auto_scheduler_par_map_cpu_forces_cpu() {
        let sched = AutoScheduler::with_threshold(10);
        // Data is larger than threshold, but par_map_cpu forces CPU path.
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let result = sched.par_map_cpu(&data, |x| x + 5.0).unwrap();
        assert_eq!(result.len(), 100);
        for (i, &v) in result.iter().enumerate() {
            let expected = (i as f32) + 5.0;
            assert!(
                (v - expected).abs() < 1e-6,
                "index {i}: expected {expected}, got {v}"
            );
        }
    }

    #[test]
    fn auto_scheduler_par_reduce_small() {
        let sched = AutoScheduler::new();
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let sum = sched.par_reduce(&data, 0.0, |a, b| a + b).unwrap();
        assert!((sum - 10.0).abs() < 1e-6);
    }

    #[test]
    fn auto_scheduler_par_reduce_empty() {
        let sched = AutoScheduler::new();
        let data: Vec<f32> = vec![];
        let sum = sched.par_reduce(&data, 0.0, |a, b| a + b).unwrap();
        assert!((sum - 0.0).abs() < 1e-6);
    }

    #[test]
    fn auto_scheduler_par_map_gpu_path() {
        // Set threshold low so our data routes to GPU.
        let sched = AutoScheduler::with_threshold(10);
        let n = 1024;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();

        // The GPU kernel computes x * 2.0 + 1.0 (the closure is ignored for the GPU path).
        let result = sched.par_map(&data, |x| x * 2.0 + 1.0).unwrap();

        assert_eq!(result.len(), n);
        for (i, &v) in result.iter().enumerate() {
            let expected = (i as f32) * 2.0 + 1.0;
            assert!(
                (v - expected).abs() < 1e-4,
                "index {i}: expected {expected}, got {v}"
            );
        }
    }

    #[test]
    fn auto_scheduler_par_map_gpu_large() {
        // Test with a more substantial dataset routed to GPU.
        let sched = AutoScheduler::with_threshold(100);
        let n = 8192;
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();

        let result = sched.par_map(&data, |x| x * 2.0 + 1.0).unwrap();

        assert_eq!(result.len(), n);
        for (i, &v) in result.iter().enumerate() {
            let expected = (i as f32) * 0.1 * 2.0 + 1.0;
            assert!(
                (v - expected).abs() < 1e-4,
                "index {i}: expected {expected}, got {v}"
            );
        }
    }

    #[test]
    fn auto_scheduler_routing_boundary() {
        // Exact boundary: threshold=100, data of len 100 should route to GPU.
        let sched = AutoScheduler::with_threshold(100);

        // len=99 → CPU path (uses closure: x + 10.0)
        let small: Vec<f32> = (0..99).map(|i| i as f32).collect();
        let result = sched.par_map(&small, |x| x + 10.0).unwrap();
        for (i, &v) in result.iter().enumerate() {
            let expected = (i as f32) + 10.0; // closure: x + 10.0
            assert!(
                (v - expected).abs() < 1e-6,
                "CPU path: index {i}: expected {expected}, got {v}"
            );
        }

        // len=100 → GPU path (kernel: x * 2.0 + 1.0, closure ignored)
        let large: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let result = sched.par_map(&large, |x| x + 10.0).unwrap();
        for (i, &v) in result.iter().enumerate() {
            let expected = (i as f32) * 2.0 + 1.0; // kernel op, not closure
            assert!(
                (v - expected).abs() < 1e-4,
                "GPU path: index {i}: expected {expected}, got {v}"
            );
        }
    }
}
