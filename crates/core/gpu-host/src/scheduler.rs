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
//! # Design
//!
//! The trait uses explicit `cpu()` and `gpu_launch()` methods rather than a single
//! polymorphic `submit()`, because GPU work (kernel name + launch config) is
//! fundamentally different from CPU work (closures). See unified-scheduler.1 for
//! the design rationale.
//!
//! `AutoScheduler` (which routes automatically based on data size heuristics) is
//! planned for unified-scheduler.3.

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
    }
}
