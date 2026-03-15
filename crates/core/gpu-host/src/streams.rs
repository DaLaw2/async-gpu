//! CUDA stream support for overlapping kernel execution.
//!
//! Provides [`GpuStream`] — a wrapper around cudarc's [`CudaStream`] that
//! enables launching kernels on independent streams for compute overlap.
//!
//! # Two-tier stream model (ADR-20)
//!
//! - **Compute streams** ([`GpuStream`]): For pure compute kernels that don't
//!   use hostcall I/O. Can overlap freely on independent streams.
//! - **Hostcall kernels**: Must use the default stream via [`GpuRuntime`] or
//!   `GpuTask`. Device-level sync is required
//!   before hostcall packet reset.
//!
//! [`CudaStream`]: cudarc::driver::CudaStream
//! [`GpuRuntime`]: crate::runtime::GpuRuntime

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction, CudaStream, LaunchAsync, LaunchConfig};

use crate::error::{GpuHostError, Result};

/// A CUDA stream for overlapping kernel execution.
///
/// Each `GpuStream` maintains its own command queue on the GPU. Kernels launched
/// on different streams can execute concurrently if they don't share resources.
///
/// On creation, the stream synchronizes with the default stream (all prior
/// default-stream work completes before this stream starts). On drop, the
/// default stream waits for this stream to finish.
///
/// # Example
///
/// ```no_run
/// # use gpu_host::runtime::GpuRuntime;
/// let rt = GpuRuntime::new(0).unwrap();
/// let stream = rt.create_stream().unwrap();
///
/// let func = rt.require_func("module", "kernel").unwrap();
/// let config = GpuRuntime::launch_config((1, 1, 1), (32, 1, 1), 0);
///
/// // Launch on this stream (overlaps with default stream work)
/// unsafe { stream.launch(func, config, (42u32,)).unwrap() };
///
/// // Wait for just this stream
/// stream.synchronize().unwrap();
/// ```
pub struct GpuStream {
    inner: CudaStream,
    dev: Arc<CudaDevice>,
}

impl GpuStream {
    /// Create a new stream that forks from the default stream.
    ///
    /// The new stream will wait for all prior default-stream work to complete
    /// before executing any operations.
    pub(crate) fn new(dev: &Arc<CudaDevice>) -> Result<Self> {
        let inner = dev.fork_default_stream().map_err(GpuHostError::Cudarc)?;
        Ok(Self {
            inner,
            dev: Arc::clone(dev),
        })
    }

    /// Launch a kernel on this stream.
    ///
    /// The kernel executes asynchronously on this stream. It can overlap with
    /// kernels on other streams or the default stream.
    ///
    /// # Safety
    ///
    /// Same requirements as [`CudaFunction::launch`] — caller must ensure
    /// kernel arguments are valid and the launch configuration is correct.
    pub unsafe fn launch<Params>(
        &self,
        func: CudaFunction,
        config: LaunchConfig,
        args: Params,
    ) -> Result<()>
    where
        CudaFunction: LaunchAsync<Params>,
    {
        func.launch_on_stream(&self.inner, config, args)
            .map_err(GpuHostError::Cudarc)
    }

    /// Synchronize this stream — block the host until all pending operations
    /// on this stream complete.
    ///
    /// Does **not** wait for work on other streams or the default stream.
    pub fn synchronize(&self) -> Result<()> {
        // Use low-level cuStreamSynchronize for per-stream sync
        unsafe {
            cudarc::driver::result::stream::synchronize(self.inner.stream)
                .map_err(GpuHostError::Cudarc)
        }
    }

    /// Make the default stream wait for this stream's pending work.
    ///
    /// After this call, any subsequent default-stream operations will execute
    /// only after this stream's current work completes. This is asynchronous
    /// with respect to the host.
    pub fn join_default(&self) -> Result<()> {
        self.dev.wait_for(&self.inner).map_err(GpuHostError::Cudarc)
    }
}

// Extend GpuRuntime with stream creation
impl crate::runtime::GpuRuntime {
    /// Create a new CUDA stream for overlapping kernel execution.
    ///
    /// The returned stream forks from the default stream — all prior
    /// default-stream work completes before this stream starts executing.
    ///
    /// Use streams for **pure compute** kernels only. Hostcall-based kernels
    /// must use the default stream (via `GpuRuntime` or `GpuTask`).
    pub fn create_stream(&self) -> Result<GpuStream> {
        GpuStream::new(self.device())
    }
}
