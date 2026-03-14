//! Async/await integration for GPU runtime.
//!
//! Provides tokio-compatible wrappers around the blocking GPU SDK:
//! - [`AsyncGpuRuntime`] — kernel launch + synchronize as async operations
//! - [`AsyncHostcallSession`] — persistent hostcall session with async event stream
//! - [`HostcallEvent`] — typed events from GPU (print, shutdown)
//!
//! # Example
//!
//! ```no_run
//! use gpu_host::async_rt::{AsyncGpuRuntime, AsyncHostcallSession};
//!
//! #[tokio::main]
//! async fn main() {
//!     let rt = AsyncGpuRuntime::new(0).unwrap();
//!     rt.load_ptx(ptx, "mod", &["kern"]).unwrap();
//!     let (session, mut events) = AsyncHostcallSession::start(16).unwrap();
//!
//!     // Launch kernel and await completion
//!     rt.synchronize().await.unwrap();
//!
//!     // Consume events asynchronously
//!     while let Some(event) = events.recv().await {
//!         match event {
//!             HostcallEvent::Print(msg) => println!("{}", String::from_utf8_lossy(&msg)),
//!             HostcallEvent::Shutdown => break,
//!         }
//!     }
//!
//!     session.shutdown().await;
//! }
//! ```

use std::sync::Arc;

use cudarc::driver::sys::CUdeviceptr;

use crate::error::{GpuHostError, Result};
use crate::hostcall::{HostcallError, HostcallSession};
use crate::runtime::GpuRuntime;

// ================================================================
// HostcallEvent — typed events from GPU hostcall listener
// ================================================================

/// An event received from the GPU via the hostcall listener.
#[derive(Debug, Clone)]
pub enum HostcallEvent {
    /// GPU issued a print message.
    Print(Vec<u8>),
    /// Listener shut down (no more events will follow).
    Shutdown,
}

// ================================================================
// AsyncGpuRuntime — tokio-compatible wrapper around GpuRuntime
// ================================================================

/// Async wrapper around [`GpuRuntime`] for tokio-compatible GPU operations.
///
/// Synchronization calls are offloaded to the tokio blocking thread pool
/// via [`tokio::task::spawn_blocking`], allowing other async tasks to
/// make progress while waiting for GPU work to complete.
pub struct AsyncGpuRuntime {
    inner: Arc<GpuRuntime>,
}

impl AsyncGpuRuntime {
    /// Initialize a CUDA device by ordinal (0 = first GPU).
    pub fn new(device_ordinal: usize) -> Result<Self> {
        let inner = GpuRuntime::new(device_ordinal)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Wrap an existing [`GpuRuntime`] in an async handle.
    pub fn from_runtime(runtime: GpuRuntime) -> Self {
        Self {
            inner: Arc::new(runtime),
        }
    }

    /// Get a reference to the underlying [`GpuRuntime`].
    pub fn inner(&self) -> &GpuRuntime {
        &self.inner
    }

    /// Get a cloneable `Arc` reference for sharing across tasks.
    pub fn inner_arc(&self) -> Arc<GpuRuntime> {
        Arc::clone(&self.inner)
    }

    /// Load a PTX module with named kernel functions.
    ///
    /// This is a fast, synchronous operation (no GPU work queued).
    pub fn load_ptx(
        &self,
        ptx_src: &str,
        module_name: &str,
        fn_names: &[&'static str],
    ) -> Result<()> {
        self.inner.load_ptx(ptx_src, module_name, fn_names)
    }

    /// Synchronize the device asynchronously.
    ///
    /// Offloads `cuCtxSynchronize` to a blocking thread so other tokio tasks
    /// continue to make progress.
    pub async fn synchronize(&self) -> Result<()> {
        let rt = Arc::clone(&self.inner);
        tokio::task::spawn_blocking(move || rt.synchronize())
            .await
            .map_err(|e| GpuHostError::Verification {
                test: "async_synchronize",
                detail: format!("spawn_blocking join error: {e}"),
            })?
    }
}

// ================================================================
// AsyncHostcallSession — persistent listener with async event stream
// ================================================================

/// A persistent hostcall session that exposes events as an async channel.
///
/// The listener thread runs in the background (as with [`HostcallSession`]),
/// but print events are forwarded to a [`tokio::sync::mpsc`] channel that
/// can be consumed asynchronously.
pub struct AsyncHostcallSession {
    inner: HostcallSession,
}

impl AsyncHostcallSession {
    /// Start a new async session with the given packet count.
    ///
    /// Returns the session handle and a receiver for [`HostcallEvent`]s.
    /// The listener thread is spawned immediately.
    pub fn start(
        num_packets: u16,
    ) -> std::result::Result<(Self, tokio::sync::mpsc::Receiver<HostcallEvent>), HostcallError>
    {
        let (tx, rx) = tokio::sync::mpsc::channel(256);

        let session = HostcallSession::start_with_print(num_packets, move |msg| {
            // blocking_send is fine here — we're on the listener thread, not tokio
            let _ = tx.blocking_send(HostcallEvent::Print(msg.to_vec()));
        })?;

        Ok((Self { inner: session }, rx))
    }

    /// Get the device pointer for kernel launch args.
    pub fn dev_ptr(&self) -> CUdeviceptr {
        self.inner.dev_ptr()
    }

    /// Get the sideband device pointer for bulk transfer args.
    pub fn sideband_dev_ptr(&self) -> CUdeviceptr {
        self.inner.sideband_dev_ptr()
    }

    /// Reinitialize packet pool between kernel launches.
    ///
    /// MUST be called after synchronize and before the next kernel launch.
    pub fn reinit_packets(&self) {
        self.inner.reinit_packets();
    }

    /// Shut down the session asynchronously.
    ///
    /// Signals the listener to stop, waits briefly, then joins the thread.
    pub async fn shutdown(self) {
        // Signal shutdown; the blocking join happens on a separate thread
        tokio::task::spawn_blocking(move || {
            self.inner.shutdown();
        })
        .await
        .ok();
    }
}
