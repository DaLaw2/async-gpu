//! Error types for gpu-host.

use cudarc::driver::sys::CUresult;
use std::fmt;

/// Top-level error type for gpu-host operations.
#[derive(Debug)]
pub enum GpuHostError {
    /// CUDA device initialization failed.
    CudaInit(cudarc::driver::DriverError),
    /// CUDA memory allocation failed.
    CudaAlloc(CUresult),
    /// Failed to obtain device pointer for host memory.
    CudaGetDevPtr(CUresult),
    /// Failed to free CUDA host memory.
    CudaFreeMem(CUresult),
    /// cudarc driver error.
    Cudarc(cudarc::driver::DriverError),
    /// GPU kernel function not found in loaded PTX module.
    KernelNotFound(&'static str),
    /// Test verification failed (expected value mismatch).
    Verification {
        /// Name of the test that failed.
        test: &'static str,
        /// Description of the mismatch.
        detail: String,
    },
    /// Operation timed out waiting for GPU response.
    Timeout {
        /// Name of the test that timed out.
        test: &'static str,
        /// Description of the timeout context.
        detail: String,
    },
    /// Hostcall buffer allocation error.
    Hostcall(crate::hostcall::HostcallError),
    /// GPU kernel returned an error via the result buffer.
    #[allow(dead_code)]
    KernelError(GpuKernelErrorInfo),
    /// GPU kernel crashed without writing to the result buffer (TAG_UNINIT).
    #[allow(dead_code)]
    KernelCrash,
}

/// Structured error info extracted from a GPU kernel result buffer.
#[derive(Debug, Clone)]
pub struct GpuKernelErrorInfo {
    /// Error category (ERR_* constant from gpu-protocol).
    pub category: u16,
    /// Raw OS errno from the host, or 0.
    pub raw_errno: u16,
    /// threadIdx.x that produced the error.
    pub thread_idx: u16,
    /// blockIdx.x that produced the error.
    pub block_idx: u16,
    /// Error message (UTF-8, up to 48 bytes).
    pub message: String,
}

impl fmt::Display for GpuKernelErrorInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} (thread={}, block={}",
            error_category_name(self.category),
            self.thread_idx,
            self.block_idx,
        )?;
        if self.raw_errno != 0 {
            write!(f, ", errno={}", self.raw_errno)?;
        }
        if !self.message.is_empty() {
            write!(f, ", msg=\"{}\"", self.message)?;
        }
        write!(f, ")")
    }
}

/// Convert a GpuKernelResult to a host-side Result.
///
/// Call after kernel launch + synchronize. Reads the result buffer and returns:
/// - `Ok(())` if TAG_OK
/// - `Err(GpuHostError::KernelError)` if TAG_ERR
/// - `Err(GpuHostError::KernelCrash)` if TAG_UNINIT (kernel crashed)
#[allow(dead_code)]
pub fn check_kernel_result(
    result: &gpu_protocol::GpuKernelResult,
) -> std::result::Result<(), GpuHostError> {
    match result.tag {
        gpu_protocol::TAG_OK => Ok(()),
        gpu_protocol::TAG_ERR => {
            let msg_bytes = result.message();
            let message = String::from_utf8_lossy(msg_bytes).into_owned();
            Err(GpuHostError::KernelError(GpuKernelErrorInfo {
                category: result.category,
                raw_errno: result.raw_errno,
                thread_idx: result.thread_idx,
                block_idx: result.block_idx,
                message,
            }))
        }
        _ => Err(GpuHostError::KernelCrash),
    }
}

/// Map an error category constant to a human-readable name.
pub fn error_category_name(category: u16) -> &'static str {
    match category {
        gpu_protocol::ERR_OTHER => "other error",
        gpu_protocol::ERR_NOT_FOUND => "not found",
        gpu_protocol::ERR_PERMISSION_DENIED => "permission denied",
        gpu_protocol::ERR_ALREADY_EXISTS => "already exists",
        gpu_protocol::ERR_INVALID_INPUT => "invalid input",
        gpu_protocol::ERR_IO_ERROR => "I/O error",
        gpu_protocol::ERR_TIMED_OUT => "timed out",
        gpu_protocol::ERR_WOULD_BLOCK => "would block",
        gpu_protocol::ERR_BROKEN_PIPE => "broken pipe",
        gpu_protocol::ERR_RESOURCE_BUSY => "resource busy",
        gpu_protocol::ERR_STORAGE_FULL => "storage full",
        gpu_protocol::ERR_TOO_MANY_FILES => "too many open files",
        gpu_protocol::ERR_OUT_OF_MEMORY => "out of memory",
        gpu_protocol::ERR_INVALID_FD => "invalid file descriptor",
        gpu_protocol::ERR_CONNECTION_REFUSED => "connection refused",
        gpu_protocol::ERR_IS_A_DIRECTORY => "is a directory",
        gpu_protocol::ERR_HOST_TIMEOUT => "host timeout",
        gpu_protocol::ERR_UNSUPPORTED => "unsupported",
        _ => "unknown error",
    }
}

impl fmt::Display for GpuHostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CudaInit(e) => write!(f, "CUDA device init failed: {e}"),
            Self::CudaAlloc(r) => write!(f, "cuMemHostAlloc failed: {r:?}"),
            Self::CudaGetDevPtr(r) => {
                write!(f, "cuMemHostGetDevicePointer_v2 failed: {r:?}")
            }
            Self::CudaFreeMem(r) => write!(f, "cuMemFreeHost failed: {r:?}"),
            Self::Cudarc(e) => write!(f, "cudarc error: {e}"),
            Self::KernelNotFound(name) => write!(f, "kernel function not found: {name}"),
            Self::Verification { test, detail } => {
                write!(f, "{test}: verification failed: {detail}")
            }
            Self::Timeout { test, detail } => {
                write!(f, "{test}: timeout: {detail}")
            }
            Self::Hostcall(e) => write!(f, "hostcall error: {e}"),
            Self::KernelError(info) => write!(f, "GPU kernel error: {info}"),
            Self::KernelCrash => write!(f, "GPU kernel crashed (result buffer uninitialized)"),
        }
    }
}

impl std::error::Error for GpuHostError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::CudaInit(e) | Self::Cudarc(e) => Some(e),
            Self::Hostcall(e) => Some(e),
            _ => None,
        }
    }
}

impl From<cudarc::driver::DriverError> for GpuHostError {
    fn from(e: cudarc::driver::DriverError) -> Self {
        Self::Cudarc(e)
    }
}

impl From<crate::hostcall::HostcallError> for GpuHostError {
    fn from(e: crate::hostcall::HostcallError) -> Self {
        Self::Hostcall(e)
    }
}

/// Convenience type alias for `Result<T, GpuHostError>`.
pub type Result<T> = std::result::Result<T, GpuHostError>;
