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
    Verification { test: &'static str, detail: String },
    /// Operation timed out waiting for GPU response.
    Timeout { test: &'static str, detail: String },
    /// Hostcall buffer allocation error.
    Hostcall(crate::hostcall::HostcallError),
}

impl fmt::Display for GpuHostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CudaInit(e) => write!(f, "CUDA device init failed: {}", e),
            Self::CudaAlloc(r) => write!(f, "cuMemHostAlloc failed: {:?}", r),
            Self::CudaGetDevPtr(r) => {
                write!(f, "cuMemHostGetDevicePointer_v2 failed: {:?}", r)
            }
            Self::CudaFreeMem(r) => write!(f, "cuMemFreeHost failed: {:?}", r),
            Self::Cudarc(e) => write!(f, "cudarc error: {}", e),
            Self::KernelNotFound(name) => write!(f, "kernel function not found: {}", name),
            Self::Verification { test, detail } => {
                write!(f, "{}: verification failed: {}", test, detail)
            }
            Self::Timeout { test, detail } => {
                write!(f, "{}: timeout: {}", test, detail)
            }
            Self::Hostcall(e) => write!(f, "hostcall error: {}", e),
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
