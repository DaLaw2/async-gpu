//! Error types for the nn module.

use std::fmt;

/// Errors specific to neural network operations.
#[derive(Debug)]
pub enum NnError {
    /// Shape mismatch between tensors (e.g., matmul dimension mismatch).
    ShapeMismatch {
        /// Description of what was expected.
        expected: String,
        /// Description of what was found.
        actual: String,
    },
    /// Dimension index out of range for the tensor's ndim.
    DimOutOfRange {
        /// The dimension index that was requested.
        dim: usize,
        /// The number of dimensions the tensor has.
        ndim: usize,
    },
    /// A required kernel function was not found in the registry.
    KernelNotFound {
        /// The kernel function name that was not found.
        name: &'static str,
    },
    /// CUDA driver error forwarded from cudarc.
    Cuda(cudarc::driver::DriverError),
}

impl fmt::Display for NnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NnError::ShapeMismatch { expected, actual } => {
                write!(f, "shape mismatch: expected {expected}, got {actual}")
            }
            NnError::DimOutOfRange { dim, ndim } => {
                write!(
                    f,
                    "dimension {dim} out of range for tensor with {ndim} dims"
                )
            }
            NnError::KernelNotFound { name } => {
                write!(f, "kernel not found in registry: {name}")
            }
            NnError::Cuda(e) => write!(f, "CUDA error: {e}"),
        }
    }
}

impl std::error::Error for NnError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            NnError::Cuda(e) => Some(e),
            _ => None,
        }
    }
}

impl From<cudarc::driver::DriverError> for NnError {
    fn from(e: cudarc::driver::DriverError) -> Self {
        NnError::Cuda(e)
    }
}

/// Result type for nn operations.
pub type Result<T> = std::result::Result<T, NnError>;
