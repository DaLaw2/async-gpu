//! ONNX Runtime — parse, optimize, and execute ONNX models on GPU.
//!
//! - [`proto`]: Protobuf message types + parser (prost-based)
//! - [`executor`]: Graph executor dispatching ONNX nodes to GPU nn ops
//! - [`fusion`]: Graph fusion pass for operator pattern matching

pub mod proto;

#[cfg(feature = "nn")]
pub mod executor;

pub mod fusion;

// Re-export key types for convenience
pub use proto::{load_onnx, parse_onnx, OnnxAttr, OnnxError, OnnxGraph, OnnxModel, OnnxNode};

#[cfg(feature = "nn")]
pub use executor::{execute_onnx, OnnxSession};

pub use fusion::{apply_fusion, count_fusion_opportunities};
