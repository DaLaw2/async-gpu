//! ONNX model parser and graph representation.
//!
//! Parses ONNX protobuf files into a typed graph IR for GPU execution.
//! Uses `prost` for protobuf decoding with manually-defined message types
//! (no prost-build/protoc required).

use std::collections::HashMap;
use std::path::Path;

/// Error type for ONNX parsing.
#[derive(Debug)]
pub enum OnnxError {
    /// I/O error reading the ONNX file.
    Io(std::io::Error),
    /// Protobuf decode error.
    Decode(prost::DecodeError),
    /// Missing or invalid data in the ONNX model.
    Invalid(String),
}

impl std::fmt::Display for OnnxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OnnxError::Io(e) => write!(f, "ONNX I/O error: {e}"),
            OnnxError::Decode(e) => write!(f, "ONNX protobuf decode error: {e}"),
            OnnxError::Invalid(msg) => write!(f, "ONNX error: {msg}"),
        }
    }
}

impl std::error::Error for OnnxError {}

impl From<std::io::Error> for OnnxError {
    fn from(e: std::io::Error) -> Self {
        OnnxError::Io(e)
    }
}

impl From<prost::DecodeError> for OnnxError {
    fn from(e: prost::DecodeError) -> Self {
        OnnxError::Decode(e)
    }
}

// ============================================================
// Protobuf message types (matching onnx.proto3)
// ============================================================

/// ONNX model container.
#[derive(Clone, prost::Message)]
pub struct ModelProto {
    #[prost(int64, tag = "1")]
    pub ir_version: i64,
    #[prost(message, repeated, tag = "8")]
    pub opset_import: Vec<OperatorSetIdProto>,
    #[prost(string, tag = "2")]
    pub producer_name: String,
    #[prost(string, tag = "3")]
    pub producer_version: String,
    #[prost(string, tag = "4")]
    pub domain: String,
    #[prost(int64, tag = "5")]
    pub model_version: i64,
    #[prost(string, tag = "6")]
    pub doc_string: String,
    #[prost(message, optional, tag = "7")]
    pub graph: Option<GraphProto>,
}

/// Operator set version.
#[derive(Clone, prost::Message)]
pub struct OperatorSetIdProto {
    #[prost(string, tag = "1")]
    pub domain: String,
    #[prost(int64, tag = "2")]
    pub version: i64,
}

/// Computation graph.
#[derive(Clone, prost::Message)]
pub struct GraphProto {
    #[prost(message, repeated, tag = "1")]
    pub node: Vec<NodeProto>,
    #[prost(string, tag = "2")]
    pub name: String,
    #[prost(message, repeated, tag = "5")]
    pub initializer: Vec<TensorProto>,
    #[prost(string, tag = "10")]
    pub doc_string: String,
    #[prost(message, repeated, tag = "11")]
    pub input: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = "12")]
    pub output: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = "13")]
    pub value_info: Vec<ValueInfoProto>,
}

/// A single computation node.
#[derive(Clone, prost::Message)]
pub struct NodeProto {
    #[prost(string, repeated, tag = "1")]
    pub input: Vec<String>,
    #[prost(string, repeated, tag = "2")]
    pub output: Vec<String>,
    #[prost(string, tag = "3")]
    pub name: String,
    #[prost(string, tag = "4")]
    pub op_type: String,
    #[prost(message, repeated, tag = "5")]
    pub attribute: Vec<AttributeProto>,
    #[prost(string, tag = "6")]
    pub doc_string: String,
    #[prost(string, tag = "7")]
    pub domain: String,
}

/// Node attribute (kernel_size, strides, pads, etc.).
#[derive(Clone, prost::Message)]
pub struct AttributeProto {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(int32, tag = "20")]
    pub r#type: i32,
    #[prost(float, tag = "2")]
    pub f: f32,
    #[prost(int64, tag = "3")]
    pub i: i64,
    #[prost(bytes = "vec", tag = "4")]
    pub s: Vec<u8>,
    #[prost(message, optional, tag = "5")]
    pub t: Option<TensorProto>,
    #[prost(message, optional, tag = "6")]
    pub g: Option<Box<GraphProto>>,
    #[prost(float, repeated, tag = "7")]
    pub floats: Vec<f32>,
    #[prost(int64, repeated, tag = "8")]
    pub ints: Vec<i64>,
    #[prost(bytes = "vec", repeated, tag = "9")]
    pub strings: Vec<Vec<u8>>,
    #[prost(message, repeated, tag = "10")]
    pub tensors: Vec<TensorProto>,
    #[prost(string, tag = "13")]
    pub doc_string: String,
    #[prost(string, tag = "21")]
    pub ref_attr_name: String,
}

/// Tensor data (weights and constants).
#[derive(Clone, prost::Message)]
pub struct TensorProto {
    #[prost(int64, repeated, tag = "1")]
    pub dims: Vec<i64>,
    #[prost(int32, tag = "2")]
    pub data_type: i32,
    #[prost(float, repeated, tag = "4")]
    pub float_data: Vec<f32>,
    #[prost(int32, repeated, tag = "5")]
    pub int32_data: Vec<i32>,
    #[prost(int64, repeated, tag = "7")]
    pub int64_data: Vec<i64>,
    #[prost(string, tag = "8")]
    pub name: String,
    #[prost(bytes = "vec", tag = "9")]
    pub raw_data: Vec<u8>,
    #[prost(double, repeated, tag = "10")]
    pub double_data: Vec<f64>,
    #[prost(uint64, repeated, tag = "11")]
    pub uint64_data: Vec<u64>,
    #[prost(string, tag = "12")]
    pub doc_string: String,
}

/// Value type info (graph input/output).
#[derive(Clone, prost::Message)]
pub struct ValueInfoProto {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(message, optional, tag = "2")]
    pub r#type: Option<TypeProto>,
}

/// Type descriptor.
#[derive(Clone, prost::Message)]
pub struct TypeProto {
    #[prost(message, optional, tag = "1")]
    pub tensor_type: Option<TypeProtoTensor>,
}

/// Tensor type descriptor.
#[derive(Clone, prost::Message)]
pub struct TypeProtoTensor {
    #[prost(int32, tag = "1")]
    pub elem_type: i32,
    #[prost(message, optional, tag = "2")]
    pub shape: Option<TensorShapeProto>,
}

/// Tensor shape.
#[derive(Clone, prost::Message)]
pub struct TensorShapeProto {
    #[prost(message, repeated, tag = "1")]
    pub dim: Vec<TensorShapeProtoDimension>,
}

/// Shape dimension (can be fixed or symbolic).
#[derive(Clone, prost::Message)]
pub struct TensorShapeProtoDimension {
    #[prost(int64, tag = "1")]
    pub dim_value: i64,
    #[prost(string, tag = "2")]
    pub dim_param: String,
}

// ONNX data types
/// Float32.
pub const ONNX_FLOAT: i32 = 1;
/// Int64.
pub const ONNX_INT64: i32 = 7;
/// Double (f64).
pub const ONNX_DOUBLE: i32 = 11;

// ============================================================
// High-level ONNX graph representation
// ============================================================

/// Parsed ONNX model ready for execution.
pub struct OnnxModel {
    /// Model metadata.
    pub opset_version: i64,
    /// Computation graph.
    pub graph: OnnxGraph,
}

/// Computation graph with nodes in topological order.
pub struct OnnxGraph {
    /// Nodes in execution order (topological sort of the ONNX graph).
    pub nodes: Vec<OnnxNode>,
    /// Initializer weights: name → f32 data + shape.
    pub initializers: HashMap<String, (Vec<f32>, Vec<usize>)>,
    /// Graph input names (excluding initializers).
    pub inputs: Vec<String>,
    /// Graph output names.
    pub outputs: Vec<String>,
}

/// A single computation node.
pub struct OnnxNode {
    /// ONNX operator type (e.g., "Conv", "MatMul", "Relu").
    pub op_type: String,
    /// Input tensor names.
    pub inputs: Vec<String>,
    /// Output tensor names.
    pub outputs: Vec<String>,
    /// Attributes (parsed into typed values).
    pub attrs: HashMap<String, OnnxAttr>,
}

/// Typed attribute value.
#[derive(Clone, Debug)]
pub enum OnnxAttr {
    /// Integer value.
    Int(i64),
    /// Float value.
    Float(f32),
    /// String value.
    String(String),
    /// List of integers.
    Ints(Vec<i64>),
    /// List of floats.
    Floats(Vec<f32>),
    /// Tensor constant.
    Tensor(Vec<f32>, Vec<usize>),
}

// ============================================================
// Parsing
// ============================================================

/// Load an ONNX model from a file.
pub fn load_onnx(path: impl AsRef<Path>) -> Result<OnnxModel, OnnxError> {
    let bytes = std::fs::read(path.as_ref()).map_err(OnnxError::Io)?;
    parse_onnx(&bytes)
}

/// Parse ONNX model from bytes.
pub fn parse_onnx(bytes: &[u8]) -> Result<OnnxModel, OnnxError> {
    use prost::Message;
    let model = ModelProto::decode(bytes)?;

    let opset_version = model
        .opset_import
        .iter()
        .filter(|op| op.domain.is_empty()) // default domain
        .map(|op| op.version)
        .max()
        .unwrap_or(0);

    let graph_proto = model
        .graph
        .ok_or_else(|| OnnxError::Invalid("ONNX model has no graph".to_string()))?;

    let graph = build_graph(graph_proto)?;

    Ok(OnnxModel {
        opset_version,
        graph,
    })
}

fn build_graph(g: GraphProto) -> Result<OnnxGraph, OnnxError> {
    // Parse initializers
    let mut initializers = HashMap::new();
    for t in &g.initializer {
        let shape: Vec<usize> = t.dims.iter().map(|&d| d as usize).collect();
        let data = extract_f32_data(t)?;
        initializers.insert(t.name.clone(), (data, shape));
    }

    // Determine real inputs (graph inputs minus initializers)
    let inputs: Vec<String> = g
        .input
        .iter()
        .filter(|vi| !initializers.contains_key(&vi.name))
        .map(|vi| vi.name.clone())
        .collect();

    let outputs: Vec<String> = g.output.iter().map(|vi| vi.name.clone()).collect();

    // Parse nodes (ONNX guarantees topological order)
    let nodes: Vec<OnnxNode> = g
        .node
        .iter()
        .map(|n| OnnxNode {
            op_type: n.op_type.clone(),
            inputs: n.input.clone(),
            outputs: n.output.clone(),
            attrs: parse_attributes(&n.attribute),
        })
        .collect();

    Ok(OnnxGraph {
        nodes,
        initializers,
        inputs,
        outputs,
    })
}

fn extract_f32_data(t: &TensorProto) -> Result<Vec<f32>, OnnxError> {
    if !t.float_data.is_empty() {
        return Ok(t.float_data.clone());
    }
    if !t.raw_data.is_empty() {
        match t.data_type {
            ONNX_FLOAT => {
                let data: Vec<f32> = t
                    .raw_data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                Ok(data)
            }
            ONNX_INT64 => {
                let data: Vec<f32> = t
                    .raw_data
                    .chunks_exact(8)
                    .map(|c| {
                        i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                    })
                    .collect();
                Ok(data)
            }
            ONNX_DOUBLE => {
                let data: Vec<f32> = t
                    .raw_data
                    .chunks_exact(8)
                    .map(|c| {
                        f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                    })
                    .collect();
                Ok(data)
            }
            other => Err(OnnxError::Invalid(format!(
                "Unsupported ONNX data type: {other}"
            ))),
        }
    } else if !t.int64_data.is_empty() {
        Ok(t.int64_data.iter().map(|&v| v as f32).collect())
    } else if !t.double_data.is_empty() {
        Ok(t.double_data.iter().map(|&v| v as f32).collect())
    } else {
        // Empty tensor (e.g., shape-only initializer)
        Ok(vec![])
    }
}

fn parse_attributes(attrs: &[AttributeProto]) -> HashMap<String, OnnxAttr> {
    let mut map = HashMap::new();
    for a in attrs {
        let val = match a.r#type {
            1 => OnnxAttr::Float(a.f),                                        // FLOAT
            2 => OnnxAttr::Int(a.i),                                          // INT
            3 => OnnxAttr::String(String::from_utf8_lossy(&a.s).to_string()), // STRING
            6 => OnnxAttr::Floats(a.floats.clone()),                          // FLOATS
            7 => OnnxAttr::Ints(a.ints.clone()),                              // INTS
            _ => continue, // Skip unsupported attribute types
        };
        map.insert(a.name.clone(), val);
    }
    map
}

// ============================================================
// Utility
// ============================================================

impl OnnxModel {
    /// Print a summary of the model.
    pub fn summary(&self) {
        println!("ONNX Model (opset {})", self.opset_version);
        println!("  Inputs: {:?}", self.graph.inputs);
        println!("  Outputs: {:?}", self.graph.outputs);
        println!(
            "  Initializers: {} ({} total params)",
            self.graph.initializers.len(),
            self.graph
                .initializers
                .values()
                .map(|(d, _)| d.len())
                .sum::<usize>()
        );
        println!("  Nodes: {}", self.graph.nodes.len());

        // Op type histogram
        let mut ops: HashMap<&str, usize> = HashMap::new();
        for n in &self.graph.nodes {
            *ops.entry(&n.op_type).or_insert(0) += 1;
        }
        let mut ops_vec: Vec<_> = ops.into_iter().collect();
        ops_vec.sort_by(|a, b| b.1.cmp(&a.1));
        for (op, count) in &ops_vec {
            println!("    {op}: {count}");
        }
    }
}

impl OnnxNode {
    /// Get an integer attribute, with default.
    pub fn attr_int(&self, name: &str, default: i64) -> i64 {
        match self.attrs.get(name) {
            Some(OnnxAttr::Int(v)) => *v,
            _ => default,
        }
    }

    /// Get a list of integers attribute, with default.
    pub fn attr_ints(&self, name: &str) -> Vec<i64> {
        match self.attrs.get(name) {
            Some(OnnxAttr::Ints(v)) => v.clone(),
            _ => vec![],
        }
    }

    /// Get a float attribute, with default.
    pub fn attr_float(&self, name: &str, default: f32) -> f32 {
        match self.attrs.get(name) {
            Some(OnnxAttr::Float(v)) => *v,
            _ => default,
        }
    }
}
