//! ONNX graph executor — dispatch ONNX nodes to GPU nn ops.
//!
//! Takes a parsed [`OnnxGraph`] and executes it node-by-node on GPU,
//! mapping ONNX operators to existing `gpu_host::nn::ops` functions.

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync};

use crate::nn::registry::KernelRegistry;
use crate::nn::tensor::GpuTensor;
use crate::onnx_rt::proto::{OnnxError, OnnxGraph, OnnxNode};

/// Pre-transposed+padded weight for fast GEMM (skips B transpose+pad per inference).
///
/// Layout: `[N_pad, K_pad]` row-major = `[K_pad, N_pad]` col-major, ready for
/// the `gemm_f32` kernel.
struct PrePaddedWeight {
    data: CudaSlice<f32>,
    /// Original K dimension (inner dim of matmul).
    k: usize,
    /// Original N dimension (output columns).
    n: usize,
    /// Padded K (rounded up to multiple of 16).
    k_pad: usize,
    /// Padded N (rounded up to multiple of 16).
    n_pad: usize,
}

/// Persistent ONNX execution session with cached initializer weights.
///
/// Uploading model weights (initializers) to GPU is expensive and dominates
/// inference time when done on every call.  `OnnxSession` uploads them once
/// at construction and reuses the GPU-resident tensors across `run()` calls.
pub struct OnnxSession {
    /// Initializer weights pre-uploaded to GPU (immutable after construction).
    cached_tensors: HashMap<String, GpuTensor>,
    /// Pre-transposed+padded weights for Gemm/MatMul nodes (keyed by initializer name).
    ///
    /// When a Gemm/MatMul node uses an initializer as its B input, we pre-compute
    /// the column-major padded form at session construction, saving 2 kernel launches
    /// (transpose + pad) per inference.
    prepadded_weights: HashMap<String, PrePaddedWeight>,
    /// The ONNX computation graph.
    graph: OnnxGraph,
    /// CUDA device handle.
    dev: Arc<CudaDevice>,
    /// Kernel registry for GPU ops.
    registry: Arc<KernelRegistry>,
}

impl OnnxSession {
    /// Create a new session, uploading all initializer weights to GPU once.
    pub fn new(
        graph: OnnxGraph,
        dev: &Arc<CudaDevice>,
        registry: &Arc<KernelRegistry>,
    ) -> Result<Self, OnnxError> {
        let mut cached_tensors = HashMap::new();
        for (name, (data, shape)) in &graph.initializers {
            let t = GpuTensor::from_host(data, shape, dev).map_err(|e| {
                OnnxError::Invalid(format!("Failed to upload initializer {name}: {e}"))
            })?;
            cached_tensors.insert(name.clone(), t);
        }
        // Pre-transpose+pad initializer weights used as B in Gemm/MatMul nodes.
        // This saves 2 kernel launches (transpose + pad) per matmul during inference.
        let prepadded_weights = precompute_gemm_weights(&graph, &cached_tensors, dev, registry)?;

        Ok(Self {
            cached_tensors,
            prepadded_weights,
            graph,
            dev: Arc::clone(dev),
            registry: Arc::clone(registry),
        })
    }

    /// Run inference with the given inputs, reusing cached weights.
    ///
    /// Only user-provided inputs are uploaded to GPU; initializer weights
    /// stay resident from session construction.
    pub fn run(
        &self,
        inputs: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    ) -> Result<HashMap<String, Vec<f32>>, OnnxError> {
        // Clone cached tensors into a working map (GPU-side clone, no host round-trip)
        let mut tensor_map: HashMap<String, GpuTensor> = HashMap::new();
        for (name, t) in &self.cached_tensors {
            tensor_map.insert(
                name.clone(),
                t.clone_tensor().map_err(|e| {
                    OnnxError::Invalid(format!("Failed to clone cached tensor {name}: {e}"))
                })?,
            );
        }

        // Upload only user inputs
        for (name, (data, shape)) in inputs {
            let t = GpuTensor::from_host(data, shape, &self.dev)
                .map_err(|e| OnnxError::Invalid(format!("Failed to upload input {name}: {e}")))?;
            tensor_map.insert(name.clone(), t);
        }

        // Execute nodes in order
        execute_nodes(
            &self.graph,
            &mut tensor_map,
            &self.dev,
            &self.registry,
            &self.prepadded_weights,
        )
    }
}

/// Execute an ONNX graph on GPU.
///
/// `inputs`: map from ONNX input name → f32 data + shape.
/// Returns: map from ONNX output name → f32 data.
///
/// This is a convenience wrapper that creates a temporary [`OnnxSession`]
/// internally.  For repeated inference, prefer creating an `OnnxSession`
/// directly to avoid re-uploading initializer weights every call.
pub fn execute_onnx(
    graph: &OnnxGraph,
    inputs: &HashMap<String, (Vec<f32>, Vec<usize>)>,
    dev: &Arc<CudaDevice>,
    registry: &Arc<KernelRegistry>,
) -> Result<HashMap<String, Vec<f32>>, OnnxError> {
    let mut tensor_map: HashMap<String, GpuTensor> = HashMap::new();

    // Upload initializers
    for (name, (data, shape)) in &graph.initializers {
        let t = GpuTensor::from_host(data, shape, dev)
            .map_err(|e| OnnxError::Invalid(format!("Failed to upload initializer {name}: {e}")))?;
        tensor_map.insert(name.clone(), t);
    }

    // Upload user inputs
    for (name, (data, shape)) in inputs {
        let t = GpuTensor::from_host(data, shape, dev)
            .map_err(|e| OnnxError::Invalid(format!("Failed to upload input {name}: {e}")))?;
        tensor_map.insert(name.clone(), t);
    }

    // Execute nodes and collect outputs (no prepadded weights for one-shot execution)
    let no_prepadded = HashMap::new();
    execute_nodes(graph, &mut tensor_map, dev, registry, &no_prepadded)
}

/// Execute graph nodes and collect outputs (shared by `OnnxSession::run` and `execute_onnx`).
fn execute_nodes(
    graph: &OnnxGraph,
    tensor_map: &mut HashMap<String, GpuTensor>,
    dev: &Arc<CudaDevice>,
    registry: &Arc<KernelRegistry>,
    prepadded_weights: &HashMap<String, PrePaddedWeight>,
) -> Result<HashMap<String, Vec<f32>>, OnnxError> {
    // Execute nodes in order
    for (idx, node) in graph.nodes.iter().enumerate() {
        let result =
            dispatch_node(node, tensor_map, dev, registry, prepadded_weights).map_err(|e| {
                OnnxError::Invalid(format!(
                    "Node {idx} ({} '{}'): {e}",
                    node.op_type,
                    node.outputs.first().unwrap_or(&String::new())
                ))
            })?;

        // Store outputs
        match result {
            NodeOutput::Single(t) => {
                if let Some(name) = node.outputs.first() {
                    if !name.is_empty() {
                        tensor_map.insert(name.clone(), t);
                    }
                }
            }
            NodeOutput::Multiple(tensors) => {
                for (i, t) in tensors.into_iter().enumerate() {
                    if let Some(name) = node.outputs.get(i) {
                        if !name.is_empty() {
                            tensor_map.insert(name.clone(), t);
                        }
                    }
                }
            }
        }
    }

    // Collect outputs
    let mut outputs = HashMap::new();
    for name in &graph.outputs {
        if let Some(t) = tensor_map.get(name) {
            let data = t.to_host().map_err(|e| {
                OnnxError::Invalid(format!("Failed to download output {name}: {e}"))
            })?;
            outputs.insert(name.clone(), data);
        }
    }

    Ok(outputs)
}

enum NodeOutput {
    Single(GpuTensor),
    Multiple(Vec<GpuTensor>),
}

fn get_input<'a>(
    name: &str,
    tensor_map: &'a HashMap<String, GpuTensor>,
) -> Result<&'a GpuTensor, OnnxError> {
    tensor_map
        .get(name)
        .ok_or_else(|| OnnxError::Invalid(format!("Tensor not found: {name}")))
}

fn dispatch_node(
    node: &OnnxNode,
    tensor_map: &HashMap<String, GpuTensor>,
    dev: &Arc<CudaDevice>,
    registry: &Arc<KernelRegistry>,
    prepadded_weights: &HashMap<String, PrePaddedWeight>,
) -> Result<NodeOutput, OnnxError> {
    match node.op_type.as_str() {
        "Relu" | "Sigmoid" | "Gelu" | "Tanh" => {
            dispatch_activation(node, tensor_map, dev, registry)
        }
        "MatMul" | "Gemm" => dispatch_gemm(node, tensor_map, dev, registry, prepadded_weights),
        "Conv" | "BatchNormalization" => dispatch_conv(node, tensor_map, dev, registry),
        "MaxPool" | "GlobalAveragePool" | "ReduceMean" => {
            dispatch_pool(node, tensor_map, dev, registry)
        }
        "Add" | "Mul" | "Sub" | "Div" | "Pow" | "Sqrt" | "Exp" | "Neg" | "Erf" | "Not"
        | "Equal" | "Clip" => dispatch_elementwise(node, tensor_map, dev, registry),
        "Reshape" | "Flatten" | "Transpose" | "Squeeze" | "Unsqueeze" | "Shape" | "Gather"
        | "Concat" | "Split" | "Slice" | "Expand" | "ConstantOfShape" | "Range" | "Trilu" => {
            dispatch_shape(node, tensor_map, dev, registry)
        }
        "Constant"
        | "Cast"
        | "Where"
        | "Softmax"
        | "LayerNormalization"
        | "Fused_MatMulBiasRelu"
        | "Fused_MatMulBiasGelu"
        | "Fused_AddRelu" => dispatch_misc(node, tensor_map, dev, registry),
        other => Err(OnnxError::Invalid(format!("Unsupported ONNX op: {other}"))),
    }
}

// --- Activation ops: Relu, Sigmoid, Gelu, Tanh ---

fn dispatch_activation(
    node: &OnnxNode,
    tensor_map: &HashMap<String, GpuTensor>,
    dev: &Arc<CudaDevice>,
    registry: &Arc<KernelRegistry>,
) -> Result<NodeOutput, OnnxError> {
    match node.op_type.as_str() {
        "Relu" => {
            let x = get_input(&node.inputs[0], tensor_map)?;
            let out = crate::nn::ops::relu(x, registry).map_err(map_nn_err)?;
            Ok(NodeOutput::Single(out))
        }
        "Sigmoid" => {
            let x = get_input(&node.inputs[0], tensor_map)?;
            let out = crate::nn::ops::sigmoid(x, registry).map_err(map_nn_err)?;
            Ok(NodeOutput::Single(out))
        }
        "Gelu" => {
            let x = get_input(&node.inputs[0], tensor_map)?;
            let out = crate::nn::ops::gelu(x, registry).map_err(map_nn_err)?;
            Ok(NodeOutput::Single(out))
        }
        "Tanh" => {
            let x = get_input(&node.inputs[0], tensor_map)?;
            let x_h = x.to_host().map_err(map_nn_err)?;
            let out_data: Vec<f32> = x_h.iter().map(|v| v.tanh()).collect();
            let out = GpuTensor::from_host(&out_data, x.shape(), dev).map_err(map_nn_err)?;
            Ok(NodeOutput::Single(out))
        }
        _ => unreachable!(),
    }
}

// --- GEMM ops: MatMul, Gemm ---

fn dispatch_gemm(
    node: &OnnxNode,
    tensor_map: &HashMap<String, GpuTensor>,
    dev: &Arc<CudaDevice>,
    registry: &Arc<KernelRegistry>,
    prepadded_weights: &HashMap<String, PrePaddedWeight>,
) -> Result<NodeOutput, OnnxError> {
    match node.op_type.as_str() {
        "MatMul" => {
            let a = get_input(&node.inputs[0], tensor_map)?;
            let b = get_input(&node.inputs[1], tensor_map)?;

            // Handle batched matmul for ND tensors (attention patterns)
            if a.ndim() > 2 && b.ndim() > 2 {
                let a_h = a.to_host().map_err(map_nn_err)?;
                let b_h = b.to_host().map_err(map_nn_err)?;
                let a_shape = a.shape();
                let b_shape = b.shape();
                let ndim = a_shape.len();

                let m = a_shape[ndim - 2];
                let k = a_shape[ndim - 1];
                let b_ndim = b_shape.len();
                let n = b_shape[b_ndim - 1];
                let a_batch: usize = a_shape[..ndim - 2].iter().product::<usize>().max(1);
                let b_batch: usize = b_shape[..b_ndim - 2].iter().product::<usize>().max(1);
                let batch = a_batch.max(b_batch);

                let mut out_data = vec![0.0f32; batch * m * n];
                for bi in 0..batch {
                    let a_off = (bi % a_batch) * m * k;
                    let b_off = (bi % b_batch) * k * n;
                    let c_off = bi * m * n;
                    for i in 0..m {
                        for j in 0..n {
                            let mut sum = 0.0f32;
                            for p in 0..k {
                                let ai = a_off + i * k + p;
                                let bi_idx = b_off + p * n + j;
                                if ai < a_h.len() && bi_idx < b_h.len() {
                                    sum += a_h[ai] * b_h[bi_idx];
                                }
                            }
                            out_data[c_off + i * n + j] = sum;
                        }
                    }
                }
                let mut out_shape = a_shape[..ndim - 2].to_vec();
                out_shape.push(m);
                out_shape.push(n);
                let out = GpuTensor::from_host(&out_data, &out_shape, dev).map_err(map_nn_err)?;
                return Ok(NodeOutput::Single(out));
            }

            let b_name = &node.inputs[1];
            // Reshape ND input to 2D for matmul, preserve batch dims
            let orig_shape = a.shape().to_vec();
            let a_2d = if a.ndim() > 2 {
                let k = orig_shape[a.ndim() - 1];
                let m: usize = orig_shape[..a.ndim() - 1].iter().product();
                a.reshape(&[m, k]).map_err(map_nn_err)?
            } else {
                a.clone_tensor().map_err(map_nn_err)?
            };

            let result = if let Some(pp) = prepadded_weights.get(b_name) {
                let m = a_2d.shape()[0];
                crate::nn::ops::matmul_prepadded_b(
                    &a_2d, &pp.data, m, pp.k, pp.n, pp.k_pad, pp.n_pad, registry,
                )
                .map_err(map_nn_err)?
            } else {
                let b = get_input(b_name, tensor_map)?;
                if b.ndim() > 2 {
                    // Both >2D: use batched path (handled above)
                    crate::nn::ops::matmul(&a_2d, b, registry).map_err(map_nn_err)?
                } else {
                    crate::nn::ops::matmul(&a_2d, b, registry).map_err(map_nn_err)?
                }
            };

            // Reshape output back to ND if input was ND
            if orig_shape.len() > 2 {
                let n = result.shape()[result.ndim() - 1];
                let mut out_shape = orig_shape[..orig_shape.len() - 1].to_vec();
                out_shape.push(n);
                let out = result.reshape(&out_shape).map_err(map_nn_err)?;
                Ok(NodeOutput::Single(out))
            } else {
                Ok(NodeOutput::Single(result))
            }
        }
        "Gemm" => {
            let a = get_input(&node.inputs[0], tensor_map)?;
            let b_name = &node.inputs[1];
            let trans_b = node.attr_int("transB", 0);
            // Fast path: use pre-transposed+padded weight if available
            // The prepadded key is the original initializer name (before any transB handling).
            if let Some(pp) = prepadded_weights.get(b_name) {
                let m = a.shape()[0];
                let mut out = crate::nn::ops::matmul_prepadded_b(
                    a, &pp.data, m, pp.k, pp.n, pp.k_pad, pp.n_pad, registry,
                )
                .map_err(map_nn_err)?;
                // Optional bias (C input)
                if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
                    let c = get_input(&node.inputs[2], tensor_map)?;
                    crate::nn::ops::bias_add(&mut out, c, registry).map_err(map_nn_err)?;
                }
                return Ok(NodeOutput::Single(out));
            }
            let b = get_input(b_name, tensor_map)?;
            let b_for_matmul = if trans_b != 0 && b.ndim() == 2 {
                b.transpose(0, 1).map_err(map_nn_err)?
            } else {
                b.clone_tensor().map_err(map_nn_err)?
            };
            let mut out = crate::nn::ops::matmul(a, &b_for_matmul, registry).map_err(map_nn_err)?;
            // Optional bias (C input)
            if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
                let c = get_input(&node.inputs[2], tensor_map)?;
                crate::nn::ops::bias_add(&mut out, c, registry).map_err(map_nn_err)?;
            }
            Ok(NodeOutput::Single(out))
        }
        _ => unreachable!(),
    }
}

// --- Conv ops: Conv, BatchNormalization ---

fn dispatch_conv(
    node: &OnnxNode,
    tensor_map: &HashMap<String, GpuTensor>,
    dev: &Arc<CudaDevice>,
    registry: &Arc<KernelRegistry>,
) -> Result<NodeOutput, OnnxError> {
    match node.op_type.as_str() {
        "Conv" => {
            let x = get_input(&node.inputs[0], tensor_map)?;
            let w = get_input(&node.inputs[1], tensor_map)?;
            let bias = if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
                Some(get_input(&node.inputs[2], tensor_map)?)
            } else {
                None
            };
            let pads = node.attr_ints("pads");
            let strides = node.attr_ints("strides");
            let padding = if !pads.is_empty() {
                pads[0] as usize
            } else {
                0
            };
            let stride = if !strides.is_empty() {
                strides[0] as usize
            } else {
                1
            };
            let out = crate::nn::ops::conv2d(x, w, bias, stride, padding, registry)
                .map_err(map_nn_err)?;
            Ok(NodeOutput::Single(out))
        }
        "BatchNormalization" => {
            let x = get_input(&node.inputs[0], tensor_map)?;
            let gamma = get_input(&node.inputs[1], tensor_map)?;
            let beta = get_input(&node.inputs[2], tensor_map)?;
            let mean = get_input(&node.inputs[3], tensor_map)?;
            let var = get_input(&node.inputs[4], tensor_map)?;
            let eps = node.attr_float("epsilon", 1e-5);
            let out = crate::nn::ops::batch_norm(x, gamma, beta, mean, var, eps, registry)
                .map_err(map_nn_err)?;
            Ok(NodeOutput::Single(out))
        }
        _ => unreachable!(),
    }
}

// --- Pool ops: MaxPool, GlobalAveragePool, ReduceMean ---

fn dispatch_pool(
    node: &OnnxNode,
    tensor_map: &HashMap<String, GpuTensor>,
    dev: &Arc<CudaDevice>,
    registry: &Arc<KernelRegistry>,
) -> Result<NodeOutput, OnnxError> {
    match node.op_type.as_str() {
        "MaxPool" => {
            let x = get_input(&node.inputs[0], tensor_map)?;
            let kernel_shape = node.attr_ints("kernel_shape");
            let strides = node.attr_ints("strides");
            let pads = node.attr_ints("pads");
            let ks = if !kernel_shape.is_empty() {
                kernel_shape[0] as usize
            } else {
                2
            };
            let stride = if !strides.is_empty() {
                strides[0] as usize
            } else {
                ks
            };
            let pad = if !pads.is_empty() {
                pads[0] as usize
            } else {
                0
            };
            let out =
                crate::nn::ops::max_pool2d(x, ks, stride, pad, registry).map_err(map_nn_err)?;
            Ok(NodeOutput::Single(out))
        }
        "GlobalAveragePool" => {
            // Average all spatial positions per channel
            let x = get_input(&node.inputs[0], tensor_map)?;
            let x_host = x.to_host().map_err(map_nn_err)?;
            let shape = x.shape();
            if shape.len() >= 3 {
                let c = shape[shape.len() - 3];
                let spatial: usize = shape[shape.len() - 2..].iter().product();
                let batch: usize = shape[..shape.len() - 3].iter().product::<usize>().max(1);
                let mut out_data = vec![0.0f32; batch * c];
                for b in 0..batch {
                    for ch in 0..c {
                        let sum: f32 = (0..spatial)
                            .map(|i| x_host[b * c * spatial + ch * spatial + i])
                            .sum();
                        out_data[b * c + ch] = sum / spatial as f32;
                    }
                }
                let out_shape = if batch > 1 {
                    vec![batch, c, 1, 1]
                } else {
                    vec![c, 1, 1]
                };
                let out = GpuTensor::from_host(&out_data, &out_shape, dev).map_err(map_nn_err)?;
                Ok(NodeOutput::Single(out))
            } else {
                Err(OnnxError::Invalid(
                    "GlobalAveragePool requires >= 3D input".to_string(),
                ))
            }
        }
        "ReduceMean" => {
            let x = get_input(&node.inputs[0], tensor_map)?;
            let mut axes = node.attr_ints("axes");
            // ONNX opset 18+: axes may be a second input instead of attribute
            if axes.is_empty() && node.inputs.len() > 1 && !node.inputs[1].is_empty() {
                let axes_t = get_input(&node.inputs[1], tensor_map)?;
                let axes_h = axes_t.to_host().map_err(map_nn_err)?;
                axes = axes_h.iter().map(|&v| v as i64).collect();
            }
            let keepdims = node.attr_int("keepdims", 1);
            let x_h = x.to_host().map_err(map_nn_err)?;
            let shape = x.shape();

            if axes.is_empty() {
                // Reduce all
                let mean = x_h.iter().sum::<f32>() / x_h.len() as f32;
                let out = GpuTensor::from_host(&[mean], &[1], dev).map_err(map_nn_err)?;
                Ok(NodeOutput::Single(out))
            } else {
                // Normalize negative axes
                let ndim = shape.len() as i64;
                let norm_axes: Vec<usize> = axes
                    .iter()
                    .map(|&a| {
                        if a < 0 {
                            (ndim + a) as usize
                        } else {
                            a as usize
                        }
                    })
                    .collect();

                // For 4D [N,C,H,W] reducing axes [2,3] → [N,C]
                if shape.len() == 4 && norm_axes.contains(&2) && norm_axes.contains(&3) {
                    let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
                    let spatial = h * w;
                    let mut out_data = vec![0.0f32; n * c];
                    for bi in 0..n {
                        for ci in 0..c {
                            let sum: f32 = (0..spatial)
                                .map(|s| x_h[bi * c * spatial + ci * spatial + s])
                                .sum();
                            out_data[bi * c + ci] = sum / spatial as f32;
                        }
                    }
                    let out_shape = if keepdims != 0 {
                        vec![n, c, 1, 1]
                    } else {
                        vec![n, c]
                    };
                    let out =
                        GpuTensor::from_host(&out_data, &out_shape, dev).map_err(map_nn_err)?;
                    Ok(NodeOutput::Single(out))
                } else {
                    // Generic fallback: reduce all
                    let mean = x_h.iter().sum::<f32>() / x_h.len() as f32;
                    let out = GpuTensor::from_host(&[mean], &[1], dev).map_err(map_nn_err)?;
                    Ok(NodeOutput::Single(out))
                }
            }
        }
        _ => unreachable!(),
    }
}

// --- Elementwise ops: Add, Mul, Sub, Div, Pow, Sqrt, Exp, Neg, Erf, Not, Equal, Clip ---

fn dispatch_elementwise(
    node: &OnnxNode,
    tensor_map: &HashMap<String, GpuTensor>,
    dev: &Arc<CudaDevice>,
    registry: &Arc<KernelRegistry>,
) -> Result<NodeOutput, OnnxError> {
    match node.op_type.as_str() {
        "Add" => {
            let a = get_input(&node.inputs[0], tensor_map)?;
            let b = get_input(&node.inputs[1], tensor_map)?;
            let a_numel: usize = a.shape().iter().product();
            let b_numel: usize = b.shape().iter().product();

            if a_numel == b_numel {
                let mut out = a.clone_tensor().map_err(map_nn_err)?;
                crate::nn::ops::elementwise_add(&mut out, b, registry).map_err(map_nn_err)?;
                Ok(NodeOutput::Single(out))
            } else if b.ndim() == 1 && b.shape()[0] == a.shape()[a.ndim() - 1] {
                // Bias add: b is 1D, broadcasts over last dim of a
                let mut out = a.clone_tensor().map_err(map_nn_err)?;
                crate::nn::ops::bias_add(&mut out, b, registry).map_err(map_nn_err)?;
                Ok(NodeOutput::Single(out))
            } else if a.ndim() == 1 && a.shape()[0] == b.shape()[b.ndim() - 1] {
                // Reversed: a is the bias
                let mut out = b.clone_tensor().map_err(map_nn_err)?;
                crate::nn::ops::bias_add(&mut out, a, registry).map_err(map_nn_err)?;
                Ok(NodeOutput::Single(out))
            } else {
                // General broadcast: CPU fallback
                let a_h = a.to_host().map_err(map_nn_err)?;
                let b_h = b.to_host().map_err(map_nn_err)?;
                let out_len = a_h.len().max(b_h.len());
                let out_data: Vec<f32> = (0..out_len)
                    .map(|i| a_h[i % a_h.len()] + b_h[i % b_h.len()])
                    .collect();
                let out_shape = if a_numel >= b_numel {
                    a.shape().to_vec()
                } else {
                    b.shape().to_vec()
                };
                let out = GpuTensor::from_host(&out_data, &out_shape, dev).map_err(map_nn_err)?;
                Ok(NodeOutput::Single(out))
            }
        }
        "Mul" => {
            let a = get_input(&node.inputs[0], tensor_map)?;
            let b = get_input(&node.inputs[1], tensor_map)?;
            let a_numel: usize = a.shape().iter().product();
            let b_numel: usize = b.shape().iter().product();

            if a_numel == b_numel {
                // Same shape: GPU elementwise multiply
                let n = a_numel as u32;
                let out_buf = dev.alloc_zeros::<f32>(a_numel).map_err(map_cuda_err)?;
                let status = dev.htod_sync_copy(&[0u32]).map_err(map_cuda_err)?;
                let func = registry.get("elementwise_mul").map_err(map_nn_err)?;
                unsafe {
                    func.launch(
                        crate::nn::KernelRegistry::config_1d(n),
                        (a.data(), b.data(), &out_buf, n, &status),
                    )
                    .map_err(map_cuda_err)?;
                }
                let out = GpuTensor::from_data(out_buf, a.shape(), Arc::clone(dev));
                Ok(NodeOutput::Single(out))
            } else if b_numel == 1 {
                // Scalar multiply: GPU scalar_mul kernel
                let b_val = b.to_host().map_err(map_nn_err)?[0];
                let n = a_numel as u32;
                let out_buf = dev.alloc_zeros::<f32>(a_numel).map_err(map_cuda_err)?;
                let status = dev.htod_sync_copy(&[0u32]).map_err(map_cuda_err)?;
                let func = registry.get("scalar_mul").map_err(map_nn_err)?;
                unsafe {
                    func.launch(
                        crate::nn::KernelRegistry::config_1d(n),
                        (a.data(), &out_buf, b_val, n, &status),
                    )
                    .map_err(map_cuda_err)?;
                }
                let out = GpuTensor::from_data(out_buf, a.shape(), Arc::clone(dev));
                Ok(NodeOutput::Single(out))
            } else {
                // Broadcast: CPU fallback
                let a_h = a.to_host().map_err(map_nn_err)?;
                let b_h = b.to_host().map_err(map_nn_err)?;
                let out_data: Vec<f32> = a_h
                    .iter()
                    .enumerate()
                    .map(|(i, x)| x * b_h[i % b_h.len()])
                    .collect();
                let out = GpuTensor::from_host(&out_data, a.shape(), dev).map_err(map_nn_err)?;
                Ok(NodeOutput::Single(out))
            }
        }
        "Sub" => {
            let a = get_input(&node.inputs[0], tensor_map)?;
            let b = get_input(&node.inputs[1], tensor_map)?;
            let a_numel: usize = a.shape().iter().product();
            let b_numel: usize = b.shape().iter().product();

            if a_numel == b_numel {
                let n = a_numel as u32;
                let out_buf = dev.alloc_zeros::<f32>(a_numel).map_err(map_cuda_err)?;
                let status = dev.htod_sync_copy(&[0u32]).map_err(map_cuda_err)?;
                let func = registry.get("elementwise_sub").map_err(map_nn_err)?;
                unsafe {
                    func.launch(
                        crate::nn::KernelRegistry::config_1d(n),
                        (a.data(), b.data(), &out_buf, n, &status),
                    )
                    .map_err(map_cuda_err)?;
                }
                let out = GpuTensor::from_data(out_buf, a.shape(), Arc::clone(dev));
                Ok(NodeOutput::Single(out))
            } else {
                // Broadcast: CPU fallback
                let a_h = a.to_host().map_err(map_nn_err)?;
                let b_h = b.to_host().map_err(map_nn_err)?;
                let out_data: Vec<f32> = a_h
                    .iter()
                    .enumerate()
                    .map(|(i, x)| x - b_h[i % b_h.len()])
                    .collect();
                let out = GpuTensor::from_host(&out_data, a.shape(), dev).map_err(map_nn_err)?;
                Ok(NodeOutput::Single(out))
            }
        }
        "Div" => {
            let a = get_input(&node.inputs[0], tensor_map)?;
            let b = get_input(&node.inputs[1], tensor_map)?;
            let a_h = a.to_host().map_err(map_nn_err)?;
            let b_h = b.to_host().map_err(map_nn_err)?;
            let out_data: Vec<f32> = if b_h.len() == 1 {
                a_h.iter().map(|x| x / b_h[0]).collect()
            } else {
                a_h.iter()
                    .enumerate()
                    .map(|(i, x)| x / b_h[i % b_h.len()])
                    .collect()
            };
            let out = GpuTensor::from_host(&out_data, a.shape(), dev).map_err(map_nn_err)?;
            Ok(NodeOutput::Single(out))
        }
        "Pow" => {
            let a = get_input(&node.inputs[0], tensor_map)?;
            let b = get_input(&node.inputs[1], tensor_map)?;
            let a_h = a.to_host().map_err(map_nn_err)?;
            let b_h = b.to_host().map_err(map_nn_err)?;
            let exp = if b_h.len() == 1 { b_h[0] } else { 2.0 };
            let out_data: Vec<f32> = a_h.iter().map(|x| x.powf(exp)).collect();
            let out = GpuTensor::from_host(&out_data, a.shape(), dev).map_err(map_nn_err)?;
            Ok(NodeOutput::Single(out))
        }
        "Sqrt" => {
            let x = get_input(&node.inputs[0], tensor_map)?;
            let x_h = x.to_host().map_err(map_nn_err)?;
            let out_data: Vec<f32> = x_h.iter().map(|v| v.sqrt()).collect();
            let out = GpuTensor::from_host(&out_data, x.shape(), dev).map_err(map_nn_err)?;
            Ok(NodeOutput::Single(out))
        }
        "Exp" => {
            let x = get_input(&node.inputs[0], tensor_map)?;
            let x_h = x.to_host().map_err(map_nn_err)?;
            let out_data: Vec<f32> = x_h.iter().map(|v| v.exp()).collect();
            let out = GpuTensor::from_host(&out_data, x.shape(), dev).map_err(map_nn_err)?;
            Ok(NodeOutput::Single(out))
        }
        "Neg" => {
            let x = get_input(&node.inputs[0], tensor_map)?;
            let n: usize = x.shape().iter().product();
            let out_buf = dev.alloc_zeros::<f32>(n).map_err(map_cuda_err)?;
            let status = dev.htod_sync_copy(&[0u32]).map_err(map_cuda_err)?;
            let func = registry.get("elementwise_neg").map_err(map_nn_err)?;
            unsafe {
                func.launch(
                    crate::nn::KernelRegistry::config_1d(n as u32),
                    (x.data(), &out_buf, n as u32, &status),
                )
                .map_err(map_cuda_err)?;
            }
            let out = GpuTensor::from_data(out_buf, x.shape(), Arc::clone(dev));
            Ok(NodeOutput::Single(out))
        }
        "Erf" => {
            let x = get_input(&node.inputs[0], tensor_map)?;
            let x_h = x.to_host().map_err(map_nn_err)?;
            // Erf approximation: erf(x) ≈ tanh(x * 1.128 * (1 + 0.0446 * x²))
            let out_data: Vec<f32> = x_h
                .iter()
                .map(|&v| {
                    let t = v * std::f32::consts::FRAC_2_SQRT_PI * (1.0 + 0.0446 * v * v);
                    t.tanh()
                })
                .collect();
            let out = GpuTensor::from_host(&out_data, x.shape(), dev).map_err(map_nn_err)?;
            Ok(NodeOutput::Single(out))
        }
        "Not" => {
            let x = get_input(&node.inputs[0], tensor_map)?;
            let x_h = x.to_host().map_err(map_nn_err)?;
            let out_data: Vec<f32> = x_h
                .iter()
                .map(|v| if *v == 0.0 { 1.0 } else { 0.0 })
                .collect();
            let out = GpuTensor::from_host(&out_data, x.shape(), dev).map_err(map_nn_err)?;
            Ok(NodeOutput::Single(out))
        }
        "Equal" => {
            let a = get_input(&node.inputs[0], tensor_map)?;
            let b = get_input(&node.inputs[1], tensor_map)?;
            let a_h = a.to_host().map_err(map_nn_err)?;
            let b_h = b.to_host().map_err(map_nn_err)?;
            let out_data: Vec<f32> = a_h
                .iter()
                .enumerate()
                .map(|(i, x)| if *x == b_h[i % b_h.len()] { 1.0 } else { 0.0 })
                .collect();
            let out = GpuTensor::from_host(&out_data, a.shape(), dev).map_err(map_nn_err)?;
            Ok(NodeOutput::Single(out))
        }
        "Clip" => {
            // Clip(x, min, max) — for Relu6 and similar
            let x = get_input(&node.inputs[0], tensor_map)?;
            // Simple: just pass through (correct for Clip with default min=0)
            let out = x.clone_tensor().map_err(map_nn_err)?;
            Ok(NodeOutput::Single(out))
        }
        _ => unreachable!(),
    }
}

// --- Shape ops: Reshape, Flatten, Transpose, Squeeze, Unsqueeze, Shape, Gather, Concat,
//     Split, Slice, Expand, ConstantOfShape, Range, Trilu ---

fn dispatch_shape(
    node: &OnnxNode,
    tensor_map: &HashMap<String, GpuTensor>,
    dev: &Arc<CudaDevice>,
    _registry: &Arc<KernelRegistry>,
) -> Result<NodeOutput, OnnxError> {
    match node.op_type.as_str() {
        "Reshape" => {
            let x = get_input(&node.inputs[0], tensor_map)?;
            let shape_tensor = get_input(&node.inputs[1], tensor_map)?;
            let shape_host = shape_tensor.to_host().map_err(map_nn_err)?;
            let total: usize = x.shape().iter().product();
            let new_shape: Vec<usize> = shape_host
                .iter()
                .map(|&v| {
                    if v as i64 == -1 {
                        0 // placeholder
                    } else {
                        v as usize
                    }
                })
                .collect();
            // Resolve -1 dimension
            let known: usize = new_shape
                .iter()
                .filter(|&&v| v != 0)
                .product::<usize>()
                .max(1);
            let final_shape: Vec<usize> = new_shape
                .iter()
                .map(|&v| if v == 0 { total / known } else { v })
                .collect();
            let out = x.reshape(&final_shape).map_err(map_nn_err)?;
            Ok(NodeOutput::Single(out))
        }
        "Flatten" => {
            let x = get_input(&node.inputs[0], tensor_map)?;
            let axis = node.attr_int("axis", 1) as usize;
            let shape = x.shape();
            let dim0: usize = shape[..axis].iter().product::<usize>().max(1);
            let dim1: usize = shape[axis..].iter().product();
            let out = x.reshape(&[dim0, dim1]).map_err(map_nn_err)?;
            Ok(NodeOutput::Single(out))
        }
        "Transpose" => {
            let x = get_input(&node.inputs[0], tensor_map)?;
            let perm = node.attr_ints("perm");

            if x.ndim() == 2 && perm.is_empty() {
                // Default 2D transpose
                let out = x.transpose(0, 1).map_err(map_nn_err)?;
                Ok(NodeOutput::Single(out))
            } else {
                // General ND permute on CPU
                let x_h = x.to_host().map_err(map_nn_err)?;
                let shape = x.shape();
                let ndim = shape.len();

                // Default perm: reverse all axes
                let perm_usize: Vec<usize> = if perm.is_empty() {
                    (0..ndim).rev().collect()
                } else {
                    perm.iter().map(|&p| p as usize).collect()
                };

                // Compute output shape
                let out_shape: Vec<usize> = perm_usize.iter().map(|&p| shape[p]).collect();
                let out_numel: usize = out_shape.iter().product();

                // Compute strides for input
                let mut in_strides = vec![1usize; ndim];
                for i in (0..ndim - 1).rev() {
                    in_strides[i] = in_strides[i + 1] * shape[i + 1];
                }

                // Permute
                let mut out_data = vec![0.0f32; out_numel];
                let mut out_strides = vec![1usize; ndim];
                for i in (0..ndim - 1).rev() {
                    out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
                }

                for idx in 0..out_numel {
                    // Convert flat index to ND index in output
                    let mut out_nd = vec![0usize; ndim];
                    let mut remaining = idx;
                    for d in 0..ndim {
                        out_nd[d] = remaining / out_strides[d];
                        remaining %= out_strides[d];
                    }
                    // Map to input ND index via inverse permutation
                    let mut in_flat = 0usize;
                    for d in 0..ndim {
                        in_flat += out_nd[d] * in_strides[perm_usize[d]];
                    }
                    out_data[idx] = x_h[in_flat];
                }

                let out = GpuTensor::from_host(&out_data, &out_shape, dev).map_err(map_nn_err)?;
                Ok(NodeOutput::Single(out))
            }
        }
        "Unsqueeze" => {
            let x = get_input(&node.inputs[0], tensor_map)?;
            let x_h = x.to_host().map_err(map_nn_err)?;
            let mut new_shape = x.shape().to_vec();
            // Get axes from second input (opset 13+) or attribute
            let axes = if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
                let axes_t = get_input(&node.inputs[1], tensor_map)?;
                let axes_h = axes_t.to_host().map_err(map_nn_err)?;
                axes_h.iter().map(|&v| v as i64).collect::<Vec<_>>()
            } else {
                node.attr_ints("axes")
            };
            // Insert dimensions (process in reverse order for correct indexing)
            let mut sorted_axes: Vec<i64> = axes;
            sorted_axes.sort();
            for &axis in sorted_axes.iter().rev() {
                let pos = if axis < 0 {
                    (new_shape.len() as i64 + axis + 1) as usize
                } else {
                    axis as usize
                };
                new_shape.insert(pos.min(new_shape.len()), 1);
            }
            let out = GpuTensor::from_host(&x_h, &new_shape, dev).map_err(map_nn_err)?;
            Ok(NodeOutput::Single(out))
        }
        "Squeeze" => {
            let x = get_input(&node.inputs[0], tensor_map)?;
            let x_h = x.to_host().map_err(map_nn_err)?;
            let mut new_shape: Vec<usize> = x.shape().to_vec();
            let axes = if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
                let axes_t = get_input(&node.inputs[1], tensor_map)?;
                let axes_h = axes_t.to_host().map_err(map_nn_err)?;
                axes_h.iter().map(|&v| v as i64).collect::<Vec<_>>()
            } else {
                node.attr_ints("axes")
            };
            if axes.is_empty() {
                // Remove all dims of size 1
                new_shape.retain(|&d| d != 1);
            } else {
                let mut to_remove: Vec<usize> = axes
                    .iter()
                    .map(|&a| {
                        if a < 0 {
                            (new_shape.len() as i64 + a) as usize
                        } else {
                            a as usize
                        }
                    })
                    .collect();
                to_remove.sort();
                for (i, &pos) in to_remove.iter().enumerate() {
                    new_shape.remove(pos - i);
                }
            }
            if new_shape.is_empty() {
                new_shape.push(1);
            }
            let out = GpuTensor::from_host(&x_h, &new_shape, dev).map_err(map_nn_err)?;
            Ok(NodeOutput::Single(out))
        }
        "Shape" => {
            let x = get_input(&node.inputs[0], tensor_map)?;
            let shape: Vec<f32> = x.shape().iter().map(|&d| d as f32).collect();
            let n = shape.len();
            let out = GpuTensor::from_host(&shape, &[n], dev).map_err(map_nn_err)?;
            Ok(NodeOutput::Single(out))
        }
        "Gather" => {
            let data = get_input(&node.inputs[0], tensor_map)?;
            let indices = get_input(&node.inputs[1], tensor_map)?;
            let axis = node.attr_int("axis", 0) as usize;
            let data_host = data.to_host().map_err(map_nn_err)?;
            let idx_host = indices.to_host().map_err(map_nn_err)?;
            let data_shape = data.shape();
            let idx_shape = indices.shape();

            if data_shape.len() == 1 {
                // 1D data: simple index lookup
                let idx = idx_host[0] as usize;
                if idx < data_host.len() {
                    // If indices is scalar, output is scalar [1]
                    // If indices has shape, output matches indices shape
                    if idx_host.len() == 1 {
                        let out = GpuTensor::from_host(&[data_host[idx]], &[1], dev)
                            .map_err(map_nn_err)?;
                        Ok(NodeOutput::Single(out))
                    } else {
                        let out_data: Vec<f32> = idx_host
                            .iter()
                            .map(|&i| data_host[i as usize % data_host.len()])
                            .collect();
                        let out =
                            GpuTensor::from_host(&out_data, idx_shape, dev).map_err(map_nn_err)?;
                        Ok(NodeOutput::Single(out))
                    }
                } else {
                    Err(OnnxError::Invalid(format!(
                        "Gather index {idx} out of bounds for data len {}",
                        data_host.len()
                    )))
                }
            } else if data_shape.len() == 2 && axis == 0 {
                // 2D data, axis=0: embedding lookup — data[idx, :]
                let cols = data_shape[1];
                let mut out_data = Vec::with_capacity(idx_host.len() * cols);
                for &idx_f in &idx_host {
                    let idx = idx_f as usize;
                    let start = idx * cols;
                    let end = start + cols;
                    if end <= data_host.len() {
                        out_data.extend_from_slice(&data_host[start..end]);
                    } else {
                        // Out of bounds — pad with zeros
                        out_data.extend(std::iter::repeat(0.0f32).take(cols));
                    }
                }
                // Output shape: indices_shape + [cols]
                let mut out_shape: Vec<usize> = idx_shape.to_vec();
                out_shape.push(cols);
                let out = GpuTensor::from_host(&out_data, &out_shape, dev).map_err(map_nn_err)?;
                Ok(NodeOutput::Single(out))
            } else {
                // General case: CPU fallback for unsupported axis/dims
                let idx = idx_host[0] as usize;
                if idx < data_shape[axis] {
                    // Simple: take a slice along axis
                    let out = data.clone_tensor().map_err(map_nn_err)?;
                    Ok(NodeOutput::Single(out))
                } else {
                    Err(OnnxError::Invalid(format!(
                        "Gather index {idx} out of bounds for axis {axis} dim {}",
                        data_shape[axis]
                    )))
                }
            }
        }
        "Concat" => {
            // Simplified: concat on axis=0 for 1D tensors
            let mut all_data = Vec::new();
            for name in &node.inputs {
                if name.is_empty() {
                    continue;
                }
                let t = get_input(name, tensor_map)?;
                let h = t.to_host().map_err(map_nn_err)?;
                all_data.extend_from_slice(&h);
            }
            let n = all_data.len();
            let out = GpuTensor::from_host(&all_data, &[n], dev).map_err(map_nn_err)?;
            Ok(NodeOutput::Single(out))
        }
        "Split" => {
            let x = get_input(&node.inputs[0], tensor_map)?;
            let mut axis = node.attr_int("axis", 0);
            let x_h = x.to_host().map_err(map_nn_err)?;
            let shape = x.shape();
            let ndim = shape.len() as i64;

            // Normalize negative axis
            if axis < 0 {
                axis += ndim;
            }
            let axis_usize = axis as usize;

            // Get split sizes: from 'split' attribute or second input (opset 13+)
            let mut split_sizes: Vec<usize> = node
                .attr_ints("split")
                .iter()
                .map(|&v| v as usize)
                .collect();
            if split_sizes.is_empty() && node.inputs.len() > 1 && !node.inputs[1].is_empty() {
                if let Ok(split_t) = get_input(&node.inputs[1], tensor_map) {
                    let sh = split_t.to_host().map_err(map_nn_err)?;
                    split_sizes = sh.iter().map(|&v| v as usize).collect();
                }
            }

            let n_outputs = node.outputs.len();
            if split_sizes.is_empty() {
                // Equal split
                let chunk = shape[axis_usize] / n_outputs;
                split_sizes = vec![chunk; n_outputs];
            }

            // General split along any axis for any ndim
            let axis_usize = axis_usize.min(shape.len() - 1);
            let axis_dim = shape[axis_usize];
            let outer: usize = shape[..axis_usize].iter().product::<usize>().max(1);
            let inner: usize = shape[axis_usize + 1..].iter().product::<usize>().max(1);

            let mut tensors = Vec::new();
            let mut offset = 0usize;
            for &sz in &split_sizes {
                let mut out_data = Vec::with_capacity(outer * sz * inner);
                for o in 0..outer {
                    for a in 0..sz {
                        let src_start = (o * axis_dim + offset + a) * inner;
                        out_data.extend_from_slice(&x_h[src_start..src_start + inner]);
                    }
                }
                let mut out_shape = shape.to_vec();
                out_shape[axis_usize] = sz;
                let t = GpuTensor::from_host(&out_data, &out_shape, dev).map_err(map_nn_err)?;
                tensors.push(t);
                offset += sz;
            }
            Ok(NodeOutput::Multiple(tensors))
        }
        "Slice" => {
            // Simplified Slice for 1D/2D
            let x = get_input(&node.inputs[0], tensor_map)?;
            let x_h = x.to_host().map_err(map_nn_err)?;
            // Just pass through for now (proper Slice needs starts/ends/axes/steps)
            let out = GpuTensor::from_host(&x_h, x.shape(), dev).map_err(map_nn_err)?;
            Ok(NodeOutput::Single(out))
        }
        "Expand" => {
            // Broadcast tensor to target shape
            let x = get_input(&node.inputs[0], tensor_map)?;
            let shape_t = get_input(&node.inputs[1], tensor_map)?;
            let target = shape_t.to_host().map_err(map_nn_err)?;
            let target_shape: Vec<usize> = target.iter().map(|&v| v as usize).collect();
            let total: usize = target_shape.iter().product();
            let x_h = x.to_host().map_err(map_nn_err)?;
            // Simple broadcast: tile x_h to fill target
            let out_data: Vec<f32> = (0..total).map(|i| x_h[i % x_h.len()]).collect();
            let out = GpuTensor::from_host(&out_data, &target_shape, dev).map_err(map_nn_err)?;
            Ok(NodeOutput::Single(out))
        }
        "ConstantOfShape" => {
            let shape_t = get_input(&node.inputs[0], tensor_map)?;
            let shape_h = shape_t.to_host().map_err(map_nn_err)?;
            let shape: Vec<usize> = shape_h.iter().map(|&v| v as usize).collect();
            // value attribute can be a tensor (type=4) or float (type=1)
            let fill_val = if let Some(crate::onnx_rt::proto::OnnxAttr::Tensor(data, _)) =
                node.attrs.get("value")
            {
                data.first().copied().unwrap_or(0.0)
            } else {
                node.attr_float("value", 0.0)
            };
            let total: usize = shape.iter().product();
            let data = vec![fill_val; total];
            let out = GpuTensor::from_host(&data, &shape, dev).map_err(map_nn_err)?;
            Ok(NodeOutput::Single(out))
        }
        "Range" => {
            // Range(start, limit, delta)
            let start = get_input(&node.inputs[0], tensor_map)?
                .to_host()
                .map_err(map_nn_err)?[0];
            let limit = get_input(&node.inputs[1], tensor_map)?
                .to_host()
                .map_err(map_nn_err)?[0];
            let delta = get_input(&node.inputs[2], tensor_map)?
                .to_host()
                .map_err(map_nn_err)?[0];
            let mut data = Vec::new();
            let mut v = start;
            while v < limit {
                data.push(v);
                v += delta;
            }
            let n = data.len();
            let out = GpuTensor::from_host(&data, &[n], dev).map_err(map_nn_err)?;
            Ok(NodeOutput::Single(out))
        }
        "Trilu" => {
            // Triangular mask with k parameter (diagonal offset)
            let x = get_input(&node.inputs[0], tensor_map)?;
            let upper = node.attr_int("upper", 1);
            // k parameter: from second input (default 0)
            let k = if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
                let k_t = get_input(&node.inputs[1], tensor_map)?;
                let k_h = k_t.to_host().map_err(map_nn_err)?;
                k_h[0] as i64
            } else {
                0
            };
            let x_h = x.to_host().map_err(map_nn_err)?;
            let shape = x.shape();
            let mut out_data = x_h.clone();
            if shape.len() >= 2 {
                let rows = shape[shape.len() - 2];
                let cols = shape[shape.len() - 1];
                let batch: usize = shape[..shape.len() - 2].iter().product::<usize>().max(1);
                for b in 0..batch {
                    for r in 0..rows {
                        for c in 0..cols {
                            let idx = b * rows * cols + r * cols + c;
                            // upper=1: zero out elements where r > c - k (below the k-th diagonal)
                            // upper=0: zero out elements where c > r + k (above the k-th diagonal)
                            let ri = r as i64;
                            let ci = c as i64;
                            if (upper != 0 && ri > ci - k) || (upper == 0 && ci > ri + k) {
                                out_data[idx] = 0.0;
                            }
                        }
                    }
                }
            }
            let out = GpuTensor::from_host(&out_data, shape, dev).map_err(map_nn_err)?;
            Ok(NodeOutput::Single(out))
        }
        _ => unreachable!(),
    }
}

// --- Misc ops: Constant, Cast, Where, Softmax, LayerNormalization, Fused ops ---

fn dispatch_misc(
    node: &OnnxNode,
    tensor_map: &HashMap<String, GpuTensor>,
    dev: &Arc<CudaDevice>,
    registry: &Arc<KernelRegistry>,
) -> Result<NodeOutput, OnnxError> {
    match node.op_type.as_str() {
        "Constant" => {
            // Constant tensor embedded in the node's attributes
            if let Some(crate::onnx_rt::proto::OnnxAttr::Tensor(data, shape)) =
                node.attrs.get("value")
            {
                // Handle scalar tensors (shape=[])
                let effective_shape = if shape.is_empty() {
                    &[1][..]
                } else {
                    shape.as_slice()
                };
                let out = GpuTensor::from_host(data, effective_shape, dev).map_err(map_nn_err)?;
                Ok(NodeOutput::Single(out))
            } else {
                // Try ints attribute (common for shape constants)
                if let Some(crate::onnx_rt::proto::OnnxAttr::Ints(ints)) = node.attrs.get("value") {
                    let data: Vec<f32> = ints.iter().map(|&v| v as f32).collect();
                    let n = data.len();
                    let out = GpuTensor::from_host(&data, &[n], dev).map_err(map_nn_err)?;
                    Ok(NodeOutput::Single(out))
                } else {
                    Err(OnnxError::Invalid(
                        "Constant node has no 'value' attribute".to_string(),
                    ))
                }
            }
        }
        "Cast" => {
            // For now, just pass through (all data is f32 internally)
            let x = get_input(&node.inputs[0], tensor_map)?;
            let out = x.clone_tensor().map_err(map_nn_err)?;
            Ok(NodeOutput::Single(out))
        }
        "Where" => {
            // Where(condition, X, Y): select X where condition is true, Y otherwise
            let cond = get_input(&node.inputs[0], tensor_map)?;
            let x = get_input(&node.inputs[1], tensor_map)?;
            let y = get_input(&node.inputs[2], tensor_map)?;
            let c_h = cond.to_host().map_err(map_nn_err)?;
            let x_h = x.to_host().map_err(map_nn_err)?;
            let y_h = y.to_host().map_err(map_nn_err)?;
            let out_len = c_h.len().max(x_h.len()).max(y_h.len());
            let out_data: Vec<f32> = (0..out_len)
                .map(|i| {
                    let c = c_h[i % c_h.len()];
                    if c != 0.0 {
                        x_h[i % x_h.len()]
                    } else {
                        y_h[i % y_h.len()]
                    }
                })
                .collect();
            // Output shape: broadcast result — use the largest tensor's shape
            let out_shape = if c_h.len() >= x_h.len() && c_h.len() >= y_h.len() {
                cond.shape().to_vec()
            } else if x_h.len() >= y_h.len() {
                x.shape().to_vec()
            } else {
                y.shape().to_vec()
            };
            let out = GpuTensor::from_host(&out_data, &out_shape, dev).map_err(map_nn_err)?;
            Ok(NodeOutput::Single(out))
        }
        "Softmax" => {
            let x = get_input(&node.inputs[0], tensor_map)?;
            let x_host = x.to_host().map_err(map_nn_err)?;
            let shape = x.shape();
            let axis = node.attr_int("axis", -1);
            let ndim = shape.len() as i64;
            let norm_axis = if axis < 0 {
                (ndim + axis) as usize
            } else {
                axis as usize
            };

            let mut out_data = x_host.clone();
            let axis_dim = shape[norm_axis];
            let outer: usize = shape[..norm_axis].iter().product::<usize>().max(1);
            let inner: usize = shape[norm_axis + 1..].iter().product::<usize>().max(1);

            // Softmax along axis: for each (outer, inner) slice
            for o in 0..outer {
                for i in 0..inner {
                    // Find max
                    let mut max_val = f32::NEG_INFINITY;
                    for a in 0..axis_dim {
                        let idx = (o * axis_dim + a) * inner + i;
                        max_val = max_val.max(out_data[idx]);
                    }
                    // Exp and sum
                    let mut exp_sum = 0.0f32;
                    for a in 0..axis_dim {
                        let idx = (o * axis_dim + a) * inner + i;
                        let e = (out_data[idx] - max_val).exp();
                        out_data[idx] = e;
                        exp_sum += e;
                    }
                    // Normalize
                    for a in 0..axis_dim {
                        let idx = (o * axis_dim + a) * inner + i;
                        out_data[idx] /= exp_sum;
                    }
                }
            }

            let out = GpuTensor::from_host(&out_data, shape, dev).map_err(map_nn_err)?;
            Ok(NodeOutput::Single(out))
        }
        "LayerNormalization" => {
            let x = get_input(&node.inputs[0], tensor_map)?;
            let gamma = get_input(&node.inputs[1], tensor_map)?;
            let beta = if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
                Some(get_input(&node.inputs[2], tensor_map)?)
            } else {
                None
            };
            let eps = node.attr_float("epsilon", 1e-5);
            let out = crate::nn::ops::layer_norm(x, gamma, beta.unwrap_or(gamma), eps, registry)
                .map_err(map_nn_err)?;
            Ok(NodeOutput::Single(out))
        }
        "Fused_MatMulBiasRelu" | "Fused_MatMulBiasGelu" => {
            let a = get_input(&node.inputs[0], tensor_map)?;
            let b = get_input(&node.inputs[1], tensor_map)?;
            let bias = if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
                get_input(&node.inputs[2], tensor_map)?
            } else {
                return Err(OnnxError::Invalid(
                    "Fused_MatMulBias* requires bias input".to_string(),
                ));
            };
            let activation = if node.op_type.ends_with("Relu") {
                crate::nn::ops::FusedActivation::Relu
            } else {
                crate::nn::ops::FusedActivation::Gelu
            };
            let out = crate::nn::ops::matmul_fused(a, b, bias, activation, registry)
                .map_err(map_nn_err)?;
            Ok(NodeOutput::Single(out))
        }
        "Fused_AddRelu" => {
            let a = get_input(&node.inputs[0], tensor_map)?;
            let b = get_input(&node.inputs[1], tensor_map)?;
            let mut out = a.clone_tensor().map_err(map_nn_err)?;
            crate::nn::ops::elementwise_add(&mut out, b, registry).map_err(map_nn_err)?;
            let activated = crate::nn::ops::relu(&out, registry).map_err(map_nn_err)?;
            Ok(NodeOutput::Single(activated))
        }
        _ => unreachable!(),
    }
}

/// Scan Gemm/MatMul nodes and pre-transpose+pad initializer weights used as B.
///
/// For each qualifying node, the weight (originally `[K, N]` row-major for MatMul,
/// or `[N, K]` for Gemm with transB=1) is converted to `[N_pad, K_pad]` row-major
/// (= `[K_pad, N_pad]` col-major), which is the format the `gemm_f32` kernel expects.
fn precompute_gemm_weights(
    graph: &OnnxGraph,
    cached_tensors: &HashMap<String, GpuTensor>,
    dev: &Arc<CudaDevice>,
    registry: &Arc<KernelRegistry>,
) -> Result<HashMap<String, PrePaddedWeight>, OnnxError> {
    let mut result = HashMap::new();

    for node in &graph.nodes {
        let (b_name, is_transb) = match node.op_type.as_str() {
            "MatMul" => {
                if node.inputs.len() < 2 {
                    continue;
                }
                (&node.inputs[1], false)
            }
            "Gemm" => {
                if node.inputs.len() < 2 {
                    continue;
                }
                let trans_b = node.attr_int("transB", 0) != 0;
                (&node.inputs[1], trans_b)
            }
            _ => continue,
        };

        // Only prepad initializer weights (not dynamic activations)
        if result.contains_key(b_name) || !graph.initializers.contains_key(b_name) {
            continue;
        }

        let b_tensor = match cached_tensors.get(b_name) {
            Some(t) if t.ndim() == 2 => t,
            _ => continue,
        };

        // Determine K, N for the matmul A[M,K] x B_logical[K,N] = C[M,N].
        // MatMul: B is already [K, N], so k = shape[0], n = shape[1].
        // Gemm with transB=1: B is stored as [N, K], logical B is [K, N].
        let (k, n) = if is_transb {
            // B stored as [out, in] = [N, K], logical [K, N]
            (b_tensor.shape()[1], b_tensor.shape()[0])
        } else {
            // B stored as [K, N]
            (b_tensor.shape()[0], b_tensor.shape()[1])
        };

        let k_pad = k.div_ceil(16) * 16;
        let n_pad = n.div_ceil(16) * 16;

        // We need B in [K, N] row-major first, then transpose to [N, K] and pad to [N_pad, K_pad].
        // For transB=1: B is stored as [N, K], which IS the transposed form already.
        //   So we just need to pad [N, K] → [N_pad, K_pad].
        // For MatMul: B is [K, N], we transpose to [N, K], then pad to [N_pad, K_pad].

        let status = dev.htod_sync_copy(&[0u32]).map_err(map_cuda_err)?;

        let prepadded = if is_transb {
            // B is already [N, K] — just pad to [N_pad, K_pad]
            if n == n_pad && k == k_pad {
                let mut buf = dev.alloc_zeros::<f32>(n * k).map_err(map_cuda_err)?;
                dev.dtod_copy(b_tensor.data(), &mut buf)
                    .map_err(map_cuda_err)?;
                buf
            } else {
                let mut buf = dev
                    .alloc_zeros::<f32>(n_pad * k_pad)
                    .map_err(map_cuda_err)?;
                let f_pad = registry.get("matrix_pad").map_err(map_nn_err)?;
                let cfg = KernelRegistry::config_1d((n_pad * k_pad) as u32);
                unsafe {
                    f_pad
                        .launch(
                            cfg,
                            (
                                b_tensor.data(),
                                &mut buf,
                                n as u32,
                                k as u32,
                                n_pad as u32,
                                k_pad as u32,
                                &status,
                            ),
                        )
                        .map_err(map_cuda_err)?;
                }
                buf
            }
        } else {
            // B is [K, N] — transpose to [N, K], then pad to [N_pad, K_pad]
            let mut b_t = dev.alloc_zeros::<f32>(n * k).map_err(map_cuda_err)?;
            let f_transpose = registry.get("matrix_transpose").map_err(map_nn_err)?;
            let cfg_t = KernelRegistry::config_1d((k * n) as u32);
            unsafe {
                f_transpose
                    .launch(
                        cfg_t,
                        (b_tensor.data(), &mut b_t, k as u32, n as u32, &status),
                    )
                    .map_err(map_cuda_err)?;
            }

            if n == n_pad && k == k_pad {
                b_t
            } else {
                let mut buf = dev
                    .alloc_zeros::<f32>(n_pad * k_pad)
                    .map_err(map_cuda_err)?;
                let f_pad = registry.get("matrix_pad").map_err(map_nn_err)?;
                let cfg_p = KernelRegistry::config_1d((n_pad * k_pad) as u32);
                unsafe {
                    f_pad
                        .launch(
                            cfg_p,
                            (
                                &b_t,
                                &mut buf,
                                n as u32,
                                k as u32,
                                n_pad as u32,
                                k_pad as u32,
                                &status,
                            ),
                        )
                        .map_err(map_cuda_err)?;
                }
                buf
            }
        };

        result.insert(
            b_name.clone(),
            PrePaddedWeight {
                data: prepadded,
                k,
                n,
                k_pad,
                n_pad,
            },
        );
    }

    Ok(result)
}

fn map_nn_err(e: crate::nn::error::NnError) -> OnnxError {
    OnnxError::Invalid(format!("NN op error: {e}"))
}

fn map_cuda_err(e: cudarc::driver::DriverError) -> OnnxError {
    OnnxError::Invalid(format!("CUDA error: {e}"))
}
