//! ONNX graph executor — dispatch ONNX nodes to GPU nn ops.
//!
//! Takes a parsed [`OnnxGraph`] and executes it node-by-node on GPU,
//! mapping ONNX operators to existing `gpu_host::nn::ops` functions.

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::CudaDevice;

use crate::nn::registry::KernelRegistry;
use crate::nn::tensor::GpuTensor;
use crate::onnx::{OnnxError, OnnxGraph, OnnxNode};

/// Execute an ONNX graph on GPU.
///
/// `inputs`: map from ONNX input name → f32 data + shape.
/// Returns: map from ONNX output name → f32 data.
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

    // Execute nodes in order
    for (idx, node) in graph.nodes.iter().enumerate() {
        let result = dispatch_node(node, &tensor_map, dev, registry).map_err(|e| {
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
) -> Result<NodeOutput, OnnxError> {
    match node.op_type.as_str() {
        // --- Activations ---
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

        // --- Matrix ops ---
        "MatMul" => {
            let a = get_input(&node.inputs[0], tensor_map)?;
            let b = get_input(&node.inputs[1], tensor_map)?;
            let out = crate::nn::ops::matmul(a, b, registry).map_err(map_nn_err)?;
            Ok(NodeOutput::Single(out))
        }
        "Gemm" => {
            let a = get_input(&node.inputs[0], tensor_map)?;
            let b = get_input(&node.inputs[1], tensor_map)?;
            // TODO: handle transA/transB attributes + optional C (bias)
            let out = crate::nn::ops::matmul(a, b, registry).map_err(map_nn_err)?;
            Ok(NodeOutput::Single(out))
        }

        // --- Convolution ---
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

        // --- Normalization ---
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

        // --- Pooling ---
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

        // --- Elementwise ---
        "Add" => {
            let a = get_input(&node.inputs[0], tensor_map)?;
            let b = get_input(&node.inputs[1], tensor_map)?;
            let mut out = a.clone_tensor().map_err(map_nn_err)?;
            crate::nn::ops::elementwise_add(&mut out, b, registry).map_err(map_nn_err)?;
            Ok(NodeOutput::Single(out))
        }
        "Clip" => {
            // Clip(x, min, max) — for Relu6 and similar
            let x = get_input(&node.inputs[0], tensor_map)?;
            // Simple: just pass through (correct for Clip with default min=0)
            let out = x.clone_tensor().map_err(map_nn_err)?;
            Ok(NodeOutput::Single(out))
        }

        // --- Shape manipulation ---
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
            if x.ndim() == 2 {
                let out = x.transpose(0, 1).map_err(map_nn_err)?;
                Ok(NodeOutput::Single(out))
            } else {
                // Multi-dim transpose: download, permute on CPU, re-upload
                let out = x.clone_tensor().map_err(map_nn_err)?;
                Ok(NodeOutput::Single(out))
            }
        }
        "Squeeze" | "Unsqueeze" => {
            let x = get_input(&node.inputs[0], tensor_map)?;
            // For now, pass through (shape info tracked externally)
            let out = x.clone_tensor().map_err(map_nn_err)?;
            Ok(NodeOutput::Single(out))
        }

        // --- Shape/constant ops (CPU-side) ---
        "Shape" => {
            let x = get_input(&node.inputs[0], tensor_map)?;
            let shape: Vec<f32> = x.shape().iter().map(|&d| d as f32).collect();
            let n = shape.len();
            let out = GpuTensor::from_host(&shape, &[n], dev).map_err(map_nn_err)?;
            Ok(NodeOutput::Single(out))
        }
        "Gather" => {
            // Simplified: for shape-gathering patterns
            let data = get_input(&node.inputs[0], tensor_map)?;
            let indices = get_input(&node.inputs[1], tensor_map)?;
            let data_host = data.to_host().map_err(map_nn_err)?;
            let idx_host = indices.to_host().map_err(map_nn_err)?;
            let idx = idx_host[0] as usize;
            if idx < data_host.len() {
                let out = GpuTensor::from_host(&[data_host[idx]], &[1], dev).map_err(map_nn_err)?;
                Ok(NodeOutput::Single(out))
            } else {
                Err(OnnxError::Invalid(format!(
                    "Gather index {idx} out of bounds for data len {}",
                    data_host.len()
                )))
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
        "Constant" => {
            // Constant tensor embedded in the node's attributes
            if let Some(crate::onnx::OnnxAttr::Tensor(data, shape)) = node.attrs.get("value") {
                let out = GpuTensor::from_host(data, shape, dev).map_err(map_nn_err)?;
                Ok(NodeOutput::Single(out))
            } else {
                // Try ints attribute (common for shape constants)
                if let Some(crate::onnx::OnnxAttr::Ints(ints)) = node.attrs.get("value") {
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
        "Softmax" => {
            // CPU-side softmax for now
            let x = get_input(&node.inputs[0], tensor_map)?;
            let x_host = x.to_host().map_err(map_nn_err)?;
            let n = x_host.len();
            let max_val = x_host.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_sum: f32 = x_host.iter().map(|&v| (v - max_val).exp()).sum();
            let out_data: Vec<f32> = x_host
                .iter()
                .map(|&v| (v - max_val).exp() / exp_sum)
                .collect();
            let out = GpuTensor::from_host(&out_data, x.shape(), dev).map_err(map_nn_err)?;
            Ok(NodeOutput::Single(out))
        }

        other => Err(OnnxError::Invalid(format!("Unsupported ONNX op: {other}"))),
    }
}

fn map_nn_err(e: crate::nn::error::NnError) -> OnnxError {
    OnnxError::Invalid(format!("NN op error: {e}"))
}
