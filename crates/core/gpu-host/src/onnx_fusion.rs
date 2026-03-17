//! ONNX graph fusion — identify and merge fusible operator patterns.
//!
//! Scans the ONNX graph for common patterns (e.g., MatMul+Add+Relu)
//! and replaces them with fused operator nodes that dispatch to
//! single-launch GPU kernels.

use crate::onnx::{OnnxGraph, OnnxNode};

/// Apply fusion passes to an ONNX graph.
///
/// Returns a new graph with fused nodes replacing matched patterns.
/// The original graph is consumed.
pub fn apply_fusion(mut graph: OnnxGraph) -> OnnxGraph {
    let nodes = std::mem::take(&mut graph.nodes);
    let fused = fuse_gemm_bias_activation(nodes);
    graph.nodes = fused;
    graph
}

/// Fuse MatMul + Add(bias) + Activation patterns.
///
/// Matches: MatMul → Add → {Relu, Gelu, Sigmoid}
/// Replaces with: FusedMatMulBiasRelu / FusedMatMulBiasGelu / etc.
fn fuse_gemm_bias_activation(nodes: Vec<OnnxNode>) -> Vec<OnnxNode> {
    let mut result = Vec::with_capacity(nodes.len());
    let mut i = 0;

    while i < nodes.len() {
        // Try 3-node pattern: MatMul → Add → Activation
        if i + 2 < nodes.len() {
            let n0 = &nodes[i];
            let n1 = &nodes[i + 1];
            let n2 = &nodes[i + 2];

            if (n0.op_type == "MatMul" || n0.op_type == "Gemm")
                && n1.op_type == "Add"
                && n1.inputs.contains(&n0.outputs[0])
                && is_activation(&n2.op_type)
                && n2.inputs.contains(&n1.outputs[0])
            {
                let fused_op = format!("Fused_MatMulBias{}", n2.op_type);
                let fused_node = OnnxNode {
                    op_type: fused_op,
                    inputs: vec![
                        n0.inputs[0].clone(), // A
                        n0.inputs[1].clone(), // B
                        // Bias: the other input to Add (not the MatMul output)
                        n1.inputs
                            .iter()
                            .find(|inp| *inp != &n0.outputs[0])
                            .cloned()
                            .unwrap_or_default(),
                    ],
                    outputs: n2.outputs.clone(),
                    attrs: n0.attrs.clone(),
                };
                result.push(fused_node);
                i += 3;
                continue;
            }
        }

        // Try 2-node pattern: Add + Activation
        if i + 1 < nodes.len() {
            let n0 = &nodes[i];
            let n1 = &nodes[i + 1];

            if n0.op_type == "Add"
                && is_activation(&n1.op_type)
                && n1.inputs.contains(&n0.outputs[0])
            {
                let fused_op = format!("Fused_Add{}", n1.op_type);
                let fused_node = OnnxNode {
                    op_type: fused_op,
                    inputs: n0.inputs.clone(),
                    outputs: n1.outputs.clone(),
                    attrs: n0.attrs.clone(),
                };
                result.push(fused_node);
                i += 2;
                continue;
            }
        }

        // No pattern matched — keep original node
        result.push(nodes[i].clone());
        i += 1;
    }

    result
}

fn is_activation(op: &str) -> bool {
    matches!(op, "Relu" | "Gelu" | "Sigmoid" | "Tanh" | "Silu")
}

/// Count how many fusion opportunities exist in the graph.
pub fn count_fusion_opportunities(graph: &OnnxGraph) -> usize {
    let mut count = 0;
    let mut i = 0;
    let nodes = &graph.nodes;

    while i < nodes.len() {
        if i + 2 < nodes.len() {
            let n0 = &nodes[i];
            let n1 = &nodes[i + 1];
            let n2 = &nodes[i + 2];
            if (n0.op_type == "MatMul" || n0.op_type == "Gemm")
                && n1.op_type == "Add"
                && n1.inputs.contains(&n0.outputs[0])
                && is_activation(&n2.op_type)
                && n2.inputs.contains(&n1.outputs[0])
            {
                count += 1;
                i += 3;
                continue;
            }
        }
        if i + 1 < nodes.len() {
            let n0 = &nodes[i];
            let n1 = &nodes[i + 1];
            if n0.op_type == "Add"
                && is_activation(&n1.op_type)
                && n1.inputs.contains(&n0.outputs[0])
            {
                count += 1;
                i += 2;
                continue;
            }
        }
        i += 1;
    }

    count
}
