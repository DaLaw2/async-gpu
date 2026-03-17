# onnx-executor.1: ONNX executor design
**Cycle**: 487 | **Theme**: onnx-executor | **Kind**: design | **Status**: done

## Summary
ONNX executor dispatches parsed graph nodes to existing nn ops via op_type matching.
Tensor lifecycle: initializers pre-uploaded to GPU, intermediates allocated on-demand.

## Architecture

```
OnnxExecutor::run(graph, input_tensors) → output_tensors
  1. Upload initializers to GPU as GpuTensor (cached across runs)
  2. Upload user input tensors
  3. For each node in topological order:
     a. Resolve input names → GpuTensor references
     b. Parse attributes (kernel_shape, strides, pads, axis, etc.)
     c. dispatch(op_type, inputs, attrs) → GpuTensor output
     d. Store output in tensor_map
  4. Return output tensors
```

## Op Dispatcher (match op_type)
- "Conv" → conv2d with parsed kernel_shape/strides/pads
- "MatMul" → matmul
- "Gemm" → matmul + optional bias + transpose
- "Relu" / "Sigmoid" / "Tanh" → activation ops
- "BatchNormalization" → batch_norm (eval mode)
- "MaxPool" → max_pool2d
- "Add" → elementwise_add
- "Reshape" → GpuTensor::reshape
- "Transpose" → matrix_transpose or GpuTensor::transpose
- "Softmax" → softmax op
- "GlobalAveragePool" → host-side average pooling
- "Flatten" → reshape to [batch, -1]
- "Shape" / "Gather" / "Unsqueeze" / "Squeeze" → shape manipulation (CPU-side)

## Tensor Lifecycle
- Initializers: uploaded once, shared across all inference runs
- Intermediate tensors: created per-inference, freed when no longer referenced
- Simple approach: keep all intermediates alive for the whole inference run
