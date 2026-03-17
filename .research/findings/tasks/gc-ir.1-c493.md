# gc-ir.1: Graph IR design + fusion rules
**Cycle**: 493 | **Theme**: gc-ir | **Kind**: design | **Status**: done

## Summary
The ONNX graph structure (OnnxGraph with nodes in topological order) serves as the base IR.
The graph compiler adds a fusion pass that identifies patterns (GEMM+Bias+Act, Conv+BN+Act)
and replaces matched subgraphs with fused dispatch nodes.

## Architecture

```
OnnxGraph (raw) → FusionPass → OptimizedGraph → Executor

FusionPass:
  1. Scan consecutive node pairs/triples
  2. Match against fusion rules (pattern → fused_op)
  3. Replace matched nodes with a single FusedNode
  4. Repeat until no more matches

FusionRules:
  - GEMM + Add(bias) + Relu → FusedGemmBiasRelu
  - GEMM + Add(bias) + Gelu → FusedGemmBiasGelu
  - Conv + BatchNormalization + Relu → FusedConvBnRelu
  - Add + Relu → FusedAddRelu
```

## Graph IR
Reuse OnnxNode directly. Add a `fused_op` field or extend `op_type` with "Fused_" prefix.
The executor already dispatches by op_type string — just add fused variants.

## Design Decision
No separate Graph IR — extend OnnxGraph with fusion annotations.
Pattern matching: simple linear scan of consecutive nodes.
Fused dispatch: map to existing PTX kernels (gemm_bias_gelu, gemm_bias_relu).
