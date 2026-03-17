# onnx-parser.1: ONNX protobuf schema investigation
**Cycle**: 483 | **Theme**: onnx-parser | **Kind**: investigation | **Status**: done

## Summary
ONNX uses protobuf3 for model serialization. Key types: ModelProto → GraphProto → NodeProto[].
Weights in TensorProto.raw_data (little-endian bytes) or float_data (repeated f32).
Use `prost` + `prost-build` for codegen, or parse manually (simpler, no build.rs complexity).

## Findings
### Q: How to parse ONNX in Rust?
A: Two options:
1. **prost**: Add prost + prost-build deps, download onnx.proto3, codegen in build.rs
2. **Manual**: ONNX protobuf is simple enough to parse with a minimal varint+wire parser

For speed of implementation, use `prost` — it handles all protobuf encoding details.
**Confidence**: high

### Q: Key ONNX data structures?
A: - ModelProto: { ir_version, opset_import, graph: GraphProto }
   - GraphProto: { node: [NodeProto], initializer: [TensorProto], input/output: [ValueInfoProto] }
   - NodeProto: { op_type: String, input: [String], output: [String], attribute: [AttributeProto] }
   - TensorProto: { dims: [i64], data_type: i32, raw_data: bytes, float_data: [f32] }
   - AttributeProto: { name, type, i/f/s/ints/floats/strings/t/graphs }
**Confidence**: high

## Design Decision
Use `prost` with onnx.proto3 from ONNX GitHub. Parse TensorProto raw_data → Vec<f32>.
Build OnnxGraph with nodes in topological order + initializers as HashMap.
