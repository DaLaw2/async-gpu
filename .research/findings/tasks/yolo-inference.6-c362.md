# yolo-inference.6: Backbone integration (Conv + BN + SiLU chain)
**Cycle**: 362 | **Theme**: yolo-inference | **Kind**: experiment | **Status**: done

## Summary

Implemented `YoloRunner` in `yolo_backbone.rs` — a complete backbone runner that chains
Conv2D (im2col+GEMM) + BN+SiLU + C2f blocks + SPPF + Upsample + Concat on GPU.
All backbone operations validated with synthetic weights on 32x32 input.

## Findings

### Q: Can the individual kernels be chained into a full backbone?
A: Yes. YoloRunner provides:
- `conv2d()`: im2col → GEMM with N-padding → CHW transpose
- `bn_silu()`: fused BN+SiLU kernel
- `conv_bn_silu()`: chains the above
- `maxpool2d()`, `upsample_2x()`, `concat()`, `chunk_split()`, `add()` (residual)

Test validates shapes through layers 0-3 + C2f + SPPF + Upsample + neck concat.
All shapes match expected YOLOv8-nano dimensions.
**Confidence**: high

### Q: Any architecture issues with the pipeline?
A: Two areas to note:
1. `chunk_split()` and `add()` currently round-trip through CPU (download + upload).
   Fine for inference correctness but adds latency. Could add GPU kernels later.
2. The transpose in `conv2d()` also round-trips through CPU. A GPU transpose kernel
   would eliminate this bottleneck for large tensors.
**Confidence**: high

## Impact on Downstream Tasks
- yolo-inference.7 (detection head + NMS) is now unblocked — needs .6 and .4 (both done)
- Need actual YOLO weights (`pip install ultralytics && python scripts/export_yolo.py`)
  for numerical validation against PyTorch reference
