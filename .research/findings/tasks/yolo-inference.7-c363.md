# yolo-inference.7: Detection head + NMS
**Cycle**: 363 | **Theme**: yolo-inference | **Kind**: experiment | **Status**: done

## Summary

Implemented detect head components: sigmoid GPU kernel, bias_add_chw GPU kernel,
DFL decode (host-side), NMS (host-side), and anchor grid generation.
All pass correctness tests.

## Findings

### Q: What new GPU kernels are needed for the detect head?
A: Two new kernels added to compute_cnn.rs:
- `sigmoid_forward`: element-wise 1/(1+exp(-x)), for class score activation
- `bias_add_chw`: per-channel bias add for bare Conv2d (no BN, no activation)
Both are simple elementwise ops (256 threads/block, 1D grid).
**Confidence**: high

### Q: Can DFL decode and NMS be done host-side?
A: Yes. DFL is a softmax+weighted-sum over only 4 bins (reg_max=4 for nano),
so host-side is trivial. NMS iterates over detections sorted by confidence,
computing IoU between pairs. For YOLOv8-nano's 8400 anchors, host-side NMS
is fast enough (< 1ms typically).
**Confidence**: high

### Q: What are the detect head's computational steps?
A: Per scale (P3/P4/P5):
1. Box branch: Conv3x3+BN+SiLU → Conv3x3+BN+SiLU → Conv1x1(bias, no BN) → 16ch
2. Class branch: Conv3x3+BN+SiLU → Conv3x3+BN+SiLU → Conv1x1(bias, no BN) → 80ch
3. Sigmoid on class scores
4. DFL decode on box logits: softmax over 4 bins → 4 coordinates per anchor
5. Box coordinate transform: offset → absolute pixel coords using anchor grid
6. NMS across all scales

## Impact on Downstream Tasks
- yolo-inference.8 (end-to-end demo) is now unblocked
- All YOLO building blocks are implemented:
  Conv2D, BN+SiLU, MaxPool, Upsample, Concat, Sigmoid, Bias add, DFL, NMS
- Missing: actual weight loading (needs ultralytics) and full-resolution run
