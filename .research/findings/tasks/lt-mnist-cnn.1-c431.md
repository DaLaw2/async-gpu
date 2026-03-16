# lt-mnist-cnn.1: MNIST CNN training example

**Cycle**: 431 | **Theme**: larger-training | **Kind**: experiment | **Status**: done

## Summary

Created MNIST CNN training example with full backward propagation through both conv layers.
Architecture: Conv(1→16,3×3,pad=1) → ReLU → AvgPool(2) → Conv(16→32,3×3,pad=1) → ReLU → AvgPool(2) → Linear(1568→10).
GPU conv2d forward, CPU conv backward (weight + input gradients), GPU matmul for FC layer.

## Results

| Mode | Per-epoch | Total (5ep) | Test accuracy |
|------|-----------|-------------|---------------|
| GPU  | 73.4s     | 368.8s      | 96.4%         |
| CPU  | 107.5s    | 541.1s      | 96.4%         |
| Speedup | 1.47x  | 1.47x       | identical     |

Loss curve: 0.748 → 0.316 → 0.234 → 0.174 → 0.138 (monotonic decrease).
Train accuracy: 79.3% → 96.0%. Test accuracy: 88.8% → 96.4%.
Deterministic: GPU = CPU bit-for-bit.

## Key Implementation Details

- Full backward chain: d_logits → FC backward (GPU matmul) → d_feat → unpool2 → relu2_mask → conv2_wgrad + conv2_igrad → unpool1 → relu1_mask → conv1_wgrad
- `cpu_conv2d_igrad()` computes input gradient for backprop through conv layers
- Per-sample conv2d (not batched) — GPU speedup limited by kernel launch overhead
- Conv backward entirely on CPU — weight and input gradients

## Unexpected Discoveries

- Without conv1 training (frozen random filters): 89.5% test accuracy ceiling
- With conv1 training: 96.4% — learned features are critical even for simple CNNs
- Loss averaging bug (compound division inside batch loop) was inherited from earlier examples — caught and fixed before first correct run

## Impact on Downstream Tasks

- Proves GPU conv2d + autograd pipeline works end-to-end for real training
- 96.4% exceeds 95% target for larger-training epic
- Batched conv2d would improve GPU speedup significantly (reduce launch overhead)
