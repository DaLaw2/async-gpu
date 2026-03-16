# tq-resnet.1: Mini-ResNet full conv training with GPU backward on CIFAR-10
**Cycle**: 455 | **Theme**: tq-resnet | **Kind**: experiment | **Status**: done

## Summary
Rewrote the resnet-cifar example to train ALL 6 conv layers + 1×1 shortcut through GPU
`conv2d_backward()` instead of FC-only linear probing. With lr=0.3, bs=16, 20 epochs on
2000 CIFAR-10 images, the model reaches 32.1% test accuracy (>30% target) with monotonically
decreasing loss from 2.292 → 1.798.

## Findings
### Q: Can mini-ResNet train all conv layers via GPU conv2d_backward?
A: Yes. The full backward chain works: GAP → relu mask → residual split → conv2d_backward
through bb2b, bb2a, shortcut, bb1b, bb1a, conv1. Each layer's dWeight is accumulated across
the batch and applied via SGD. The 1×1 shortcut uses `conv2d_backward` with padding=0.
**Confidence**: high

### Q: What accuracy does full conv training achieve vs FC-only?
A: FC-only (linear probing) reached ~11.5% after 5 epochs (barely above random).
Full conv training reaches 32.1% after 20 epochs — 2.8× improvement, proving the
conv layers are learning useful features.
**Confidence**: high

## Unexpected Discoveries
- Without batch normalization, convergence is slow (~1.5% accuracy gain per epoch).
  BN would likely push accuracy much higher.
- Training is ~19-24s per epoch (2000 samples, bs=16) = ~150ms per sample per backward
  pass through 6 conv layers. Dominated by per-sample GPU conv2d_backward calls.

## Open Questions
- Batched conv2d backward for the ResNet architecture could reduce per-epoch time.
- Adding BN would improve convergence speed significantly.

## Impact on Downstream Tasks
- training-quality epic criterion "Mini-ResNet trains conv layers with residual gradient
  flow" and ">30% CIFAR-10" are both met.
