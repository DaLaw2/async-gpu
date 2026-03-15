# yolo-inference.1: YOLOv8-nano architecture mapping
**Cycle**: 357 | **Theme**: yolo-inference | **Kind**: investigation | **Status**: done

## Summary

YOLOv8-nano uses depth_multiple=0.33, width_multiple=0.25, max_channels=1024 applied to the base YOLOv8 config. The result is a 3.2M-parameter, 8.7 GFLOPs model with 16/32/64/128/256 channel progression through the backbone, C2f blocks with 1-2 bottleneck repeats, and a decoupled anchor-free detect head with separate box-regression and classification branches at three scales (P3/P4/P5).

## Findings

### Q: What is the exact layer-by-layer graph of YOLOv8-nano?

A: The base YOLOv8 YAML defines 23 indexed layers (0-22). The nano variant applies width_multiple=0.25 (channels scaled then rounded to nearest multiple of 8) and depth_multiple=0.33 (repeat counts scaled with min=1). Below is the full graph with nano-scaled values.

**Scaling rules:**
- Channels: `make_divisible(min(base_ch, 1024) * 0.25, 8)`
  - 64 * 0.25 = 16
  - 128 * 0.25 = 32
  - 256 * 0.25 = 64
  - 512 * 0.25 = 128
  - 1024 * 0.25 = 256
- Repeats: `max(round(n * 0.33), 1)`
  - 3 * 0.33 = 0.99 -> round -> 1
  - 6 * 0.33 = 1.98 -> round -> 2

**Conv block** = Conv2d + BatchNorm2d + SiLU (default activation).

**C2f block** internals (for output channels `c` and `n` bottleneck repeats):
- `cv1`: Conv 1x1, in_ch -> 2*c_hidden (where c_hidden = c // 2 for default e=0.5... but actually c_hidden = int(c * 0.5) for the hidden dim, and cv1 outputs `2 * c_hidden`)
  - Actually: `self.c = int(c2 * e)` where e=0.5 by default, so hidden_ch = c_out // 2
  - cv1: Conv 1x1, c_in -> 2 * hidden_ch  (i.e., c_in -> c_out)
  - The output is chunk-split into 2 halves of hidden_ch each
- `n` Bottleneck modules, each:
  - Conv 3x3, hidden_ch -> hidden_ch
  - Conv 3x3, hidden_ch -> hidden_ch
  - Optional residual add (shortcut=True in backbone, False in head neck)
- `cv2`: Conv 1x1, (2 + n) * hidden_ch -> c_out

**SPPF block** internals (for channels `c`, pool kernel `k=5`):
- cv1: Conv 1x1, c_in -> c_hidden (c_hidden = c_in // 2)
- MaxPool2d(k=5, s=1, p=2) applied 3 times sequentially
- Concatenate: [cv1_out, pool1, pool2, pool3] -> 4 * c_hidden channels
- cv2: Conv 1x1, 4 * c_hidden -> c_out

**Detect head** (decoupled, anchor-free):
- For each scale (P3, P4, P5), two parallel branches:
  - **cv2 (box regression)**: Conv 3x3 -> Conv 3x3 -> Conv2d 1x1, output = 4 * reg_max channels
    - c2 = max(16, ch[0]//4, reg_max*4) where reg_max = ch[0]//16
    - For nano: ch[0]=64, reg_max = 64//16 = 4, c2 = max(16, 16, 16) = 16
    - Output per scale: 4 * 4 = 16 channels (box distribution)
  - **cv3 (classification)**: Conv 3x3 -> Conv 3x3 -> Conv2d 1x1, output = nc channels
    - c3 = max(ch[0], min(nc, 100)) = max(64, min(80, 100)) = max(64, 80) = 80
    - Output per scale: 80 channels (COCO classes)

**Confidence**: high

### Q: What are the tensor shapes at each stage?

A: Input 640x640x3. Spatial dimensions halve at each stride-2 conv. Channel counts are nano-scaled.

See the Layer Graph section below for complete shapes.

| Stage | Layer Index | Spatial (HxW) | Channels |
|-------|------------|---------------|----------|
| Input | - | 640x640 | 3 |
| P1/2 | 0 | 320x320 | 16 |
| P2/4 | 1 | 160x160 | 32 |
| C2f | 2 | 160x160 | 32 |
| P3/8 | 3 | 80x80 | 64 |
| C2f | 4 | 80x80 | 64 |
| P4/16 | 5 | 40x40 | 128 |
| C2f | 6 | 40x40 | 128 |
| P5/32 | 7 | 20x20 | 256 |
| C2f | 8 | 20x20 | 256 |
| SPPF | 9 | 20x20 | 256 |
| Upsample | 10 | 40x40 | 256 |
| Concat(+P4) | 11 | 40x40 | 384 (256+128) |
| C2f | 12 | 40x40 | 128 |
| Upsample | 13 | 80x80 | 128 |
| Concat(+P3) | 14 | 80x80 | 192 (128+64) |
| C2f (P3 out) | 15 | 80x80 | 64 |
| Conv s2 | 16 | 40x40 | 64 |
| Concat(+12) | 17 | 40x40 | 192 (64+128) |
| C2f (P4 out) | 18 | 40x40 | 128 |
| Conv s2 | 19 | 20x20 | 128 |
| Concat(+9) | 20 | 20x20 | 384 (128+256) |
| C2f (P5 out) | 21 | 20x20 | 256 |
| Detect | 22 | multi-scale | see below |

**Detect head output grid sizes:**
- P3: 80x80 = 6400 anchors (small objects)
- P4: 40x40 = 1600 anchors (medium objects)
- P5: 20x20 = 400 anchors (large objects)
- Total: 8400 predictions, each with 4*reg_max + nc = 16 + 80 = 96 values

**Confidence**: high

### Q: Which layers share the same kernel type?

A: Grouped by operation type:

| Operation | Count | Details |
|-----------|-------|---------|
| Conv2d 3x3 s2 + BN + SiLU | 6 | Backbone: layers 0,1,3,5,7; Head: layers 16, 19 — total 7 |
| Conv2d 1x1 s1 + BN + SiLU | ~14 | C2f cv1 and cv2 in each of 8 C2f blocks (16 total); SPPF cv1, cv2 (2 total) = 18 |
| Conv2d 3x3 s1 + BN + SiLU | ~18 | Inside C2f bottlenecks: each bottleneck has 2x Conv 3x3. Backbone C2f blocks have (1+2+2+1)=6 bottlenecks = 12 conv3x3. Head C2f blocks have (1+1+1+1)=4 bottlenecks = 8 conv3x3. Plus detect head cv2/cv3 branches: 6 scales * 2 branches * 2 conv = 12 more (these are Conv 3x3) |
| Conv2d 1x1 (no BN, no act) | 6 | Detect head final projection: 3 scales * 2 branches = 6 bare Conv2d 1x1 |
| MaxPool2d 5x5 s1 p2 | 1 | SPPF (applied 3 times but it's one nn.MaxPool2d instance) |
| nn.Upsample nearest 2x | 2 | Layers 10, 13 |
| Concat | 4 | Layers 11, 14, 17, 20 |

**Detailed Conv breakdown (nano):**

Backbone Conv 3x3 stride 2 (downsampling):
1. 3 -> 16, k3 s2 p1
2. 16 -> 32, k3 s2 p1
3. 32 -> 64, k3 s2 p1
4. 64 -> 128, k3 s2 p1
5. 128 -> 256, k3 s2 p1

Head Conv 3x3 stride 2 (downsampling):
6. 64 -> 64, k3 s2 p1
7. 128 -> 128, k3 s2 p1

**Confidence**: high

### Q: Total parameter count and memory footprint?

A: YOLOv8n has **3,157,200 parameters** (3.2M) and 8.7-8.9 GFLOPs at 640x640 input.

**Parameter breakdown by component (approximate):**

| Component | Parameters (approx) | % of total |
|-----------|---------------------|-----------|
| Backbone Conv layers (5 downsampling convs) | ~30K | 1.0% |
| Backbone C2f blocks (4 blocks) | ~180K | 5.7% |
| SPPF block | ~200K | 6.3% |
| Neck/Head C2f blocks (4 blocks) | ~350K | 11.1% |
| Neck Conv layers (2 downsampling) | ~45K | 1.4% |
| Detect cv2 (box) branches (3 scales) | ~30K | 1.0% |
| Detect cv3 (class) branches (3 scales) | ~2,300K | 72.9% |
| BatchNorm parameters | ~20K | 0.6% |

The detect head cv3 (classification) branches dominate because c3=80 channels with 3x3 convs at each of the three scales, especially at P3 (64-in) contributing ~80*80*3*3 + 80*80*3*3 + 80*80*1 per scale.

**Memory footprint (FP32):**
- Parameters: 3.2M * 4 bytes = ~12.8 MB
- Parameters (FP16): ~6.4 MB

**Activation memory at batch=1 FP32 (approximate peak):**
- Layer 0 output: 320*320*16*4 = 6.25 MB
- Layer 4 output (P3): 80*80*64*4 = 1.56 MB
- Layer 6 output (P4): 40*40*128*4 = 0.78 MB
- Layer 9 output (SPPF/P5): 20*20*256*4 = 0.39 MB
- Neck intermediate tensors: various, ~2-4 MB total
- Peak activation memory: ~15-20 MB (when early large tensors are live)

**Confidence**: medium (parameter breakdown is approximate; detect head dominance needs verification)

### Q: Optimal execution order considering memory reuse?

A: The model is a DAG (not purely sequential) due to skip connections in the FPN/PAN neck. Key constraints:

**Skip connections requiring tensor persistence:**
- Layer 4 output (P3, 80x80x64) must persist until layer 14 (Concat)
- Layer 6 output (P4, 40x40x128) must persist until layer 11 (Concat)
- Layer 9 output (SPPF, 20x20x256) must persist until layer 20 (Concat)
- Layer 12 output (40x40x128) must persist until layer 17 (Concat)

**Tensor lifetime analysis (layer indices):**

| Tensor | Shape (B=1, FP32) | Size | Created | Last used | Lifetime |
|--------|-------------------|------|---------|-----------|----------|
| L0 out | 320x320x16 | 6.25 MB | 0 | 1 | short |
| L1 out | 160x160x32 | 3.13 MB | 1 | 2 | short |
| L2 out | 160x160x32 | 3.13 MB | 2 | 3 | short |
| L3 out | 80x80x64 | 1.56 MB | 3 | 4 | short |
| **L4 out** | **80x80x64** | **1.56 MB** | **4** | **14** | **long** |
| L5 out | 40x40x128 | 0.78 MB | 5 | 6 | short |
| **L6 out** | **40x40x128** | **0.78 MB** | **6** | **11** | **long** |
| L7 out | 20x20x256 | 0.39 MB | 7 | 8 | short |
| L8 out | 20x20x256 | 0.39 MB | 8 | 9 | short |
| **L9 out** | **20x20x256** | **0.39 MB** | **9** | **20** | **long** |
| L10 out | 40x40x256 | 1.56 MB | 10 | 11 | short |
| L11 out | 40x40x384 | 2.34 MB | 11 | 12 | short |
| **L12 out** | **40x40x128** | **0.78 MB** | **12** | **17** | **medium** |
| L13 out | 80x80x128 | 3.13 MB | 13 | 14 | short |
| L14 out | 80x80x192 | 4.69 MB | 14 | 15 | short |
| L15 out | 80x80x64 | 1.56 MB | 15 | 16,detect | medium |
| L16 out | 40x40x64 | 0.39 MB | 16 | 17 | short |
| L17 out | 40x40x192 | 1.17 MB | 17 | 18 | short |
| L18 out | 40x40x128 | 0.78 MB | 18 | 19,detect | medium |
| L19 out | 20x20x128 | 0.10 MB | 19 | 20 | short |
| L20 out | 20x20x384 | 0.59 MB | 20 | 21 | short |
| L21 out | 20x20x256 | 0.39 MB | 21 | detect | short |

**Memory reuse opportunities:**
1. L0 buffer (6.25 MB) can be reused after layer 1 completes — this is the largest single tensor
2. L1 and L2 buffers (3.13 MB each) can be reused after layer 3
3. Short-lived tensors in the head can reuse backbone buffers
4. The concat outputs (L11, L14, L17, L20) are large but short-lived

**Peak memory = live tensors at the most memory-intensive point:**
- At layer 14 (Concat P3): L4(1.56) + L6(0.78) + L9(0.39) + L12(0.78) + L13(3.13) + L14(4.69) = ~11.3 MB activations
- Plus parameters: ~12.8 MB
- **Total peak: ~25-30 MB** at FP32, batch=1

**Optimal buffer allocation (5 reusable buffers):**
- Buffer A (6.25 MB): L0 -> L2 -> L13 -> L14 (large spatial tensors)
- Buffer B (3.13 MB): L1 -> L3 -> L10 -> L11 (medium spatial)
- Buffer C (1.56 MB): L5 -> L8 -> L15 -> L16 -> L19 (small, short-lived)
- Buffer D (0.78 MB): pinned for L6 (persists until layer 11)
- Buffer E (0.39 MB): pinned for L9 (persists until layer 20)
- Plus L4 pinned (1.56 MB, persists until layer 14)
- Plus L12 pinned (0.78 MB, persists until layer 17)

**Confidence**: medium (buffer scheme is a proposal; actual optimal allocation depends on implementation constraints)

## Layer Graph

Complete layer-by-layer graph for YOLOv8-nano (640x640 input, 80 COCO classes):

| Idx | From | Module | Kernel | Stride | Pad | Ch_in | Ch_out | Repeats | Output Shape | Notes |
|-----|------|--------|--------|--------|-----|-------|--------|---------|-------------|-------|
| 0 | -1 | Conv(Conv2d+BN+SiLU) | 3x3 | 2 | 1 | 3 | 16 | 1 | 320x320x16 | P1/2 |
| 1 | -1 | Conv(Conv2d+BN+SiLU) | 3x3 | 2 | 1 | 16 | 32 | 1 | 160x160x32 | P2/4 |
| 2 | -1 | C2f | - | - | - | 32 | 32 | 1 bottleneck | 160x160x32 | shortcut=True |
| 3 | -1 | Conv(Conv2d+BN+SiLU) | 3x3 | 2 | 1 | 32 | 64 | 1 | 80x80x64 | P3/8 |
| 4 | -1 | C2f | - | - | - | 64 | 64 | 2 bottlenecks | 80x80x64 | shortcut=True |
| 5 | -1 | Conv(Conv2d+BN+SiLU) | 3x3 | 2 | 1 | 64 | 128 | 1 | 40x40x128 | P4/16 |
| 6 | -1 | C2f | - | - | - | 128 | 128 | 2 bottlenecks | 40x40x128 | shortcut=True |
| 7 | -1 | Conv(Conv2d+BN+SiLU) | 3x3 | 2 | 1 | 128 | 256 | 1 | 20x20x256 | P5/32 |
| 8 | -1 | C2f | - | - | - | 256 | 256 | 1 bottleneck | 20x20x256 | shortcut=True |
| 9 | -1 | SPPF | 5x5 pool | 1 | 2 | 256 | 256 | 1 | 20x20x256 | 3x sequential MaxPool |
| 10 | -1 | nn.Upsample | - | - | - | 256 | 256 | 1 | 40x40x256 | nearest, scale=2 |
| 11 | [-1,6] | Concat | - | - | - | 256+128 | 384 | 1 | 40x40x384 | cat P4 features |
| 12 | -1 | C2f | - | - | - | 384 | 128 | 1 bottleneck | 40x40x128 | shortcut=False |
| 13 | -1 | nn.Upsample | - | - | - | 128 | 128 | 1 | 80x80x128 | nearest, scale=2 |
| 14 | [-1,4] | Concat | - | - | - | 128+64 | 192 | 1 | 80x80x192 | cat P3 features |
| 15 | -1 | C2f | - | - | - | 192 | 64 | 1 bottleneck | 80x80x64 | shortcut=False, P3 output |
| 16 | -1 | Conv(Conv2d+BN+SiLU) | 3x3 | 2 | 1 | 64 | 64 | 1 | 40x40x64 | downsample |
| 17 | [-1,12] | Concat | - | - | - | 64+128 | 192 | 1 | 40x40x192 | cat head P4 |
| 18 | -1 | C2f | - | - | - | 192 | 128 | 1 bottleneck | 40x40x128 | shortcut=False, P4 output |
| 19 | -1 | Conv(Conv2d+BN+SiLU) | 3x3 | 2 | 1 | 128 | 128 | 1 | 20x20x128 | downsample |
| 20 | [-1,9] | Concat | - | - | - | 128+256 | 384 | 1 | 20x20x384 | cat head P5 |
| 21 | -1 | C2f | - | - | - | 384 | 256 | 1 bottleneck | 20x20x256 | shortcut=False, P5 output |
| 22 | [15,18,21] | Detect | - | - | - | 64,128,256 | 96 each | 1 | multi-scale | anchor-free, decoupled |

### C2f Internal Structure (example: layer 4, 64->64, 2 bottlenecks)

```
Input (64 ch)
  |
  cv1: Conv 1x1 (64 -> 64) + BN + SiLU
  |
  chunk split -> [branch_0: 32ch] [branch_1: 32ch]
                                    |
                              Bottleneck_0:
                                Conv 3x3 (32->32) + BN + SiLU
                                Conv 3x3 (32->32) + BN + SiLU
                                + residual add (shortcut=True)
                                    |
                              Bottleneck_1:
                                Conv 3x3 (32->32) + BN + SiLU
                                Conv 3x3 (32->32) + BN + SiLU
                                + residual add (shortcut=True)
                                    |
  Concat([branch_0, branch_1, bn0_out, bn1_out]) -> (2+2)*32 = 128ch
  |
  cv2: Conv 1x1 (128 -> 64) + BN + SiLU
  |
  Output (64 ch)
```

### SPPF Internal Structure (layer 9, 256->256)

```
Input (256 ch)
  |
  cv1: Conv 1x1 (256 -> 128) + BN + SiLU
  |
  x0 = cv1_out (128 ch)
  x1 = MaxPool2d(k=5, s=1, p=2)(x0)  -- 128 ch, same spatial
  x2 = MaxPool2d(k=5, s=1, p=2)(x1)  -- 128 ch, same spatial
  x3 = MaxPool2d(k=5, s=1, p=2)(x2)  -- 128 ch, same spatial
  |
  Concat([x0, x1, x2, x3]) -> 512 ch
  |
  cv2: Conv 1x1 (512 -> 256) + BN + SiLU
  |
  Output (256 ch)
```

### Detect Head Structure (per scale, e.g. P3: 64 ch input)

```
Input (64 ch)
  |
  +--- cv2 (box regression) ---+--- cv3 (classification) ---+
  |                            |                             |
  Conv 3x3 (64->16)+BN+SiLU   Conv 3x3 (64->80)+BN+SiLU    |
  Conv 3x3 (16->16)+BN+SiLU   Conv 3x3 (80->80)+BN+SiLU    |
  Conv2d 1x1 (16->16, no BN)  Conv2d 1x1 (80->80, no BN)   |
  |                            |                             |
  DFL(reg_max=4) -> 4 values   Sigmoid -> 80 class probs     |
  |                            |                             |
  +--- Concat -> 84 values per anchor ----------------------+

Grid sizes: P3=80x80 (6400), P4=40x40 (1600), P5=20x20 (400)
Total predictions: 8400 x 84 (4 box coords + 80 class scores)
```

### Operation Count Summary

| Operation Type | Count (instances) |
|----------------|-------------------|
| Conv2d 3x3 s2 + BN + SiLU (downsampling) | 7 |
| Conv2d 1x1 s1 + BN + SiLU (C2f cv1/cv2, SPPF cv1/cv2) | 18 |
| Conv2d 3x3 s1 + BN + SiLU (bottleneck convs) | 20 |
| Conv2d 3x3 s1 + BN + SiLU (detect branches) | 12 |
| Conv2d 1x1 (bare, detect final) | 6 |
| MaxPool2d 5x5 s1 p2 | 1 (applied 3x) |
| nn.Upsample nearest 2x | 2 |
| Concat (dim=1) | 4 |
| Sigmoid | 3 (one per detect scale, on class branch) |
| DFL (linear projection) | 3 (one per detect scale, on box branch) |
| **Total Conv2d layers** | **~63** |
| **Total BN layers** | **~57** |

## Unexpected Discoveries

1. **Detect head dominates parameters**: The classification branch (cv3) uses c3=80 hidden channels even when the backbone P3 output is only 64 channels. This means the detect head's classification convolutions are wider than the backbone features they process at the smallest scale.

2. **Asymmetric C2f repeats**: Backbone uses shortcut=True (residual connections) in all C2f blocks, while the neck/head C2f blocks use shortcut=False. This means neck C2f blocks are pure feedforward without skip connections.

3. **reg_max=4 for nano**: The DFL distribution uses only 4 bins per coordinate (vs 16 for larger models), meaning box regression is coarser but much cheaper. Total box output = 4*4 = 16 channels per scale.

4. **SPPF is parameter-light**: The three sequential MaxPool2d operations reuse a single nn.MaxPool2d module (no learnable parameters in pooling), so SPPF's parameters come only from two 1x1 convolutions.

5. **Channel bottleneck at P3 output**: The P3 feature map (80x80x64) is the highest-resolution output but has the fewest channels (64). This creates a natural bottleneck where the detect head must expand from 64 to 80 channels for classification.

## Impact on Downstream Tasks

1. **GPU kernel mapping**: The model has ~63 Conv2d layers but only a few unique kernel configurations (3x3s2, 3x3s1, 1x1s1). A GPU implementation needs at most 3-4 optimized convolution kernels plus MaxPool, Upsample, Concat, SiLU, BN, Sigmoid, and DFL.

2. **Memory planning**: Peak activation memory is ~15-20 MB at FP32 batch=1, dominated by the 80x80 spatial resolution tensors in the neck. Four tensors must be pinned across long lifetimes for skip connections (L4, L6, L9, L12).

3. **Fusion opportunities**:
   - Conv+BN+SiLU can be fused into a single kernel (standard practice)
   - SPPF's 3 sequential MaxPool passes could be fused
   - Detect head cv2 and cv3 branches are independent and can run in parallel
   - DFL is a simple 1D convolution (1x1 over reg_max dimension) followed by softmax-weighted sum

4. **Quantization**: The small channel counts (16, 32, 64) in early layers make INT8 quantization sensitive to precision loss. The detect head with 80-channel classification convolutions is more amenable to quantization.

5. **Batch size scaling**: At batch=1, the model is memory-light (~30 MB total). The bottleneck for GPU utilization will be kernel launch overhead rather than memory, making kernel fusion and operator batching critical for inference performance.
