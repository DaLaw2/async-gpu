# gl-schema.1: Weight Loading Patterns Analysis

## 1. GPT-2 Weight Loading

### SafeTensors Key Naming Convention
Global tensors:
- `wte.weight` — token embeddings [50257, 768]
- `wpe.weight` — positional embeddings [1024, 768]
- `ln_f.weight`, `ln_f.bias` — final LayerNorm [768]

Per-layer tensors (i = 0..11):
- `h.{i}.ln_1.weight`, `h.{i}.ln_1.bias` — pre-attention LayerNorm [768]
- `h.{i}.attn.c_attn.weight` — QKV projection [768, 2304]
- `h.{i}.attn.c_attn.bias` — QKV bias [2304]
- `h.{i}.attn.c_proj.weight` — attention output [768, 768]
- `h.{i}.attn.c_proj.bias` — [768]
- `h.{i}.ln_2.weight`, `h.{i}.ln_2.bias` — pre-FFN LayerNorm [768]
- `h.{i}.mlp.c_fc.weight` — FFN up [768, 3072]
- `h.{i}.mlp.c_fc.bias` — [3072]
- `h.{i}.mlp.c_proj.weight` — FFN down [3072, 768]
- `h.{i}.mlp.c_proj.bias` — [768]

### Weight Shape Transformations
**Conv1D-to-Linear transpose**: GPT-2's safetensors stores weights in Conv1D layout `[in_features, out_features]`, but the nn Linear layer expects `[out_features, in_features]`. All 4 weight matrices per layer are transposed:
- `c_attn.weight` [768, 2304] -> transposed to [2304, 768]
- `c_proj.weight` [768, 768] -> transposed to [768, 768]
- `mlp.c_fc.weight` [768, 3072] -> transposed to [3072, 768]
- `mlp.c_proj.weight` [3072, 768] -> transposed to [768, 3072]

**Weight tying**: LM head reuses `wte.weight` directly (already [vocab, n_embd] = [out, in], no transpose needed).

### Weight-to-Module Mapping
| SafeTensors key pattern | Module type | Notes |
|---|---|---|
| `wte.weight`, `wpe.weight` | `Embedding` | Combined token + position embeddings |
| `h.{i}.ln_*.weight/bias` | `LayerNorm` | eps=1e-5 |
| `h.{i}.attn.c_attn.*` | `MultiHeadAttention` (QKV) | Transpose required |
| `h.{i}.attn.c_proj.*` | `MultiHeadAttention` (out) | Transpose required |
| `h.{i}.mlp.c_fc.*` | `Linear` (ffn_up) | Transpose required |
| `h.{i}.mlp.c_proj.*` | `Linear` (ffn_down) | Transpose required |
| `ln_f.*` | `LayerNorm` | Final norm |
| `wte.weight` (tied) | `Linear` (lm_head) | No bias, no transpose |

---

## 2. YOLOv8-Nano Weight Loading

### SafeTensors Key Naming Convention
Ultralytics-exported names with numeric layer indices:

**Simple Conv+BN layers** (backbone conv blocks):
- `model.{idx}.conv.weight` — Conv2d [C_out, C_in, kH, kW]
- `model.{idx}.bn.weight` — BN gamma [C]
- `model.{idx}.bn.bias` — BN beta [C]
- `model.{idx}.bn.running_mean` — [C]
- `model.{idx}.bn.running_var` — [C]

**C2f sub-convolutions** (cv1, cv2):
- `model.{idx}.cv1.conv.weight`, `model.{idx}.cv1.bn.*`
- `model.{idx}.cv2.conv.weight`, `model.{idx}.cv2.bn.*`

**C2f bottleneck convolutions**:
- `model.{idx}.m.{j}.cv1.conv.weight`, `model.{idx}.m.{j}.cv1.bn.*`
- `model.{idx}.m.{j}.cv2.conv.weight`, `model.{idx}.m.{j}.cv2.bn.*`

**Detect head** (layer 22):
- Conv+BN layers: `model.22.{branch}.{scale}.{sub}.conv.weight`, `model.22.{branch}.{scale}.{sub}.bn.*`
  - branch = "cv2" (box) or "cv3" (class), scale = 0..2, sub = 0..1
- Final bare Conv2d: `model.22.{branch}.{scale}.2.weight`, `model.22.{branch}.{scale}.2.bias`
  - Note: sub=2 does NOT have `.conv.` prefix — naming inconsistency

### Weight Shape Transformations
**None** — Conv2d weights are stored as [C_out, C_in, kH, kW], which matches the Conv2d layer's expected layout directly. No transpose or reshape needed.

### Weight-to-Module Mapping
| SafeTensors key pattern | Module type | Notes |
|---|---|---|
| `model.{idx}.conv.weight` + `model.{idx}.bn.*` | `ConvBnSilu` (Conv2d + BatchNorm2d + SiLU) | Stride/padding from architecture |
| `model.{idx}.cv{1,2}.conv.weight` + `bn.*` | `ConvBnSilu` (inside C2f/SPPF) | |
| `model.{idx}.m.{j}.cv{1,2}.*` | `ConvBnSilu` (bottleneck) | Shortcut add is architecture-level |
| `model.22.{branch}.{scale}.{0,1}.*` | `ConvBnSilu` (detect intermediate) | |
| `model.22.{branch}.{scale}.2.*` | `ConvBias` (Conv2d with bias, no BN) | Final detect projection |

### Architectural Quirks
- **Generic loader**: Uses `load_all_tensors()` to load everything into a HashMap, then extracts by name. No shape validation at load time.
- **Shape carried alongside data**: `ConvBnSiluWeights.conv_shape` stores [C_out, C_in, kH, kW] so the architecture code can extract dims.
- **Stride/padding not in weights**: Determined by architecture position, not stored in safetensors.

---

## 3. ResNet-18 Weight Loading

### SafeTensors Key Naming Convention
ResNet-18 currently has **no safetensors loader** — it only uses `ResNet18Weights` structs populated from random data or (future) manual construction. However, the weight structure implies a PyTorch-style naming that would be:

Expected PyTorch convention (if loaded from safetensors):
- `conv1.weight` — [64, 3, 3, 3]
- `bn1.weight`, `bn1.bias`, `bn1.running_mean`, `bn1.running_var` — [64]
- `layer{s}.{b}.conv1.weight` — [out, in, 3, 3] (s=1..4, b=0..1)
- `layer{s}.{b}.bn1.weight/bias/running_mean/running_var` — [out]
- `layer{s}.{b}.conv2.weight` — [out, out, 3, 3]
- `layer{s}.{b}.bn2.*` — [out]
- `layer{s}.{b}.downsample.0.weight` — [out, in, 1, 1] (shortcut conv, only when stride>1 or channels change)
- `layer{s}.{b}.downsample.1.*` — shortcut BN
- `fc.weight` — [num_classes, 512]
- `fc.bias` — [num_classes]

### Weight Shape Transformations
**None** — Conv2d weights [C_out, C_in, kH, kW] and Linear weights [out, in] are already in the expected layout.

### Weight-to-Module Mapping
| Weight field | Module type | Notes |
|---|---|---|
| `conv1_w` | `Conv2d` | 3x3, stride=1 (CIFAR variant) |
| `bn1_*` | `BatchNorm2d` | eps=1e-5, silu=false |
| `layer{s} -> conv{1,2}_w` | `Conv2d` (inside BasicBlock) | 3x3 |
| `layer{s} -> bn{1,2}_*` | `BatchNorm2d` | |
| `layer{s} -> shortcut_conv_w` | `Conv2d` (1x1 downsample) | Optional |
| `layer{s} -> shortcut_bn_*` | `BatchNorm2d` | Optional |
| `fc_w`, `fc_b` | `Linear` | Final classifier |

---

## 4. Cross-Model Comparison

### Layer Types Used

| Layer | GPT-2 | YOLOv8 | ResNet-18 |
|---|---|---|---|
| Linear | Yes (4 per block + lm_head) | No | Yes (fc) |
| Conv2d | No | Yes (everywhere) | Yes (everywhere) |
| LayerNorm | Yes (2 per block + final) | No | No |
| BatchNorm2d | No | Yes (with SiLU) | Yes (with ReLU) |
| Embedding | Yes (wte + wpe) | No | No |
| MaxPool2d | No | Yes (SPPF) | No |

### Transform Requirements

| Transform | Used by | Description |
|---|---|---|
| Transpose 2D | GPT-2 only | Conv1D [in, out] -> Linear [out, in] |
| Weight tying | GPT-2 only | wte reused as lm_head |
| Shape extraction | YOLO only | Conv shape [C_out, C_in, kH, kW] carried with data |
| Optional sub-modules | ResNet only | Shortcut Conv+BN present only when stride/channels change |

### Key Naming Patterns

| Model | Pattern | Repeat structure |
|---|---|---|
| GPT-2 | `h.{layer}.{submodule}.{param}` | Fixed 12 layers, flat |
| YOLO | `model.{layer}.{sub}.{subsub}.{param}` | Numeric indices, deep nesting |
| ResNet | `layer{stage}.{block}.{submodule}.{param}` | 4 stages x 2 blocks |

---

## 5. Proposed `ModelDef` Schema

### Core Insight
The weight loading process for all 3 models follows the same abstract pattern:
1. **Enumerate** safetensors keys matching a pattern
2. **Extract** f32 data + shape for each key
3. **Transform** some weights (transpose, reshape)
4. **Map** groups of weights to nn module constructors (Linear, Conv2d, LayerNorm, etc.)
5. **Wire** modules into an architecture (sequential, residual, multi-scale)

### Proposed Fields

```rust
/// A declarative model definition that drives weight loading + module construction.
struct ModelDef {
    /// Model name for error messages.
    name: String,

    /// Ordered list of module definitions.
    /// Execution order is defined separately (in architecture).
    modules: Vec<ModuleDef>,

    /// Architecture: how modules connect (sequential, residual, branch+concat).
    /// This is the "forward graph" — optional for weight loading, needed for inference.
    architecture: Option<Architecture>,
}

/// A single module with its weight mapping.
struct ModuleDef {
    /// Unique name for this module instance (e.g., "block.0.attn.qkv").
    name: String,

    /// What kind of nn layer this is.
    kind: ModuleKind,

    /// How to find weights in the safetensors file.
    weight_map: Vec<WeightMapping>,
}

/// Supported module types with their constructor parameters.
enum ModuleKind {
    Linear { in_features: usize, out_features: usize },
    Conv2d { c_out: usize, c_in: usize, kh: usize, kw: usize, stride: usize, padding: usize },
    LayerNorm { dim: usize, eps: f32 },
    BatchNorm2d { channels: usize, eps: f32, silu: bool },
    Embedding { vocab_size: usize, max_seq: usize, dim: usize },
}

/// Maps a safetensors tensor to a module constructor parameter.
struct WeightMapping {
    /// SafeTensors key (supports `{i}` for repeated layers).
    tensor_key: String,

    /// Which constructor parameter this feeds into (e.g., "weight", "bias", "bn_mean").
    param_name: String,

    /// Transform to apply before passing to constructor.
    transform: Transform,

    /// Whether this weight is optional (e.g., bias, shortcut).
    optional: bool,
}

/// Weight transformations.
enum Transform {
    /// No transformation — use as-is.
    None,

    /// Transpose 2D: [rows, cols] -> [cols, rows].
    /// Used by GPT-2 Conv1D->Linear conversion.
    Transpose { rows: usize, cols: usize },

    /// Reshape without data movement (just reinterpret shape).
    Reshape { target_shape: Vec<usize> },

    /// Reuse another module's weight (weight tying).
    /// E.g., GPT-2 lm_head ties to wte.
    TieFrom { source_module: String, source_param: String },
}
```

### Handling Architecture-Specific Quirks

**1. GPT-2 Conv1D Transpose**
```
WeightMapping {
    tensor_key: "h.{i}.attn.c_attn.weight",
    param_name: "weight",
    transform: Transform::Transpose { rows: 768, cols: 2304 },
    optional: false,
}
```

**2. GPT-2 Weight Tying (lm_head = wte)**
```
WeightMapping {
    tensor_key: "",  // no tensor to load
    param_name: "weight",
    transform: Transform::TieFrom {
        source_module: "embedding".into(),
        source_param: "wte".into(),
    },
    optional: false,
}
```

**3. YOLO Detect Head Naming Inconsistency**
The final detect convs (sub=2) omit `.conv.` in their key names. This can be handled by explicit key specification per module — no special transform needed:
```
// sub=0,1: "model.22.cv2.0.0.conv.weight"
// sub=2:   "model.22.cv2.0.2.weight"
```

**4. ResNet Optional Shortcut**
```
WeightMapping {
    tensor_key: "layer{s}.{b}.downsample.0.weight",
    param_name: "shortcut_weight",
    transform: Transform::None,
    optional: true,  // only present when stride>1 or channels change
}
```

**5. Stride/Padding (Not in Weights)**
Conv2d stride and padding are part of `ModuleKind::Conv2d`, not `WeightMapping`. They come from the architecture definition, not the safetensors file. This is correct — architecture hyperparameters belong in `ModuleDef`, not in the weight file.

### Repeated Layers
All 3 models have repeated structures. The schema should support a `Repeat` wrapper:
```rust
struct RepeatDef {
    /// Template module definitions (use `{i}` in tensor keys).
    template: Vec<ModuleDef>,
    /// Number of repetitions.
    count: usize,
    /// Variable name for substitution (e.g., "i" for `{i}`).
    var: String,
}
```
- GPT-2: 12 transformer blocks, `h.{i}.*`
- YOLO: numeric layer indices, `model.{idx}.*` — but layers aren't uniform (Conv, C2f, SPPF mix), so repeat is per-type
- ResNet: 4 stages x 2 blocks, `layer{s}.{b}.*`

### What ModelDef Does NOT Need
- **Forward pass logic**: The execution graph (residual connections, concat, upsample) is architecture code, not weight loading. ModelDef handles weight loading and module construction only.
- **Activation functions**: GELU, SiLU, ReLU, Sigmoid have no weights. They're part of the architecture graph.
- **Post-processing**: NMS, DFL decode, argmax — these are inference-time operations.

### Summary

A `ModelDef` with ~4 types (`ModuleDef`, `ModuleKind`, `WeightMapping`, `Transform`) can handle all 3 architectures' weight loading generically. The key insight is separating:
1. **What to load** (tensor keys) — `WeightMapping.tensor_key`
2. **How to transform** (transpose, tie) — `Transform`
3. **Where to put it** (module constructor params) — `ModuleKind` + `param_name`
4. **How to connect** (architecture) — separate concern, not in ModelDef
