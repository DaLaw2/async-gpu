# Auto-Fusion

Tape-level kernel fusion detection and fused kernel execution on GPU.

## What It Demonstrates

- Fusion pattern detection: Matmul+BiasAdd+Gelu, ElemAdd+LayerNorm, elementwise chains
- FusionOptimizer analyzing an autograd tape and producing a FusionPlan
- FusionCodegen JIT-compiling fused CUDA kernels via NVRTC
- Fused vs unfused performance comparison with correctness verification
- Fan-out detection (intermediate tensor consumed by multiple ops blocks fusion)

## Running

```bash
cargo run -p auto-fusion --release
```

## Key APIs

- `FusionOptimizer::analyze(&tape)` — detect fusable op sequences
- `FusionPlan` — describes fusion groups and launches saved
- `FusionCodegen::get_or_compile(ops, n_cols, dev)` — JIT compile fused kernel
- `FusedOpKind` — MatmulBiasGelu, ElemAddLayerNorm, MatmulBias, ElementwiseChain

## Expected Output

```
=== Auto-Fusion Example ===

--- Demo 1: Fusion Detection (GPT-2 Block) ---
  Fusion groups detected: 5
  Kernel launches saved:  6
  14 ops -> 8 fused ops (saved 6 kernel launches)

--- Demo 2: Elementwise Chain Detection ---
  3 ops fused into 1 kernel launch

--- Demo 3: Fused Kernel Execution (GPU) ---
  PASSED: Fused ElemAdd+Gelu matches CPU reference

--- Demo 4: Fused vs Unfused Performance ---
  Speedup: >1.0x
  PASSED: Fused kernel is faster than unfused

--- Demo 5: Fan-Out Blocks Fusion ---
  PASSED: Fan-out correctly detected, no fusion applied

=== All demos complete! ===
```
