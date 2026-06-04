# fusion-analysis.3 — Experiment: Fusion Candidate Detection in Tape Optimizer

## Status: done

## Summary

Implemented `FusionOptimizer` in `crates/core/gpu-host/src/nn/fusion.rs` — a standalone analysis module that scans the autograd tape and identifies kernel fusion candidates using a greedy longest-match forward scan algorithm. This is detection only; no kernel compilation or execution.

## What was implemented

### Module: `crates/core/gpu-host/src/nn/fusion.rs`

**Core types:**

- `OpClass` — classifies every `OpKind` into one of four fusion categories: `Elementwise`, `ComputeBound`, `Reduction`, `Structural`.
- `FusedOpKind` — enum of fused operation kinds the optimizer can detect: `MatmulBiasGelu`, `ElemAddLayerNorm`, `MatmulBias`.
- `FusionPattern` — a concrete pattern definition: ordered `OpKind` sequence + replacement + priority.
- `FusionGroup` — a detected fusion opportunity: tape range `[start..end)`, fused op, input/output tensor IDs.
- `FusionPlan` — the result of analysis: list of `FusionGroup`s with a `launches_saved()` metric.
- `FusionOptimizer` — the analyzer, holds the pattern catalog and exposes `analyze(&[TapeEntry]) -> FusionPlan`.

**Patterns implemented (top 3 from design):**

| ID | Pattern | Replacement | Priority |
|----|---------|-------------|----------|
| P1 | Matmul → BiasAdd → Gelu | `MatmulBiasGelu` | 100 |
| P3 | ElemAdd → LayerNorm | `ElemAddLayerNorm` | 90 |
| P4 | Matmul → BiasAdd | `MatmulBias` | 80 |

**Algorithm:** Greedy forward scan with patterns sorted by priority desc, then length desc. At each tape position, the first matching pattern wins. Fusability predicates checked per consecutive pair:
1. **Data flow** — producer's output is in consumer's inputs
2. **Single consumer** — producer's output has ref_count == 1 (no fan-out)

**Prerequisite change:** Added `PartialEq, Eq, Hash` derives to `OpKind` enum in `tape.rs` — required for pattern matching comparisons.

### Tests (13 tests, all passing)

| Test | What it verifies |
|------|-----------------|
| `test_classify_ops` | OpClass assignment for all op categories |
| `test_empty_tape` | Empty tape produces empty plan |
| `test_single_op_no_fusion` | Single op cannot form a fusion group |
| `test_p1_matmul_bias_gelu` | P1 pattern detected with correct inputs/output |
| `test_p3_elemadd_layernorm` | P3 pattern detected |
| `test_p4_matmul_bias` | P4 pattern detected |
| `test_p1_preferred_over_p4` | Longer P1 wins over shorter P4 at same position |
| `test_fan_out_blocks_fusion` | ref_count > 1 prevents fusion |
| `test_broken_data_flow_blocks_fusion` | Non-connected ops don't fuse |
| `test_multiple_fusion_groups` | Three patterns in one tape (P1 + P4 + P3) |
| `test_unfusable_ops_pass_through` | Barriers (Embedding, Attention) correctly break chains |
| `test_ref_counts` | Reference count computation correctness |
| `test_gpt2_full_block_fusion` | Full 14-op GPT-2 block → 5 fusion groups, 6 launches saved |

## GPT-2 full-block validation

The `test_gpt2_full_block_fusion` test simulates a complete transformer block:

```
LayerNorm → Matmul→BiasAdd → Attention → Matmul→BiasAdd → ElemAdd→LayerNorm → Matmul→BiasAdd→Gelu → Matmul→BiasAdd → ElemAdd
```

Detected 5 fusion groups:
1. QKV projection: Matmul→BiasAdd (P4)
2. Output projection: Matmul→BiasAdd (P4)
3. Residual + LN2: ElemAdd→LayerNorm (P3)
4. FFN up: Matmul→BiasAdd→Gelu (P1)
5. FFN down: Matmul→BiasAdd (P4)

Result: 14 ops → 8 effective kernels (6 launches saved per block).

## Files changed

- `crates/core/gpu-host/src/nn/fusion.rs` — **new** — fusion optimizer module
- `crates/core/gpu-host/src/nn/mod.rs` — added `pub mod fusion;`
- `crates/core/gpu-host/src/nn/autograd/tape.rs` — added `PartialEq, Eq, Hash` to `OpKind` derive

## CI

`scripts/ci-lint.sh` passes (fmt + clippy + check for all crates/examples).
