# dyn-control.3: Why these patterns are impossible with CUDA graphs
**Cycle**: 563 | **Theme**: dyn-control | **Kind**: design | **Status**: done

## Summary
Documents the fundamental architectural difference between CUDA graphs (static)
and async-gpu (dynamic), with concrete examples from dyn-control.1 and .2.

## CUDA Graphs: What They Are

CUDA graphs capture a fixed DAG of kernel launches during a "capture" phase, then
replay the exact same sequence in an "instantiate + launch" phase. This provides:
- Near-zero CPU launch overhead (no per-kernel API calls)
- Deterministic execution order
- Optimized scheduling by the driver

## What CUDA Graphs Cannot Do

### 1. Data-Dependent Loop Count
```
// This is IMPOSSIBLE with CUDA graphs:
while model.generate_next_token() != EOS {
    // The iteration count depends on model output
}
```
CUDA graphs require a fixed number of kernel launches at capture time. Variable-length
generation (dyn-control.1) where each prompt stops at a different token count cannot
be captured as a static graph.

### 2. Data-Dependent Branching
```
// This is IMPOSSIBLE with CUDA graphs:
if confidence(logits) > threshold {
    skip_remaining_layers();
}
```
Early-exit inference (dyn-control.2) requires runtime decisions about which layers
to execute. The layer count per token ranges from 1 to 12 depending on the input.

### 3. Stochastic Control Flow
Top-k sampling with different random seeds produces different execution paths.
Each seed leads to different token choices, which cascade into different loop counts
and different model states. The execution graph is non-deterministic.

### 4. Per-Sample Heterogeneous Compute
In batch inference, different samples may need different amounts of compute.
One sample might hit EOS after 10 tokens while another needs 100. CUDA graphs
execute the same graph for all samples.

## What async-gpu Provides Instead

async-gpu compiles real Rust to GPU code via the PTX backend. Control flow
(loops, branches, early exits) executes as native GPU instructions, not replayed
from a captured trace. This means:

| Pattern | CUDA Graphs | async-gpu |
|---------|-------------|-----------|
| Fixed kernel sequence | Yes (optimal) | Yes |
| Variable-length loops | No | Yes (natural Rust loop) |
| Data-dependent branching | No | Yes (native if/else) |
| Early exit | No | Yes (break/return) |
| Stochastic paths | No | Yes (runtime sampling) |

## Demonstrated Results

| Demo | What It Shows | Impossible with CUDA Graphs? |
|------|--------------|------------------------------|
| Variable-length gen | 63-100 tokens per prompt | Yes — loop count varies |
| Top-k sampling | Same prompt, different outputs per seed | Yes — stochastic branching |
| Temperature sweep | Different text at different temperatures | Yes — data-dependent paths |
| Early-exit inference | 1-12 layers per token | Yes — layer count is data-dependent |
| Confidence probing | "easy" tokens exit at layer 1, "hard" at layer 12 | Yes — per-token adaptive compute |

## Trade-offs

CUDA graphs win on:
- Launch overhead for fixed workloads
- Driver-level scheduling optimization
- Maturity and tooling

async-gpu wins on:
- Any workload with data-dependent control flow
- Dynamic shapes and variable-length sequences
- Adaptive compute (early exit, convergence checking)
- GPU-autonomous operation (no host round-trip for decisions)

## Impact
This design doc supports the gpu-dynamic epic's criterion: "At least one demo
that is impossible with CUDA graphs / TensorRT static compilation." We have
demonstrated FOUR such demos.
