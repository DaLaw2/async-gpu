# showcase-readme.1 — Feature Matrix

## Summary

Added a comprehensive Feature Matrix section to README.md covering all 26 identified GPU capabilities organized into 4 groups (Language & Runtime, I/O & Networking, Compute Patterns, ML/AI) plus a kernel summary line. Each row provides the feature name, a one-line description, and a link to the relevant example.

## Findings

### Design Decisions

1. **Four-group taxonomy**: Language & Runtime (11 features), I/O & Networking (3), Compute Patterns (8), ML/AI (8). This matches how developers think: "what can I write?" vs "what can it compute?" vs "what models can I run?"

2. **Three-column table**: Feature | Description | Example. Dropped a "Status" column — all features are shipped and working, so a checkmark column would be pure noise. The presence of an example link already signals maturity.

3. **Parallel iterator listed without example link**: `GpuParallelIterator` has a full 1230-line implementation but zero examples. Listed it with `—` to be honest about coverage rather than omitting a shipped feature.

4. **Sub-mode examples noted inline**: Kernel fusion (`--bench-fused`), INT8 (`--bench-int8`), and ONNX (`--onnx`) are CLI flags on other examples. Noted the flag directly in the Example column rather than pretending they're standalone.

5. **Kernel summary as prose, not table**: 130+ kernels don't fit in a table. A one-line summary below the ML table lists the kernel families (GEMM, FlashAttention, Conv2D, etc.) without bloating the page.

6. **Placement**: Inserted between Quick Start and "Real Rust std on GPU". This gives new readers: intro -> how to run it -> what it can do -> deeper feature explanations.

## Unexpected Discoveries

- The existing "All examples" `<details>` block (22 rows) focuses on example names and descriptions. The feature matrix is intentionally higher-level: it maps *capabilities* not examples. Some capabilities span multiple examples (autograd has 4), some examples demonstrate multiple capabilities.

## Open Questions

- Should the parallel iterator row stay in the matrix without an example, or be removed until an example exists? Kept it for completeness since the implementation is real.

## Impact on Downstream Tasks

- **showcase-readme.2 (performance table)**: The Performance section already exists lower in the README. The perf table task should format it to match the feature matrix style.
- **showcase-readme.3+ (hero examples)**: The feature matrix provides the inventory of what to showcase — hero snippets should cover at least one feature from each group.
