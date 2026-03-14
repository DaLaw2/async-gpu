# iter-workflow.1: Iterative convergence kernel
**Cycle**: 226 | **Theme**: iter-workflow | **Kind**: experiment | **Status**: done

## Summary
Implemented a GPU kernel that autonomously iterates until convergence using Newton's method for integer square root. Iteration count is data-dependent: sqrt(4) takes 2 iterations, sqrt(1000000) takes 13. Also implemented a 2-stage autonomous pipeline combining Pipeline API with convergence computation across different datasets.

## Findings

### Q: Can GPU kernels autonomously loop with data-dependent iteration counts?
A: Yes. The convergence_kernel iterates Newton's method (x = (x + n/x)/2) until x_next >= x. Different inputs produce different iteration counts, confirmed on GPU:
- sqrt(4): 2 iters, sqrt(9): 3, sqrt(16): 4, sqrt(100): 5, sqrt(10000): 10, sqrt(1000000): 13
**Confidence**: high

### Q: Does the Pipeline API work for multi-stage convergence workflows?
A: Yes. run_autonomous_pipeline_test runs two stages with different datasets (small: [4,25,144] → 12 total iters; large: [10000,1000000,99999999] → 41 total iters). The iteration counts differ between stages, confirming data-dependent behavior across a Pipeline.
**Confidence**: high
