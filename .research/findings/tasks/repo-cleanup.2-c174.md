# repo-cleanup.2: Extract async-io example from pipeline tests
**Cycle**: 174 | **Theme**: repo-cleanup | **Kind**: experiment | **Status**: done

## Summary
Created `examples/async-io/` — a standalone example demonstrating multi-step
file I/O from GPU kernels. Two kernels: write_pipeline (writes 3 files) and
transform_pipeline (reads file → GPU uppercase → writes result). Both compile
and pass validation.

## Findings

### Q: Can the file transform pipeline become a standalone example using gpu-host public API?
A: Yes. The example uses `HostcallBuffer` from gpu-host, `cudarc` for CUDA device
management, and a self-compiled kernel via `build.rs`. No test infrastructure needed.
**Confidence**: high

### Q: What minimal kernel subset is needed for the example?
A: Only `gpu_runtime::prelude::*` — provides PRINT, OPEN, WRITE, CLOSE, BULK_READ
hostcall services. No compute kernels needed.
**Confidence**: high
