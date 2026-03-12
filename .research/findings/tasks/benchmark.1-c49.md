# benchmark.1: Benchmark methodology design
**Cycle**: 49 | **Theme**: benchmark | **Kind**: investigation | **Status**: done

## Summary
Designed a comprehensive benchmark methodology for GPU hostcall performance
characterization. Identified available tools (nvidia-smi, globaltimer, PTX
parsing), missing tools (cuobjdump not in PATH, ncu not available), and
proposed benchmark kernel designs with per-thread latency measurement.

## Findings

### Q: What metrics to measure?
A: Four categories:
1. **Hostcall latency** — per-packet round-trip time decomposed into allocation
   (CAS loop), submission (membar + push), and wait (spin for response). Measured
   via gpu_instant_nanos() (%globaltimer) on GPU side.
2. **CAS retry rate** — contention counter in hc_pop_free. Instrumented in kernel.
3. **Register pressure** — hardware register count per kernel via cuobjdump or PTX
   parsing fallback (grep for .reg declarations).
4. **Host listener responsiveness** — response time distribution, duplicate rate,
   CPU utilization.

**Confidence**: high

### Q: How many samples for statistical rigor?
A: 1000+ samples per configuration for p50/p95/p99/p999 percentiles. Test matrix:
thread counts [1, 32, 128, 512], pool sizes [2x, 4x threads], sync vs async modes.
Each run produces per-thread timing data written to mapped result arrays.

**Confidence**: high

### Q: What tools to use?
A: Available: nvidia-smi (driver 581.57, CUDA 13.0), gpu_instant_nanos() via
%globaltimer, PTX text parsing for register counts. NOT available: cuobjdump
(not in PATH, may exist in CUDA toolkit dir), ncu (Nsight Compute, not installed).
Workaround: parse .reg declarations from PTX output, calculate occupancy manually.

**Confidence**: high

### Q: What baselines to compare against?
A: Three comparisons for benchmark.4:
1. Rust hostcall vs equivalent CUDA C++ polling notification
2. Sync hostcall vs async hostcall at same thread count
3. Single-thread vs multi-thread scaling curve

**Confidence**: high

## Benchmark Kernel Design

### Latency microbenchmark kernel
- Each thread runs N iterations of hostcall (NOP service for minimal host processing)
- Per-iteration: record t_alloc, t_submit, t_wait via globaltimer
- Write per-thread timing array to mapped memory
- Host computes percentiles from all samples

### CAS contention kernel
- Instrumented hc_pop_free counts retries
- Per-thread retry count written to result array
- Host computes retry rate = total_retries / successful_pops

### Register profiling
- Parse PTX for `.reg .b32 %r<N>` and `.reg .b64 %rd<N>` declarations
- Map virtual → estimated hardware registers (roughly 1:1 for simple kernels)
- Calculate theoretical occupancy from SM specs

## Impact on Downstream Tasks
- benchmark.2 can implement the latency kernel directly from this methodology
- benchmark.3 needs cuobjdump — may need to ask user to add CUDA bin to PATH
- benchmark.4 needs CUDA C++ baseline — requires nvcc (not in PATH)
