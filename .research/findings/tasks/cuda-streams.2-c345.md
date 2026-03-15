# cuda-streams.2: Stream-aware launch wrapper + per-stream sync API design
**Cycle**: 345 | **Theme**: cuda-streams | **Kind**: design | **Status**: done

## Summary
Designed a two-tier stream architecture: GpuStream for pure compute overlap, GpuTask stays on default stream for hostcall safety. cudarc 0.12.1 already has `launch_on_stream()` and `fork_default_stream()` — no low-level API workaround needed. ADR-19 written.

## Findings

### Architecture: Two-Tier Stream Model
1. **Tier 1 (Compute Streams)**: `GpuStream` wraps cudarc's `CudaStream`. Enables overlapping compute kernels on independent streams with per-stream sync.
2. **Tier 2 (Hostcall Streams)**: Hostcall kernels stay on default stream. Device-level sync before packet reset. No changes to GpuTask.

### API Design
- `GpuRuntime::create_stream() -> Result<GpuStream>` — wraps `dev.fork_default_stream()`
- `GpuStream::launch(func, config, args)` — wraps `func.launch_on_stream(&stream, config, args)`
- `GpuStream::synchronize()` — per-stream wait
- `AsyncGpuStream` — tokio-compatible wrapper using `spawn_blocking`

### cudarc API (corrected from initial investigation)
- `CudaDevice::fork_default_stream()` — creates new non-blocking stream
- `CudaFunction::launch_on_stream(&CudaStream, config, args)` — launches on specific stream
- `CudaDevice::wait_for(&stream)` — waits for stream on default stream
- Stream Drop automatically synchronizes with default stream

### Hostcall Safety Model
- Hostcall packet reinit requires device-idle (`cuCtxSynchronize`)
- Compute streams can overlap freely (no hostcall buffers)
- Between hostcall launches: `dev.synchronize()` → `reinit_packets()` → launch

### New file: `crates/core/gpu-host/src/streams.rs`

## Open Questions
- Should GpuStream own `Arc<CudaDevice>` or borrow?
- Benchmark: how much overlap benefit for typical workloads?

## Impact on Downstream Tasks
- cuda-streams.3 experiment: implement GpuStream wrapper + basic test
- Forward-compatible with per-stream hostcall buffers (future)
