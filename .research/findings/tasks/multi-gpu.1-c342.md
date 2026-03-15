# multi-gpu.1: Current GpuRuntime device handling and cudarc multi-device support
**Cycle**: 342 | **Theme**: multi-gpu | **Kind**: investigation | **Status**: done

## Summary
GpuRuntime already accepts device ordinal and works for any valid device. cudarc handles primary contexts automatically per-device. Multiple GpuRuntime instances can coexist. The only missing piece for true multi-GPU is per-device hostcall listener threads and buffer indexing.

## Findings

### Q: Does GpuRuntime::new(n) already work for n > 0?
A: **YES.** `GpuRuntime::new(device_ordinal: usize)` passes the ordinal directly to `CudaDevice::new(device_ordinal)`. No hardcoding of device 0 in library code. All examples hardcode `new(0)` but that's client code, not library.
**Confidence**: high

### Q: Can two CudaDevice instances coexist for different ordinals?
A: **YES.** cudarc uses `cuCtxPrimaryRetain` for context management. Multiple `CudaDevice` objects for different ordinals have separate CUDA contexts. Multiple `Arc<CudaDevice>` for the same ordinal share the primary context safely. `CudaDevice` is Send + Sync.
**Confidence**: high

### Q: What are the CUDA context implications of multi-device?
A: cudarc handles primary context per-device automatically via `cuCtxPrimaryRetain(ordinal)`. Context switching happens transparently on kernel launch. No explicit context management needed — cudarc hides the complexity.
**Confidence**: high

### Q: Does cudarc handle primary context vs explicit context?
A: cudarc (0.12.x) uses **primary context only** via `cuCtxPrimaryRetain()`. No support for explicit contexts created with `cuCtxCreate()`. This is sufficient for multi-GPU — each device gets its primary context automatically.
**Confidence**: high

## Unexpected Discoveries
- `AsyncGpuRuntime::new(n)` also works for any ordinal — wraps `GpuRuntime::new(n)` in `Arc`
- The real blocker is hostcall, not runtime: `HostcallSession` has no device index, single listener thread per session. Each session is already independent, but there's no multi-device orchestration layer.

## Open Questions
- Should we create a `MultiGpuRuntime` struct managing N devices, or let users compose multiple `GpuRuntime` instances manually?
- Should examples accept device ordinal as CLI arg?

## Impact on Downstream Tasks
- multi-gpu.2 experiment: GpuRuntime already works for multi-device. Focus should be on (1) device enumeration API, (2) multi-device hostcall validation, (3) example with device selection.
