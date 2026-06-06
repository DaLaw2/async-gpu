# td-design: Transparent Vec<T> Design Synthesis

## Current State
GpuVec<T> wraps pinned-mapped memory (MappedBuffer). Zero-copy after
creation, but user must: (1) explicitly create GpuVec, (2) specify PTX
and kernel name for map_gpu, (3) know GPU exists. Three separate memory
models coexist: CudaSlice (explicit copy), GpuVec (pinned-mapped),
MappedBuffer (raw).

## Recommended Approach: Lazy Residency Wrapper (Option A)
A newtype `GpuArray<T>` that derefs to `[T]`, tracks residency
(Host|Device|Synced), and lazily migrates data at kernel boundaries.
The runtime accepts `&GpuArray<T>` and auto-transfers. Users write
`GpuArray::from(vec)` once; after that it looks like `&[T]`.

## Key Design Constraints
- Must coexist with existing GpuVec/CudaSlice (incremental adoption)
- Kernel selection still requires user intent (which kernel to run)
- Auto-sync before host reads (no manual synchronize)
- Size-based backend selection: pinned-mapped for small, device-copy for large
- T: Copy + Send + 'static (same as current GpuVec constraint)

## Blocked By
- Scheduler must evolve to accept GpuArray as input (not just &[f32])
- Kernel dispatch needs "operation descriptor" instead of raw PTX+name
