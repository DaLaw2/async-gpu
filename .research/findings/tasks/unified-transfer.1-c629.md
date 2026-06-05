# unified-transfer.1: Automatic Data Transfer Design

Investigation: ManagedBuffer + affinity tracking for transparent host-device data transfer.

## Status: COMPLETE

## Executive Summary

The recommended design is a **two-tier buffer hierarchy** built on top of the existing
`MappedBuffer<T>` (zero-copy pinned memory) as the default path, with an explicit
`DeviceBuffer<T>` escape hatch for performance-critical GPU-only computation. A new
`GpuVec<T>` type wraps both tiers behind a unified API that eliminates manual `cudaMemcpy`.

The key insight: **MappedBuffer is already the North Star for the read-compute-write
demo**. It is zero-copy, requires no explicit transfers, and for the GTX 1660's
PCIe 3.0 x16 bandwidth (~12 GB/s), the penalty vs device memory is modest for
streaming workloads. Adding location-tracking state machines would be premature
complexity. The simplest approach that eliminates manual cudaMemcpy is to make
MappedBuffer the default and provide an opt-in `to_device()` for hot paths.

## 1. Existing Infrastructure Analysis

### MappedBuffer<T> (gpu-host/src/memory.rs)
- `cuMemHostAlloc` with `DEVICEMAP | PORTABLE` flags
- Dual pointers: `host_ptr` (CPU access) + `dev_ptr` (GPU access via PCIe)
- Zero-copy: GPU reads/writes host RAM directly, no explicit transfer
- Volatile reads/writes for synchronization
- Limitations:
  - GPU accesses go over PCIe (slower than device memory for GPU-intensive loops)
  - No `From<Vec<T>>` or safe slice API without `unsafe`
  - No integration with par_iter (par_iter uses `GpuSlice<T>` from raw pointers)

### gpu::custom() builder API (gpu-host/src/gpu.rs)
- `GpuContext::upload(&[T])` wraps `dev.htod_sync_copy()` -> `CudaSlice<T>`
- `GpuContext::mapped_buffer::<T>(n)` wraps `MappedBuffer::new_zeroed(n)`
- `GpuResult::download(&CudaSlice<T>)` wraps `dev.dtoh_sync_copy()`
- All transfers are explicit and synchronous

### par_iter (gpu-runtime/src/par_iter.rs)
- `GpuSlice<T>` wraps `*const T` + `len` -- raw device pointer
- `GpuSliceMut<T>` wraps `*mut T` + `len` -- raw device pointer
- Created via `GpuSlice::from_raw_parts(ptr, len)` inside kernel code
- The kernel receives raw pointers as kernel arguments
- par_iter is a GPU-side API -- it runs on the device, not the host

### Data flow today (par_iter test as example)
```
Host:  Vec<f32> --htod_sync_copy--> CudaSlice<f32> (device memory)
                                       |
Kernel: GpuSlice::from_raw_parts(ptr, len)  <-- raw pointer from kernel arg
                                       |
                par_iter().map(|x| x * 2.0).collect_into(output)
                                       |
Host:  dtoh_sync_copy(output) --> Vec<f32>
```

## 2. Buffer Type Hierarchy (Proposed)

### Recommendation: Two-tier, NOT three-tier

Rejected: `GpuBuffer<T>` with location enum { Host, Device, Both } and lazy
migration. This adds a state machine, invalidation tracking, and runtime branches
for what should be a compile-time decision. It is the CUDA Unified Memory approach
(cuMemAllocManaged), which has known performance unpredictability.

**Proposed hierarchy:**

```
GpuVec<T>     -- user-facing type, wraps MappedBuffer by default
  |
  +-- MappedBuffer<T>  (pinned zero-copy, default)
  +-- DeviceBuffer<T>  (explicit device memory, opt-in for hot paths)
```

### GpuVec<T> -- the user-facing type

```rust
/// A GPU-accessible vector. Data is automatically available to both
/// CPU and GPU with no explicit memory copies.
///
/// By default, uses pinned zero-copy memory (MappedBuffer).
/// For GPU-intensive computation, call `.to_device()` to get a
/// DeviceBuffer with full GPU memory bandwidth.
pub struct GpuVec<T: Copy + Send + Sync> {
    inner: GpuVecInner<T>,
}

enum GpuVecInner<T: Copy> {
    /// Zero-copy pinned memory: CPU and GPU access same physical pages.
    /// No transfer needed. GPU reads go over PCIe (~12 GB/s on GTX 1660).
    Mapped(MappedBuffer<T>),
    /// Explicit device memory: full GPU bandwidth (~192 GB/s on GTX 1660).
    /// Requires explicit download to read on CPU.
    Device(DeviceBuf<T>),
}

// DeviceBuf wraps cudarc::CudaSlice<T> with a cached host copy
struct DeviceBuf<T> {
    device: CudaSlice<T>,
    host_cache: Option<Vec<T>>,   // populated after download
    dirty_device: bool,           // true after kernel writes
}
```

### Key API

```rust
impl<T: Copy + Send + Sync> GpuVec<T> {
    /// Create from a Vec<T>. Data is copied into pinned memory.
    /// Both CPU and GPU can access immediately -- no transfers.
    pub fn from_vec(data: Vec<T>) -> Result<Self>;

    /// Create zeroed buffer of `n` elements.
    pub fn zeroed(n: usize) -> Result<Self>;

    /// Read as a CPU slice. Safe after GPU synchronization.
    /// For Mapped: returns pinned memory directly (zero-copy).
    /// For Device: downloads from GPU if dirty.
    pub fn as_slice(&self) -> &[T];

    /// Write as a CPU mutable slice. Safe when no kernel is running.
    pub fn as_mut_slice(&mut self) -> &mut [T];

    /// Get the device pointer for kernel arguments.
    /// For Mapped: returns the DEVICEMAP pointer.
    /// For Device: returns the CudaSlice device pointer.
    pub fn dev_ptr(&self) -> u64;

    /// Number of elements.
    pub fn len(&self) -> usize;

    /// Move data to device memory for maximum GPU bandwidth.
    /// This is the opt-in escape hatch for performance-critical paths.
    pub fn to_device(self) -> Result<GpuVec<T>>;

    /// Move data back to pinned memory for zero-copy access.
    pub fn to_mapped(self) -> Result<GpuVec<T>>;
}

// Conversion traits
impl<T: Copy + Send + Sync> From<Vec<T>> for GpuVec<T> { ... }
impl<T: Copy + Send + Sync> Into<Vec<T>> for GpuVec<T> { ... }
```

## 3. Affinity Model: Zero-copy by Default

### Decision: NO automatic location tracking

The affinity model is **static, not dynamic**:

| Variant | Where data lives | GPU access cost | CPU access cost |
|---------|-----------------|-----------------|-----------------|
| `Mapped` (default) | Host RAM, pinned | PCIe read (~12 GB/s) | Direct (~50 GB/s) |
| `Device` (opt-in) | GPU VRAM | Full BW (~192 GB/s) | Requires download |

There is no automatic migration. The user starts with `Mapped` (zero-copy, no
transfers) and can explicitly call `.to_device()` if GPU bandwidth matters.

### Why NOT lazy transfer with dirty tracking:
1. **Unpredictable performance** -- you don't know when a copy will happen
2. **Synchronization complexity** -- when does the GPU "need" the data?
3. **MappedBuffer already works** -- for streaming (read once, compute, write once),
   PCIe bandwidth is sufficient
4. **CUDA Unified Memory exists** -- if we wanted lazy migration, we'd just use
   `cuMemAllocManaged`. But NVIDIA's own guidance says explicit is faster.

### Performance analysis for GTX 1660

- PCIe 3.0 x16 theoretical: ~15.8 GB/s
- Measured htod+dtoh: ~12 GB/s (from iter-demo.2 benchmark data)
- Device memory bandwidth: ~192 GB/s

For the North Star demo (read -> compute -> write):
- **Read**: File -> host memory -> (zero-copy) GPU reads via PCIe
- **Compute**: Each element read once, computed, written once
- **Write**: GPU writes to pinned memory -> host reads -> File::write

With MappedBuffer, the pipeline is **bandwidth-bound by PCIe at ~12 GB/s**.
With DeviceBuffer, you'd pay ~12 GB/s for htod + kernel at ~192 GB/s + ~12 GB/s
for dtoh = dominated by transfer anyway.

**For single-pass streaming workloads, MappedBuffer and DeviceBuffer have
essentially the same end-to-end throughput.** The DeviceBuffer path only wins
when the kernel reads the same data multiple times (e.g., matrix multiply with
tiling, iterative solvers).

## 4. Integration with par_iter

### Current state
par_iter operates on `GpuSlice<T>` / `GpuSliceMut<T>`, which are GPU-side types
wrapping raw device pointers. They're constructed inside kernel code from raw
pointers passed as kernel arguments.

### Proposed integration

On the **host side**, `GpuVec<T>` provides `.dev_ptr()` which returns the device
pointer suitable for kernel arguments. The kernel code constructs `GpuSlice` from
this pointer as it does today.

On the **kernel side**, no changes needed. `GpuSlice::from_raw_parts(ptr, len)`
works with both MappedBuffer dev pointers and CudaSlice device pointers -- they're
both valid device addresses.

### Future: host-side par_iter dispatch

For the full North Star where users never write kernel code, a host-side par_iter
entry point would look like:

```rust
let data = GpuVec::from_vec(vec![1.0f32; 1_000_000]);
let result: GpuVec<f32> = data.par_iter_gpu()
    .map(|x| x * 2.0 + 1.0)
    .collect();
let output: Vec<f32> = result.into();
```

This requires:
1. **GpuVec** to provide a host-side `.par_iter_gpu()` that generates a kernel
   launch (either JIT or from a pre-compiled kernel library)
2. The closure `|x| x * 2.0 + 1.0` to be compiled to GPU code (this is the
   auto-fusion / kernel codegen problem from the gpu-iterator and auto-fusion epics)

This is **out of scope for unified-transfer.2** but is the eventual goal for
unified-demo.1. For now, the transfer layer focuses on eliminating manual
cudaMemcpy from the existing gpu::custom() workflow.

## 5. Integration with BlockScope

BlockScope allocates shared memory within a GPU block. It is orthogonal to
host-device transfer:

- BlockScope's `alloc()` carves out shared memory for intra-block cooperation
- Global data (input/output buffers) comes from kernel arguments, which come from
  the host via `GpuVec::dev_ptr()`
- No integration needed at the BlockScope level

The connection point is the **kernel launch**: host code passes `GpuVec::dev_ptr()`
as a kernel argument, the kernel constructs `GpuSlice` from it, and BlockScope
operates on shared memory copies of that data.

```rust
// Host
let input = GpuVec::from_vec(data);
let output = GpuVec::zeroed(n);
unsafe {
    ctx.launch((input.dev_ptr(), output.dev_ptr(), n as u32))?;
}
let result = output.as_slice();  // zero-copy read

// Kernel (unchanged)
pub unsafe extern "gpu-kernel" fn my_kernel(input: *const f32, output: *mut f32, n: u32) {
    thread::gpu_main(|| {
        init_shared_mem_allocator(512);
        let src = GpuSlice::from_raw_parts(input, n as usize);
        let dst = GpuSliceMut::from_raw_parts(output, n as usize);
        src.par_iter().map(|x| x * 2.0).collect_into(dst);
    });
}
```

## 6. Zero-copy vs Transfer Tradeoffs

### Decision matrix

| Workload pattern | Best approach | Why |
|-----------------|---------------|-----|
| Read-once, compute, write-once (streaming) | MappedBuffer | PCIe bandwidth = transfer bandwidth; no copy saves nothing |
| Read-many (iterative solver, tiling) | DeviceBuffer | GPU reads from VRAM at 192 GB/s vs 12 GB/s over PCIe |
| Small data (< 1MB) | MappedBuffer | Transfer overhead dominates; zero-copy avoids it |
| Large data, single-pass | MappedBuffer | End-to-end identical to htod + kernel + dtoh |
| GPU-to-GPU pipeline (output of kernel A is input to kernel B) | DeviceBuffer | Data never needs to return to host |

### The North Star demo: read -> matmul -> write

For matrix multiply specifically:
- Input matrices are read MANY times (tiling reads each tile multiple times)
- **DeviceBuffer is better for matmul**
- But for the DEMO, the user shouldn't need to know this

**Recommendation**: Default to `Mapped`, but the scheduler (from unified-scheduler
theme) should automatically call `.to_device()` when dispatching to a GPU-compute
kernel with multi-read access patterns. This is a unified-scheduler.2 concern,
not a transfer concern.

For **simpler compute (element-wise map)**, MappedBuffer is optimal:
- Read each element once over PCIe
- Compute in registers
- Write each element once over PCIe
- Same total PCIe traffic as htod + kernel + dtoh

## 7. User-Facing API Proposal (Pragmatic)

### Phase 1: GpuVec with zero-copy default (unified-transfer.2)

```rust
use async_gpu::GpuVec;

fn main() -> async_gpu::Result<()> {
    // Create from Vec -- data is in pinned zero-copy memory
    let input = GpuVec::from_vec(vec![1.0f32; 1_000_000])?;

    // Launch kernel -- input.dev_ptr() is the device-visible address
    let mut output = GpuVec::<f32>::zeroed(1_000_000)?;
    let ctx = gpu::custom("par_iter_map_collect")
        .threads(128)
        .prepare()?;
    let result = unsafe {
        ctx.launch((input.dev_ptr(), output.dev_ptr(), 1_000_000u32, ...))?
    };

    // Read results -- zero-copy, no download needed
    let sum: f32 = output.as_slice().iter().sum();
    println!("sum = {sum}");

    Ok(())
}
```

Compare to today:
```rust
// TODAY: explicit copies everywhere
let input_dev = dev.htod_sync_copy(&input)?;          // explicit htod
let mut output_dev = dev.alloc_zeros::<f32>(n)?;       // explicit device alloc
unsafe { func.launch(cfg, (&input_dev, &mut output_dev, n))?; }
dev.synchronize()?;
let output: Vec<f32> = dev.dtoh_sync_copy(&output_dev)?;  // explicit dtoh
```

### Phase 2: GpuVec integration with gpu::custom() (unified-transfer.3)

```rust
let ctx = gpu::custom("my_kernel")
    .threads(128)
    .input(&input)       // accepts &GpuVec<T>, extracts dev_ptr automatically
    .output::<f32>(n)    // returns GpuVec<f32> after launch
    .prepare()?;
```

### Phase 3: Full pipeline (unified-demo.1)

```rust
// North Star: zero GPU concepts
let data: Vec<f32> = read_file("input.bin")?;
let result = gpu::map(&data, |x| x * 2.0 + 1.0)?;  // returns Vec<f32>
write_file("output.bin", &result)?;
```

## 8. Implementation Plan for unified-transfer.2

### Where to put GpuVec

- **Crate**: `gpu-host` (alongside MappedBuffer in `memory.rs`)
- **Re-export**: via `async-gpu` facade crate (`pub use gpu_host::GpuVec`)

### Concrete steps

1. **Add `GpuVec<T>` type** in `crates/core/gpu-host/src/memory.rs`:
   - `GpuVec::from_vec(Vec<T>) -> Result<Self>` -- alloc MappedBuffer, copy data in
   - `GpuVec::zeroed(n) -> Result<Self>` -- alloc MappedBuffer, zero-init
   - `GpuVec::dev_ptr() -> u64` -- return device-mapped pointer
   - `GpuVec::as_slice() -> &[T]` -- return host pointer as slice (unsafe sync)
   - `GpuVec::as_mut_slice() -> &mut [T]` -- return host pointer as mut slice
   - `GpuVec::len() -> usize`
   - `impl From<Vec<T>> for GpuVec<T>` (calls from_vec, panics on error)
   - `impl Into<Vec<T>> for GpuVec<T>` (copies from pinned to owned Vec)

2. **Add `DeviceBuffer<T>` variant** (optional, can defer to transfer.3):
   - `GpuVec::to_device(self) -> Result<GpuVec<T>>` -- htod copy to CudaSlice
   - `GpuVec::to_mapped(self) -> Result<GpuVec<T>>` -- dtoh copy back

3. **Add `GpuVec` support to `GpuContext`**:
   - `GpuContext::gpu_vec::<T>(n) -> Result<GpuVec<T>>` -- convenience like `mapped_buffer`

4. **Re-export** from `async-gpu/src/lib.rs`

5. **Test**: Modify one par_iter test to use `GpuVec` instead of
   `htod_sync_copy` + `dtoh_sync_copy`

### What NOT to build in transfer.2
- No lazy migration / dirty tracking
- No automatic `.to_device()` based on access patterns
- No host-side par_iter dispatch
- No CUDA Unified Memory (cuMemAllocManaged)

## 9. Risks and Mitigations

### Risk: MappedBuffer pinned memory limit
- CUDA limits pinned memory to a fraction of system RAM
- Mitigation: GpuVec should report allocation failures gracefully
- For very large datasets (> system RAM), a streaming approach with smaller
  buffers would be needed (out of scope)

### Risk: Synchronization safety
- `as_slice()` is only safe after GPU synchronization
- Today: MappedBuffer uses `unsafe fn read()` / `unsafe fn as_slice()`
- Mitigation: GpuVec can track whether a kernel is in-flight and panic on
  unsynchronized access, OR use a `GpuResult` token that proves sync happened.
  Simplest: require explicit `dev.synchronize()` before reads, same as today.

### Risk: Performance expectations
- Users may expect zero-copy to be as fast as device memory
- Mitigation: Document that `.to_device()` exists for GPU-bandwidth-sensitive code
- For the demo, zero-copy streaming is good enough

## 10. Recommendation Summary

| Question | Answer |
|----------|--------|
| Buffer hierarchy? | Two-tier: GpuVec (Mapped default + Device opt-in) |
| Affinity model? | Static, not lazy. Mapped by default, explicit .to_device() |
| Automatic transfers? | None. MappedBuffer IS the transfer -- it's zero-copy |
| par_iter integration? | GpuVec::dev_ptr() -> kernel arg -> GpuSlice::from_raw_parts |
| BlockScope integration? | Orthogonal. Kernel code unchanged. |
| North Star demo fit? | MappedBuffer is optimal for single-pass streaming |
| Next step? | unified-transfer.2: implement GpuVec<T> in gpu-host |
