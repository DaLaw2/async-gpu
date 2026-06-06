# td-design.1: Audit GpuVec API and Map Transfer Points

Status: COMPLETE
Task kind: investigation

## Summary

Audited the complete GpuVec API surface, mapped all host-device transfer points
across the codebase, and analyzed example usage patterns to inform the transparent
Vec<T> replacement design.

## Findings

### 1. GpuVec<T> Full API

**Struct**: `GpuVec<T>` wraps `MappedBuffer<T>` (pinned device-mapped memory).

**Constructors**:
- `GpuVec::from_vec(Vec<T>) -> Result<Self>` — copies data into pinned memory
- `GpuVec::zeroed(len) -> Result<Self>` — allocates zeroed pinned memory
- `TryFrom<Vec<T>>` and `TryFrom<&[T]>` trait impls

**Accessors**:
- `dev_ptr() -> u64` — device-visible address for kernel args
- `as_slice() -> &[T]` — zero-copy host read (caller must sync)
- `as_mut_slice() -> &mut [T]` — zero-copy host write (no concurrent GPU access)
- `len() -> usize`, `is_empty() -> bool`

**Compute**:
- `map_gpu(ptx, kernel_name, threads) -> Result<GpuVec<T>>` — one-liner transform
- `map_gpu_cubin(ptx, cubin, kernel_name, threads) -> Result<GpuVec<T>>`

**Conversion**:
- `into_vec() -> Vec<T>` — copies pinned memory back to owned Vec

**Key trait**: Requires `T: Copy`.

### 2. Transfer Points Map

There are **three distinct memory models** in the codebase:

**Model A: CudaSlice (explicit copy, separate address spaces)**
Used by: `gpu::run_with_output()`, `gpu::launch()`, `gpu::custom()`, `GpuRuntime`,
`AutoScheduler::gpu_par_map()`, `GpuTensor`, all nn/ops, all examples except gpuvec.

- Host→Device: `dev.htod_sync_copy(data)` / `ctx.upload(data)`
- Device→Host: `dev.dtoh_sync_copy(&buf)` / `ctx.download(&buf)` / `result.download(&buf)`
- Allocation: `dev.alloc_zeros::<T>(n)` / `ctx.alloc_zeros::<T>(n)`

Trigger: explicit API call. User must call upload before launch, download after sync.

**Model B: GpuVec (pinned mapped, single address space)**
Used by: `GpuVec::from_vec()`, `launch_with_gpuvec()`, `gpuvec_pipeline` example.

- Host→Device: `GpuVec::from_vec(data)` (copies into pinned mem; GPU reads over PCIe)
- Device→Host: `as_slice()` (zero-copy volatile read after sync)
- No explicit cudaMemcpy at any point.

Trigger: construction (from_vec) and kernel launch (dev_ptr passed as arg).

**Model C: MappedBuffer (raw pinned mapped)**
Used by: `GpuContext::mapped_buffer()`, `HostcallBuffer`, test harness.

- Direct raw pointer access: `host_ptr()`, `dev_ptr()`
- Unsafe read/write with volatile semantics
- Lowest level, used for hostcall protocol buffers

### 3. Example Usage Patterns

**Pattern 1: gpu::custom() builder (most examples)**
```
Vec<f32> → ctx.upload() → CudaSlice → launch → sync → result.download() → Vec<f32>
```
Used by: vector-math, monte-carlo, benchmark, all nn examples.
5 explicit steps, user manages upload/download/sync.

**Pattern 2: GpuVec zero-copy (gpuvec_pipeline)**
```
Vec<f32> → GpuVec::from_vec() → map_gpu() → as_slice() → done
```
Used by: gpuvec_pipeline example, integration tests.
3 steps, no explicit transfer calls, but user still writes "GpuVec".

**Pattern 3: gpu::run()/gpu::launch() (simple kernels)**
```
gpu::launch("kernel", n, threads) → Vec<T>
```
Output-only, no input data. Already transparent but limited.

**Pattern 4: AutoScheduler::par_map() (transparent routing)**
```
sched.par_map(&data, |x| x * 2.0 + 1.0) → Vec<f32>
```
Already transparent — but limited to a fixed pre-compiled kernel.

### 4. Transparent Vec<T> Design Options

**Option A: Newtype wrapper with Deref**
```rust
pub struct GpuArray<T>(Vec<T>, Option<DeviceState>);
impl<T> Deref for GpuArray<T> { type Target = [T]; ... }
```
- Looks like Vec<T> to the user via Deref
- Internal residency tracking (Host, Device, Both)
- Lazy transfer: data stays on host until kernel launch needs dev_ptr
- Pro: safe, explicit creation point
- Con: not literally `Vec<T>`

**Option B: Extension trait on Vec<T>**
```rust
trait GpuExt<T> { fn dev_ptr(&self) -> DevicePtr; }
impl<T: Copy> GpuExt<T> for Vec<T> { ... }
```
- User writes `Vec<f32>` everywhere
- Global registry maps Vec allocations to pinned memory
- Pro: literally Vec<T>
- Con: requires global state, thread safety issues, can't intercept drop

**Option C: Modified runtime that accepts &[T]**
```rust
gpu::custom("kernel").input(&my_vec).launch()
```
- Runtime handles upload/download transparently
- Vec<T> is never modified, runtime manages CudaSlice internally
- Pro: zero API surface change for Vec<T>
- Con: runtime must track lifetimes, can't do lazy/cached transfers

**Option D: GpuVec as a drop-in with From<Vec<T>> auto-coercion**
- Make GpuVec implement enough Vec<T> traits to be interchangeable
- Add blanket impl or macro to auto-convert at kernel boundary
- Pro: builds on working code
- Con: still a different type

### 5. Key Design Decisions

1. **Eager vs lazy transfer**: GpuVec currently does eager copy-to-pinned on `from_vec()`.
   A transparent type should be lazy — only transfer when a kernel needs dev_ptr.

2. **Pinned-mapped vs copy model**: GpuVec uses pinned-mapped (zero-copy over PCIe).
   CudaSlice uses separate device memory (cudaMemcpy). Pinned-mapped is simpler but
   slower for repeated GPU access. The transparent type should choose automatically.

3. **Ownership**: GpuVec consumes the Vec on creation. A transparent wrapper should
   allow borrowing (&[T]) for read-only kernel inputs.

4. **Sync model**: Currently caller must know when GPU is done (after synchronize()).
   A transparent type should handle sync automatically before host reads.

5. **Multiple GPU access**: If the same data is used in multiple kernel launches,
   should it stay in device memory? Current GpuVec re-maps every time.

### 6. Existing Abstractions That Overlap

- **AutoScheduler::par_map()**: Already provides transparent CPU/GPU routing, but only
  for a fixed pre-compiled kernel. The closure is ignored on GPU path.

- **GpuTensor**: Device-resident tensor with from_host()/to_host(). Similar transfer
  pattern but specialized for nn module (f32 only, shape metadata).

- **GpuContext::upload()/download()**: Named to feel transparent but still explicit.

- **GpuVec itself**: Already halfway to transparent — zero-copy read, no explicit
  cudaMemcpy. The gap is: (a) user must write `GpuVec` not `Vec`, (b) creation is
  explicit, (c) kernel launch still requires PTX/kernel name.

## Open Questions

1. Should the transparent type be a true wrapper (newtype) or should we modify
   the runtime to accept `&[T]` / `&mut [T]` directly?

2. How to handle the kernel selection problem? Currently GpuVec::map_gpu requires
   PTX source and kernel name. A transparent Vec<T> can't know which kernel to run.
   This ties into the compiler/scheduler story.

3. Should we support both pinned-mapped (zero-copy) and device-copy modes, with
   automatic selection based on access patterns and data size?

4. How does this interact with the async runtime? The transparent type should
   work with both sync and async kernel launches.

5. What is the minimum viable change that makes existing examples simpler while
   maintaining full GPU performance?
