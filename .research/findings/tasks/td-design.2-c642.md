# td-design.2: Design TransparentVec<T> with Residency State Machine and Deref<Target=[T]>

Status: COMPLETE
Task kind: design

## Summary

Concrete type design for `GpuArray<T>` — a transparent data container that tracks
host/device residency internally, provides `Deref<Target=[T]>` for ergonomic host
access, and auto-migrates data at kernel launch boundaries. Users write
`GpuArray::from(vec)` once and then treat it like `&[T]` everywhere.

## Design

### 1. Type Definition

```rust
/// Transparent GPU-aware array with automatic residency management.
///
/// Wraps data that may live on the host, the device, or both. Provides
/// `Deref<Target=[T]>` so host code reads the array as a normal `&[T]`.
/// When passed to a kernel launch, the runtime automatically ensures
/// data is resident on the device.
pub struct GpuArray<T: Copy + Send + 'static> {
    /// Host-side data. Always valid when residency is Host or Synced.
    /// After a device-only kernel write, this becomes stale until sync.
    host: UnsafeCell<Vec<T>>,

    /// Device-side state. None until first device upload.
    /// Uses MappedBuffer for small arrays, CudaSlice for large.
    device: UnsafeCell<Option<DeviceStorage<T>>>,

    /// Current residency state.
    residency: Cell<Residency>,

    /// Length of the array (immutable after construction).
    len: usize,
}
```

**Generic bounds**: `T: Copy + Send + 'static` — same as existing `GpuVec<T>`.
`Copy` is required for host-device memcpy semantics. `Send` for cross-thread
transfer. `'static` because device memory outlives any stack frame.

**No lifetime parameters**: The array owns its data on both sides. Borrows
are only produced via `Deref` (host reads) and `dev_ptr()` (kernel args).
This keeps the type simple and `Send + Sync`-compatible.

**Interior mutability**: `UnsafeCell` + `Cell<Residency>` rather than `RefCell`
because:
- `Deref` cannot return a `Ref<'_, [T]>` (wrong return type for the trait)
- Residency transitions are simple enum swaps, no runtime borrow tracking needed
- Safety invariant: host data is only mutated in `sync_to_host()` which is
  called exclusively when residency is `DeviceOnly`

### 2. Device Storage Enum

```rust
/// Backend storage on the device, chosen by size threshold.
enum DeviceStorage<T> {
    /// Pinned device-mapped memory (zero-copy over PCIe).
    /// Used for arrays <= SIZE_THRESHOLD elements.
    /// Pro: no explicit cudaMemcpy, GPU reads host memory directly.
    /// Con: every GPU access traverses PCIe, slower for repeated reads.
    Mapped(MappedBuffer<T>),

    /// Separate device memory with explicit host<->device copies.
    /// Used for arrays > SIZE_THRESHOLD elements.
    /// Pro: GPU accesses fast VRAM, not PCIe.
    /// Con: requires explicit cudaMemcpy at transition points.
    DeviceMem {
        /// Device-side buffer (cudarc CudaSlice).
        slice: CudaSlice<T>,
        /// Cached CudaDevice handle for transfers.
        dev: Arc<CudaDevice>,
    },
}
```

### 3. Residency State Machine

```
                    ┌─────────────────────────────┐
                    │         HostOnly             │
                    │  (host valid, no device buf) │
                    └──────────┬──────────────────┘
                               │
                   ensure_device() — first kernel launch
                   allocates DeviceStorage, copies host→device
                               │
                               ▼
                    ┌──────────────────────────────┐
          ┌────────│           Synced              │────────┐
          │        │  (host valid, device valid)   │        │
          │        └──────────────────────────────-┘        │
          │                                                 │
  kernel writes to              host writes via
  device (mark dirty)           deref_mut (mark dirty)
          │                                                 │
          ▼                                                 ▼
┌──────────────────┐                          ┌──────────────────┐
│   DeviceOnly     │                          │   HostDirty      │
│ (device valid,   │                          │ (host valid+     │
│  host stale)     │                          │  modified,       │
│                  │                          │  device stale)   │
└────────┬─────────┘                          └────────┬─────────┘
         │                                             │
   Deref (host read)                          ensure_device()
   triggers sync_to_host:                     copies host→device:
   device→host copy                           host→device copy
         │                                             │
         ▼                                             ▼
┌──────────────────┐                          ┌──────────────────┐
│      Synced      │                          │      Synced      │
└──────────────────┘                          └──────────────────┘
```

```rust
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Residency {
    /// Data exists only on host. No device buffer allocated yet.
    HostOnly,

    /// Host and device are in sync. Both contain the same data.
    Synced,

    /// Device has been written by a kernel. Host copy is stale.
    /// A host read (Deref) must trigger device→host sync first.
    DeviceOnly,

    /// Host has been modified (via deref_mut or explicit write).
    /// Device copy is stale. Next kernel launch must re-upload.
    HostDirty,
}
```

**Transition triggers:**

| From        | Trigger                          | To          | Action                    |
|-------------|----------------------------------|-------------|---------------------------|
| HostOnly    | `ensure_device()`               | Synced      | Alloc device + H→D copy  |
| Synced      | `mark_device_dirty()`           | DeviceOnly  | (no copy, just flag)      |
| Synced      | `deref_mut()` / `as_mut_slice()`| HostDirty   | (no copy, just flag)      |
| DeviceOnly  | `Deref` / `as_slice()`          | Synced      | D→H copy                 |
| DeviceOnly  | `ensure_device()`               | DeviceOnly  | (no-op, already on device)|
| HostDirty   | `ensure_device()`               | Synced      | H→D copy                 |
| HostDirty   | `Deref` / `as_slice()`          | HostDirty   | (no-op, host is valid)    |

### 4. Deref<Target=[T]> Implementation

```rust
impl<T: Copy + Send + 'static> Deref for GpuArray<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        // If device has been written and host is stale, sync back.
        if self.residency.get() == Residency::DeviceOnly {
            // SAFETY: We hold &self so no &mut exists. The sync_to_host
            // call writes to self.host through UnsafeCell, which is safe
            // because no other reference to the Vec's contents exists
            // (device is done writing — the user called synchronize()
            // or the runtime did it automatically after kernel launch).
            unsafe { self.sync_to_host() };
        }
        // SAFETY: host data is valid (HostOnly, Synced, or HostDirty).
        // The UnsafeCell access is safe because we hold &self (shared ref),
        // and no concurrent mutation can occur (single-threaded access
        // enforced by the Cell<Residency> protocol).
        unsafe { &*self.host.get() }
    }
}
```

**Why UnsafeCell, not RefCell?**

The `Deref` trait requires returning `&[T]`, not `Ref<'_, Vec<T>>`. A `RefCell`
would require returning a `Ref` guard, which is incompatible with the trait
signature. `UnsafeCell` lets us return a plain reference with the correct
lifetime tied to `&self`.

**Safety argument**: The `UnsafeCell<Vec<T>>` is only mutated in two places:
1. `sync_to_host()` — called from `deref()` when `residency == DeviceOnly`.
   At this point, no `&[T]` reference from a previous `deref()` can exist
   because transitioning to `DeviceOnly` requires either (a) constructing a
   new `GpuArray` or (b) calling `mark_device_dirty()` which takes `&self`.
   The key invariant: once `deref()` returns `&[T]`, the residency is
   `Synced` or `HostDirty`, so subsequent `deref()` calls skip the mutation path.
2. `deref_mut()` on `DerefMut` — takes `&mut self`, exclusive access guaranteed.

**DerefMut for host writes:**

```rust
impl<T: Copy + Send + 'static> DerefMut for GpuArray<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        // If device wrote, sync back first so we don't lose GPU results.
        if self.residency.get() == Residency::DeviceOnly {
            unsafe { self.sync_to_host() };
        }
        // Mark host as dirty — device copy is now stale.
        self.residency.set(Residency::HostDirty);
        // SAFETY: &mut self guarantees exclusive access.
        unsafe { &mut *self.host.get() }
    }
}
```

### 5. Kernel Interface

The runtime needs to know a `GpuArray` is a kernel input/output. We introduce
a trait that `GpuArray<T>` implements:

```rust
/// Trait for types that can provide a device pointer for kernel arguments.
///
/// Implemented by GpuArray<T>, GpuVec<T>, and CudaSlice<T> wrappers.
pub trait AsDevicePtr {
    /// Ensure data is on the device and return the device pointer.
    ///
    /// For GpuArray: triggers HostOnly→Synced or HostDirty→Synced transition.
    /// For GpuVec: no-op (always mapped).
    /// For CudaSlice wrappers: no-op (always on device).
    fn ensure_device(&self) -> Result<u64>;

    /// Number of elements.
    fn device_len(&self) -> usize;

    /// Mark device contents as dirty (kernel wrote to this buffer).
    ///
    /// Called by the runtime after a kernel launch that uses this as output.
    fn mark_device_dirty(&self);
}

impl<T: Copy + Send + 'static> AsDevicePtr for GpuArray<T> {
    fn ensure_device(&self) -> Result<u64> {
        match self.residency.get() {
            Residency::HostOnly => {
                // Allocate device storage + copy host→device
                let storage = self.allocate_device()?;
                self.copy_host_to_device(&storage)?;
                // SAFETY: single-threaded mutation through UnsafeCell
                unsafe { *self.device.get() = Some(storage) };
                self.residency.set(Residency::Synced);
                Ok(unsafe { (*self.device.get()).as_ref().unwrap().dev_ptr() })
            }
            Residency::HostDirty => {
                // Re-upload modified host data
                let device = unsafe { (*self.device.get()).as_ref().unwrap() };
                self.copy_host_to_device(device)?;
                self.residency.set(Residency::Synced);
                Ok(device.dev_ptr())
            }
            Residency::Synced | Residency::DeviceOnly => {
                // Already on device
                Ok(unsafe { (*self.device.get()).as_ref().unwrap().dev_ptr() })
            }
        }
    }

    fn device_len(&self) -> usize {
        self.len
    }

    fn mark_device_dirty(&self) {
        if self.residency.get() == Residency::Synced {
            self.residency.set(Residency::DeviceOnly);
        }
    }
}
```

**Integration with gpu::custom()**: The builder gains a method that accepts
`AsDevicePtr` inputs:

```rust
impl GpuContext {
    /// Upload a GpuArray (or any AsDevicePtr) as a kernel argument.
    ///
    /// Calls ensure_device() to handle residency transitions, then
    /// returns the device pointer for use in launch args.
    pub fn bind<D: AsDevicePtr>(&self, data: &D) -> Result<u64> {
        data.ensure_device()
    }

    /// Mark a GpuArray as written-by-kernel after launch.
    pub fn mark_output<D: AsDevicePtr>(&self, data: &D) {
        data.mark_device_dirty();
    }
}
```

### 6. API Sketch — Full Lifecycle

```rust
// ── Creation ──────────────────────────────────────────────
// From Vec (most common)
let data = GpuArray::from(vec![1.0f32, 2.0, 3.0, 4.0]);

// From slice (copies)
let data = GpuArray::from_slice(&[1.0f32, 2.0, 3.0]);

// Zeroed output buffer
let mut output = GpuArray::<f32>::zeroed(1024);

// ── Host reads (Deref) ───────────────────────────────────
// Transparently reads host data. If a kernel has written to device,
// this triggers an automatic device→host sync.
let sum: f32 = data.iter().sum();
assert_eq!(data[0], 1.0);
println!("length = {}", data.len());  // Deref to [T]

// ── Host writes (DerefMut) ───────────────────────────────
// Marks host as dirty — device will be re-uploaded on next kernel launch.
output[0] = 42.0;
output.iter_mut().for_each(|x| *x += 1.0);

// ── Pass to kernel ───────────────────────────────────────
let ctx = gpu::custom("my_kernel")
    .threads(256)
    .elements(data.len() as u32)
    .prepare()?;

// bind() triggers HostOnly→Synced (first use) or HostDirty→Synced
let in_ptr = ctx.bind(&data)?;
let out_ptr = ctx.bind(&output)?;

let result = unsafe {
    ctx.launch((in_ptr, out_ptr, data.len() as u32))?
};

// After launch, mark output as device-dirty
output.mark_device_dirty();

// ── Read results ─────────────────────────────────────────
// Deref triggers DeviceOnly→Synced (automatic device→host copy)
println!("result[0] = {}", output[0]);

// ── Re-use for another kernel ────────────────────────────
// data is still Synced, no re-upload needed.
// output is now Synced (after the Deref read above), but if we
// modify it on the host, it goes to HostDirty again.
```

### 7. Size Threshold for Backend Selection

```rust
/// Elements threshold for choosing pinned-mapped vs device-copy backend.
///
/// Below this: MappedBuffer (zero-copy, GPU reads over PCIe).
///   - Pro: no explicit H→D/D→H copies, simpler lifecycle
///   - Con: every GPU memory access traverses PCIe (~12 GB/s)
///
/// At or above: CudaSlice (separate device VRAM with explicit copies).
///   - Pro: GPU reads at VRAM bandwidth (~900 GB/s for HBM)
///   - Con: requires H→D copy before kernel, D→H copy after
///
/// The threshold is set at 64 KiB worth of elements. For f32 (4 bytes),
/// that's 16,384 elements. Below this, PCIe latency of the copy itself
/// (~5-10 μs) dominates any bandwidth benefit of device memory.
const SIZE_THRESHOLD_BYTES: usize = 64 * 1024; // 64 KiB

impl<T: Copy + Send + 'static> GpuArray<T> {
    fn should_use_mapped(&self) -> bool {
        self.len * std::mem::size_of::<T>() < SIZE_THRESHOLD_BYTES
    }

    fn allocate_device(&self) -> Result<DeviceStorage<T>> {
        if self.should_use_mapped() {
            Ok(DeviceStorage::Mapped(MappedBuffer::new_zeroed(self.len)?))
        } else {
            let dev = CudaDevice::new(0).map_err(GpuHostError::CudaInit)?;
            let slice = dev.alloc_zeros::<T>(self.len).map_err(GpuHostError::Cudarc)?;
            Ok(DeviceStorage::DeviceMem {
                slice,
                dev: dev.clone(),
            })
        }
    }
}
```

### 8. Integration with Existing APIs

**Coexistence**: `GpuArray<T>` is additive. Existing `GpuVec<T>`, `CudaSlice<T>`,
and `MappedBuffer<T>` continue to work unchanged. `GpuArray` is the recommended
type for new code; existing code migrates at its own pace.

**gpu::custom() builder**: Works today with `ctx.bind(&gpu_array)` returning a `u64`
device pointer. No changes to the builder itself — only `GpuContext` gains
`bind()` and `mark_output()` convenience methods.

**AutoScheduler**: Future work — `par_map` gains an overload accepting `&GpuArray<f32>`
instead of `&[f32]`. The scheduler calls `ensure_device()` on the GPU path and
`Deref` on the CPU path. This is not part of the initial implementation.

**gpu::run() / gpu::launch()**: These are output-only APIs (no input data). They
don't interact with `GpuArray` directly. Users who need input+output use
`gpu::custom()`.

**GpuVec interop**: `GpuArray` can be constructed from a `GpuVec` (takes ownership
of the underlying `MappedBuffer`). Conversely, `GpuArray` can produce a `GpuVec`
view when the backend is `Mapped`.

### 9. Error Handling

| Error condition         | Behavior                                      |
|-------------------------|-----------------------------------------------|
| CUDA OOM on alloc       | `ensure_device()` returns `Err(GpuHostError::CudaAlloc)` |
| CUDA OOM on H→D copy   | `ensure_device()` returns `Err(GpuHostError::Cudarc)`    |
| D→H sync failure       | `sync_to_host()` panics (called from `Deref`)            |
| No CUDA device          | `ensure_device()` returns `Err(GpuHostError::CudaInit)`  |
| Zero-length array       | Allowed; no device allocation performed                   |

**Deref panic policy**: `Deref` cannot return `Result`. If a device→host sync
fails inside `deref()`, the implementation panics with a descriptive message.
This matches Rust conventions (e.g., `Mutex::lock()` panics on poison).
Users who need fallible reads can call `try_sync_to_host() -> Result<&[T]>`
explicitly before `Deref`.

```rust
impl<T: Copy + Send + 'static> GpuArray<T> {
    /// Explicitly sync device→host, returning an error on failure.
    ///
    /// Call this before Deref if you need to handle sync errors gracefully.
    pub fn try_sync_to_host(&self) -> Result<&[T]> {
        if self.residency.get() == Residency::DeviceOnly {
            unsafe { self.sync_to_host_fallible()? };
        }
        Ok(unsafe { &*self.host.get() })
    }
}
```

### 10. Thread Safety

`GpuArray<T>` is `!Sync` because `Cell<Residency>` and `UnsafeCell` are `!Sync`.
This is correct: concurrent reads from multiple threads while a device→host
sync might mutate the host buffer would be unsound.

`GpuArray<T>` is `Send` (where `T: Send`): ownership can be transferred across
threads, which is the common pattern (create on main thread, move to a
worker that does GPU work).

If multi-threaded shared access is needed in the future, a `SharedGpuArray<T>`
wrapper using `RwLock<Residency>` + `Mutex<Option<DeviceStorage<T>>>` can be
added. This is not part of the initial design.

### 11. Incremental Implementation Plan

**Phase 1: Core type + Deref (this feature)**
- `GpuArray<T>` struct with `Residency` enum
- `From<Vec<T>>`, `from_slice()`, `zeroed()` constructors
- `Deref<Target=[T]>` with auto-sync
- `DerefMut` with dirty tracking
- Unit tests (host-only lifecycle, no GPU needed)

**Phase 2: Device storage + ensure_device()**
- `DeviceStorage` enum (Mapped + DeviceMem backends)
- `ensure_device() -> Result<u64>` with size-based backend selection
- `AsDevicePtr` trait
- Integration test: create → bind → launch → read

**Phase 3: GpuContext integration**
- `ctx.bind()` and `ctx.mark_output()` methods
- End-to-end example: `GpuArray` with `gpu::custom()` builder
- Migrate one existing example (e.g., vector-math) to use `GpuArray`

**Phase 4: AutoScheduler integration (future story)**
- `par_map(&GpuArray<f32>, ...)` overload
- Scheduler calls `ensure_device()` on GPU path, `Deref` on CPU path

## Open Questions

1. **Cached CudaDevice handle**: Currently `allocate_device()` calls
   `CudaDevice::new(0)` which creates/retains a context. Should `GpuArray`
   cache the device handle to avoid repeated initialization? The `GpuRuntime`
   already exists for this. Resolution: defer to Phase 2, use lazy init.

2. **Multi-GPU**: The design assumes device 0. Multi-GPU support (device
   affinity per `GpuArray`) is a separate story. The `DeviceStorage::DeviceMem`
   variant already stores an `Arc<CudaDevice>` which can target any device.

3. **Async kernel launches**: With async/await on GPU, the `mark_device_dirty()`
   call must happen after the kernel future resolves, not at launch time.
   This requires integration with the async executor (future work).

4. **Drop ordering**: If `GpuArray` is dropped while device storage still
   exists, the `DeviceStorage::drop()` frees CUDA memory. This is correct
   as long as no kernel is in-flight using the pointer. The runtime must
   synchronize before dropping output `GpuArray`s.
