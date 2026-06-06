//! Transparent GPU-aware array with automatic residency management.
//!
//! [`GpuArray<T>`] wraps data that may live on the host, the device, or both.
//! It provides `Deref<Target=[T]>` so host code reads the array as a normal
//! `&[T]`. When passed to a kernel launch via [`AsDevicePtr::ensure_device()`],
//! the runtime automatically ensures data is resident on the device.
//!
//! # Design
//!
//! - **Residency tracking**: A 4-state machine (`HostOnly`, `Synced`,
//!   `DeviceOnly`, `HostDirty`) determines which copy is authoritative.
//! - **Size threshold**: Arrays below 64 KiB use pinned mapped memory
//!   (zero-copy over PCIe); larger arrays use separate device VRAM with
//!   explicit copies.
//! - **Interior mutability**: `UnsafeCell` + `Cell<Residency>` for `Deref`
//!   compatibility (cannot return `Ref<'_, [T]>` through `Deref`).
//!
//! # Example
//!
//! ```no_run
//! use gpu_host::gpu_array::GpuArray;
//!
//! // Create from a Vec — data lives on host only
//! let mut data = GpuArray::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]);
//!
//! // Deref transparently reads host data
//! assert_eq!(data[0], 1.0);
//!
//! // DerefMut marks host as dirty for next kernel upload
//! data[0] = 42.0;
//! ```

use std::cell::Cell;
use std::cell::UnsafeCell;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DeviceRepr, ValidAsZeroBits};

use crate::error::{GpuHostError, Result};
use crate::memory::MappedBuffer;

/// Size threshold for choosing pinned-mapped vs device-copy backend.
///
/// Below this: `MappedBuffer` (zero-copy, GPU reads over PCIe).
/// At or above: `CudaSlice` (separate device VRAM with explicit copies).
///
/// Set at 64 KiB. For `f32` (4 bytes) that is 16,384 elements. Below this
/// threshold, the PCIe latency of a copy (~5-10 us) dominates any bandwidth
/// benefit of device memory.
const SIZE_THRESHOLD_BYTES: usize = 64 * 1024;

/// Current residency state of a [`GpuArray`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Residency {
    /// Data exists only on host. No device buffer allocated yet.
    HostOnly,

    /// Host and device are in sync. Both contain the same data.
    Synced,

    /// Device has been written by a kernel. Host copy is stale.
    /// A host read (`Deref`) must trigger device-to-host sync first.
    DeviceOnly,

    /// Host has been modified (via `DerefMut`). Device copy is stale.
    /// Next kernel launch must re-upload.
    HostDirty,
}

/// Backend storage on the device, chosen by size threshold.
enum DeviceStorage<T> {
    /// Pinned device-mapped memory (zero-copy over PCIe).
    Mapped(MappedBuffer<T>),

    /// Separate device memory with explicit host-device copies.
    DeviceMem {
        /// Device-side buffer (cudarc `CudaSlice`).
        slice: CudaSlice<T>,
        /// Cached `CudaDevice` handle for transfers.
        dev: Arc<CudaDevice>,
    },
}

impl<T: DeviceRepr> DeviceStorage<T> {
    /// Return the device pointer as a `u64` suitable for kernel arguments.
    fn dev_ptr(&self) -> u64 {
        match self {
            DeviceStorage::Mapped(buf) => buf.dev_ptr(),
            DeviceStorage::DeviceMem { slice, .. } => *slice.device_ptr(),
        }
    }
}

/// Transparent GPU-aware array with automatic residency management.
///
/// Wraps data that may live on the host, the device, or both. Provides
/// `Deref<Target=[T]>` so host code reads the array as a normal `&[T]`.
/// When passed to a kernel launch, the runtime automatically ensures
/// data is resident on the device.
///
/// # Generic bounds
///
/// - `Copy` — required for host-device memcpy semantics.
/// - `Send` — for cross-thread transfer.
/// - `DeviceRepr` — required by cudarc for device memory operations.
/// - `Unpin` — required by cudarc for host-device copy operations.
/// - `'static` — device memory outlives any stack frame.
///
/// All primitive numeric types (`f32`, `u32`, `i64`, etc.) satisfy these bounds.
///
/// # Thread safety
///
/// `GpuArray<T>` is `Send` (where `T: Send`) but `!Sync`. Ownership can
/// transfer across threads, but concurrent shared access is not supported
/// because `Cell<Residency>` and `UnsafeCell` are `!Sync`.
pub struct GpuArray<T: Copy + Send + DeviceRepr + Unpin + 'static> {
    /// Host-side data. Always valid when residency is `HostOnly`, `Synced`,
    /// or `HostDirty`. Stale when `DeviceOnly`.
    host: UnsafeCell<Vec<T>>,

    /// Device-side state. `None` until first device upload.
    device: UnsafeCell<Option<DeviceStorage<T>>>,

    /// Current residency state.
    residency: Cell<Residency>,

    /// Length of the array (immutable after construction).
    len: usize,
}

// SAFETY: GpuArray owns all its data (host Vec + device storage). Transferring
// ownership across threads is safe when T: Send. The UnsafeCell fields are only
// mutated through controlled transition points (ensure_device, sync_to_host,
// deref_mut) which are sequenced by the single-owner &self/&mut self protocol.
unsafe impl<T: Copy + Send + DeviceRepr + Unpin + 'static> Send for GpuArray<T> {}

impl<T: Copy + Send + DeviceRepr + Unpin + 'static> GpuArray<T> {
    /// Create a `GpuArray` from an existing `Vec<T>`.
    ///
    /// The data starts in `HostOnly` residency. No device allocation occurs
    /// until [`ensure_device()`] is called.
    pub fn from_vec(data: Vec<T>) -> Self {
        let len = data.len();
        Self {
            host: UnsafeCell::new(data),
            device: UnsafeCell::new(None),
            residency: Cell::new(Residency::HostOnly),
            len,
        }
    }

    /// Create a `GpuArray` by copying from a slice.
    pub fn from_slice(data: &[T]) -> Self {
        Self::from_vec(data.to_vec())
    }

    /// Return the number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Return whether the array is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Return the current residency state.
    pub fn residency(&self) -> Residency {
        self.residency.get()
    }

    /// Whether this array should use pinned-mapped memory (below threshold).
    fn should_use_mapped(&self) -> bool {
        self.len * std::mem::size_of::<T>() < SIZE_THRESHOLD_BYTES
    }

    /// Synchronize device data back to host (device-to-host copy).
    ///
    /// # Safety
    ///
    /// Caller must ensure no other reference to the host `Vec` contents is live.
    /// This is called from `deref()` when `residency == DeviceOnly`, at which
    /// point no `&[T]` from a previous `deref()` can exist (entering `DeviceOnly`
    /// requires `mark_device_dirty` which only flags the state).
    unsafe fn sync_to_host(&self) {
        self.sync_to_host_fallible()
            .expect("GpuArray: device-to-host sync failed (like Mutex::lock poison)")
    }

    /// Fallible device-to-host sync implementation.
    ///
    /// # Safety
    ///
    /// Same preconditions as [`sync_to_host`](Self::sync_to_host).
    unsafe fn sync_to_host_fallible(&self) -> Result<()> {
        let device_opt = &*self.device.get();
        let storage = device_opt
            .as_ref()
            .expect("GpuArray: DeviceOnly but no device storage");

        let host_vec = &mut *self.host.get();

        match storage {
            DeviceStorage::Mapped(buf) => {
                // Zero-copy: read from the pinned memory into host Vec.
                let src = buf.as_slice();
                host_vec.copy_from_slice(src);
            }
            DeviceStorage::DeviceMem { slice, dev } => {
                // Explicit device-to-host copy.
                let downloaded = dev.dtoh_sync_copy(slice).map_err(GpuHostError::Cudarc)?;
                *host_vec = downloaded;
            }
        }

        self.residency.set(Residency::Synced);
        Ok(())
    }

    /// Explicitly sync device data to host, returning an error on failure.
    ///
    /// Call this before `Deref` if you need to handle sync errors gracefully
    /// instead of panicking.
    pub fn try_sync_to_host(&self) -> Result<&[T]> {
        if self.residency.get() == Residency::DeviceOnly {
            // SAFETY: We hold &self so no &mut exists. The sync writes to
            // self.host through UnsafeCell, which is safe because no other
            // reference to the Vec contents exists (DeviceOnly means the host
            // copy is stale and no one has read it yet).
            unsafe { self.sync_to_host_fallible()? };
        }
        // SAFETY: host data is valid (HostOnly, Synced, or HostDirty).
        Ok(unsafe { &*self.host.get() })
    }

    /// Mark the device contents as dirty (kernel wrote to this buffer).
    ///
    /// Call this after a kernel launch that uses this `GpuArray` as output.
    /// The next `Deref` will trigger an automatic device-to-host sync.
    pub fn mark_device_dirty(&self) {
        let r = self.residency.get();
        debug_assert!(
            r == Residency::Synced || r == Residency::DeviceOnly,
            "mark_device_dirty called in unexpected state: {r:?}"
        );
        if r == Residency::Synced {
            self.residency.set(Residency::DeviceOnly);
        }
    }

    /// Ensure data is on the device and return the device pointer.
    ///
    /// - `HostOnly` -> allocates device storage, copies host-to-device, transitions to `Synced`.
    /// - `HostDirty` -> re-uploads host data to existing device storage, transitions to `Synced`.
    /// - `Synced` / `DeviceOnly` -> no-op, returns existing device pointer.
    ///
    /// # Arguments
    ///
    /// * `dev` - The CUDA device to allocate on. Required for the first upload;
    ///   subsequent calls use the cached device handle (for `DeviceMem` backend).
    pub fn ensure_device(&self, dev: &Arc<CudaDevice>) -> Result<u64> {
        match self.residency.get() {
            Residency::HostOnly => {
                // SAFETY: We hold &self. No other mutation path is active
                // (HostOnly means no device storage exists yet).
                let host_data = unsafe { &*self.host.get() };
                if self.len == 0 {
                    // Zero-length arrays: no allocation needed.
                    self.residency.set(Residency::Synced);
                    return Ok(0);
                }

                let storage = if self.should_use_mapped() {
                    let buf = MappedBuffer::<T>::new_zeroed(self.len)?;
                    // Copy host data into the mapped buffer.
                    // SAFETY: buf was just allocated with the correct length,
                    // and no GPU kernel is running yet.
                    unsafe {
                        std::ptr::copy_nonoverlapping(host_data.as_ptr(), buf.host_ptr(), self.len);
                    }
                    DeviceStorage::Mapped(buf)
                } else {
                    let slice = dev
                        .htod_sync_copy(host_data)
                        .map_err(GpuHostError::Cudarc)?;
                    DeviceStorage::DeviceMem {
                        slice,
                        dev: Arc::clone(dev),
                    }
                };

                let ptr = storage.dev_ptr();
                // SAFETY: single-owner mutation through UnsafeCell. No other
                // reference to the device storage exists.
                unsafe { *self.device.get() = Some(storage) };
                self.residency.set(Residency::Synced);
                Ok(ptr)
            }
            Residency::HostDirty => {
                // SAFETY: We hold &self. Device storage exists (was created in
                // a prior HostOnly->Synced transition). Host data is valid.
                let host_data = unsafe { &*self.host.get() };
                let device_opt = unsafe { &mut *self.device.get() };
                let storage = device_opt
                    .as_mut()
                    .expect("GpuArray: HostDirty but no device storage");

                match storage {
                    DeviceStorage::Mapped(buf) => {
                        // Copy host data into the mapped buffer.
                        // SAFETY: We hold &self, no kernel is in-flight (caller's
                        // responsibility), and the buffer is the correct length.
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                host_data.as_ptr(),
                                buf.host_ptr(),
                                self.len,
                            );
                        }
                    }
                    DeviceStorage::DeviceMem { slice, dev } => {
                        dev.htod_sync_copy_into(host_data, slice)
                            .map_err(GpuHostError::Cudarc)?;
                    }
                }

                self.residency.set(Residency::Synced);
                Ok(storage.dev_ptr())
            }
            Residency::Synced | Residency::DeviceOnly => {
                // SAFETY: device storage was created in a prior transition.
                let device_opt = unsafe { &*self.device.get() };
                let storage = device_opt
                    .as_ref()
                    .expect("GpuArray: Synced/DeviceOnly but no device storage");
                Ok(storage.dev_ptr())
            }
        }
    }
}

impl<T: Copy + Send + DeviceRepr + Unpin + ValidAsZeroBits + 'static> GpuArray<T> {
    /// Create a zeroed `GpuArray` of `len` elements.
    ///
    /// Useful for output buffers that a GPU kernel will write into.
    pub fn zeroed(len: usize) -> Self {
        // SAFETY: ValidAsZeroBits guarantees that all-zeros is a valid bit pattern.
        Self::from_vec(vec![unsafe { std::mem::zeroed() }; len])
    }
}

impl<T: Copy + Send + DeviceRepr + Unpin + 'static> Deref for GpuArray<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        // If device has been written and host is stale, sync back.
        if self.residency.get() == Residency::DeviceOnly {
            // SAFETY: We hold &self so no &mut exists. The sync_to_host call
            // writes to self.host through UnsafeCell, which is safe because no
            // other reference to the Vec contents exists. Entering DeviceOnly
            // requires mark_device_dirty which only sets a flag — it doesn't
            // produce any references.
            unsafe { self.sync_to_host() };
        }
        // SAFETY: host data is valid (HostOnly, Synced, or HostDirty).
        // The UnsafeCell access is safe because we hold &self (shared ref)
        // and the only mutation path (sync_to_host) was completed above.
        unsafe { &*self.host.get() }
    }
}

impl<T: Copy + Send + DeviceRepr + Unpin + 'static> DerefMut for GpuArray<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        // If device wrote, sync back first so we don't lose GPU results.
        if self.residency.get() == Residency::DeviceOnly {
            // SAFETY: &mut self guarantees exclusive access.
            unsafe { self.sync_to_host() };
        }
        // Mark host as dirty — device copy is now stale.
        self.residency.set(Residency::HostDirty);
        // SAFETY: &mut self guarantees exclusive access.
        unsafe { &mut *self.host.get() }
    }
}

impl<T: Copy + Send + DeviceRepr + Unpin + 'static> From<Vec<T>> for GpuArray<T> {
    fn from(data: Vec<T>) -> Self {
        Self::from_vec(data)
    }
}

impl<T: Copy + Send + DeviceRepr + Unpin + 'static> From<&[T]> for GpuArray<T> {
    fn from(data: &[T]) -> Self {
        Self::from_slice(data)
    }
}

/// Trait for types that can provide a device pointer for kernel arguments.
///
/// Implemented by [`GpuArray<T>`], enabling automatic residency management
/// when binding data to kernel launches.
pub trait AsDevicePtr {
    /// Ensure data is on the device and return the device pointer.
    ///
    /// For `GpuArray`: triggers `HostOnly -> Synced` or `HostDirty -> Synced`.
    fn ensure_device(&self, dev: &Arc<CudaDevice>) -> Result<u64>;

    /// Number of elements.
    fn device_len(&self) -> usize;

    /// Mark device contents as dirty (kernel wrote to this buffer).
    fn mark_device_dirty(&self);
}

impl<T: Copy + Send + DeviceRepr + Unpin + 'static> AsDevicePtr for GpuArray<T> {
    fn ensure_device(&self, dev: &Arc<CudaDevice>) -> Result<u64> {
        self.ensure_device(dev)
    }

    fn device_len(&self) -> usize {
        self.len
    }

    fn mark_device_dirty(&self) {
        self.mark_device_dirty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Host-only lifecycle tests (no GPU needed) ─────────────

    #[test]
    fn from_vec_creates_host_only() {
        let arr = GpuArray::from_vec(vec![1.0f32, 2.0, 3.0]);
        assert_eq!(arr.residency(), Residency::HostOnly);
        assert_eq!(arr.len(), 3);
        assert!(!arr.is_empty());
    }

    #[test]
    fn deref_reads_host_data() {
        let arr = GpuArray::from_vec(vec![10u32, 20, 30]);
        assert_eq!(&arr[..], &[10, 20, 30]);
        // Residency stays HostOnly — no device involved.
        assert_eq!(arr.residency(), Residency::HostOnly);
    }

    #[test]
    fn deref_mut_marks_host_dirty() {
        let mut arr = GpuArray::from_vec(vec![1.0f32, 2.0, 3.0]);
        arr[0] = 99.0;
        assert_eq!(arr.residency(), Residency::HostDirty);
        assert_eq!(arr[0], 99.0);
    }

    #[test]
    fn from_slice_copies_data() {
        let data = [5u8, 10, 15];
        let arr = GpuArray::from_slice(&data);
        assert_eq!(&arr[..], &[5, 10, 15]);
    }

    #[test]
    fn zeroed_creates_zero_filled() {
        let arr = GpuArray::<f32>::zeroed(100);
        assert_eq!(arr.len(), 100);
        assert!(arr.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn empty_array() {
        let arr = GpuArray::<f32>::from_vec(vec![]);
        assert!(arr.is_empty());
        assert_eq!(arr.len(), 0);
        assert_eq!(&arr[..], &[] as &[f32]);
    }

    #[test]
    fn from_trait_vec() {
        let arr: GpuArray<i32> = vec![1, 2, 3].into();
        assert_eq!(&arr[..], &[1, 2, 3]);
    }

    #[test]
    fn from_trait_slice() {
        let data = [4.0f64, 5.0, 6.0];
        let arr: GpuArray<f64> = data.as_slice().into();
        assert_eq!(&arr[..], &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn size_threshold_small() {
        // 100 f32 = 400 bytes, well below 64 KiB
        let arr = GpuArray::from_vec(vec![0.0f32; 100]);
        assert!(arr.should_use_mapped());
    }

    #[test]
    fn size_threshold_large() {
        // 64 KiB / 4 bytes = 16384 elements. At 16384 we should NOT use mapped.
        let arr = GpuArray::from_vec(vec![0.0f32; 16384]);
        assert!(!arr.should_use_mapped());
    }

    #[test]
    fn size_threshold_just_below() {
        // 16383 f32 = 65532 bytes, just below 65536
        let arr = GpuArray::from_vec(vec![0.0f32; 16383]);
        assert!(arr.should_use_mapped());
    }

    // ── GPU integration tests ──────────────────────────────────

    #[test]
    fn ensure_device_small_uses_mapped() {
        // Small array: should use MappedBuffer backend.
        let arr = GpuArray::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]);
        let dev = CudaDevice::new(0).expect("CUDA device init");
        let ptr = arr.ensure_device(&dev).expect("ensure_device");
        assert_ne!(ptr, 0);
        assert_eq!(arr.residency(), Residency::Synced);
    }

    #[test]
    fn ensure_device_large_uses_device_mem() {
        // Large array (above 64 KiB threshold): should use CudaSlice backend.
        let n = 32768; // 128 KiB of f32
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let arr = GpuArray::from_vec(data);
        let dev = CudaDevice::new(0).expect("CUDA device init");
        let ptr = arr.ensure_device(&dev).expect("ensure_device");
        assert_ne!(ptr, 0);
        assert_eq!(arr.residency(), Residency::Synced);
    }

    #[test]
    fn ensure_device_idempotent_when_synced() {
        let arr = GpuArray::from_vec(vec![1.0f32; 100]);
        let dev = CudaDevice::new(0).expect("CUDA device init");
        let ptr1 = arr.ensure_device(&dev).expect("first ensure");
        let ptr2 = arr.ensure_device(&dev).expect("second ensure");
        assert_eq!(ptr1, ptr2);
        assert_eq!(arr.residency(), Residency::Synced);
    }

    #[test]
    fn host_dirty_triggers_reupload() {
        let mut arr = GpuArray::from_vec(vec![1.0f32; 100]);
        let dev = CudaDevice::new(0).expect("CUDA device init");

        // First upload
        arr.ensure_device(&dev).expect("first ensure");
        assert_eq!(arr.residency(), Residency::Synced);

        // Modify host data
        arr[0] = 42.0;
        assert_eq!(arr.residency(), Residency::HostDirty);

        // Re-upload
        arr.ensure_device(&dev).expect("reupload");
        assert_eq!(arr.residency(), Residency::Synced);
    }

    #[test]
    fn device_dirty_deref_syncs_small() {
        // Small array with MappedBuffer backend.
        let arr = GpuArray::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]);
        let dev = CudaDevice::new(0).expect("CUDA device init");

        arr.ensure_device(&dev).expect("upload");
        assert_eq!(arr.residency(), Residency::Synced);

        // Simulate kernel writing to device
        arr.mark_device_dirty();
        assert_eq!(arr.residency(), Residency::DeviceOnly);

        // Deref should trigger sync and return host data.
        // For mapped memory, the host sees GPU writes immediately
        // (they share the same physical memory).
        let _slice: &[f32] = &arr;
        assert_eq!(arr.residency(), Residency::Synced);
    }

    #[test]
    fn device_dirty_deref_syncs_large() {
        // Large array with CudaSlice backend.
        let n = 32768;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let arr = GpuArray::from_vec(data);
        let dev = CudaDevice::new(0).expect("CUDA device init");

        arr.ensure_device(&dev).expect("upload");
        arr.mark_device_dirty();
        assert_eq!(arr.residency(), Residency::DeviceOnly);

        // Deref triggers D2H copy
        let slice: &[f32] = &arr;
        assert_eq!(arr.residency(), Residency::Synced);
        // Data should match what was uploaded (kernel didn't actually modify it)
        for i in 0..n {
            assert_eq!(slice[i], i as f32, "mismatch at index {i}");
        }
    }

    #[test]
    fn full_lifecycle() {
        // Create -> Deref (read) -> ensure_device (upload) -> modify host ->
        // ensure_device (re-upload) -> mark_device_dirty -> Deref (sync)
        let mut arr = GpuArray::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]);
        let dev = CudaDevice::new(0).expect("CUDA device init");

        // Step 1: Read on host
        assert_eq!(arr[0], 1.0);
        assert_eq!(arr.residency(), Residency::HostOnly);

        // Step 2: Upload to device
        let ptr = arr.ensure_device(&dev).expect("upload");
        assert_ne!(ptr, 0);
        assert_eq!(arr.residency(), Residency::Synced);

        // Step 3: Modify host
        arr[0] = 99.0;
        assert_eq!(arr.residency(), Residency::HostDirty);

        // Step 4: Re-upload
        arr.ensure_device(&dev).expect("re-upload");
        assert_eq!(arr.residency(), Residency::Synced);

        // Step 5: Simulate kernel write
        arr.mark_device_dirty();
        assert_eq!(arr.residency(), Residency::DeviceOnly);

        // Step 6: Read back (triggers sync)
        let _val = arr[0];
        assert_eq!(arr.residency(), Residency::Synced);
    }

    #[test]
    fn try_sync_to_host_graceful() {
        let arr = GpuArray::from_vec(vec![10.0f32, 20.0]);
        let dev = CudaDevice::new(0).expect("CUDA device init");
        arr.ensure_device(&dev).expect("upload");

        // Not DeviceOnly, so try_sync is a no-op
        let slice = arr.try_sync_to_host().expect("sync ok");
        assert_eq!(slice, &[10.0, 20.0]);

        // Mark dirty, then gracefully sync
        arr.mark_device_dirty();
        let slice = arr.try_sync_to_host().expect("sync ok");
        assert_eq!(slice, &[10.0, 20.0]);
        assert_eq!(arr.residency(), Residency::Synced);
    }

    #[test]
    fn zero_length_ensure_device() {
        let arr = GpuArray::<f32>::from_vec(vec![]);
        let dev = CudaDevice::new(0).expect("CUDA device init");
        let ptr = arr.ensure_device(&dev).expect("empty ensure");
        assert_eq!(ptr, 0);
        assert_eq!(arr.residency(), Residency::Synced);
    }

    #[test]
    fn as_device_ptr_trait() {
        let arr = GpuArray::from_vec(vec![1.0f32; 100]);
        let dev = CudaDevice::new(0).expect("CUDA device init");

        let dyn_ref: &dyn AsDevicePtr = &arr;
        assert_eq!(dyn_ref.device_len(), 100);

        let ptr = dyn_ref.ensure_device(&dev).expect("upload via trait");
        assert_ne!(ptr, 0);

        dyn_ref.mark_device_dirty();
        assert_eq!(arr.residency(), Residency::DeviceOnly);
    }

    // ── GpuArray + gpu::custom() integration tests ────────────────
    //
    // These tests use a minimal inline PTX kernel (f(x) = x * 2.0 + 1.0)
    // that JIT-compiles in milliseconds, avoiding the 10+ minute full
    // kernel PTX compilation.

    /// Minimal PTX for a map kernel: f(x) = x * 2.0 + 1.0
    ///
    /// Kernel signature: `fn(input: *const f32, output: *mut f32, n: u32)`
    /// Grid-stride loop for multi-block correctness.
    const MAP_KERNEL_PTX: &str = "\
.version 7.8\n\
.target sm_75\n\
.address_size 64\n\
\n\
.visible .entry gpu_array_map_f32(\n\
    .param .u64 input,\n\
    .param .u64 output,\n\
    .param .u32 n\n\
)\n\
{\n\
    .reg .u32  %r<10>;\n\
    .reg .u64  %rd<6>;\n\
    .reg .f32  %f<3>;\n\
    .reg .pred %p;\n\
\n\
    ld.param.u64    %rd0, [input];\n\
    ld.param.u64    %rd1, [output];\n\
    ld.param.u32    %r0,  [n];\n\
\n\
    mov.u32         %r1,  %tid.x;\n\
    mov.u32         %r2,  %ntid.x;\n\
    mov.u32         %r3,  %ctaid.x;\n\
    mov.u32         %r4,  %nctaid.x;\n\
\n\
    mad.lo.u32      %r5,  %r3, %r2, %r1;\n\
    mul.lo.u32      %r6,  %r4, %r2;\n\
\n\
LOOP:\n\
    setp.ge.u32     %p, %r5, %r0;\n\
    @%p bra         DONE;\n\
\n\
    mul.wide.u32    %rd2, %r5, 4;\n\
    add.u64         %rd3, %rd0, %rd2;\n\
    ld.global.f32   %f0,  [%rd3];\n\
\n\
    fma.rn.f32      %f1,  %f0, 0f40000000, 0f3F800000;\n\
\n\
    add.u64         %rd4, %rd1, %rd2;\n\
    st.global.f32   [%rd4], %f1;\n\
\n\
    add.u32         %r5, %r5, %r6;\n\
    bra             LOOP;\n\
\n\
DONE:\n\
    ret;\n\
}\n\
";

    /// End-to-end: GpuArray<f32> with gpu::custom() — auto H2D, kernel, auto D2H.
    ///
    /// This is the core story criterion test: a program that uses only
    /// GpuArray<f32> with zero explicit cudaMemcpy, zero GpuVec, zero CudaSlice.
    #[test]
    fn gpu_array_custom_launch_small() {
        let n: usize = 1024;
        let input_data: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();

        // Create GpuArrays — just normal Rust data, no GPU concepts visible
        let input = GpuArray::from_vec(input_data.clone());
        let output = GpuArray::<f32>::zeroed(n);

        // Prepare kernel
        let ctx = crate::gpu::custom("gpu_array_map_f32")
            .ptx(MAP_KERNEL_PTX)
            .threads(256)
            .elements(n as u32)
            .prepare()
            .expect("prepare");

        // Bind GpuArrays — auto H2D transfer
        let input_ptr = ctx.bind_gpu_array(&input).expect("bind input");
        let output_ptr = ctx.bind_gpu_array(&output).expect("bind output");

        assert_eq!(input.residency(), Residency::Synced);
        assert_eq!(output.residency(), Residency::Synced);
        assert_ne!(input_ptr, 0);
        assert_ne!(output_ptr, 0);

        // Launch kernel with raw u64 pointers
        let _result = unsafe {
            ctx.launch((input_ptr, output_ptr, n as u32))
                .expect("launch")
        };

        // Mark output as device-dirty (kernel wrote to it)
        output.mark_device_dirty();
        assert_eq!(output.residency(), Residency::DeviceOnly);

        // Read results — Deref triggers auto D2H sync
        let results: &[f32] = &output;
        assert_eq!(output.residency(), Residency::Synced);

        for i in 0..n {
            let expected = input_data[i] * 2.0 + 1.0;
            assert!(
                (results[i] - expected).abs() < 1e-5,
                "mismatch at index {i}: got {}, expected {expected}",
                results[i]
            );
        }
    }

    /// GpuArray + gpu::custom() with large data (above 64 KiB threshold).
    ///
    /// Uses the CudaSlice backend (not MappedBuffer) for device storage.
    /// Verifies that the auto-transfer works correctly for both backends.
    #[test]
    fn gpu_array_custom_launch_large() {
        let n: usize = 32768; // 128 KiB of f32, above 64 KiB threshold
        let input_data: Vec<f32> = (0..n).map(|i| i as f32 * 0.001).collect();

        let input = GpuArray::from_vec(input_data.clone());
        let output = GpuArray::<f32>::zeroed(n);

        // Verify large arrays use DeviceMem backend
        assert!(!input.should_use_mapped());
        assert!(!output.should_use_mapped());

        let ctx = crate::gpu::custom("gpu_array_map_f32")
            .ptx(MAP_KERNEL_PTX)
            .threads(256)
            .elements(n as u32)
            .prepare()
            .expect("prepare");

        let input_ptr = ctx.bind_gpu_array(&input).expect("bind input");
        let output_ptr = ctx.bind_gpu_array(&output).expect("bind output");

        let _result = unsafe {
            ctx.launch((input_ptr, output_ptr, n as u32))
                .expect("launch")
        };

        output.mark_device_dirty();

        // Deref triggers D2H — for large arrays, this is a real cudaMemcpy
        let results: &[f32] = &output;
        for i in (0..n).step_by(1000) {
            let expected = input_data[i] * 2.0 + 1.0;
            assert!(
                (results[i] - expected).abs() < 1e-3,
                "mismatch at index {i}: got {}, expected {expected}",
                results[i]
            );
        }
    }

    /// Full transparent lifecycle: modify -> re-upload -> kernel -> read back.
    ///
    /// Tests the HostDirty -> Synced -> DeviceOnly -> Synced state transitions
    /// through the gpu::custom() builder API.
    #[test]
    fn gpu_array_modify_reupload_cycle() {
        let n: usize = 256;

        let mut input = GpuArray::from_vec(vec![1.0f32; n]);
        let output = GpuArray::<f32>::zeroed(n);

        // First pass: all 1.0 -> kernel -> f(1.0) = 1.0 * 2.0 + 1.0 = 3.0
        {
            let ctx = crate::gpu::custom("gpu_array_map_f32")
                .ptx(MAP_KERNEL_PTX)
                .threads(256)
                .elements(n as u32)
                .prepare()
                .expect("prepare");

            let in_ptr = ctx.bind_gpu_array(&input).expect("bind");
            let out_ptr = ctx.bind_gpu_array(&output).expect("bind");

            let _result = unsafe { ctx.launch((in_ptr, out_ptr, n as u32)).expect("launch") };
            output.mark_device_dirty();

            let results: &[f32] = &output;
            assert!(
                (results[0] - 3.0).abs() < 1e-5,
                "expected 3.0, got {}",
                results[0]
            );
        }

        // Modify input on host: all 1.0 -> all 10.0
        input[0] = 10.0;
        assert_eq!(input.residency(), Residency::HostDirty);

        // Second pass: 10.0 -> kernel -> f(10.0) = 10.0 * 2.0 + 1.0 = 21.0
        {
            let ctx = crate::gpu::custom("gpu_array_map_f32")
                .ptx(MAP_KERNEL_PTX)
                .threads(256)
                .elements(n as u32)
                .prepare()
                .expect("prepare");

            // bind_gpu_array triggers re-upload of HostDirty data
            let in_ptr = ctx.bind_gpu_array(&input).expect("re-bind");
            assert_eq!(input.residency(), Residency::Synced);

            let out_ptr = ctx.bind_gpu_array(&output).expect("bind");

            let _result = unsafe { ctx.launch((in_ptr, out_ptr, n as u32)).expect("launch") };
            output.mark_device_dirty();

            let results: &[f32] = &output;
            assert!(
                (results[0] - 21.0).abs() < 1e-5,
                "expected 21.0, got {}",
                results[0]
            );
            // Other elements still 1.0 -> f(1.0) = 3.0
            assert!(
                (results[1] - 3.0).abs() < 1e-5,
                "expected 3.0, got {}",
                results[1]
            );
        }
    }

    /// Transparent demo: uses only GpuArray<f32> — zero explicit cudaMemcpy,
    /// zero GpuVec, zero CudaSlice in user code.
    ///
    /// This is the story criterion demo: "a program that uses only Vec<f32>
    /// and runs compute on GPU transparently."
    #[test]
    fn gpu_array_transparent_demo() {
        // Step 1: User writes normal Rust code — just Vec<f32>
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let n = data.len();

        // Step 2: Wrap in GpuArray (the only "GPU" concept)
        let input = GpuArray::from_vec(data);
        let result = GpuArray::<f32>::zeroed(n);

        // Step 3: Launch kernel — auto H2D transfers
        let ctx = crate::gpu::custom("gpu_array_map_f32")
            .ptx(MAP_KERNEL_PTX)
            .threads(256)
            .elements(n as u32)
            .prepare()
            .expect("prepare");

        let in_ptr = ctx.bind_gpu_array(&input).expect("bind input");
        let out_ptr = ctx.bind_gpu_array(&result).expect("bind output");

        let _res = unsafe { ctx.launch((in_ptr, out_ptr, n as u32)).expect("launch") };
        result.mark_device_dirty();

        // Step 4: Read results — auto D2H sync, Deref makes it feel like a slice
        assert_eq!(result[0], 1.0 * 2.0 + 1.0); // 3.0
        assert_eq!(result[1], 2.0 * 2.0 + 1.0); // 5.0
        assert_eq!(result[7], 8.0 * 2.0 + 1.0); // 17.0

        // The user never wrote: cudaMemcpy, htod, dtoh, CudaSlice, or GpuVec
    }

    /// Test bind_gpu_array via the AsDevicePtr trait (dynamic dispatch).
    #[test]
    fn gpu_array_bind_via_trait() {
        let n: usize = 64;
        let input = GpuArray::from_vec(vec![5.0f32; n]);
        let output = GpuArray::<f32>::zeroed(n);

        let ctx = crate::gpu::custom("gpu_array_map_f32")
            .ptx(MAP_KERNEL_PTX)
            .threads(256)
            .elements(n as u32)
            .prepare()
            .expect("prepare");

        // Use AsDevicePtr trait references
        let input_ref: &dyn AsDevicePtr = &input;
        let output_ref: &dyn AsDevicePtr = &output;

        let in_ptr = ctx.bind_gpu_array(input_ref).expect("bind input");
        let out_ptr = ctx.bind_gpu_array(output_ref).expect("bind output");

        let _result = unsafe { ctx.launch((in_ptr, out_ptr, n as u32)).expect("launch") };

        output_ref.mark_device_dirty();

        // f(5.0) = 5.0 * 2.0 + 1.0 = 11.0
        let results: &[f32] = &output;
        assert!(
            (results[0] - 11.0).abs() < 1e-5,
            "expected 11.0, got {}",
            results[0]
        );
    }

    /// Test try_sync_to_host after a real kernel launch.
    ///
    /// Verifies that the fallible sync path works after actual GPU writes.
    #[test]
    fn gpu_array_try_sync_after_kernel() {
        let n: usize = 128;
        let input = GpuArray::from_vec(vec![3.0f32; n]);
        let output = GpuArray::<f32>::zeroed(n);

        let ctx = crate::gpu::custom("gpu_array_map_f32")
            .ptx(MAP_KERNEL_PTX)
            .threads(256)
            .elements(n as u32)
            .prepare()
            .expect("prepare");

        let in_ptr = ctx.bind_gpu_array(&input).expect("bind");
        let out_ptr = ctx.bind_gpu_array(&output).expect("bind");

        let _result = unsafe { ctx.launch((in_ptr, out_ptr, n as u32)).expect("launch") };
        output.mark_device_dirty();

        // Use try_sync_to_host instead of Deref
        let results = output.try_sync_to_host().expect("try_sync");
        assert_eq!(output.residency(), Residency::Synced);

        // f(3.0) = 3.0 * 2.0 + 1.0 = 7.0
        for &val in results.iter() {
            assert!((val - 7.0).abs() < 1e-5, "expected 7.0, got {val}",);
        }
    }
}
