//! RAII wrappers for pinned, device-mapped host memory.
//!
//! Provides `MappedBuffer<T>` — a typed, automatically-freed handle for
//! GPU-CPU shared memory allocated via `cuMemHostAlloc(DEVICEMAP|PORTABLE)`.

use cudarc::driver::sys::{self, lib as cuda_lib};

use crate::error::{GpuHostError, Result};

/// RAII handle for pinned, device-mapped host memory.
///
/// Provides both a host-side pointer (for CPU reads/writes) and a device-side
/// pointer (for passing to GPU kernels). Memory is zero-initialized on
/// allocation and automatically freed on drop.
///
/// # Example
/// ```no_run
/// use gpu_host::memory::MappedBuffer;
///
/// let mut buf = MappedBuffer::<u32>::new_zeroed(1024).unwrap();
/// // Pass buf.dev_ptr() to a kernel as a launch argument
/// // After kernel completes, read results:
/// let value = unsafe { buf.read(0) };
/// ```
pub struct MappedBuffer<T> {
    host_ptr: *mut T,
    dev_ptr: sys::CUdeviceptr,
    len: usize,
}

// SAFETY: The buffer is pinned memory shared between host and GPU.
// Single-writer access is ensured by the caller's synchronization protocol.
unsafe impl<T: Send> Send for MappedBuffer<T> {}
unsafe impl<T: Sync> Sync for MappedBuffer<T> {}

impl<T> MappedBuffer<T> {
    /// Allocate a zero-initialized mapped buffer with `len` elements.
    pub fn new_zeroed(len: usize) -> Result<Self> {
        let size = len * std::mem::size_of::<T>();
        assert!(size > 0, "cannot allocate zero-sized mapped buffer");

        // SAFETY: cuda_lib() returns a lazily-loaded CUDA driver function table.
        let cu = unsafe { cuda_lib() };
        let flags = sys::CU_MEMHOSTALLOC_DEVICEMAP | sys::CU_MEMHOSTALLOC_PORTABLE;

        let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: cuMemHostAlloc writes to `host_ptr`; flags request pinned device-mapped memory.
        // The returned pointer is valid for `size` bytes until freed with cuMemFreeHost.
        let result = unsafe { cu.cuMemHostAlloc(&mut host_ptr, size, flags) };
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(GpuHostError::CudaAlloc(result));
        }

        let mut dev_ptr: sys::CUdeviceptr = 0;
        // SAFETY: host_ptr was successfully allocated above with DEVICEMAP flag,
        // so cuMemHostGetDevicePointer_v2 can retrieve the GPU-visible address.
        let result = unsafe { cu.cuMemHostGetDevicePointer_v2(&mut dev_ptr, host_ptr, 0) };
        if result != sys::CUresult::CUDA_SUCCESS {
            // SAFETY: host_ptr was allocated by cuMemHostAlloc above; freeing on error path.
            unsafe { cu.cuMemFreeHost(host_ptr) };
            return Err(GpuHostError::CudaGetDevPtr(result));
        }

        // Zero-initialize
        // SAFETY: host_ptr is valid for `size` bytes (just allocated). No GPU kernel
        // is running yet, so there is no concurrent access.
        unsafe { std::ptr::write_bytes(host_ptr as *mut u8, 0, size) };

        Ok(Self {
            host_ptr: host_ptr as *mut T,
            dev_ptr,
            len,
        })
    }

    /// Get the device pointer for passing to GPU kernel arguments.
    pub fn dev_ptr(&self) -> sys::CUdeviceptr {
        self.dev_ptr
    }

    /// Get the host pointer for direct CPU access.
    pub fn host_ptr(&self) -> *mut T {
        self.host_ptr
    }

    /// Number of elements in the buffer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the buffer has zero length (never true for valid buffers).
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Read a value at the given index using a volatile read.
    ///
    /// # Safety
    /// The caller must ensure the index is in bounds and that the GPU has
    /// finished writing to this location (e.g., after `dev.synchronize()`).
    pub unsafe fn read(&self, index: usize) -> T {
        assert!(index < self.len, "index out of bounds");
        // SAFETY: Caller guarantees index is in bounds (asserted above) and that the
        // GPU has finished writing. host_ptr is valid for self.len elements (pinned
        // CUDA memory). Volatile read is used because the GPU may have written this
        // location and we need to observe the latest value.
        unsafe { std::ptr::read_volatile(self.host_ptr.add(index)) }
    }

    /// Write a value at the given index using a volatile write.
    ///
    /// # Safety
    /// The caller must ensure the index is in bounds and that no GPU kernel
    /// is concurrently reading this location.
    pub unsafe fn write(&mut self, index: usize, value: T) {
        assert!(index < self.len, "index out of bounds");
        // SAFETY: Caller guarantees index is in bounds (asserted above) and that no
        // GPU kernel is concurrently reading this location. host_ptr is valid for
        // self.len elements. Volatile write ensures the store is not elided.
        unsafe { std::ptr::write_volatile(self.host_ptr.add(index), value) };
    }

    /// Get a slice view of the host-side memory.
    ///
    /// # Safety
    /// The caller must ensure the GPU is not concurrently writing to this memory.
    pub unsafe fn as_slice(&self) -> &[T] {
        // SAFETY: Caller guarantees the GPU is not concurrently writing. host_ptr
        // is non-null, properly aligned, and valid for self.len elements for the
        // lifetime of this borrow (pinned CUDA memory is not freed until Drop).
        unsafe { std::slice::from_raw_parts(self.host_ptr, self.len) }
    }

    /// Get a mutable slice view of the host-side memory.
    ///
    /// # Safety
    /// The caller must ensure no GPU kernel is concurrently accessing this memory.
    pub unsafe fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: Caller guarantees no GPU kernel is concurrently accessing this
        // memory. host_ptr is non-null, properly aligned, and valid for self.len
        // elements. The &mut self borrow prevents aliasing from Rust code.
        unsafe { std::slice::from_raw_parts_mut(self.host_ptr, self.len) }
    }
}

/// High-level GPU buffer with zero-copy semantics.
///
/// Wraps [`MappedBuffer<T>`] to provide automatic host-device visibility.
/// No explicit transfers needed — data is accessible from both CPU and GPU
/// through CUDA pinned device-mapped memory.
///
/// # Zero-copy model
///
/// `GpuVec` uses pinned host memory that is mapped into the GPU's address space.
/// The GPU reads/writes this memory over PCIe — no `cudaMemcpy` is needed.
/// After kernel completion (and `dev.synchronize()`), results are immediately
/// visible on the host via [`as_slice()`](GpuVec::as_slice).
///
/// # Example
///
/// ```no_run
/// use gpu_host::memory::GpuVec;
///
/// let data = GpuVec::from_vec(vec![1.0f32; 1024]).unwrap();
/// // Pass data.dev_ptr() to a GPU kernel — zero-copy, no cudaMemcpy
/// // After kernel + synchronize, read results:
/// // let results = data.as_slice();
/// ```
pub struct GpuVec<T> {
    inner: MappedBuffer<T>,
}

impl<T: Copy> GpuVec<T> {
    /// Create from an existing `Vec<T>`, copying data into pinned memory.
    ///
    /// The data is immediately accessible from both CPU and GPU after creation.
    pub fn from_vec(data: Vec<T>) -> Result<Self> {
        let len = data.len();
        let buf = MappedBuffer::new_zeroed(len)?;
        // SAFETY: No GPU kernel is running yet; we just allocated the buffer.
        // The copy is from a valid Vec slice into pinned memory of the same length.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), buf.host_ptr(), len);
        }
        Ok(Self { inner: buf })
    }

    /// Create a zeroed buffer of `len` elements.
    ///
    /// Useful for output buffers that the GPU kernel will write into.
    pub fn zeroed(len: usize) -> Result<Self> {
        Ok(Self {
            inner: MappedBuffer::new_zeroed(len)?,
        })
    }

    /// Device pointer for GPU kernel access.
    ///
    /// Returns the device-visible address of this buffer. Pass this value
    /// as a kernel argument so the GPU can read/write the data.
    pub fn dev_ptr(&self) -> u64 {
        self.inner.dev_ptr()
    }

    /// Host-side slice for CPU access (zero-copy read).
    ///
    /// # Safety contract
    ///
    /// The caller must ensure the GPU has finished writing to this buffer
    /// (e.g., after `dev.synchronize()`) before reading the returned slice.
    /// This method itself is safe because it returns an immutable view;
    /// however, reading stale data is possible if the GPU has not been
    /// synchronized.
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: The borrow rules of &self prevent concurrent &mut access
        // from Rust code. GPU synchronization is the caller's responsibility
        // (same as MappedBuffer). The host pointer is valid for self.inner.len()
        // elements for the lifetime of this borrow.
        unsafe { self.inner.as_slice() }
    }

    /// Mutable host-side slice for CPU writes.
    ///
    /// # Safety contract
    ///
    /// The caller must ensure no GPU kernel is concurrently accessing this
    /// buffer. The `&mut self` borrow prevents aliasing from Rust code.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: The &mut self borrow prevents aliasing from Rust code.
        // GPU synchronization is the caller's responsibility (same as
        // MappedBuffer). The host pointer is valid for self.inner.len() elements.
        unsafe { self.inner.as_mut_slice() }
    }

    /// Number of elements in the buffer.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns true if the buffer has zero length (never true for valid buffers).
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Convert back to a `Vec<T>`, copying from pinned memory to an owned Vec.
    pub fn into_vec(self) -> Vec<T> {
        let len = self.inner.len();
        let mut v = Vec::with_capacity(len);
        // SAFETY: We are consuming self, so no kernel should be in-flight.
        // The host pointer is valid for `len` elements. We copy into the Vec's
        // uninitialized buffer and then set its length.
        unsafe {
            std::ptr::copy_nonoverlapping(self.inner.host_ptr(), v.as_mut_ptr(), len);
            v.set_len(len);
        }
        v
    }
}

impl<T: Copy> TryFrom<Vec<T>> for GpuVec<T> {
    type Error = GpuHostError;

    fn try_from(data: Vec<T>) -> Result<Self> {
        Self::from_vec(data)
    }
}

impl<T: Copy> TryFrom<&[T]> for GpuVec<T> {
    type Error = GpuHostError;

    fn try_from(data: &[T]) -> Result<Self> {
        Self::from_vec(data.to_vec())
    }
}

impl<T> Drop for MappedBuffer<T> {
    fn drop(&mut self) {
        // SAFETY: cuda_lib() returns the CUDA driver function table.
        let cu = unsafe { cuda_lib() };
        // SAFETY: host_ptr was allocated by cuMemHostAlloc in new_zeroed() and has
        // not been freed yet (this is the Drop impl, called exactly once).
        let result = unsafe { cu.cuMemFreeHost(self.host_ptr as *mut std::ffi::c_void) };
        if result != sys::CUresult::CUDA_SUCCESS {
            eprintln!(
                "WARNING: cuMemFreeHost failed during MappedBuffer drop: {:?}",
                result
            );
        }
    }
}
