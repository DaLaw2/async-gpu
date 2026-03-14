//! Shared utility functions for pinned, device-mapped host memory allocation.

use std::sync::Arc;

use cudarc::driver::sys::{self, lib as cuda_lib};
use cudarc::driver::CudaDevice;

use crate::error::{GpuHostError, Result};

/// Allocates a single u32 of pinned, device-mapped host memory.
/// Returns (host_ptr, device_ptr).
///
/// # Safety
/// Caller must free the returned host pointer via [`free_mapped_mem`].
/// The CUDA context must be initialized.
pub unsafe fn alloc_mapped_u32(_dev: &Arc<CudaDevice>) -> Result<(*mut u32, sys::CUdeviceptr)> {
    let cu = cuda_lib();

    let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let flags = sys::CU_MEMHOSTALLOC_DEVICEMAP | sys::CU_MEMHOSTALLOC_PORTABLE;
    let result = cu.cuMemHostAlloc(&mut host_ptr, std::mem::size_of::<u32>(), flags);
    if result != sys::CUresult::CUDA_SUCCESS {
        return Err(GpuHostError::CudaAlloc(result));
    }

    let mut dev_ptr: sys::CUdeviceptr = 0;
    let result = cu.cuMemHostGetDevicePointer_v2(&mut dev_ptr, host_ptr, 0);
    if result != sys::CUresult::CUDA_SUCCESS {
        cu.cuMemFreeHost(host_ptr);
        return Err(GpuHostError::CudaGetDevPtr(result));
    }

    Ok((host_ptr as *mut u32, dev_ptr))
}

/// Frees pinned host memory allocated with cuMemHostAlloc.
///
/// # Safety
/// `host_ptr` must have been returned by [`alloc_mapped_u32`] or
/// [`alloc_mapped_result_array`] and not yet freed.
pub unsafe fn free_mapped_mem(host_ptr: *mut u32) -> Result<()> {
    let cu = cuda_lib();
    let result = cu.cuMemFreeHost(host_ptr as *mut std::ffi::c_void);
    if result != sys::CUresult::CUDA_SUCCESS {
        return Err(GpuHostError::CudaFreeMem(result));
    }
    Ok(())
}

/// Allocates an array of u32 values in pinned, device-mapped host memory.
///
/// # Safety
/// Caller must free the returned host pointer via [`free_mapped_mem`].
/// The CUDA context must be initialized.
pub unsafe fn alloc_mapped_result_array(
    _dev: &Arc<CudaDevice>,
    count: usize,
) -> Result<(*mut u32, sys::CUdeviceptr)> {
    let cu = cuda_lib();
    let size = count * std::mem::size_of::<u32>();

    let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let flags = sys::CU_MEMHOSTALLOC_DEVICEMAP | sys::CU_MEMHOSTALLOC_PORTABLE;
    let result = cu.cuMemHostAlloc(&mut host_ptr, size, flags);
    if result != sys::CUresult::CUDA_SUCCESS {
        return Err(GpuHostError::CudaAlloc(result));
    }

    let mut dev_ptr: sys::CUdeviceptr = 0;
    let result = cu.cuMemHostGetDevicePointer_v2(&mut dev_ptr, host_ptr, 0);
    if result != sys::CUresult::CUDA_SUCCESS {
        cu.cuMemFreeHost(host_ptr);
        return Err(GpuHostError::CudaGetDevPtr(result));
    }

    std::ptr::write_bytes(host_ptr as *mut u8, 0, size);
    Ok((host_ptr as *mut u32, dev_ptr))
}

/// Allocate a mapped u64 array for benchmark results.
///
/// # Safety
/// Caller must free the returned host pointer via [`free_mapped_u64_array`].
/// The CUDA context must be initialized.
pub unsafe fn alloc_mapped_u64_array(
    _dev: &Arc<CudaDevice>,
    count: usize,
) -> Result<(*mut u64, sys::CUdeviceptr)> {
    let cu = cuda_lib();
    let size = count * std::mem::size_of::<u64>();

    let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let flags = sys::CU_MEMHOSTALLOC_DEVICEMAP | sys::CU_MEMHOSTALLOC_PORTABLE;
    let result = cu.cuMemHostAlloc(&mut host_ptr, size, flags);
    if result != sys::CUresult::CUDA_SUCCESS {
        return Err(GpuHostError::CudaAlloc(result));
    }

    let mut dev_ptr: sys::CUdeviceptr = 0;
    let result = cu.cuMemHostGetDevicePointer_v2(&mut dev_ptr, host_ptr, 0);
    if result != sys::CUresult::CUDA_SUCCESS {
        cu.cuMemFreeHost(host_ptr);
        return Err(GpuHostError::CudaGetDevPtr(result));
    }

    std::ptr::write_bytes(host_ptr as *mut u8, 0, size);
    Ok((host_ptr as *mut u64, dev_ptr))
}

/// Free a mapped u64 array.
///
/// # Safety
/// `host_ptr` must have been returned by [`alloc_mapped_u64_array`] and not yet freed.
pub unsafe fn free_mapped_u64_array(host_ptr: *mut u64) -> Result<()> {
    let cu = cuda_lib();
    let result = cu.cuMemFreeHost(host_ptr as *mut std::ffi::c_void);
    if result != sys::CUresult::CUDA_SUCCESS {
        return Err(GpuHostError::CudaFreeMem(result));
    }
    Ok(())
}
