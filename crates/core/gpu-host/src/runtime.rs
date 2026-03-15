//! High-level GPU runtime — wraps CudaDevice with PTX loading and kernel launch.

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, LaunchConfig, ValidAsZeroBits};

use crate::error::{GpuHostError, Result};

/// High-level GPU runtime handle.
///
/// Wraps a CUDA device with convenience methods for PTX loading, memory
/// allocation, kernel launching, and data transfer.
pub struct GpuRuntime {
    dev: Arc<CudaDevice>,
    ordinal: usize,
}

impl GpuRuntime {
    /// Initialize a CUDA device by ordinal (0 = first GPU).
    pub fn new(device_ordinal: usize) -> Result<Self> {
        let dev = CudaDevice::new(device_ordinal).map_err(GpuHostError::CudaInit)?;
        Ok(Self {
            dev,
            ordinal: device_ordinal,
        })
    }

    /// Return the number of available CUDA devices.
    pub fn device_count() -> Result<usize> {
        let count = CudaDevice::count().map_err(GpuHostError::Cudarc)?;
        Ok(count as usize)
    }

    /// Return the device ordinal this runtime is bound to.
    pub fn device_ordinal(&self) -> usize {
        self.ordinal
    }

    /// Return the human-readable name of this device (e.g. "NVIDIA GeForce RTX 4090").
    pub fn device_name(&self) -> Result<String> {
        let cu_dev = cudarc::driver::result::device::get(self.ordinal as i32)
            .map_err(GpuHostError::Cudarc)?;
        cudarc::driver::result::device::get_name(cu_dev).map_err(GpuHostError::Cudarc)
    }

    /// Get the underlying `CudaDevice` for advanced operations.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.dev
    }

    /// Get a clone of the device `Arc` for sharing across threads.
    pub fn device_arc(&self) -> Arc<CudaDevice> {
        Arc::clone(&self.dev)
    }

    /// Load a PTX module with named kernel functions.
    ///
    /// The `ptx_src` is a PTX source string (typically embedded via `include_str!`).
    /// The `module_name` is used to look up functions later.
    /// The `fn_names` list declares which kernel functions to extract.
    pub fn load_ptx(
        &self,
        ptx_src: &str,
        module_name: &str,
        fn_names: &[&'static str],
    ) -> Result<()> {
        let ptx = cudarc::nvrtc::Ptx::from_src(ptx_src);
        self.dev
            .load_ptx(ptx, module_name, fn_names)
            .map_err(GpuHostError::Cudarc)?;
        Ok(())
    }

    /// Get a kernel function handle for manual launch.
    ///
    /// Returns `None` if the function is not found in the loaded module.
    pub fn get_func(&self, module: &str, func: &str) -> Option<cudarc::driver::CudaFunction> {
        self.dev.get_func(module, func)
    }

    /// Get a kernel function handle, returning an error if not found.
    ///
    /// This is the preferred way to look up kernel functions — avoids the
    /// `get_func().ok_or(KernelNotFound(...))` boilerplate.
    pub fn require_func(
        &self,
        module: &str,
        func: &'static str,
    ) -> Result<cudarc::driver::CudaFunction> {
        self.dev
            .get_func(module, func)
            .ok_or(GpuHostError::KernelNotFound(func))
    }

    /// Allocate device memory initialized to zero.
    pub fn alloc_zeros<T: DeviceRepr + ValidAsZeroBits>(&self, len: usize) -> Result<CudaSlice<T>> {
        self.dev.alloc_zeros::<T>(len).map_err(GpuHostError::Cudarc)
    }

    /// Upload data from host to device.
    pub fn htod_sync_copy<T: DeviceRepr + Unpin>(&self, data: &[T]) -> Result<CudaSlice<T>> {
        self.dev.htod_sync_copy(data).map_err(GpuHostError::Cudarc)
    }

    /// Download data from device to host.
    pub fn dtoh_sync_copy<T: DeviceRepr + Unpin>(&self, buf: &CudaSlice<T>) -> Result<Vec<T>> {
        self.dev.dtoh_sync_copy(buf).map_err(GpuHostError::Cudarc)
    }

    /// Create a launch configuration.
    pub fn launch_config(
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem_bytes: u32,
    ) -> LaunchConfig {
        LaunchConfig {
            grid_dim: grid,
            block_dim: block,
            shared_mem_bytes,
        }
    }

    /// Synchronize the device — wait for all pending kernels to complete.
    pub fn synchronize(&self) -> Result<()> {
        self.dev.synchronize().map_err(GpuHostError::Cudarc)
    }
}
