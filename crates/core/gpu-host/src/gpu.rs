//! One-liner GPU launch API.
//!
//! Each call creates a fresh CUDA module to avoid stale global state
//! between kernel launches (GPU statics persist within a module).
//!
//! # API
//!
//! - `gpu::run("kernel")` — hostcall-enabled kernel (println!, file I/O)
//! - `gpu::run_with_output("kernel", n)` — hostcall + output buffer
//! - `gpu::launch("kernel", n, threads)` — pure compute, output only

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};

use crate::error::{GpuHostError, Result};
use crate::hostcall::HostcallSession;

/// Unique module counter to avoid shared globals across launches.
static MODULE_SEQ: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

fn fresh_module_name() -> String {
    let seq = MODULE_SEQ.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    format!("gpu_{seq}")
}

fn get_kernel(
    dev: &std::sync::Arc<CudaDevice>,
    kernel_name: &'static str,
) -> Result<cudarc::driver::CudaFunction> {
    let ptx = cudarc::nvrtc::Ptx::from_src(crate::ptx::KERNEL);
    let module = fresh_module_name();
    dev.load_ptx(ptx, &module, &[kernel_name])
        .map_err(|e| GpuHostError::Verification {
            test: "ptx_load",
            detail: format!("{e}"),
        })?;
    dev.get_func(&module, kernel_name)
        .ok_or(GpuHostError::KernelNotFound(kernel_name))
}

/// Launch a hostcall-enabled kernel (supports println!, file I/O).
///
/// Uses 4 warps (128 threads) for thread::spawn support.
pub fn run(kernel_name: &'static str) -> Result<()> {
    let dev = CudaDevice::new(0).map_err(GpuHostError::CudaInit)?;
    let func = get_kernel(&dev, kernel_name)?;
    let session = HostcallSession::start(64)?;

    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        func.launch(config, (session.dev_ptr(),))
            .map_err(|e| GpuHostError::Verification {
                test: kernel_name,
                detail: format!("launch: {e}"),
            })?;
    }

    dev.synchronize().map_err(|e| GpuHostError::Verification {
        test: kernel_name,
        detail: format!("sync: {e}"),
    })?;

    session.shutdown();
    Ok(())
}

/// Launch a hostcall-enabled kernel that writes to an output buffer.
///
/// Kernel signature: `fn(hostcall_buf: *mut u8, output: *mut T)`
pub fn run_with_output<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Clone>(
    kernel_name: &'static str,
    n_elements: usize,
) -> Result<Vec<T>> {
    let dev = CudaDevice::new(0).map_err(GpuHostError::CudaInit)?;
    let func = get_kernel(&dev, kernel_name)?;
    let session = HostcallSession::start(64)?;
    let mut output: CudaSlice<T> =
        dev.alloc_zeros::<T>(n_elements)
            .map_err(|e| GpuHostError::Verification {
                test: kernel_name,
                detail: format!("alloc: {e}"),
            })?;

    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        func.launch(config, (session.dev_ptr(), &mut output))
            .map_err(|e| GpuHostError::Verification {
                test: kernel_name,
                detail: format!("launch: {e}"),
            })?;
    }

    dev.synchronize().map_err(|e| GpuHostError::Verification {
        test: kernel_name,
        detail: format!("sync: {e}"),
    })?;

    let result = dev
        .dtoh_sync_copy(&output)
        .map_err(|e| GpuHostError::Verification {
            test: kernel_name,
            detail: format!("dtoh: {e}"),
        })?;

    session.shutdown();
    Ok(result)
}

/// Launch a pure compute kernel (no hostcall) with an output buffer.
///
/// Kernel signature: `fn(output: *mut T)`
pub fn launch<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Clone>(
    kernel_name: &'static str,
    n_elements: usize,
    threads_per_block: u32,
) -> Result<Vec<T>> {
    let dev = CudaDevice::new(0).map_err(GpuHostError::CudaInit)?;
    let func = get_kernel(&dev, kernel_name)?;
    let mut output: CudaSlice<T> =
        dev.alloc_zeros::<T>(n_elements)
            .map_err(|e| GpuHostError::Verification {
                test: kernel_name,
                detail: format!("alloc: {e}"),
            })?;

    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        func.launch(config, (&mut output,))
            .map_err(|e| GpuHostError::Verification {
                test: kernel_name,
                detail: format!("launch: {e}"),
            })?;
    }

    dev.synchronize().map_err(|e| GpuHostError::Verification {
        test: kernel_name,
        detail: format!("sync: {e}"),
    })?;

    dev.dtoh_sync_copy(&output)
        .map_err(|e| GpuHostError::Verification {
            test: kernel_name,
            detail: format!("dtoh: {e}"),
        })
}

/// Backwards-compatible alias for `launch`.
#[deprecated(note = "use gpu::launch() instead")]
pub fn compute<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Clone>(
    kernel_name: &'static str,
    n_elements: usize,
    threads_per_block: u32,
) -> Result<Vec<T>> {
    launch(kernel_name, n_elements, threads_per_block)
}
