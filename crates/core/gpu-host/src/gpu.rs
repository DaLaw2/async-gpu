//! One-liner GPU launch API — `gpu::run()`.
//!
//! Hides all CUDA boilerplate (device init, PTX loading, hostcall setup,
//! launch config, synchronization). Users call `gpu::run("kernel_name")`
//! and the kernel executes with full hostcall support.
//!
//! # Example
//!
//! ```no_run
//! use gpu_host::gpu;
//!
//! // Launch a thread-based kernel (uses thread::spawn internally)
//! gpu::run("thread_spawn_test").unwrap();
//!
//! // Launch with explicit output buffer
//! let results: Vec<u32> = gpu::run_with_output("thread_spawn_test", 4).unwrap();
//! ```

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};

use crate::error::{GpuHostError, Result};
use crate::hostcall::HostcallSession;

/// Launch a GPU kernel by name. Handles all setup automatically:
/// - CUDA device initialization
/// - PTX module loading (from embedded kernel.ptx)
/// - Hostcall session (for println!, file I/O, etc.)
/// - Launch with 4 warps (128 threads) for thread::spawn support
/// - Synchronization
///
/// The kernel receives a hostcall buffer pointer as its first argument.
pub fn run(kernel_name: &'static str) -> Result<()> {
    let dev = CudaDevice::new(0).map_err(GpuHostError::CudaInit)?;
    let ptx = cudarc::nvrtc::Ptx::from_src(crate::ptx::KERNEL);
    dev.load_ptx(ptx, "gpu_run", &[kernel_name])
        .map_err(|e| GpuHostError::Verification {
            test: "ptx_load",
            detail: format!("{e}"),
        })?;

    let func = dev
        .get_func("gpu_run", kernel_name)
        .ok_or(GpuHostError::KernelNotFound(kernel_name))?;

    let session = HostcallSession::start(64)?;

    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (128, 1, 1), // 4 warps for thread::spawn support
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

/// Launch a GPU kernel that writes results to an output buffer.
///
/// The kernel receives `(hostcall_buf: *mut u8, output: *mut T)` as arguments.
/// Returns the output buffer as a Vec<T> after kernel completion.
pub fn run_with_output<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Clone>(
    kernel_name: &'static str,
    n_elements: usize,
) -> Result<Vec<T>> {
    let dev = CudaDevice::new(0).map_err(GpuHostError::CudaInit)?;
    let ptx = cudarc::nvrtc::Ptx::from_src(crate::ptx::KERNEL);
    dev.load_ptx(ptx, "gpu_run", &[kernel_name])
        .map_err(|e| GpuHostError::Verification {
            test: "ptx_load",
            detail: format!("{e}"),
        })?;

    let func = dev
        .get_func("gpu_run", kernel_name)
        .ok_or(GpuHostError::KernelNotFound(kernel_name))?;

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
/// The kernel receives `(output: *mut T)` as its only argument.
/// Uses the specified number of threads per block.
pub fn compute<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Clone>(
    kernel_name: &'static str,
    n_elements: usize,
    threads_per_block: u32,
) -> Result<Vec<T>> {
    let dev = CudaDevice::new(0).map_err(GpuHostError::CudaInit)?;
    let ptx = cudarc::nvrtc::Ptx::from_src(crate::ptx::KERNEL);
    dev.load_ptx(ptx, "gpu_run", &[kernel_name])
        .map_err(|e| GpuHostError::Verification {
            test: "ptx_load",
            detail: format!("{e}"),
        })?;

    let func = dev
        .get_func("gpu_run", kernel_name)
        .ok_or(GpuHostError::KernelNotFound(kernel_name))?;

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
