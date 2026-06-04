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

/// Load a CUDA module from cubin (fast) or PTX (slow JIT fallback).
///
/// If `cubin` is non-empty, loads the precompiled cubin directly (sub-second).
/// Otherwise, JIT-compiles the PTX string (can take 10+ minutes for large PTX).
///
/// Returns the loaded CUmodule handle.
///
/// # Safety
///
/// The caller must ensure the cubin/PTX contains the expected kernel functions.
unsafe fn load_module_cubin_or_ptx(
    ptx_src: &str,
    cubin: &[u8],
) -> Result<cudarc::driver::sys::CUmodule> {
    use cudarc::driver::sys::{self, lib as cuda_lib};

    let cu = cuda_lib();
    let mut module: sys::CUmodule = std::ptr::null_mut();

    if !cubin.is_empty() {
        // Fast path: load pre-compiled cubin (sub-second)
        let result = cu.cuModuleLoadData(&mut module, cubin.as_ptr() as *const std::ffi::c_void);
        if result == sys::CUresult::CUDA_SUCCESS {
            return Ok(module);
        }
        // Cubin load failed (e.g., architecture mismatch) — fall through to PTX
        eprintln!(
            "cubin load failed ({result:?}), falling back to PTX JIT compilation (this may take several minutes)"
        );
    }

    // Slow path: JIT-compile PTX
    let ptx_cstring = std::ffi::CString::new(ptx_src).map_err(|_| GpuHostError::Verification {
        test: "ptx_load",
        detail: "PTX source contains null byte".to_string(),
    })?;
    let result = cu.cuModuleLoadData(&mut module, ptx_cstring.as_ptr() as *const std::ffi::c_void);
    if result != sys::CUresult::CUDA_SUCCESS {
        return Err(GpuHostError::Verification {
            test: "ptx_load",
            detail: format!("cuModuleLoadData failed: {result:?}"),
        });
    }
    Ok(module)
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

// ============================================================================
// Zero-param kernel launch — hostcall injected via device global
// ============================================================================

/// Launch a zero-parameter kernel with hostcall injected via `__HOSTCALL_BUF` device global.
///
/// Instead of passing the hostcall buffer as a kernel parameter, the host writes
/// the pointer to a device global (`__HOSTCALL_BUF`) in the loaded PTX module
/// via `cuModuleGetGlobal_v2` + `cuMemcpyHtoD`. The kernel reads it at entry
/// via `gpu_runtime::entry::auto_init()`.
///
/// This uses the raw CUDA driver API because cudarc does not expose `CUmodule`
/// handles or `cuModuleGetGlobal_v2`.
///
/// # Arguments
///
/// * `ptx_src` - PTX source string (e.g., `ptx::KERNEL_STD`)
/// * `kernel_name` - Name of the kernel function in the PTX
///
/// # Example
///
/// ```no_run
/// use gpu_host::gpu;
/// use gpu_host::ptx;
///
/// gpu::run_zero_param(ptx::KERNEL_STD, "zero_param_hello").unwrap();
/// ```
pub fn run_zero_param(ptx_src: &str, kernel_name: &'static str) -> Result<()> {
    run_zero_param_with_config(ptx_src, kernel_name, 128, (1, 1, 1))
}

/// Launch a zero-parameter kernel with custom thread count and grid dimensions.
///
/// See [`run_zero_param`] for details.
pub fn run_zero_param_with_config(
    ptx_src: &str,
    kernel_name: &'static str,
    threads_per_block: u32,
    grid_dim: (u32, u32, u32),
) -> Result<()> {
    use cudarc::driver::sys::{self, lib as cuda_lib};
    use std::ffi::CString;

    // Initialize CUDA context via cudarc (this handles cuInit, context creation, etc.)
    let dev = CudaDevice::new(0).map_err(GpuHostError::CudaInit)?;

    let effective_cubin: &[u8] = &[];

    let cu_module: sys::CUmodule = unsafe { load_module_cubin_or_ptx(ptx_src, effective_cubin)? };

    // Get kernel function handle
    let func_name = CString::new(kernel_name).map_err(|_| GpuHostError::Verification {
        test: kernel_name,
        detail: "kernel name contains null byte".to_string(),
    })?;

    let cu_func: sys::CUfunction;
    unsafe {
        let cu = cuda_lib();
        let mut func: sys::CUfunction = std::ptr::null_mut();
        let result = cu.cuModuleGetFunction(&mut func, cu_module, func_name.as_ptr());
        if result != sys::CUresult::CUDA_SUCCESS {
            cu.cuModuleUnload(cu_module);
            return Err(GpuHostError::KernelNotFound(kernel_name));
        }
        cu_func = func;
    }

    // Start hostcall session
    let session = HostcallSession::start(64)?;

    // Inject hostcall pointer via device global
    let global_name = CString::new("__HOSTCALL_BUF").unwrap();
    unsafe {
        let cu = cuda_lib();
        let mut global_dptr: sys::CUdeviceptr = 0;
        let mut global_size: usize = 0;
        let result = cu.cuModuleGetGlobal_v2(
            &mut global_dptr,
            &mut global_size,
            cu_module,
            global_name.as_ptr(),
        );
        if result != sys::CUresult::CUDA_SUCCESS {
            cu.cuModuleUnload(cu_module);
            return Err(GpuHostError::Verification {
                test: kernel_name,
                detail: format!(
                    "cuModuleGetGlobal_v2(__HOSTCALL_BUF) failed: {result:?}. \
                     The kernel PTX must contain the __HOSTCALL_BUF symbol \
                     (add `gpu_runtime::entry` dependency)."
                ),
            });
        }

        // Sanity check: the global should be 8 bytes (AtomicU64)
        if global_size != 8 {
            cu.cuModuleUnload(cu_module);
            return Err(GpuHostError::Verification {
                test: kernel_name,
                detail: format!("__HOSTCALL_BUF has unexpected size {global_size} (expected 8)"),
            });
        }

        // Write the hostcall device pointer value to the global
        let hc_ptr_val: u64 = session.dev_ptr();
        let result = cu.cuMemcpyHtoD_v2(
            global_dptr,
            &hc_ptr_val as *const u64 as *const std::ffi::c_void,
            8,
        );
        if result != sys::CUresult::CUDA_SUCCESS {
            cu.cuModuleUnload(cu_module);
            return Err(GpuHostError::Verification {
                test: kernel_name,
                detail: format!("cuMemcpyHtoD to __HOSTCALL_BUF failed: {result:?}"),
            });
        }
    }

    // Launch kernel with ZERO arguments
    unsafe {
        let cu = cuda_lib();
        // cuLaunchKernel with no arguments (params = null)
        let result = cu.cuLaunchKernel(
            cu_func,
            grid_dim.0,
            grid_dim.1,
            grid_dim.2, // grid
            threads_per_block,
            1,
            1,                    // block
            0,                    // shared mem
            std::ptr::null_mut(), // stream (default)
            std::ptr::null_mut(), // kernel params (none!)
            std::ptr::null_mut(), // extra (none)
        );
        if result != sys::CUresult::CUDA_SUCCESS {
            cu.cuModuleUnload(cu_module);
            return Err(GpuHostError::Verification {
                test: kernel_name,
                detail: format!("cuLaunchKernel failed: {result:?}"),
            });
        }
    }

    // Synchronize
    dev.synchronize().map_err(|e| GpuHostError::Verification {
        test: kernel_name,
        detail: format!("sync: {e}"),
    })?;

    // Clean up
    session.shutdown();
    unsafe {
        let cu = cuda_lib();
        cu.cuModuleUnload(cu_module);
    }

    Ok(())
}

// ============================================================================
// GpuStdModule: load PTX with __HOSTCALL_BUF device global injection
// ============================================================================

/// A loaded GPU module with hostcall buffer injected via `__HOSTCALL_BUF` device global.
///
/// Unlike `run_zero_param` which only supports zero-argument kernels, this
/// provides the raw `CUmodule` and `CUfunction` handles so callers can launch
/// kernels that still have data parameters (e.g., `result: *mut u32`) while
/// reading the hostcall buffer from the device global.
///
/// # Example
///
/// ```no_run
/// use gpu_host::gpu::GpuStdModule;
/// use gpu_host::ptx;
///
/// let module = GpuStdModule::load(ptx::KERNEL_STD, "my_kernel", 128, (1,1,1)).unwrap();
/// // Launch with kernel-specific params via module.launch_raw(...)
/// module.finish();
/// ```
pub struct GpuStdModule {
    dev: std::sync::Arc<CudaDevice>,
    cu_module: cudarc::driver::sys::CUmodule,
    cu_func: cudarc::driver::sys::CUfunction,
    session: HostcallSession,
    threads_per_block: u32,
    grid_dim: (u32, u32, u32),
}

impl GpuStdModule {
    /// Load PTX, inject `__HOSTCALL_BUF` device global, and prepare for launch.
    pub fn load(
        ptx_src: &str,
        kernel_name: &'static str,
        threads_per_block: u32,
        grid_dim: (u32, u32, u32),
    ) -> Result<Self> {
        Self::load_with_print(ptx_src, kernel_name, threads_per_block, grid_dim, None)
    }

    /// Load PTX with a custom print callback for capturing GPU println! output.
    ///
    /// If `ptx_src` matches the unified kernel PTX, automatically tries the
    /// pre-compiled cubin first for fast loading (sub-second vs 10+ minutes).
    #[allow(clippy::type_complexity)]
    pub fn load_with_print(
        ptx_src: &str,
        kernel_name: &'static str,
        threads_per_block: u32,
        grid_dim: (u32, u32, u32),
        print_cb: Option<Box<dyn Fn(&[u8]) + Send + 'static>>,
    ) -> Result<Self> {
        Self::load_with_cubin(
            ptx_src,
            &[],
            kernel_name,
            threads_per_block,
            grid_dim,
            print_cb,
        )
    }

    /// Load PTX or cubin with optional print callback.
    ///
    /// If `cubin` is non-empty, loads the pre-compiled binary directly.
    /// Otherwise falls back to JIT-compiling the PTX source.
    #[allow(clippy::type_complexity)]
    pub fn load_with_cubin(
        ptx_src: &str,
        cubin: &[u8],
        kernel_name: &'static str,
        threads_per_block: u32,
        grid_dim: (u32, u32, u32),
        print_cb: Option<Box<dyn Fn(&[u8]) + Send + 'static>>,
    ) -> Result<Self> {
        use cudarc::driver::sys::{self, lib as cuda_lib};
        use std::ffi::CString;

        let dev = CudaDevice::new(0).map_err(GpuHostError::CudaInit)?;

        let effective_cubin = if !cubin.is_empty() { cubin } else { &[] };

        let cu_module: sys::CUmodule =
            unsafe { load_module_cubin_or_ptx(ptx_src, effective_cubin)? };

        let func_name = CString::new(kernel_name).map_err(|_| GpuHostError::Verification {
            test: kernel_name,
            detail: "kernel name contains null byte".to_string(),
        })?;

        let cu_func: sys::CUfunction;
        unsafe {
            let cu = cuda_lib();
            let mut func: sys::CUfunction = std::ptr::null_mut();
            let result = cu.cuModuleGetFunction(&mut func, cu_module, func_name.as_ptr());
            if result != sys::CUresult::CUDA_SUCCESS {
                cu.cuModuleUnload(cu_module);
                return Err(GpuHostError::KernelNotFound(kernel_name));
            }
            cu_func = func;
        }

        let session = match print_cb {
            Some(cb) => HostcallSession::start_with_print(64, cb)?,
            None => HostcallSession::start(64)?,
        };

        // Inject hostcall pointer via device global
        let global_name = CString::new("__HOSTCALL_BUF").unwrap();
        unsafe {
            let cu = cuda_lib();
            let mut global_dptr: sys::CUdeviceptr = 0;
            let mut global_size: usize = 0;
            let result = cu.cuModuleGetGlobal_v2(
                &mut global_dptr,
                &mut global_size,
                cu_module,
                global_name.as_ptr(),
            );
            if result != sys::CUresult::CUDA_SUCCESS {
                cu.cuModuleUnload(cu_module);
                return Err(GpuHostError::Verification {
                    test: kernel_name,
                    detail: format!("cuModuleGetGlobal_v2(__HOSTCALL_BUF) failed: {result:?}"),
                });
            }

            let hc_ptr_val: u64 = session.dev_ptr();
            let result = cu.cuMemcpyHtoD_v2(
                global_dptr,
                &hc_ptr_val as *const u64 as *const std::ffi::c_void,
                8,
            );
            if result != sys::CUresult::CUDA_SUCCESS {
                cu.cuModuleUnload(cu_module);
                return Err(GpuHostError::Verification {
                    test: kernel_name,
                    detail: format!("cuMemcpyHtoD to __HOSTCALL_BUF failed: {result:?}"),
                });
            }
        }

        Ok(Self {
            dev,
            cu_module,
            cu_func,
            session,
            threads_per_block,
            grid_dim,
        })
    }

    /// Launch the kernel with raw kernel parameter pointers.
    ///
    /// `params` is an array of pointers to each kernel argument, in order.
    /// For a zero-param kernel, pass an empty slice.
    ///
    /// # Safety
    ///
    /// The param pointers must match the kernel's parameter signature.
    pub unsafe fn launch_raw(&self, params: &[*mut std::ffi::c_void]) -> Result<()> {
        use cudarc::driver::sys::{self, lib as cuda_lib};

        let cu = cuda_lib();
        let params_ptr = if params.is_empty() {
            std::ptr::null_mut()
        } else {
            params.as_ptr() as *mut *mut std::ffi::c_void
        };

        let result = cu.cuLaunchKernel(
            self.cu_func,
            self.grid_dim.0,
            self.grid_dim.1,
            self.grid_dim.2,
            self.threads_per_block,
            1,
            1,
            0,
            std::ptr::null_mut(),
            params_ptr,
            std::ptr::null_mut(),
        );
        if result != sys::CUresult::CUDA_SUCCESS {
            return Err(GpuHostError::Verification {
                test: "launch",
                detail: format!("cuLaunchKernel failed: {result:?}"),
            });
        }

        self.dev
            .synchronize()
            .map_err(|e| GpuHostError::Verification {
                test: "sync",
                detail: format!("{e}"),
            })?;

        Ok(())
    }

    /// Get the underlying CUDA device for memory allocation.
    pub fn device(&self) -> &std::sync::Arc<CudaDevice> {
        &self.dev
    }

    /// Get the hostcall session's device pointer.
    pub fn hostcall_ptr(&self) -> u64 {
        self.session.dev_ptr()
    }

    /// Shut down the hostcall session and unload the module.
    pub fn finish(self) {
        self.session.shutdown();
        unsafe {
            let cu = cudarc::driver::sys::lib();
            cu.cuModuleUnload(self.cu_module);
        }
    }
}

// ============================================================================
// Builder API: gpu::custom("kernel") → CustomLaunchBuilder → GpuContext → GpuResult
// ============================================================================

use crate::memory::MappedBuffer;
use cudarc::driver::{DeviceRepr, ValidAsZeroBits};

/// Create a builder for launching a custom-signature kernel.
///
/// Returns a [`CustomLaunchBuilder`] that configures launch parameters,
/// then call `.prepare()` to get a [`GpuContext`] for uploading data and
/// launching the kernel.
///
/// # Example
///
/// ```no_run
/// use gpu_host::gpu;
///
/// let ctx = gpu::custom("my_kernel")
///     .ptx(include_str!("kernel.ptx"))
///     .threads(256)
///     .elements(1024)
///     .prepare()
///     .unwrap();
///
/// let input = ctx.upload(&[1.0f32; 1024]).unwrap();
/// let mut output = ctx.alloc_zeros::<f32>(1024).unwrap();
/// let result = unsafe { ctx.launch((&input, &mut output, 1024u32)).unwrap() };
/// let data = result.download(&output).unwrap();
/// ```
pub fn custom(kernel_name: &'static str) -> CustomLaunchBuilder {
    CustomLaunchBuilder {
        kernel_name,
        ptx_src: None,
        threads: 128,
        grid: (1, 1, 1),
        shared_mem: 0,
        hostcall: false,
        hostcall_packets: 64,
    }
}

/// Builder for custom-signature kernel launches.
///
/// Configures launch parameters (threads, grid, hostcall, PTX source),
/// then call `.prepare()` to initialize the GPU context.
pub struct CustomLaunchBuilder {
    kernel_name: &'static str,
    ptx_src: Option<&'static str>,
    threads: u32,
    grid: (u32, u32, u32),
    shared_mem: u32,
    hostcall: bool,
    hostcall_packets: u16,
}

impl CustomLaunchBuilder {
    /// Use a custom PTX source instead of the embedded `kernel.ptx`.
    ///
    /// Required for examples that compile their own kernels via build scripts.
    pub fn ptx(mut self, src: &'static str) -> Self {
        self.ptx_src = Some(src);
        self
    }

    /// Set threads per block (default: 128).
    pub fn threads(mut self, n: u32) -> Self {
        self.threads = n;
        self
    }

    /// Set grid dimensions (default: `(1,1,1)`).
    pub fn grid(mut self, dim: (u32, u32, u32)) -> Self {
        self.grid = dim;
        self
    }

    /// Set 1D grid to cover `n` elements with the current thread count.
    ///
    /// Equivalent to `.grid((n.div_ceil(threads), 1, 1))`.
    pub fn elements(mut self, n: u32) -> Self {
        self.grid = (n.div_ceil(self.threads), 1, 1);
        self
    }

    /// Set shared memory bytes (default: 0).
    pub fn shared_mem(mut self, bytes: u32) -> Self {
        self.shared_mem = bytes;
        self
    }

    /// Enable hostcall support (spawns a [`HostcallSession`]).
    pub fn hostcall(mut self) -> Self {
        self.hostcall = true;
        self
    }

    /// Set hostcall packet count (default: 64). Implies `.hostcall()`.
    pub fn hostcall_packets(mut self, n: u16) -> Self {
        self.hostcall = true;
        self.hostcall_packets = n;
        self
    }

    /// Prepare the launch context.
    ///
    /// Initializes the CUDA device, loads PTX, and optionally starts a
    /// hostcall session. Returns a [`GpuContext`] for uploading data and
    /// launching the kernel.
    pub fn prepare(self) -> Result<GpuContext> {
        let dev = CudaDevice::new(0).map_err(GpuHostError::CudaInit)?;

        let ptx_src = self.ptx_src.unwrap_or(crate::ptx::KERNEL);
        let module = fresh_module_name();

        let ptx = cudarc::nvrtc::Ptx::from_src(ptx_src);
        dev.load_ptx(ptx, &module, &[self.kernel_name])
            .map_err(|e| GpuHostError::Verification {
                test: "ptx_load",
                detail: format!("{e}"),
            })?;

        let func = dev
            .get_func(&module, self.kernel_name)
            .ok_or(GpuHostError::KernelNotFound(self.kernel_name))?;

        let session = if self.hostcall {
            Some(HostcallSession::start(self.hostcall_packets)?)
        } else {
            None
        };

        let config = LaunchConfig {
            grid_dim: self.grid,
            block_dim: (self.threads, 1, 1),
            shared_mem_bytes: self.shared_mem,
        };

        Ok(GpuContext {
            dev,
            func,
            config,
            session,
            kernel_name: self.kernel_name,
        })
    }
}

/// A prepared GPU context with device, function, and optional hostcall session.
///
/// Provides methods to upload data and launch kernels with arbitrary arguments.
/// Created by [`CustomLaunchBuilder::prepare()`].
pub struct GpuContext {
    dev: std::sync::Arc<CudaDevice>,
    func: cudarc::driver::CudaFunction,
    config: LaunchConfig,
    session: Option<HostcallSession>,
    kernel_name: &'static str,
}

impl GpuContext {
    /// Upload a slice to device memory (host-to-device copy).
    pub fn upload<T: DeviceRepr + Unpin>(&self, data: &[T]) -> Result<CudaSlice<T>> {
        self.dev.htod_sync_copy(data).map_err(GpuHostError::Cudarc)
    }

    /// Allocate zeroed device memory.
    pub fn alloc_zeros<T: DeviceRepr + ValidAsZeroBits>(&self, n: usize) -> Result<CudaSlice<T>> {
        self.dev.alloc_zeros::<T>(n).map_err(GpuHostError::Cudarc)
    }

    /// Allocate a mapped buffer (pinned host+device memory, GPU-visible).
    pub fn mapped_buffer<T>(&self, n: usize) -> Result<MappedBuffer<T>> {
        MappedBuffer::<T>::new_zeroed(n)
    }

    /// Get the hostcall device pointer as `u64`.
    ///
    /// # Panics
    ///
    /// Panics if hostcall was not enabled on the builder.
    pub fn hostcall_ptr(&self) -> u64 {
        self.session
            .as_ref()
            .expect("hostcall not enabled — call .hostcall() on the builder")
            .dev_ptr()
    }

    /// Get the sideband device pointer as `u64`.
    ///
    /// # Panics
    ///
    /// Panics if hostcall was not enabled on the builder.
    pub fn sideband_ptr(&self) -> u64 {
        self.session
            .as_ref()
            .expect("hostcall not enabled — call .hostcall() on the builder")
            .sideband_dev_ptr()
    }

    /// Download device memory to host.
    ///
    /// Can be called before launch (e.g., to verify uploaded data).
    /// For post-launch downloads, use [`GpuResult::download()`].
    pub fn download<T: DeviceRepr + Unpin + Clone>(&self, buf: &CudaSlice<T>) -> Result<Vec<T>> {
        self.dev.dtoh_sync_copy(buf).map_err(GpuHostError::Cudarc)
    }

    /// Launch the kernel with the given argument tuple, synchronize,
    /// and return a [`GpuResult`] handle for downloading output data.
    ///
    /// # Safety
    ///
    /// The `args` tuple must match the kernel's parameter signature.
    /// This is the same tuple type that cudarc's `LaunchAsync` accepts.
    pub unsafe fn launch<P>(self, args: P) -> Result<GpuResult>
    where
        cudarc::driver::CudaFunction: LaunchAsync<P>,
    {
        self.func
            .launch(self.config, args)
            .map_err(|e| GpuHostError::Verification {
                test: self.kernel_name,
                detail: format!("launch: {e}"),
            })?;

        self.dev
            .synchronize()
            .map_err(|e| GpuHostError::Verification {
                test: self.kernel_name,
                detail: format!("sync: {e}"),
            })?;

        Ok(GpuResult {
            dev: self.dev,
            _session: self.session,
        })
    }
}

/// Handle returned after a successful kernel launch + synchronize.
///
/// Use [`download()`](GpuResult::download) to copy device buffers back to host,
/// then drop or call [`finish()`](GpuResult::finish) to shut down the hostcall
/// session (if any).
pub struct GpuResult {
    dev: std::sync::Arc<CudaDevice>,
    /// Held for its `Drop` impl which shuts down the hostcall listener.
    _session: Option<HostcallSession>,
}

impl GpuResult {
    /// Download device memory to host.
    pub fn download<T: DeviceRepr + Unpin + Clone>(&self, buf: &CudaSlice<T>) -> Result<Vec<T>> {
        self.dev.dtoh_sync_copy(buf).map_err(GpuHostError::Cudarc)
    }

    /// Explicitly shut down the hostcall session.
    ///
    /// Called automatically on drop, but an explicit call may be useful
    /// for ordering guarantees.
    pub fn finish(self) {
        // session dropped → HostcallSession::drop() handles shutdown
    }
}
