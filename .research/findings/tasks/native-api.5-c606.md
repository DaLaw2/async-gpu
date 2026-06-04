# native-api.5: Design — gpu::run_custom() builder API for multi-argument kernels

## Summary
Designed a builder API (`gpu::custom()`) that handles arbitrary kernel signatures through a fluent interface. The API hides CudaDevice init, PTX loading, LaunchConfig, unsafe kernel launch, and synchronization behind type-safe builder methods. Four unconverted examples were assessed for migration feasibility — all four can use the builder API without kernel-side changes.

## API Design

### Core Types

```rust
// -- crates/core/gpu-host/src/gpu.rs --

use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig, ValidAsZeroBits};
use crate::error::{GpuHostError, Result};
use crate::hostcall::HostcallSession;
use crate::memory::MappedBuffer;

/// Entry point: `gpu::custom("kernel_name")` returns a builder.
pub fn custom(kernel_name: &'static str) -> CustomLaunchBuilder {
    CustomLaunchBuilder {
        kernel_name,
        ptx_src: None,        // None => use embedded kernel.ptx
        module_name: None,    // auto-generated if None
        threads: 128,
        grid: (1, 1, 1),
        shared_mem: 0,
        hostcall: false,
        hostcall_packets: 64,
    }
}

/// Builder for custom-signature kernel launches.
///
/// Configures the launch parameters, then call `.args(...)` to bind
/// kernel arguments and get back a runnable launcher.
pub struct CustomLaunchBuilder {
    kernel_name: &'static str,
    ptx_src: Option<&'static str>,
    module_name: Option<&'static str>,
    threads: u32,
    grid: (u32, u32, u32),
    shared_mem: u32,
    hostcall: bool,
    hostcall_packets: u16,
}
```

### Builder Methods

```rust
impl CustomLaunchBuilder {
    /// Use a custom PTX source instead of the embedded kernel.ptx.
    /// Required for examples that compile their own kernels.
    pub fn ptx(mut self, src: &'static str) -> Self {
        self.ptx_src = Some(src);
        self
    }

    /// Set the module name for PTX loading (default: auto-generated unique name).
    pub fn module(mut self, name: &'static str) -> Self {
        self.module_name = Some(name);
        self
    }

    /// Set threads per block (default: 128).
    pub fn threads(mut self, n: u32) -> Self {
        self.threads = n;
        self
    }

    /// Set grid dimensions (default: (1,1,1)).
    pub fn grid(mut self, dim: (u32, u32, u32)) -> Self {
        self.grid = dim;
        self
    }

    /// Convenience: set 1D grid to cover `n` elements with current thread count.
    pub fn elements(mut self, n: u32) -> Self {
        self.grid = (n.div_ceil(self.threads), 1, 1);
        self
    }

    /// Set shared memory bytes (default: 0).
    pub fn shared_mem(mut self, bytes: u32) -> Self {
        self.shared_mem = bytes;
        self
    }

    /// Enable hostcall support (spawns HostcallSession).
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

    /// Prepare the launch context. Returns a `GpuContext` that can
    /// upload data and build the kernel argument tuple.
    pub fn prepare(self) -> Result<GpuContext> {
        let dev = CudaDevice::new(0).map_err(GpuHostError::CudaInit)?;

        let ptx_src = self.ptx_src.unwrap_or(crate::ptx::KERNEL);
        let module = self.module_name
            .map(|s| s.to_string())
            .unwrap_or_else(fresh_module_name);

        let ptx = cudarc::nvrtc::Ptx::from_src(ptx_src);
        dev.load_ptx(ptx, &module, &[self.kernel_name])
            .map_err(|e| GpuHostError::Verification {
                test: "ptx_load",
                detail: format!("{e}"),
            })?;

        let func = dev.get_func(&module, self.kernel_name)
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
```

### GpuContext — The Runtime Handle

```rust
/// A prepared GPU context with device, function, and optional hostcall session.
/// Provides methods to upload data and launch kernels with arbitrary arguments.
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
    pub fn alloc_zeros<T: DeviceRepr + ValidAsZeroBits>(
        &self,
        n: usize,
    ) -> Result<CudaSlice<T>> {
        self.dev.alloc_zeros::<T>(n).map_err(GpuHostError::Cudarc)
    }

    /// Allocate a mapped buffer (pinned host+device memory, GPU-visible).
    pub fn mapped_buffer<T>(&self, n: usize) -> Result<MappedBuffer<T>> {
        MappedBuffer::<T>::new_zeroed(n)
    }

    /// Get the hostcall device pointer (panics if hostcall not enabled).
    pub fn hostcall_ptr(&self) -> u64 {
        self.session.as_ref()
            .expect("hostcall not enabled — call .hostcall() on the builder")
            .dev_ptr()
    }

    /// Get the sideband device pointer (panics if hostcall not enabled).
    pub fn sideband_ptr(&self) -> u64 {
        self.session.as_ref()
            .expect("hostcall not enabled — call .hostcall() on the builder")
            .sideband_dev_ptr()
    }

    /// Launch the kernel with the given argument tuple, synchronize,
    /// and shut down the hostcall session (if any).
    ///
    /// The `args` tuple must match the kernel's parameter signature.
    /// This is the same tuple type that cudarc's LaunchAsync accepts.
    pub unsafe fn launch<P>(self, args: P) -> Result<()>
    where
        cudarc::driver::CudaFunction: LaunchAsync<P>,
    {
        self.func.launch(self.config, args)
            .map_err(|e| GpuHostError::Verification {
                test: self.kernel_name,
                detail: format!("launch: {e}"),
            })?;

        self.dev.synchronize().map_err(|e| GpuHostError::Verification {
            test: self.kernel_name,
            detail: format!("sync: {e}"),
        })?;

        if let Some(session) = self.session {
            session.shutdown();
        }

        Ok(())
    }

    /// Download device memory to host after kernel completion.
    /// Call this on slices AFTER launch() — but you'll need a split design
    /// (see launch_ref below) since launch() consumes self.
    pub fn download<T: DeviceRepr + Unpin>(
        &self,
        buf: &CudaSlice<T>,
    ) -> Result<Vec<T>> {
        self.dev.dtoh_sync_copy(buf).map_err(GpuHostError::Cudarc)
    }

    /// Launch the kernel, synchronize, then return self for post-launch
    /// data download. Caller is responsible for calling .finish() after
    /// downloading results.
    pub unsafe fn launch_ref<P>(&mut self, args: P) -> Result<()>
    where
        cudarc::driver::CudaFunction: LaunchAsync<P>,
    {
        // Need to clone func since launch() consumes it
        // Actually CudaFunction is Clone (it's just a handle + Arc<CudaDevice>)
        let func = self.dev.get_func(/* ... */);
        // ... implementation detail — see alternative design below
        todo!()
    }

    /// Shut down the hostcall session (if any). Call after downloading results.
    pub fn finish(self) {
        if let Some(session) = self.session {
            session.shutdown();
        }
    }
}
```

### Refined Design: Split launch/download with `GpuLaunch`

The core tension: `launch()` needs the `CudaFunction` (consumed by cudarc), but `download()` needs the `CudaDevice` reference. Solution: split into two phases.

```rust
/// Phase 1: Build args and launch.
/// Phase 2: Download results and finish.
impl GpuContext {
    /// Launch the kernel with arbitrary args. Returns a `GpuResult` handle
    /// for downloading output data.
    pub unsafe fn launch<P>(self, args: P) -> Result<GpuResult>
    where
        cudarc::driver::CudaFunction: LaunchAsync<P>,
    {
        self.func.launch(self.config, args)
            .map_err(|e| GpuHostError::Verification {
                test: self.kernel_name,
                detail: format!("launch: {e}"),
            })?;

        self.dev.synchronize().map_err(|e| GpuHostError::Verification {
            test: self.kernel_name,
            detail: format!("sync: {e}"),
        })?;

        Ok(GpuResult {
            dev: self.dev,
            session: self.session,
        })
    }
}

/// Handle returned after kernel launch + sync.
/// Use to download results, then drop to clean up.
pub struct GpuResult {
    dev: std::sync::Arc<CudaDevice>,
    session: Option<HostcallSession>,
}

impl GpuResult {
    /// Download device memory to host.
    pub fn download<T: DeviceRepr + Unpin>(
        &self,
        buf: &CudaSlice<T>,
    ) -> Result<Vec<T>> {
        self.dev.dtoh_sync_copy(buf).map_err(GpuHostError::Cudarc)
    }

    /// Shut down the hostcall session. Called automatically on drop,
    /// but explicit call allows error handling.
    pub fn finish(self) {
        // session dropped → shutdown
    }
}

impl Drop for GpuResult {
    fn drop(&mut self) {
        // HostcallSession's Drop handles shutdown
    }
}
```

## Example Rewrites

### SAXPY (vector-math) — Before: ~60 lines, After: ~20 lines

```rust
use gpu_host::gpu;

const KERNEL_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernel.ptx"));

fn main() -> gpu_host::Result<()> {
    const N: usize = 1024;
    let x: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let y_orig: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();
    let a = 2.0f32;

    // Prepare GPU context with custom PTX
    let ctx = gpu::custom("saxpy")
        .ptx(KERNEL_PTX)
        .threads(256)
        .elements(N as u32)
        .prepare()?;

    // Upload data
    let x_dev = ctx.upload(&x)?;
    let mut y_dev = ctx.upload(&y_orig)?;

    // Launch kernel: saxpy(x: *const f32, y: *mut f32, a: f32, n: u32)
    let result = unsafe {
        ctx.launch((&x_dev, &mut y_dev, a, N as u32))?
    };

    // Download result
    let y_result = result.download(&y_dev)?;

    let ok = (0..N).all(|i| (y_result[i] - (a * x[i] + y_orig[i])).abs() < 0.001);
    println!("SAXPY: {}", if ok { "PASSED" } else { "FAILED" });
    Ok(())
}
```

### TCP Echo — Before: ~100 lines, After: ~35 lines

```rust
use gpu_host::gpu;

const KERNEL_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernel.ptx"));

fn main() -> gpu_host::Result<()> {
    // ... TCP server setup (same as before, app-specific) ...
    let port: u16 = /* ... */;

    let ctx = gpu::custom("tcp_echo_kernel")
        .ptx(KERNEL_PTX)
        .threads(32)
        .hostcall_packets(8)
        .prepare()?;

    let mut output_buf = ctx.mapped_buffer::<u32>(1)?;

    // Launch: tcp_echo_kernel(buf, sideband, port, output)
    let result = unsafe {
        ctx.launch((
            ctx.hostcall_ptr(),
            ctx.sideband_ptr(),
            port as u32,
            output_buf.dev_ptr() as u64,
        ))?
    };

    let response_len = unsafe { output_buf.read(0) };
    println!("Response length: {response_len}");
    Ok(())
}
```

**Wait — there's a borrow issue here.** `ctx.launch()` consumes `ctx`, but `ctx.hostcall_ptr()` borrows it. The tuple construction borrows `ctx` to get the pointers, then `launch()` tries to move `ctx`. This won't compile.

### Revised API: Extract pointers before launch

```rust
impl GpuContext {
    /// Get the hostcall device pointer value (u64).
    /// Safe to call before launch — the value is just a number.
    pub fn hostcall_ptr(&self) -> u64 { /* ... */ }
    pub fn sideband_ptr(&self) -> u64 { /* ... */ }
}

// Usage: extract values first, then launch
let hc_ptr = ctx.hostcall_ptr();
let sb_ptr = ctx.sideband_ptr();
let out_ptr = output_buf.dev_ptr() as u64;

let result = unsafe {
    ctx.launch((hc_ptr, sb_ptr, port as u32, out_ptr))?
};
```

This works because the pointer values (u64) are Copy. The borrow of `ctx` ends before `launch()` consumes it.

### Parallel Search — Before: ~90 lines, After: ~40 lines

```rust
use gpu_host::gpu;

const KERNEL_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernel.ptx"));

fn main() -> gpu_host::Result<()> {
    let pattern = b"GPU";
    // ... create input file ...

    let ctx = gpu::custom("parallel_search")
        .ptx(KERNEL_PTX)
        .threads(32)          // full warp
        .hostcall_packets(8)
        .prepare()?;

    let result_buf = ctx.mapped_buffer::<u32>(1)?;
    let mut pattern_buf = ctx.mapped_buffer::<u8>(pattern.len())?;
    for (i, &b) in pattern.iter().enumerate() {
        unsafe { pattern_buf.write(i, b) };
    }
    let data_buf = ctx.mapped_buffer::<u8>(4096)?;

    let hc = ctx.hostcall_ptr();
    let sb = ctx.sideband_ptr();

    let _done = unsafe {
        ctx.launch((
            hc, sb,
            pattern_buf.dev_ptr() as u64,
            pattern.len() as u32,
            data_buf.dev_ptr() as u64,
            result_buf.dev_ptr() as u64,
        ))?
    };

    let gpu_count = unsafe { result_buf.read(0) };
    println!("GPU found {gpu_count} matches");
    Ok(())
}
```

### Warp-Cooperative — Before: ~130 lines, After: ~25 lines

```rust
use gpu_host::gpu;

const SIMPLE_PTX: &str = include_str!("../minimal.ptx");

fn main() -> gpu_host::Result<()> {
    let ctx = gpu::custom("test_simple_warp")
        .ptx(SIMPLE_PTX)
        .threads(32)
        .prepare()?;

    let mut output = ctx.alloc_zeros::<u32>(32)?;

    let result = unsafe { ctx.launch((&mut output,))? };
    let values = result.download(&output)?;

    let ok = (0..32).all(|tid| values[tid] == tid as u32 + 1);
    println!("test_simple_warp: {}", if ok { "PASSED" } else { "FAILED" });
    Ok(())
}
```

## Assessment Per Unconverted Example

| Example | Kernel Args | Hostcall? | Builder API Works? | Notes |
|---------|------------|-----------|-------------------|-------|
| **vector-math** | `(x, y, a, n)` etc | No | Yes | Pure compute, multiple kernels in one PTX module. Needs `.module()` reuse or multiple `prepare()` calls. |
| **tcp-echo** | `(buf, sideband, port, output)` | Yes | Yes | Hostcall + sideband + MappedBuffer output. The pointer extraction pattern handles this cleanly. |
| **parallel-search** | `(buf, sideband, pattern, pattern_len, data, result)` | Yes | Yes | 6 args, all fit within cudarc's 12-arg tuple limit. Full warp launch. |
| **warp-cooperative** | `(&mut output,)` | No | Yes | Already simple — just needs custom PTX support. Three separate kernels need three `prepare()` calls. |

## Key Design Decisions

### 1. Two-phase API: `prepare()` → `launch()`
**Why**: The user needs a handle to upload data before launching. A pure one-liner (`gpu::custom("k").input(&x).run()`) would require runtime-built `Vec<*mut c_void>` — losing type safety and requiring dynamic dispatch. The two-phase design keeps cudarc's compile-time tuple checking.

### 2. `launch()` consumes `GpuContext`, returns `GpuResult`
**Why**: The `CudaFunction` is consumed by cudarc's `LaunchAsync::launch()`. By consuming the context and returning a `GpuResult`, we enforce the correct lifecycle: prepare → upload → launch → download → finish.

### 3. No named parameters (`.input("x", &data)`)
**Why rejected**: CUDA kernels are positional, not named. Named params would need a registry mapping names to positions — fragile and error-prone. The raw tuple approach matches what cudarc does and what GPU programmers expect.

### 4. `hostcall_ptr()` returns `u64`, not a magic token
**Why**: The kernel expects a raw pointer. Returning u64 lets the user place it at the correct position in the argument tuple. A magic `.with_hostcall()` that auto-prepends would break kernels that don't put the hostcall buffer first.

### 5. Custom PTX via `.ptx(src)` instead of requiring embedded kernel.ptx
**Why**: The unconverted examples compile their own kernels (not part of the embedded monolith PTX). They use `include_str!(concat!(env!("OUT_DIR"), "/kernel.ptx"))`. The builder must accept arbitrary PTX sources.

## Multi-kernel Launches (vector-math softmax pattern)

Vector-math launches 4 different kernels sequentially (saxpy, elementwise_mul, softmax_exp, softmax_normalize) against the same device. Each `prepare()` creates a new CudaDevice — wasteful.

**Solution**: Add a `GpuContext::next()` method on `GpuResult` that reuses the device:

```rust
impl GpuResult {
    /// Prepare a new kernel launch on the same device.
    /// Reuses the CUDA device and (optionally) the hostcall session.
    pub fn next(self, kernel_name: &'static str) -> Result<GpuContext> {
        let ptx = cudarc::nvrtc::Ptx::from_src(/* need to know PTX source */);
        // ... load new function on existing device ...
    }
}
```

**Alternative** (simpler, recommended for v1): Let each `prepare()` create its own device. CudaDevice::new(0) returns the same physical device — cudarc handles context reuse internally via thread-local state. The overhead is minimal (cudarc caches the primary context). Keep the API simple; optimize later if profiling shows a problem.

## Risks

1. **CudaFunction consumed by launch**: cudarc's `LaunchAsync::launch(self, ...)` takes ownership of the function handle. This prevents re-launching the same kernel. For vector-math's multi-launch pattern, users must call `prepare()` again. This is a cudarc limitation, not a design flaw.

2. **Unsafe remains**: `launch()` is inherently unsafe because kernel argument correctness cannot be verified at compile time. The builder reduces boilerplate but cannot eliminate the fundamental unsafety.

3. **MappedBuffer lifetime**: MappedBuffers allocated via `ctx.mapped_buffer()` outlive the `GpuContext` (they're independent CUDA allocations). This is correct — the user reads from them after launch — but could confuse users who expect RAII cleanup to invalidate the buffers.

4. **Module name conflicts**: If two `prepare()` calls use the same PTX with auto-generated module names, cudarc loads duplicate modules. This is harmless but wastes memory. The `.module()` builder method allows explicit control.

## Recommendation

Implement the two-phase `prepare()` / `launch()` API as described. Start with the minimal surface area:
- `gpu::custom(name)` → `CustomLaunchBuilder`
- Builder: `.ptx()`, `.threads()`, `.grid()`, `.elements()`, `.hostcall()`, `.hostcall_packets()`, `.shared_mem()`
- `.prepare()` → `GpuContext`
- `GpuContext`: `.upload()`, `.alloc_zeros()`, `.mapped_buffer()`, `.hostcall_ptr()`, `.sideband_ptr()`, `.launch(tuple)` → `GpuResult`
- `GpuResult`: `.download()`, `.finish()`

Skip `.module()` and multi-kernel chaining for v1. Each example can call `gpu::custom()` separately for each kernel — the overhead is negligible.
