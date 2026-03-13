# host-sdk.1: API Surface Design
**Cycle**: 1 | **Theme**: host-sdk | **Kind**: design | **Status**: done

## Summary
Designed the public API surface for extracting a library crate from gpu-host. The key insight is that gpu-host already has a `lib.rs` exporting `error` and `hostcall` modules, but the test infrastructure (PTX loading, kernel launching, memory management) is all in `tests_*.rs` modules with `pub(crate)` visibility. The design extracts common patterns into reusable public types while keeping the existing test code working.

## Findings

### Q: What is the minimal public API surface that covers kernel launch + hostcall + memory?
A: Three core abstractions cover 95% of use cases:

1. **`GpuRuntime`** — wraps `CudaDevice` + PTX module loading + hostcall lifecycle
2. **`DeviceBuffer<T>`** — typed wrapper around `CudaSlice<T>` for device memory
3. **`MappedBuffer<T>`** — typed RAII wrapper around pinned mapped memory (replaces raw `alloc_mapped_*` + `free_mapped_*`)

Plus the already-exported `HostcallBuffer`, `HostcallError`, `GpuHostError`.

The current test pattern is:
```rust
// 1. Init device
let dev = CudaDevice::new(0)?;
// 2. Load PTX
let ptx = cudarc::nvrtc::Ptx::from_src(KERNEL_PTX);
dev.load_ptx(ptx, "kernel", &["my_func"]);
let f = dev.get_func("kernel", "my_func")?;
// 3. Allocate memory
let output: CudaSlice<f32> = dev.alloc_zeros::<f32>(n)?;
// 4. Launch
unsafe { f.launch(cfg, (&mut output,))?; }
dev.synchronize()?;
// 5. Read back
let result: Vec<f32> = dev.dtoh_sync_copy(&output)?;
```

With the SDK, this becomes:
```rust
let rt = GpuRuntime::new(0)?;
rt.load_ptx(KERNEL_PTX, "kernel", &["my_func"])?;
let mut output = rt.alloc_zeros::<f32>(n)?;
rt.launch("kernel", "my_func", (1,1,1), (32,1,1), 0, (&mut output.inner,))?;
rt.synchronize()?;
let result = output.to_host()?;
```

**Confidence**: high

### Q: How to handle PTX loading (embed vs file vs builder)?
A: Support all three via the same `load_ptx` method:
- `rt.load_ptx(include_str!("kernel.ptx"), "module", &[...])` — embedded at compile time (current approach)
- `rt.load_ptx(&std::fs::read_to_string("kernel.ptx")?, "module", &[...])` — loaded from file
- Build automation (build.rs) is deferred to build-tooling theme

No special abstraction needed — PTX is just a `&str`.

**Confidence**: high

### Q: What error types should be exposed?
A: Keep the existing `GpuHostError` and `HostcallError` as-is. They already have `Display`, `Error`, and `From` impls. Add one new variant:
- `GpuHostError::Launch(String)` — for kernel launch failures with descriptive messages

Remove `Verification` and `Timeout` variants from the public error type — these are test-specific, not SDK-level.

**Confidence**: high

## Proposed Public API

### Module: `gpu_host` (library root)

```rust
// Re-exports
pub mod error;      // GpuHostError, Result<T>
pub mod hostcall;   // HostcallBuffer, HostcallError, StdinSource, etc.
pub mod runtime;    // NEW: GpuRuntime
pub mod memory;     // NEW: MappedBuffer<T> (extracted from mapped_mem)
```

### Type: `GpuRuntime`

```rust
pub struct GpuRuntime {
    dev: Arc<CudaDevice>,
}

impl GpuRuntime {
    /// Initialize CUDA device by ordinal.
    pub fn new(device_ordinal: usize) -> Result<Self>;

    /// Get the underlying CudaDevice (for advanced usage).
    pub fn device(&self) -> &Arc<CudaDevice>;

    /// Load a PTX module with named kernel functions.
    pub fn load_ptx(&self, ptx_src: &str, module: &str, fn_names: &[&str]) -> Result<()>;

    /// Allocate device memory initialized to zero.
    pub fn alloc_zeros<T: cudarc::driver::DeviceRepr>(&self, len: usize) -> Result<CudaSlice<T>>;

    /// Upload data from host to device.
    pub fn htod_sync_copy<T: cudarc::driver::DeviceRepr>(&self, data: &[T]) -> Result<CudaSlice<T>>;

    /// Download data from device to host.
    pub fn dtoh_sync_copy<T: cudarc::driver::DeviceRepr>(&self, buf: &CudaSlice<T>) -> Result<Vec<T>>;

    /// Launch a kernel function.
    pub unsafe fn launch<P: cudarc::driver::LaunchAsync>(
        &self, module: &str, func: &str,
        grid: (u32, u32, u32), block: (u32, u32, u32), smem: u32,
        params: P,
    ) -> Result<()>;

    /// Synchronize the device (wait for all kernels to complete).
    pub fn synchronize(&self) -> Result<()>;
}
```

### Type: `MappedBuffer<T>`

```rust
/// RAII handle for pinned, device-mapped host memory.
/// Provides both host-side and device-side access.
pub struct MappedBuffer<T> {
    host_ptr: *mut T,
    dev_ptr: CUdeviceptr,
    len: usize,
}

impl<T> MappedBuffer<T> {
    /// Allocate a zero-initialized mapped buffer.
    pub fn new_zeroed(len: usize) -> Result<Self>;

    /// Get device pointer for kernel arguments.
    pub fn dev_ptr(&self) -> CUdeviceptr;

    /// Read a value at index (volatile read for GPU-written data).
    pub unsafe fn read(&self, index: usize) -> T;

    /// Write a value at index (volatile write for GPU-readable data).
    pub unsafe fn write(&mut self, index: usize, value: T);

    /// Get a slice view of the host memory.
    pub unsafe fn as_slice(&self) -> &[T];
}

impl<T> Drop for MappedBuffer<T> {
    fn drop(&mut self) { /* cuMemFreeHost */ }
}
```

## Design Decisions

### Keep cudarc types in public API
Decision: Expose `CudaSlice<T>` and `CudaDevice` in the public API rather than wrapping them.
Rationale: cudarc is already a well-maintained public crate. Wrapping it adds complexity without value. Users who need advanced features can access `rt.device()` directly. The SDK is a convenience layer, not an abstraction boundary.

### Binary + Library dual mode
Decision: gpu-host becomes both `lib.rs` (library) and `main.rs` (test binary).
Rationale: Cargo supports this natively. `main.rs` uses the library API. No crate splitting needed. The `tests_*.rs` modules become `pub(crate)` test infrastructure that uses the public API internally.

### No GpuTensor<T> abstraction yet
Decision: Defer tensor abstraction (shape, strides, dtype) to inference code.
Rationale: A tensor type is domain-specific to ML. The SDK should provide raw memory management; inference code adds its own tensor wrapper. Premature abstraction here would couple the SDK to ML use cases.

## Migration Path

1. Create `runtime.rs` with `GpuRuntime` struct
2. Move `mapped_mem.rs` → `memory.rs`, make functions `pub` with RAII wrapper
3. Update `lib.rs` to export new modules
4. Keep `main.rs` and `tests_*.rs` unchanged (they use internal `pub(crate)` helpers)
5. Gradually migrate test code to use SDK API (optional, not required)

## Impact on Downstream Tasks
- **host-sdk.2**: Implement `GpuRuntime` + `MappedBuffer<T>` based on this design
- **host-sdk.3**: Create standalone example using the SDK
- **model-loading.2-4**: Will benefit from `GpuRuntime` for weight upload and kernel management
- **full-inference**: Will use SDK API for the 12-layer inference pipeline
