# gpu-error-propagation.1: Design GpuKernelResult ABI + error buffer layout
**Cycle**: 181 | **Theme**: gpu-error-propagation | **Kind**: design | **Status**: done

## Summary
Designed a 64-byte GpuKernelResult ABI for GPU→host error propagation. The error buffer
is passed as a kernel parameter (not global state). GPU-side `?` operator works naturally
through Rust's Result type — a wrapper function at the kernel entry catches errors and
writes them to the buffer. Host reads the buffer after synchronization.

## Findings

### Q: What layout for GpuKernelResult: tag + category + errno + message? How many bytes?
A: 64 bytes (single cache line, clean alignment for mapped memory).

```
GpuKernelResult (64 bytes, repr(C)):
  Offset 0:   tag        (u32)  — TAG_OK=0, TAG_ERR=1, TAG_UNINIT=0xDEAD_BEEF
  Offset 4:   category   (u16)  — ERR_* constant from gpu-protocol
  Offset 6:   raw_errno  (u16)  — OS errno (optional, 0 = not provided)
  Offset 8:   thread_idx (u16)  — threadIdx.x that produced the error
  Offset 10:  block_idx  (u16)  — blockIdx.x that produced the error
  Offset 12:  msg_len    (u32)  — message byte count (0..48)
  Offset 16:  msg_bytes  [48]   — UTF-8 message (truncated if needed)
```

TAG_UNINIT serves as sentinel: if kernel crashes (trap) before writing, host sees
TAG_UNINIT and knows the kernel died without reporting an error. This replaces the
current pattern where trap → CUDA_ERROR_LAUNCH_FAILED with no diagnostic info.

**Confidence**: high

### Q: Should error buffer be a kernel parameter or embedded in hostcall buffer?
A: **Kernel parameter.** Rationale:

1. **Explicit**: each kernel launch has its own error buffer — no global state
2. **Composable**: multiple concurrent kernel launches each have independent error buffers
3. **Simple**: host allocates mapped memory, passes device pointer as last kernel arg
4. **No protocol change**: hostcall buffer layout remains unchanged

The alternative (embedded in hostcall buffer header) would require protocol versioning
and break existing kernel compatibility. Not worth it.

**Confidence**: high

### Q: How does `?` operator desugar: write to buffer + early return?
A: Through Rust's standard `Result` + `From` + `?` mechanics — no macro magic needed.

**GPU-side types** (in gpu-protocol or new gpu-error crate):

```rust
/// Error type for GPU operations that can be propagated via `?`.
#[repr(C)]
pub struct GpuError {
    pub category: u16,
    pub raw_errno: u16,
    // msg is formatted at the kernel entry wrapper, not here
}

impl From<HostcallError> for GpuError { ... }  // hostcall failures
impl From<IoError> for GpuError { ... }        // future: std::io::Error
```

**Kernel pattern (what the user writes):**

```rust
fn my_kernel_body(buf: *mut u8, data: *const f32, len: u32) -> Result<(), GpuError> {
    let fd = gpu_fs_open(buf, "output.txt", FILE_OPEN_WRITE_CREATE)?;
    gpu_fs_write(buf, fd, b"hello from GPU")?;
    gpu_fs_close(buf, fd)?;
    Ok(())
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn my_kernel(
    buf: *mut u8,
    data: *const f32,
    len: u32,
    __result: *mut GpuKernelResult,
) {
    // Initialize as UNINIT — if we trap before finishing, host sees this
    (*__result).tag = TAG_UNINIT;

    match my_kernel_body(buf, data, len) {
        Ok(()) => (*__result).set_ok(),
        Err(e) => (*__result).set_err(e),
    }
}
```

**Future: proc macro** could generate the extern wrapper automatically:
```rust
#[gpu_entry]
fn my_kernel(buf: *mut u8, data: *const f32, len: u32) -> Result<(), GpuError> {
    // ...
}
```
But the proc macro is optional sugar — the manual pattern works today.

**Host pattern:**

```rust
// Allocate error buffer (mapped memory, visible to both GPU and CPU)
let result_buf = alloc_mapped::<GpuKernelResult>(&dev)?;
(*result_buf.host_ptr).tag = TAG_UNINIT;

// Launch kernel with error buffer as last param
unsafe { kernel.launch(cfg, (buf_dev, data_dev, len, result_buf.dev_ptr))? };
dev.synchronize()?;

// Read result
match (*result_buf.host_ptr).tag {
    TAG_OK => Ok(output),
    TAG_ERR => {
        let r = &*result_buf.host_ptr;
        let msg = core::str::from_utf8(&r.msg_bytes[..r.msg_len as usize])
            .unwrap_or("<invalid utf8>");
        Err(GpuKernelError {
            category: r.category,
            raw_errno: r.raw_errno,
            thread_idx: r.thread_idx,
            block_idx: r.block_idx,
            message: msg.to_string(),
        })
    }
    TAG_UNINIT => Err(GpuKernelError::crashed("kernel exited without writing result")),
    _ => Err(GpuKernelError::crashed("unknown result tag")),
}
```

**Confidence**: high

## ADR

### ADR-13: GPU→Host Error Propagation via Result Buffer
- **Date**: 2026-03-13
- **Status**: proposed
- **Context**: GPU kernels currently have no way to report errors to the host except
  via panic (which traps and kills the CUDA context) or writing ad-hoc status values
  to output buffers. std::io returns Result, so proper error handling is needed before
  real std can work on GPU.
- **Decision**: Define a 64-byte `GpuKernelResult` struct passed as the last kernel
  parameter. TAG_OK/TAG_ERR/TAG_UNINIT tag field. GPU writes error info (category,
  errno, message). Host reads after synchronization. `?` operator works through
  standard Rust Result mechanics with a wrapper function at kernel entry.
- **Rationale**: (1) Kernel parameter is explicit and composable. (2) No protocol
  change needed. (3) Works with both no_std and std kernels. (4) TAG_UNINIT detects
  kernel crashes. (5) Standard Rust error handling patterns.
- **Alternatives**: Embed in hostcall buffer header (breaks protocol), use hostcall
  channel for errors (adds latency, trap still kills context), global error buffer
  (not composable).

## Impact on Downstream Tasks
- gpu-error-propagation.2: implement GpuError + From impls + hostcall Result returns
- gpu-error-propagation.3: implement host-side GpuKernelError + result buffer allocation
- gpu-error-propagation.4: modify panic handler to write to result buffer before trap
- std-migration.3: async pipeline kernels can use Result<(), GpuError>
