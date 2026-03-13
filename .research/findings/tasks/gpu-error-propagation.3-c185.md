# gpu-error-propagation.3: Host-side error extraction from kernel result buffer
**Cycle**: 185 | **Theme**: gpu-error-propagation | **Kind**: experiment | **Status**: done

## Summary
Added `GpuKernelErrorInfo` struct and `check_kernel_result()` function to gpu-host. Host can now
read the 64-byte `GpuKernelResult` buffer after kernel launch and get structured error info
including category, errno, thread/block index, and message.

## Findings

### Q: How does host read GpuKernelResult after kernel launch?
A: The pattern is:
1. Allocate mapped memory for `GpuKernelResult` (host-accessible, device-accessible)
2. Initialize `.tag = TAG_UNINIT` before launch
3. Pass device pointer as last kernel parameter
4. After `synchronize()`, call `check_kernel_result(&result)` which returns:
   - `Ok(())` for TAG_OK
   - `Err(KernelError(info))` for TAG_ERR with full error details
   - `Err(KernelCrash)` for TAG_UNINIT (kernel crashed before reporting)

**Confidence**: high

### Q: Can KernelBuilder::launch() return Result<(), GpuKernelError>?
A: Not yet implemented at the KernelBuilder level — that's a host-sdk task. The `check_kernel_result`
function is a standalone helper that can be called after any kernel launch. Integration with
KernelBuilder API is deferred to host-sdk theme.

**Confidence**: high

## Types Added

### GpuKernelErrorInfo (gpu-host, host-only)
Human-readable error info with category name, thread/block IDs, errno, and message string.
Implements Display for nice formatting.

### GpuHostError::KernelError / KernelCrash
Two new variants on the host error enum.

### check_kernel_result() / error_category_name()
Helper functions for decoding kernel results.

## Impact on Downstream Tasks
- host-sdk.2 can integrate this into KernelBuilder API
- std-fs.4 can use this for end-to-end error testing
