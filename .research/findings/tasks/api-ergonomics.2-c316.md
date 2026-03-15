# api-ergonomics.2: Builder pattern for HostcallBuffer + session management
**Cycle**: 316 | **Theme**: api-ergonomics | **Kind**: experiment | **Status**: done

## Summary
Promoted `HostcallSession` and `Pipeline` from dead code to first-class public API. Added `require_func()` to `GpuRuntime`. Updated hello-gpu example to demonstrate the improved API, reducing boilerplate by ~15 lines.

## Changes Made

### 1. `GpuRuntime::require_func()` (runtime.rs)
- New method: `require_func(&self, module: &str, func: &'static str) -> Result<CudaFunction>`
- Replaces `get_func().ok_or(KernelNotFound(...))` boilerplate

### 2. Promoted HostcallSession + Pipeline (hostcall.rs)
- Removed `#[allow(dead_code)]` from `HostcallSession` struct and impl
- Removed `#[allow(dead_code)]` from `Pipeline` struct and impl
- These types were already fully implemented but unused

### 3. Re-exports (lib.rs)
- Added `HostcallSession` and `Pipeline` to crate root re-exports
- Users can now `use gpu_host::HostcallSession` directly

### 4. Updated hello-gpu example
- Replaced manual `thread::scope` + `spawn` + `listen` + `signal_shutdown` with `HostcallSession::start(8)?` + `session.shutdown()`
- Replaced 4x `get_func().ok_or(KernelNotFound(...))` with `require_func()`
- Removed `GpuHostError` import (no longer needed)
- Replaced `hcbuf.dev_ptr as u64` with `session.dev_ptr()`
- Net reduction: ~15 lines of boilerplate

## Impact on Downstream Tasks
- api-ergonomics.3 (high-level launch helper) can now build on HostcallSession
- Other examples should be updated to use HostcallSession + require_func in future
