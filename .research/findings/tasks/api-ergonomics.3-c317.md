# api-ergonomics.3: Update remaining examples to use HostcallSession + require_func
**Cycle**: 317 | **Theme**: api-ergonomics | **Kind**: experiment | **Status**: done

## Summary
Updated all 5 remaining examples to use the new `HostcallSession` and `require_func()` API,
matching the pattern established in hello-gpu. Each example replaced manual `thread::scope` +
`spawn` + `listen` + `signal_shutdown` with `HostcallSession::start(8)?` / `session.shutdown()`,
and replaced `get_func().ok_or(GpuHostError::KernelNotFound(...))` with `require_func()`.
All CI checks pass (`scripts/pre-push.sh`).

## Changes per Example

### async-io
- Lines before: 114 → Lines after: 100
- Changes:
  - Import: `{GpuHostError, HostcallBuffer}` → `HostcallSession`
  - `HostcallBuffer::new(8)` → `HostcallSession::start(8)?`
  - Removed `thread::scope` + `spawn` + `listen` wrapper
  - 2x `get_func().ok_or(KernelNotFound)` → `require_func()`
  - `hcbuf.dev_ptr as u64` → `session.dev_ptr()`
  - `hcbuf.sideband_dev_ptr as u64` → `session.sideband_dev_ptr()`
  - `hcbuf.signal_shutdown()` → `session.shutdown()`

### parallel-search
- Lines before: 153 → Lines after: 140
- Changes:
  - Import: `{GpuHostError, HostcallBuffer}` → `HostcallSession`
  - `HostcallBuffer::new(8)` → `HostcallSession::start(8)?`
  - Removed `thread::scope` + `spawn` + `listen` wrapper
  - 1x `get_func().ok_or(KernelNotFound)` → `require_func()`
  - `hcbuf.dev_ptr as u64` → `session.dev_ptr()`
  - `hcbuf.sideband_dev_ptr as u64` → `session.sideband_dev_ptr()`
  - `hcbuf.signal_shutdown()` → `session.shutdown()`

### tcp-echo
- Lines before: 115 → Lines after: 105
- Changes:
  - Import: `{GpuHostError, HostcallBuffer}` → `HostcallSession`
  - `HostcallBuffer::new(8)` → `HostcallSession::start(8)?`
  - Removed `thread::scope` + `spawn` + `listen` wrapper
  - 1x `get_func().ok_or(KernelNotFound)` → `require_func()`
  - `hcbuf.dev_ptr as u64` → `session.dev_ptr()`
  - `hcbuf.sideband_dev_ptr as u64` → `session.sideband_dev_ptr()`
  - `hcbuf.signal_shutdown()` → `session.shutdown()`

### vector-math
- Lines before: 147 → Lines after: 138
- Changes:
  - Import: removed `GpuHostError` (no longer needed)
  - 4x `get_func().ok_or(KernelNotFound)` → `require_func()` (saxpy, elementwise_mul, softmax_exp, softmax_normalize)
  - No hostcall changes (pure compute example)

### async-pipeline
- Lines before: 200 → Lines after: 174
- Changes:
  - Import: `{GpuHostError, HostcallBuffer}` → `HostcallSession`
  - 2x `HostcallBuffer::new(8)` → `HostcallSession::start(8)?` (one per demo fn)
  - Removed 2x `thread::scope` + `spawn` + `listen` wrappers
  - 2x `get_func().ok_or(KernelNotFound)` → `require_func()`
  - `hcbuf.dev_ptr as u64` → `session.dev_ptr()`
  - `hcbuf.sideband_dev_ptr as u64` → `session.sideband_dev_ptr()`
  - 2x `hcbuf.signal_shutdown()` → `session.shutdown()`

## Impact on Downstream Tasks
- All examples now use a consistent API pattern matching hello-gpu
- No example imports `GpuHostError` or `HostcallBuffer` anymore
- async-pipeline excluded from CI (requires patched rustc) but follows identical pattern
- `warp-cooperative` intentionally NOT updated (uses raw cudarc)
