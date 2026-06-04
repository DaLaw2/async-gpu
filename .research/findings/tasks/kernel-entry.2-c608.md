# kernel-entry.2: Implicit hostcall injection for zero-param kernel entry

**Status:** done
**Kind:** experiment
**Date:** 2024-06-04

## Summary

Implemented implicit hostcall buffer injection via `__HOSTCALL_BUF` device global,
enabling zero-parameter GPU kernel entry. The host writes the hostcall session's
device pointer to a well-known device global via `cuModuleGetGlobal_v2` +
`cuMemcpyHtoD` before kernel launch. The kernel reads it at entry via
`gpu_runtime::entry::auto_init()`. End-to-end test passes: `zero_param_hello()`
kernel uses `println!` and `Vec` with no kernel parameters.

## Approach

Based on investigation (kernel-entry.1): use device globals to inject the hostcall
buffer address. No compiler changes needed.

### Phase 1: GPU-side device global + auto_init

- Added `gpu_runtime::entry` module with:
  - `__HOSTCALL_BUF: AtomicU64` — `#[no_mangle] #[used]` canonical device global
  - `hostcall_buf_ptr()` — read the global, return `*mut u8`
  - `auto_init()` — read global and initialize panic handler
- Added `stdio_auto_init()` in `gpu-kernel-std` that reads `__HOSTCALL_BUF`
  and initializes stdio, panic, and libc I/O subsystems

### Phase 2: Host-side device global injection

- Added `gpu::run_zero_param()` and `gpu::run_zero_param_with_config()` in `gpu-host/src/gpu.rs`
- Uses raw CUDA driver API (since cudarc doesn't expose `CUmodule` handles):
  1. `cuModuleLoadData` — load PTX, get `CUmodule`
  2. `cuModuleGetFunction` — get kernel function handle
  3. `cuModuleGetGlobal_v2` — find `__HOSTCALL_BUF` symbol, get device address
  4. `cuMemcpyHtoD_v2` — write hostcall `dev_ptr` value to the global
  5. `cuLaunchKernel` — launch with zero kernel arguments (`params = null`)

### Phase 3: Zero-param kernel + end-to-end test

- Added `zero_param_hello()` kernel in `gpu-kernel-std`:
  ```rust
  pub unsafe extern "gpu-kernel" fn zero_param_hello() {
      let buf = stdio_auto_init();  // reads __HOSTCALL_BUF device global
      gpu_runtime::thread::gpu_main_poll(|| {
          println!("Hello from zero-param kernel!");
          let v: Vec<i32> = (1..=5).collect();
          println!("Vec on GPU: {:?}, sum = {}", v, v.iter().sum::<i32>());
      });
  }
  ```
- PTX confirms: `.visible .entry zero_param_hello()` — no parameters
- Test via `ONLY_TEST=zero_param cargo run --release -p gpu-test-harness` — PASSES

## PTX Evidence

```
.visible .global .align 8 .b8 __HOSTCALL_BUF[8];
	ld.relaxed.sys.global.b64 	%rd4, [__HOSTCALL_BUF];
...
.visible .entry zero_param_hello()      // no parameters!
```

## Key Design Decisions

1. **Raw CUDA API for host side** — cudarc's `CudaDevice::load_ptx()` stores
   the `CUmodule` in a private field (`pub(crate)`). We need the module handle
   for `cuModuleGetGlobal_v2`, so we load via raw `cuModuleLoadData` instead.

2. **AtomicU64 (not raw pointer)** — The device global uses `AtomicU64` because
   raw pointers are not `Sync` in `no_std` (needed for `static`). On the PTX
   side this compiles to the same 8-byte global.

3. **Backward compatible** — Old kernels with explicit `buf: *mut u8` parameter
   still work unchanged. The device global is only used when `stdio_auto_init()`
   or `entry::auto_init()` is called.

4. **Module cleanup** — The raw `CUmodule` is explicitly unloaded via
   `cuModuleUnload` after kernel completion, avoiding leaks.

## Files Changed

- `crates/core/gpu-runtime/src/entry.rs` — NEW: device global + auto_init
- `crates/core/gpu-runtime/src/lib.rs` — added `pub mod entry`
- `crates/kernel/gpu-kernel-std/src/lib.rs` — added `stdio_auto_init()` + `zero_param_hello` kernel
- `crates/core/gpu-host/src/gpu.rs` — added `run_zero_param()` + `run_zero_param_with_config()`
- `crates/test/gpu-test-harness/src/main.rs` — added `run_zero_param_test()` + ONLY_TEST=zero_param
