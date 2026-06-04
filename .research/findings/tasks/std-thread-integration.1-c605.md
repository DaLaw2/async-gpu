# std-thread-integration.1: Investigation — std::thread::spawn hang

## Summary
The std::thread::spawn hang has TWO root causes: (1) the build system uses stock (unpatched) std from rustup's rust-src, not the patched std at `patched-std/`, so `std::thread::spawn()` routes to `unsupported::Thread::new()` which returns `Err(UNSUPPORTED_PLATFORM)` and panics; (2) even kernels using `gpu_runtime::thread::spawn()` (not std::thread) hang when loaded from `kernel_std.ptx`, indicating the hang is in the PTX module itself, not the thread spawn path. The `std_thread_spawn_demo` kernel hangs on `dev.synchronize()` despite using the poll-based `gpu_main_poll()` that avoids `bar.sync`.

## Findings

### Q1: What is the std::thread::spawn code path?
**Confidence: 95%**

`std::thread::spawn(f)` → `Builder::new().spawn(f)` → `spawn_unchecked()` (lifecycle.rs) → `imp::Thread::new(stack_size, init)`. For nvptx64-nvidia-cuda, `imp` resolves to `sys::thread::unsupported` (stock std) or `sys::thread::cuda` (patched std). The patched cuda.rs calls `gpu_thread_spawn_raw()` FFI into gpu_runtime.

### Q2: Why does the build use stock std?
**Confidence: 99%**

The `.cargo/config.toml` for gpu-kernel-std specifies `build-std = ["std", "core", "panic_abort"]` but does NOT specify a custom library source path. Cargo's `-Zbuild-std` resolves to:
```
/home/dalaw2/.rustup/toolchains/nightly-2026-06-03-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/std
```
This is the UNPATCHED stock std. The patched std at `patched-std/` is never used. The installed rust-src has NO `cuda.rs`, NO `gpu_threads.rs`, and NO `target_os = "cuda"` branches.

### Q3: What thread implementation does stock std use for nvptx64?
**Confidence: 99%**

Stock std's `sys::thread::mod.rs` has no `target_os = "cuda"` branch. nvptx64-nvidia-cuda falls to the `_ => { mod unsupported; ... }` catch-all. `unsupported::Thread::new()` returns `Err(io::Error::UNSUPPORTED_PLATFORM)`. So `std::thread::spawn()` will call `.expect("failed to spawn thread")` and PANIC.

### Q4: Does the `std_thread_spawn_demo` actually use std::thread::spawn?
**Confidence: 99%**

NO. Despite the name, `std_thread_spawn_demo` uses `gpu_runtime::thread::gpu_main_poll()` and `gpu_runtime::thread::spawn()` directly. It NEVER calls `std::thread::spawn()`. The comment "identical to CPU Rust" is misleading.

### Q5: Where does the hang actually occur?
**Confidence: 85%**

The hang occurs during `dev.synchronize()` after launching `std_thread_spawn_demo` from `kernel_std.ptx`. The kernel never completes. Possible causes:
- The kernel_std.ptx module (4.5MB) contains massive std machinery. Module globals may interact unexpectedly.
- The thread-local implementation falls to `os` module (OS-provided TLS keys) which requires `pthread_key_create` — stubbed as no-ops in gpu-libc.
- LLVM optimizer may have miscompiled the `gpu_main_poll` inlining when combined with std code.
- The `membar.cta` instruction replacing `bar.sync` may not provide sufficient synchronization.

### Q6: Does kernel_std.ptx contain bar.sync?
**Confidence: 99%**

NO. `kernel_std.ptx` has ZERO `bar.sync` instructions. `kernel.ptx` (non-std) has 39. This is because `std_thread_spawn_demo` uses `gpu_main_poll()` (atomic polling) instead of `gpu_main()` (bar.sync). However, the `membar.cta` that appears may be insufficient.

### Q7: What about the thread_local implementation?
**Confidence: 90%**

Stock std for nvptx64 uses the `os` TLS module (since `target_thread_local` is not set and no specific branch matches). This requires OS TLS keys via `pthread_key_create` etc. These are stubbed as no-ops in gpu-libc, so TLS initialization silently fails but doesn't crash. The patched std's `gpu_threads.rs` (thread-ID-indexed arrays) is NOT used.

## Unexpected Discoveries

1. **kernel_std.ptx is built from stock std, not patched std.** The entire patched-std infrastructure (`std-patches/`, `patched-std/`, `apply-std-patches.sh`) is not wired into the build. The patches exist but are never applied to the build pipeline.

2. **The `std_thread_spawn_demo` kernel name is misleading.** It uses `gpu_runtime::thread::spawn`, not `std::thread::spawn`. There is no kernel that actually tests `std::thread::spawn` on GPU.

3. **pthread_create exists in kernel_std.ptx** from gpu-libc stubs, returning ENOSYS. Stock std may attempt to call this for thread operations.

4. **Compiler reordering**: The NUM_WARPS store happens BEFORE the WARP_STATUS initialization in the PTX, opposite to the Rust source order. This is allowed by Relaxed ordering but could cause timing issues.

5. **STATUS_COOPERATIVE (5) check is missing from the worker loop** in the PTX — the worker only checks for STATUS_ASSIGNED (1) and STATUS_EXIT (4). The `STATUS_ASSIGNED | STATUS_COOPERATIVE` match arm in the source is compiled to check only for 1, not 5.

## Open Questions

1. **Why does kernel_std.ptx hang even without using std::thread?** The `std_thread_spawn_demo` uses only gpu_runtime::thread, yet it hangs. Is it a PTX JIT issue, a miscompilation, or a global state interaction?

2. **How to wire patched std into the build?** The `-Zbuild-std` flag needs a custom library source path. Can `__CARGO_TESTS_ONLY_SRC_ROOT` or a Cargo path override be used?

3. **Is the hang in PTX JIT compilation or kernel execution?** The PTX loads OK (no CUDA_ERROR_INVALID_PTX for kernel_std.ptx unlike std_build_test.ptx), but the kernel never completes.

4. **Does the OS TLS fallback cause silent corruption?** The stock std's `os` TLS module may silently corrupt data when the pthread stubs return 0/ENOSYS.

## Impact on Downstream Tasks

- **std-thread-integration.2 (implement std::thread routing)**: BLOCKED until the build system is fixed to use patched std. The patches exist but are not applied.
- **std-thread-integration.3 (test std::thread::spawn)**: Cannot test until both the build system AND the kernel hang are fixed.
- The kernel_std.ptx hang affects ALL kernels loaded from that module, not just thread-related ones. This is a prerequisite blocker.
