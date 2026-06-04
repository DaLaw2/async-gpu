# Patched std — CUDA/nvptx64 patches for Rust std library

Patches for `target_os = "cuda"` (nvptx64-nvidia-cuda).
Baseline: `rustc 1.96.0-nightly (3b1b0ef4d 2026-03-11)`

## Setup

```bash
./scripts/apply-std-patches.sh    # copies rustc-src/library/std/ + applies patches → patched-std/
```

## Regenerate patches after editing patched-std/

```bash
./scripts/gen-std-patches.sh      # diffs patched-std/ vs rustc-src/library/std/
```

## Modified files (patches)

- `src/io/stdio.rs` → `std-patches/io_stdio.patch`
- `src/os/fd/raw.rs` → `std-patches/os_fd_raw.patch`
- `src/sys/alloc/mod.rs` → `std-patches/sys_alloc_mod.patch`
- `src/sys/fs/mod.rs` → `std-patches/sys_fs_mod.patch`
- `src/sys/io/error/mod.rs` → `std-patches/sys_io_error_mod.patch`
- `src/sys/net/connection/sgx.rs` → `std-patches/sys_net_connection_sgx.patch`
- `src/sys/random/mod.rs` → `std-patches/sys_random_mod.patch`
- `src/sys/stdio/mod.rs` → `std-patches/sys_stdio_mod.patch`
- `src/sys/thread/mod.rs` → `std-patches/sys_thread_mod.patch`
- `src/sys/thread_local/mod.rs` → `std-patches/sys_thread_local_mod.patch`

## New files

- `src/sys/alloc/cuda.rs` → `std-patches/sys_alloc_cuda.rs`
- `src/sys/fs/cuda.rs` → `std-patches/sys_fs_cuda.rs`
- `src/sys/io/error/cuda.rs` → `std-patches/sys_io_error_cuda.rs`
- `src/sys/stdio/cuda.rs` → `std-patches/sys_stdio_cuda.rs`
- `src/sys/thread/cuda.rs` → `std-patches/sys_thread_cuda.rs`
- `src/sys/thread_local/gpu_threads.rs` → `std-patches/sys_thread_local_gpu_threads.rs`
