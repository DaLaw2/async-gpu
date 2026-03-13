# Patched std — CUDA/nvptx64 patches for Rust std library

Patches for `target_os = "cuda"` (nvptx64-nvidia-cuda).
Nightly: `rustc 1.96.0-nightly (3b1b0ef4d 2026-03-11)`

## Setup

```bash
./std-patches/apply.sh            # copies sysroot + applies patches
# Then build with:
__CARGO_TESTS_ONLY_SRC_ROOT="patched-std/library" cargo +nightly build --release
```

## Regenerate patches after editing patched-std/

```bash
./std-patches/gen-patches.sh      # diffs patched-std/ vs sysroot, updates std-patches/
```

## Modified files (patches)

- `std/src/io/stdio.rs` → `std-patches/io_stdio.patch`
- `std/src/sys/alloc/mod.rs` → `std-patches/sys_alloc_mod.patch`
- `std/src/sys/fs/mod.rs` → `std-patches/sys_fs_mod.patch`
- `std/src/sys/io/error/mod.rs` → `std-patches/sys_io_error_mod.patch`
- `std/src/sys/random/mod.rs` → `std-patches/sys_random_mod.patch`
- `std/src/sys/stdio/mod.rs` → `std-patches/sys_stdio_mod.patch`
- `std/src/sys/thread_local/mod.rs` → `std-patches/sys_thread_local_mod.patch`

## New files

- `std/src/sys/alloc/cuda.rs` → `std-patches/sys_alloc_cuda.rs`
- `std/src/sys/fs/cuda.rs` → `std-patches/sys_fs_cuda.rs`
- `std/src/sys/io/error/cuda.rs` → `std-patches/sys_io_error_cuda.rs`
- `std/src/sys/stdio/cuda.rs` → `std-patches/sys_stdio_cuda.rs`
