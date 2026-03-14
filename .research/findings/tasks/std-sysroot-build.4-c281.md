# std-sysroot-build.4: End-to-end test — File::create + write on GPU via hostcall
**Cycle**: 281 | **Theme**: std-sysroot-build | **Kind**: experiment | **Status**: done

## Summary
End-to-end GPU File I/O test passed: `std::fs::File::create` + `write_all` and `File::open` + `read` execute correctly on GPU through hostcall. The key fix was forcing gpu-libc symbols (`open`, `close`, `read`, `write`, `__errno_location`) to survive LTO by referencing their function pointers in a `#[used]` static array.

## Findings

### Q: Does std::fs::File work end-to-end on GPU?
A: **Yes.** Both write and read operations pass:
- `std_file_write_kernel`: `File::create("gpu_test_output.txt")` → hostcall OPEN (flags=1) → `write_all(b"Hello from GPU std::fs::File!")` → hostcall WRITE (29 bytes) → hostcall CLOSE. File verified on host with correct content.
- `std_file_read_kernel`: `File::open("gpu_test_input.txt")` → hostcall OPEN (flags=0) → `read(&mut buf)` → hostcall READ (18 bytes) → hostcall CLOSE. First byte verified as 'G' (0x47).

**Confidence**: high

### Q: What was the root cause of CUDA_ERROR_INVALID_PTX?
A: **Unresolved `.extern .func` declarations in PTX.** The patched std PAL (`sys_fs_cuda.rs`) declares `extern "C" fn open/close/read/write/__errno_location` which are implemented in gpu-libc. However, LTO removed gpu-libc's `#[no_mangle]` functions because:
1. std-build-test depends on gpu-libc in Cargo.toml
2. But no Rust code directly calls these functions — only std's PAL does via `extern "C"` blocks
3. LLVM's LTO pass sees no direct references from Rust and eliminates the symbols
4. The resulting PTX has `.extern .func open/close/read/write/__errno_location` with no definitions

**Fix**: Added `#[used] static FORCE_LINK_GPU_LIBC` array with function pointers to all 5 gpu-libc functions, preventing LTO elimination.

Also needed: `gpu_libc::gpu_libc_io_init(buf)` call in `stdio_init()` to set the hostcall buffer pointer for file I/O operations.

**Confidence**: high

### Q: PTX characteristics after fix?
A:
- 17 kernel entries (unchanged)
- 61,648 lines (up from 54,492 — gpu-libc function bodies now included)
- 0 `.extern .func` (all resolved)
- 0 `.ptr .align` (post-processed)
- PTX version 7.8, target sm_86

**Confidence**: high

## Unexpected Discoveries
- LTO aggressively removes `#[no_mangle] extern "C"` functions even when they have the same symbol name as `extern "C"` declarations in other compilation units. The `#[used]` attribute with function pointer references is needed to force retention.
- `gpu_libc::gpu_libc_io_init(buf)` was not being called for the File I/O kernels — `stdio_init()` only initialized the stdout/stdin hostcall buffer, not the gpu-libc I/O subsystem.

## Impact on Downstream Tasks
- **std-sysroot epic**: ALL 4 CRITERIA MET — epic completed
- **std-sysroot-build theme**: ALL 3 criteria met — theme completed
- **async-std epic**: File I/O foundation solid — ready for warp-cooperative async File I/O
