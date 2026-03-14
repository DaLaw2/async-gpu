# std-sysroot-build.1: Investigation — attempt x.py build with std patches, catalog all errors
**Cycle**: 276 | **Theme**: std-sysroot-build | **Kind**: investigation | **Status**: done

## Summary
Investigated building patched std for nvptx64-nvidia-cuda. Found that `x.py build` does NOT build std for nvptx64 because the target spec has `"std": false`. However, `-Zbuild-std=["std", "core", "panic_abort"]` with `__CARGO_TESTS_ONLY_SRC_ROOT` pointing to patched library source works perfectly — std compiles for nvptx64 with **zero errors**. All 15 test kernels using Vec, String, format!, println!, stdin, File I/O compile to valid PTX (51K lines, 1.9MB).

## Findings

### Q: Does x.py build library succeed for nvptx64 with patched std?
A: **Partial.** `x.py build compiler library` succeeds, but only builds `core`, `alloc`, and `compiler_builtins` for nvptx64. The nvptx64-nvidia-cuda target spec has `"std": false`, which tells the bootstrap system to skip std. The sysroot at `stage1/lib/rustlib/nvptx64-nvidia-cuda/lib/` contains only:
- `liballoc-*.rlib` (190 KB)
- `libcompiler_builtins-*.rlib` (1.1 MB)
- `libcore-*.rlib` (1.5 MB)

No `libstd`, `libpanic_abort`, or other std dependencies.

**Confidence**: high

### Q: Which PAL modules fail? How many errors?
A: **Zero errors with -Zbuild-std path.** When using the patched compiler with `-Zbuild-std=["std", "core", "panic_abort"]` and `__CARGO_TESTS_ONLY_SRC_ROOT=patched-rustc/library`, std compiles cleanly for nvptx64. The build output shows all crates compile successfully:
```
Compiling core v0.0.0
Compiling alloc v0.0.0
Compiling compiler_builtins v0.1.160
Compiling std v0.0.0
Compiling panic_abort v0.0.0
Compiling unwind v0.0.0
Compiling panic_unwind v0.0.0
Compiling hashbrown v0.16.1
Compiling std_detect v0.1.5
Compiling proc_macro v0.0.0
```
Build time: ~29 seconds (with cached dependencies).

The patched PAL modules (`sys_fs_cuda.rs`, `sys_stdio_cuda.rs`, `sys_alloc_cuda.rs`, `sys_io_error_cuda.rs`, `sys_thread_local_gpu_threads.rs`) plus the 9 patch files correctly gate all cuda-specific code via `target_os = "cuda"`. Modules we don't patch (net, process, env, thread) fall through to existing `unsupported` stubs.

**Confidence**: high

### Q: Is -Zbuild-std a viable fallback if sysroot build fails?
A: **-Zbuild-std is the PREFERRED path**, not a fallback. It works perfectly and avoids the complexity of modifying the nvptx64 target spec to `"std": true`. Key advantages:
1. No need to rebuild the entire toolchain
2. Patched std source just needs to be available on disk
3. Each kernel crate can independently choose `build-std = ["std"]` vs `build-std = ["core"]`
4. Build time is fast (~29s including std compilation)

To use: set `__CARGO_TESTS_ONLY_SRC_ROOT` (or the stable equivalent `RUST_LIB_SRC` once stabilized) to point to `patched-rustc/library/`.

**Confidence**: high

### Q: PTX output characteristics?
A: The compiled PTX has:
- **15 kernel entries** — all original test kernels compile to valid PTX
- **51,651 lines / 1.9 MB** — significant size due to std's fmt machinery
- **30 `.ptr .align` occurrences** — all on kernel entry parameters (known LLVM issue)
- **0 `panic_const` references** — panic_abort profile correctly eliminates these
- **PTX version 7.1, target sm_86**

**Confidence**: high

## Unexpected Discoveries
- `panic_const` issue is completely gone when using `panic_abort` profile. This was previously a major concern but doesn't manifest with proper panic strategy.
- PTX size is large (1.9 MB) but this is for 15 kernels with full std support. Per-kernel PTX should be smaller.
- `__CARGO_TESTS_ONLY_SRC_ROOT` env var works for overriding std source location. This is an internal cargo feature but reliable.

## Open Questions
- Can we make the `-Zbuild-std` path seamless in the build system? (e.g., build.rs sets env var automatically)
- Should we pursue `"std": true` in the target spec for a cleaner sysroot experience, or is `-Zbuild-std` sufficient?
- What is the per-kernel PTX size? 1.9 MB is for 15 kernels — need to measure individual kernel PTX.

## Impact on Downstream Tasks
- **std-sysroot-build.2**: Partially bypassed — no compilation errors to fix! The -Zbuild-std path works out of the box.
- **std-sysroot-build.3**: Unblocked — can immediately test kernel with `use std::fs::File`
- **std-sysroot epic criterion 1**: PATH A (x.py sysroot) not met, but PATH B (-Zbuild-std) fully works
- **ptx-codegen-fix.1**: Confirmed 30 `.ptr .align` instances in std-enabled PTX
