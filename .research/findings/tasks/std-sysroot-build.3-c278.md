# std-sysroot-build.3: Experiment — build test kernel with use std::fs::File, verify PTX
**Cycle**: 278 | **Theme**: std-sysroot-build | **Kind**: experiment | **Status**: done

## Summary
Added two new test kernels to std-build-test that use `std::fs::File`:
- `std_file_write_kernel`: `File::create("gpu_test_output.txt")` + `write_all(data)`
- `std_file_read_kernel`: `File::open("gpu_test_input.txt")` + `read(&mut buf)`

Both compile successfully to valid PTX via `-Zbuild-std=["std"]` with the patched compiler and patched std source. Total: 17 kernel entries in PTX output.

## Findings

### Q: Does `use std::fs::File` compile to PTX?
A: **Yes.** Both `File::create` + `write_all` and `File::open` + `read` compile to valid PTX without any errors. The compilation chain is:
1. `std::fs::File::create()` → patched `sys_fs_cuda.rs` `File::create()`
2. `File::create()` → `OpenOptions::new().write(true).create(true).truncate(true).open(path)`
3. `open()` → `extern "C" fn open()` from gpu-libc
4. gpu-libc `open()` → hostcall OPEN service

The patched PAL correctly routes all filesystem operations through the hostcall framework.

**Confidence**: high

### Q: PTX output characteristics?
A: Build with 17 kernels produces valid PTX. The File kernels are visible entries:
```ptx
.visible .entry std_file_read_kernel(
.visible .entry std_file_write_kernel(
```

Build time: ~7 seconds (incremental, only std-build-test recompiled).

**Confidence**: high

## Impact on Downstream Tasks
- **std-sysroot-build.4**: Unblocked — can now do end-to-end test (load PTX, run kernel on GPU, verify file I/O)
- **std-sysroot epic criterion 2**: MET — "GPU kernel with `use std::fs::File` compiles to valid PTX"
