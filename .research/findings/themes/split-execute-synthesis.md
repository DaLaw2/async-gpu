# split-execute — Execute kernel crate split

**Epic**: kernel-split (T0)
**Status**: completed

## Completed tasks (5/5)

| Task | What | Crate |
|------|------|-------|
| .1 | Extract stdio to gpu-runtime | gpu-runtime |
| .2 | Create gpu-kernel-core (helpers, basic, compute_math) | gpu-kernel-core |
| .3 | Create gpu-kernel-compute (9 compute modules, 84 entries) | gpu-kernel-compute |
| .4 | Create gpu-kernel-io (hostcall, hybrid, pipeline) | gpu-kernel-io |
| .5 | Rename gpu-kernel-std -> gpu-kernel-test (test/demo) | gpu-kernel-test |

## Final layout
```
crates/kernel/
  gpu-kernel-core/     rlib+cdylib  shared helpers, basic kernels
  gpu-kernel-compute/  cdylib       fused/physics/transformer/persistent compute
  gpu-kernel-io/       cdylib       hostcall, hybrid executor, async pipeline
  gpu-kernel-test/     cdylib       test/demo kernels (std-based, highest churn)
```

## Patterns
- Cross-crate helpers: `pub mod helpers` in core, `extern crate gpu_kernel_core;` in cdylib crates
- Compute-only crates skip stdio_auto_init and gpu-libc (no hostcall I/O)
- PTX/cubin filenames kept as kernel_std.ptx/cubin for backward compat
