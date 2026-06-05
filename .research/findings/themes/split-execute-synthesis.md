# split-execute — Execute kernel crate split

**Epic**: kernel-split (T0)
**Status**: active

## Completed tasks

### split-execute.1 — Extract stdio to gpu-runtime (c614)
Moved stdio statics + functions to gpu-runtime. Force-link via #[used] confirmed.

### split-execute.2 — Create gpu-kernel-core crate (c615)
Created gpu-kernel-core with helpers (pub), basic, compute_math. gpu-kernel-std
depends on it via extern crate. All 181 PTX kernel symbols preserved.

### split-execute.3 — Create gpu-kernel-compute crate (c616)
Moved 9 compute_* modules to gpu-kernel-compute (cdylib only). 84 entry points.
Removed dead stdio_auto_init and gpu-libc dep. sm_80 feature gate preserved.

## Patterns established
- Cross-crate helper access: `pub mod helpers` in core, `use gpu_kernel_core::helpers::*` elsewhere
- Kernel symbol linking: `extern crate gpu_kernel_core;` in cdylib crates
- Each kernel crate needs: #![no_main], dynamic_smem global_asm, stdio force-link statics
- Compute-only crates skip stdio_auto_init and gpu-libc (no hostcall I/O needed)
