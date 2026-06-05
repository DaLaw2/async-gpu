# split-execute — Execute kernel crate split

**Epic**: kernel-split (T0)
**Status**: active

## Completed tasks

### split-execute.1 — Extract stdio to gpu-runtime (c614)
Moved stdio statics + functions to gpu-runtime. Force-link via #[used] confirmed.

### split-execute.2 — Create gpu-kernel-core crate (c615)
Created gpu-kernel-core with helpers (pub), basic, compute_math. gpu-kernel-std
depends on it and imports helpers via gpu_kernel_core::helpers::*. Removed
helpers.rs/basic.rs/compute_math.rs from gpu-kernel-std. All 181 PTX kernel
symbols preserved. extern crate force-links rlib into cdylib.

## Patterns established
- Cross-crate helper access: `pub` functions + `pub mod helpers` in core crate
- Kernel symbol linking: `extern crate gpu_kernel_core;` in cdylib crates
- Each kernel crate needs: #![no_main], dynamic_smem global_asm, stdio force-link statics
