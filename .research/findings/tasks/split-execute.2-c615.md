# split-execute.2: Create gpu-kernel-core crate

**Status**: done
**Cycle**: 615

## What was done

Created `crates/kernel/gpu-kernel-core/` — the shared library crate containing helpers,
basic asm/atomic kernels, and compute_math kernels. Other kernel crates depend on this
for shared helpers via `gpu_kernel_core::helpers::*`.

### Crate structure
- `Cargo.toml`: crate-type = ["rlib", "cdylib"], depends on gpu-atomics, gpu-libc, gpu-protocol, gpu-runtime
- `.cargo/config.toml`: nvptx64 target, sm_75, build-std
- `src/lib.rs`: #![no_main], restricted_std, abi_gpu_kernel, dynamic_smem global_asm,
  pub mod helpers, mod basic, mod compute_math, stdio force-link statics, pub stdio_auto_init()
- `src/helpers.rs`: all functions changed from pub(crate) to pub
- `src/basic.rs`: copied from gpu-kernel-std (asm/atomic test kernels)
- `src/compute_math.rs`: copied from gpu-kernel-std (f32 math validation kernel)

### gpu-kernel-std changes
- Added dependency: `gpu-kernel-core = { path = "../gpu-kernel-core" }`
- Removed `mod helpers;`, `mod basic;`, `mod compute_math;` — replaced with `extern crate gpu_kernel_core;`
- Updated 8 files to use `gpu_kernel_core::helpers::*` instead of `crate::helpers::*`:
  compute_fused, compute_mma, compute_gemm, compute_cnn, compute_search,
  compute_transformer, pipeline, hostcall_kernels
- Updated 5 inline `crate::helpers::` path references in compute_cnn and compute_transformer
- Deleted helpers.rs, basic.rs, compute_math.rs from gpu-kernel-std

## Verification
- `cargo build --release -p gpu-kernel-core` (from crate dir): OK, produces PTX with 17 kernel entries
- `cargo build --release -p gpu-kernel-std` (from crate dir): OK, produces PTX with 181 kernel entries
  (same count as before — basic/compute_math kernels linked in via extern crate)
- `cargo +stable fmt --check` for both crates: OK
- `scripts/ci-lint.sh`: all checks passed

## Key decisions
- Made helpers.rs functions `pub` (not `pub(crate)`) for cross-crate access
- Used `extern crate gpu_kernel_core;` in gpu-kernel-std lib.rs to force-link kernel
  symbols from the rlib into the cdylib PTX output
- Made `stdio_auto_init()` pub in gpu-kernel-core so other kernel crates can use it
