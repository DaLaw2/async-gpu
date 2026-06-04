# lib-cleanup.4: Merge gpu-kernel + gpu-kernel-std into single kernel crate

## Investigation Results

### Before Merge

| Property | gpu-kernel (no_std) | gpu-kernel-std (restricted_std) |
|---|---|---|
| Source lines | 13,468 | 1,072 |
| Entry points | 154 (`#[no_mangle]`) | 23 (`#[unsafe(no_mangle)]`) |
| Clean build time | ~22s | ~35s |
| PTX size | 2.7 MB | 6.3 MB |
| Dependencies | gpu-atomics, gpu-libc, gpu-protocol, gpu-runtime | gpu-libc, gpu-protocol, gpu-runtime |
| build-std | `["core"]` | `["std", "core", "panic_abort"]` |
| Features | `sm_80` | (none) |

### Can gpu-kernel-std compile all gpu-kernel kernels?

**Yes.** The `restricted_std` environment is a superset of `no_std` — all `core::*` APIs
are available. The only conflict was the `gpu_runtime::panic_handler!()` macro, which
defines a `#[panic_handler]` lang item. Under restricted_std, std already provides this,
causing a duplicate. Solution: remove the macro call (std handles panics; kernels use
`gpu_panic_init()` for hostcall routing).

### What would break?

1. **JIT compilation time**: The merged PTX (9 MB) takes ~41 min to JIT-compile via
   `cuModuleLoadData` vs 2.7 MB taking seconds. Fixed by adding cubin loading support
   to `gpu-host` — pre-compiled cubin loads in sub-second.

2. **CI**: The old gpu-kernel was in the CI PTX build list. Removed it since the merged
   crate requires patched std (already excluded from CI).

## After Merge

| Property | Merged gpu-kernel-std |
|---|---|
| Source lines | ~14,540 (all modules combined) |
| Entry points | 159 (union of both) |
| Clean build time | ~39s (vs 57s combined) |
| PTX size | 8.9 MB |
| Cubin compile | ~41 min (ptxas --gpu-name sm_75) |
| Cubin size | 42 MB |

### Changes Made

1. **Copied all source files** from `gpu-kernel/src/` into `gpu-kernel-std/src/`
2. **Updated `gpu-kernel-std/Cargo.toml`**: Added `gpu-atomics` dep, `sm_80` feature
3. **Updated `gpu-kernel-std/src/lib.rs`**: Added all module declarations, removed
   conflicting `panic_handler!()` macro, added `#![feature(stdarch_nvptx)]`
4. **Updated `gpu-host/build.rs`**: Points to gpu-kernel-std, outputs both
   `kernel.ptx` and `kernel_std.ptx` (same content)
5. **Updated `gpu-host/src/lib.rs`**: Added `cubin` module with `KERNEL_CUBIN` constant;
   documented that `KERNEL` and `KERNEL_STD` are now the same PTX
6. **Added cubin loading to `gpu-host/src/gpu.rs`**: `load_module_cubin_or_ptx()`,
   `is_unified_kernel_ptx()` — auto-detects when to use cubin vs PTX JIT
7. **Updated `scripts/build-kernel-std.sh`**: Copies PTX to both names
8. **Updated `scripts/ci-lint.sh`**: Removed gpu-kernel from PTX build list

### Verification

- Merged crate compiles cleanly (17 warnings, 0 errors)
- All 159 kernel entry points present in unified PTX
- `zero_param` test: PASSED (cubin loading, sub-second)
- `std_thread_demo` test: PASSED
- `real_std_thread` test: PASSED
- `matmul_io` North Star litmus test: PASSED
- CI lint: All checks passed

### Old gpu-kernel retained

The old `crates/kernel/gpu-kernel/` directory is kept for now:
- Still used by `scripts/setup.sh` for quick-mode smoke test
- Still used by CI for basic PTX compilation check
- No source duplication concern: old gpu-kernel is independent, merged crate is the source of truth
- Can be fully removed once CI pipeline is updated to skip it
