# split-design.1: Kernel-to-Crate Dependency Map & Verification

**Task**: Map kernel→crate assignments, verify no circular dependencies, count entry points, assess risks.
**Date**: 2026-06-05 | **Cycle**: 612

## Summary

The proposed 4-crate split is **dependency-safe** with **zero circular dependencies**. All cross-file dependencies flow strictly downward: `core ← compute`, `core ← io`, `core ← test`. The one lateral dependency (`pipeline.rs → hybrid.rs`) stays within the same proposed crate (gpu-kernel-io). The `dynamic_smem` global_asm declaration must be duplicated in each crate that uses shared memory, and lib.rs infrastructure (stdio_auto_init, gpu_stdout_write, etc.) must be extracted into gpu-kernel-core.

## 1. Dependency Matrix

### Cross-file `use crate::` imports

| Source File | Imports From | Specific Items |
|---|---|---|
| compute_math.rs | helpers.rs | `gpu_sqrtf` |
| compute_cnn.rs | helpers.rs | `gpu_exp_f32`, `gpu_sqrtf` (via `crate::helpers::gpu_sqrtf` inline) |
| compute_fused.rs | helpers.rs | `bar_sync`, `get_dynamic_smem_ptr`, `gpu_exp_f32` |
| compute_gemm.rs | helpers.rs | `bar_sync`, `get_dynamic_smem_ptr`, `gpu_exp_f32` |
| compute_transformer.rs | helpers.rs | `bar_sync`, `get_dynamic_smem_ptr`, `gpu_exp_f32`, `gpu_sqrtf` |
| compute_mma.rs | helpers.rs | `bar_sync`, `get_dynamic_smem_ptr` |
| compute_search.rs | helpers.rs | `gpu_sqrtf` |
| hostcall_kernels.rs | helpers.rs | `gpu_hostcall_close`, `gpu_hostcall_open`, `gpu_hostcall_read`, `gpu_hostcall_stdin_read`, `gpu_hostcall_time`, `gpu_hostcall_write`, `gpu_instant_nanos`, `hc_pop_free_counted`, `hc_pop_free_counted_v2` |
| pipeline.rs | helpers.rs | `gpu_hostcall_close`, `gpu_hostcall_open`, `grep_buffer` |
| pipeline.rs | hybrid.rs | `hybrid_warp_print_init`, `hybrid_warp_wait` |
| compute_demo.rs | *(none)* | — |
| compute_persistent.rs | *(none)* | — |
| compute_physics.rs | *(none)* | — |
| hybrid.rs | *(none)* | — |
| warp.rs | *(none)* | — |
| thread_test.rs | *(none)* | — |
| sc_demo.rs | *(none)* | — |
| par_iter_demo.rs | *(none)* | — |
| basic.rs | *(none)* | — |

### Dependency direction under proposed split

```
gpu-kernel-core (helpers.rs, basic.rs, compute_math.rs)
    ↑               ↑               ↑
    |               |               |
gpu-kernel-compute  gpu-kernel-io   gpu-kernel-test
(uses helpers)      (uses helpers   (uses nothing from
                    + hybrid→pipeline  other kernel crates)
                    is internal)
```

**No circular dependencies exist.** All arrows point upward to core.

## 2. Shared Infrastructure in lib.rs

### Infrastructure functions (NOT kernel entry points)
These are `#[no_mangle]` but are called by the std PAL, not launched as kernels:
- `gpu_stdout_write()` — called by std Stdout::write
- `gpu_stdin_read()` — called by std Stdin::read
- `stdio_print_buffer_init()` — print buffer init
- `gpu_print_buffer_flush()` — print buffer flush

### Internal helpers (private, not `#[no_mangle]`)
- `stdio_init()` — sets STDIO_HOSTCALL_BUF
- `stdio_auto_init()` — reads __HOSTCALL_BUF device global, inits stdio+panic+libc
- `get_tid()` — inline PTX for thread ID
- `matmul_callback()` — callback for cooperative_map
- `matmul_io_inner()` — matmul implementation
- `std_pipeline_inner()` — pipeline implementation

### Global state
- `STDIO_HOSTCALL_BUF: AtomicU64` — hostcall pointer for stdio
- `STDIO_SIDEBAND_PTR: AtomicU64` — sideband pointer
- `STDIO_PRINT_BUF_READY: AtomicU32` — print buffer ready flag

### Global assembly
- `core::arch::global_asm!(".extern .shared .align 4 .b8 dynamic_smem[];")` — dynamic shared memory symbol

### `mod` declarations
All 18 submodules declared in lib.rs, plus features: `#[cfg(feature = "sm_80")] mod compute_mma`.

### Extraction requirement
`stdio_auto_init`, `gpu_stdout_write`, `gpu_stdin_read`, `stdio_print_buffer_init`, `gpu_print_buffer_flush`, and the three static atomics MUST move to **gpu-kernel-core** because:
1. They are used by 22 kernel entry points in lib.rs
2. Any crate with kernels that call `println!()` or `stdio_auto_init()` needs them
3. The `dynamic_smem` global_asm must be emitted in EVERY crate that uses `get_dynamic_smem_ptr()`

## 3. Entry Point Counts per Proposed Crate

| Proposed Crate | Files | Entry Points | Lines |
|---|---|---|---|
| **gpu-kernel-core** | helpers.rs (387), basic.rs (266), compute_math.rs (54), + lib.rs infrastructure (~174) | 18 (basic: 17, compute_math: 1) + 4 infra no_mangle | ~881 |
| **gpu-kernel-compute** | compute_gemm.rs (2609), compute_transformer.rs (2288), compute_cnn.rs (854), compute_search.rs (1142), compute_fused.rs (285), compute_mma.rs (227), compute_physics.rs (205), compute_demo.rs (235), compute_persistent.rs (183) | 81 (gemm:18, transformer:29, cnn:17, search:2, fused:2, mma:3, physics:3, demo:6, persistent:1) | ~8,028 |
| **gpu-kernel-io** | hostcall_kernels.rs (2202), pipeline.rs (1080), hybrid.rs (458) | 38 (hostcall:32, pipeline:4, hybrid:2) | ~3,740 |
| **gpu-kernel-test** | lib.rs test kernels (~1031), warp.rs (920), thread_test.rs (396), sc_demo.rs (972), par_iter_demo.rs (517) | 61 (lib.rs:22, warp:13, thread_test:11, sc_demo:6, par_iter:9) | ~3,836 |

**Total**: 198 entry points (173 `#[no_mangle]` + some overlap from unsafe(no_mangle) variant).

## 4. External Dependency Audit

| Proposed Crate | gpu-runtime | gpu-atomics | gpu-protocol | gpu-libc |
|---|---|---|---|---|
| **gpu-kernel-core** | YES (hostcall, panic, entry, print_buffer, thread, index, prelude) | YES (sys_cas_u64, sys_load_acquire_u64) | YES (*) | YES (gpu_libc_io_init, malloc, gpu_heap_init) |
| **gpu-kernel-compute** | YES (index, math, warp, block, nn, warp_future) | YES (membar_sys) | YES (*) | NO |
| **gpu-kernel-io** | YES (hostcall, warp_future, sideband) | YES (sys_fetch_add_u64, membar_sys, activemask, sys_spin_load_acquire_u32, sys_store_release_u32) | YES (*) | NO |
| **gpu-kernel-test** | YES (thread, scope, block_channel, par_iter, channel, index, std_future) | NO | NO (only via gpu_protocol in warp.rs) | NO |

All four external crates are already published as path dependencies. No issues.

## 5. Risk Assessment

### RISK 1 (Medium): `dynamic_smem` global_asm duplication
The `core::arch::global_asm!(".extern .shared .align 4 .b8 dynamic_smem[];")` in lib.rs is needed by any crate calling `get_dynamic_smem_ptr()`. Affected: gpu-kernel-core (helpers.rs), gpu-kernel-compute (via helpers). Each new crate's lib.rs MUST emit this declaration.

### RISK 2 (Medium): lib.rs infrastructure extraction
`stdio_auto_init()` and friends (gpu_stdout_write, gpu_stdin_read, etc.) are used by test kernels in lib.rs AND could be needed by any crate doing println!. These must become public API in gpu-kernel-core. The `#[no_mangle]` functions `gpu_stdout_write` and `gpu_stdin_read` are PAL callbacks — they must be defined exactly once in the final linked binary. If each crate is a separate cdylib (separate PTX), this is fine — each gets its own copy. If they're linked together, there would be symbol conflicts.

### RISK 3 (Low): `pipeline.rs → hybrid.rs` dependency
Both are in the same proposed crate (gpu-kernel-io), so this is a non-issue.

### RISK 4 (Low): Feature flag `sm_80` for compute_mma.rs
Only gpu-kernel-compute needs this feature flag. Clean separation.

### RISK 5 (Low): `warp.rs` uses `gpu_protocol` directly (not via helpers)
`warp.rs` does `use gpu_protocol;` at the top level for `NULL_INDEX` and packet layout constants. Under the proposed split (gpu-kernel-test), it would need gpu-protocol as a direct dependency. This is trivial to add.

### RISK 6 (None): No file straddles crate boundaries
Every source file maps cleanly to exactly one proposed crate. No file needs to be split.

## 6. Open Questions

1. **Crate type**: Should gpu-kernel-core be `rlib + cdylib` (allowing dependent crates to link against it) or `cdylib` only? If cdylib only, dependent crates can't `use gpu_kernel_core::helpers::bar_sync` — they'd need to copy the helpers or use a separate `-sys` crate.

2. **Build system**: Each cdylib crate produces its own PTX/cubin. The host loader needs to know which cubin contains which kernel. Does the loader already support multi-cubin?

3. **Incremental build measurement**: What's the current full build time? Need a baseline to measure improvement.
