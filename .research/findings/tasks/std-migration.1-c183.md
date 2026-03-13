# std-migration.1: Create gpu-kernel-std crate + verify println/Vec/format work
**Cycle**: 183 | **Theme**: std-migration | **Kind**: experiment | **Status**: done

## Summary
Created gpu-kernel-std crate and updated patched-std to nightly-2026-03-11. Build succeeds,
producing PTX with 3 visible kernel entry points using real Rust std.

## Findings

### Q: Can a new crate build with -Zbuild-std=std and produce valid PTX?
A: YES. After updating patched-std/library/ to nightly-2026-03-11 and re-applying CUDA patches,
gpu-kernel-std builds successfully with `-Zbuild-std=std,core,panic_abort`. PTX output is ~1MB
with 3 `.visible .entry` kernels: `std_println_test`, `std_vec_format_test`, `std_alloc_stress_test`.

**Patches applied to fresh nightly source:**
1. `sys/alloc/cuda.rs` — slab+bitmap allocator (8 size classes, 16B-4096B)
2. `sys/alloc/mod.rs` — `target_os = "cuda"` in cfg_select + nvptx64 in MIN_ALIGN
3. `sys/stdio/cuda.rs` — routes stdout/stderr through extern `gpu_stdout_write`
4. `sys/stdio/mod.rs` — `target_os = "cuda"` in cfg_select
5. `sys/io/error/mod.rs` — cuda added to generic error group
6. `sys/random/mod.rs` — cuda added to unsupported random group + hashmap_random_keys exclusion
7. `sys/thread_local/mod.rs` — cuda added to `no_threads` group + guard no-op group
8. `io/stdio.rs` — OnceLock bypass for `target_arch = "nvptx64"` in `_print()` and `_eprint()`

**Confidence**: high (build succeeds, PTX emitted)

### Q: Do println!, Vec, String, format! all work without duplicating hostcall code?
A: The crate uses `gpu_runtime::hostcall::gpu_hostcall_print()` for the PAL `gpu_stdout_write()`
function, eliminating the 430+ lines of duplicated inline PTX from std-build-test. Needs
end-to-end GPU execution test to confirm runtime behavior.

**Confidence**: medium (PTX compiles but runtime not yet tested)

### Q: What is the PTX size difference vs no_std kernel?
A: gpu-kernel-std.s is ~1MB (1,067,842 bytes). For comparison, gpu-kernel.s (no_std, 65+ kernels)
is typically ~200-400KB. The std overhead is significant (~600KB+) due to formatting machinery,
allocator, and panic infrastructure.

**Confidence**: high

## Unexpected Discoveries
- nightly-2026-03-11 requires 8 separate patches (vs 4-5 expected). New requirements:
  `sys/io/error/mod.rs` (no catch-all), `sys/random/mod.rs` (no catch-all for fill_bytes +
  hashmap_random_keys exclusion), `sys/thread_local/mod.rs` (no_threads group + guard no-op).
- The `guard::key` submodule depends on `thread_local::key::{LazyKey, set}` which is empty
  for unknown targets — required explicit cuda entry in guard's cfg_select.

## Impact on Downstream Tasks
- std-migration.2 unblocked (shared hostcall protocol design)
- std-fs.3 unblocked (OnceLock fix for stdin)
- std-migration.3 still needs gpu-error-propagation.2
