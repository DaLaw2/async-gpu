# showcase-examples.2: Clean up warp-cooperative as cooperative compute showcase

## Summary

Transformed the warp-cooperative example from a two-crate MIR pass verification test into a single-crate cooperative compute showcase with 5 demos. Added cubin loading support to `CustomLaunchBuilder` to avoid 10+ minute PTX JIT compilation. Fixed `gpu_main` -> `gpu_main_poll` in thread_test.rs kernel entry points to prevent bar.sync deadlocks in std-compiled kernels.

## Findings

### Q: Can the existing warp-cooperative example serve as a cooperative compute showcase?
**A (confidence: 10/10):** No. The old example was a raw no_std kernel crate testing compiler MIR pass behavior (shfl.sync, bar.warp.sync). It had nothing to do with the `gpu_runtime::thread::cooperative()` API family. Complete restructuring was needed.

### Q: Are there existing kernel entry points for cooperative compute APIs?
**A (confidence: 10/10):** Yes, in `gpu-kernel-std/src/thread_test.rs`: `cooperative_compute_test`, `cooperative_map_test`, `cooperative_reduce_test`, `cooperative_map_ext_test`, `cooperative_matmul_test`. All are compiled into the unified PTX/cubin.

### Q: Why do showcase examples using `gpu::custom()` appear to hang?
**A (confidence: 10/10):** Not a deadlock — it's PTX JIT compilation. The unified kernel PTX is 9.5MB and takes 10+ minutes to JIT compile. The pre-compiled cubin (148MB) loads in <1 second but `gpu::custom().prepare()` had no cubin support. Fixed by adding `.cubin_file()` method.

### Q: Do the thread_test.rs kernels work correctly with the current cubin?
**A (confidence: 10/10):** Yes, all 5 cooperative kernels pass with the current cubin. The kernels use `gpu_main()` with `bar.sync`, which works because the cubin was compiled before the std migration introduced warp divergence. For future PTX rebuilds, the kernels have been switched to `gpu_main_poll()` (polling-based, no bar.sync).

## Unexpected Discoveries

1. **All `gpu::custom()` examples silently suffer from 10+ minute startup** — the structured-concurrency and gpu-channels examples also trigger PTX JIT. This was masked because verification was done with sufficient patience. The `.cubin_file()` API added here benefits all showcase examples.

2. **`gpu_main()` vs `gpu_main_poll()` is critical for std-compiled kernels** — The bar.sync in `gpu_main()` can deadlock when std initialization causes warp divergence before the barrier. All thread_test.rs kernels used `gpu_main()` and would deadlock after a PTX rebuild. Fixed proactively.

3. **cudarc's `Ptx::from_file()` can load cubin files** — `cuModuleLoad` (the underlying CUDA API) handles both PTX and cubin files from disk, making cubin integration trivial via cudarc's existing API.

## Open Questions

1. Should the structured-concurrency and gpu-channels examples also use `.cubin_file()` for fast startup? (Likely yes — same 10+ minute JIT issue.)
2. The sc_demo.rs kernels also use `gpu_main()` (not `gpu_main_poll`) — should those be migrated too? They will deadlock after the next PTX rebuild.

## Impact on Downstream Tasks

- Unblocks showcase-examples theme: cooperative compute now has a clean showcase.
- The `.cubin_file()` API enables fast-loading for all future showcase examples.
- The `gpu_main` -> `gpu_main_poll` fix in thread_test.rs prevents future deadlocks after kernel rebuilds.
