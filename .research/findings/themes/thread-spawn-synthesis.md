# thread-spawn: thread::spawn() maps to warp
**Epic**: std-thread-gpu | **Status**: completed | **Updated**: 2026-06-04

## Progress
- gpu_runtime::thread module fully implemented: gpu_main(), spawn(), JoinHandle, cooperative()
- Test kernels verified: thread_spawn_test (42, 99), thread_reuse_test (10+20+30+40=100)
- C-FFI entry points for std patches: gpu_thread_spawn_raw, gpu_thread_join_warp, etc.
- std-patches/sys_thread_cuda.rs created for std::thread integration
- cooperative() wakes all warps for data-parallel execution

## Verified Conclusions
- Warp-as-thread model works: 1 warp = 1 logical thread, warp 0 = main
- Closure data must go to global memory (SCRATCH buffers), NOT local stack
- Cooperative closures CANNOT capture local variables (per-warp memory isolation)
- Data passed via global atomics for cooperative mode

## Rejected Approaches
- Closure captures for cooperative: GPU local memory per-warp isolation causes ILLEGAL_ADDRESS
- std::thread::spawn via patched std: hangs at bar.sync + std init interaction (unresolved)

## Open Questions
- std::thread::spawn hang: bar.sync interacts with std initialization. Needs investigation.

## Key Metrics
- thread_spawn_test: spawn 2 threads, join → [42, 99, 3, 0] ✓
- thread_reuse_test: spawn 4 on 3 warps → [10, 20, 30, 40, 100] ✓
- cooperative_debug: 4 warps → [100, 101, 102, 103] ✓

## Next Steps
- Fix std::thread::spawn hang for full std integration (tracked under std-thread-gpu epic)
