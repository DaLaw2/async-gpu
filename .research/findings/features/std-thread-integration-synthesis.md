# std-thread-integration: std::thread::spawn via patched std
**Epic**: std-thread-gpu | **Status**: completed | **Updated**: 2026-06-04

## Progress
- Root cause identified: SIMT multi-lane execution of main_fn + trampoline
- Lane-0 guard added to gpu_main() and gpu_main_poll()
- Lane masking added to sys_thread_cuda.rs thread_trampoline
- kernel_std.ptx rebuilt with patched sysroot, pre-compiled to cubin
- All tests pass: smoke_test, println_smoke, pool_smoke, real_std_thread, std_thread_demo

## Verified Conclusions
- std::thread::spawn routes through cuda.rs → gpu_thread_spawn_raw (sysroot patched)
- Lane-0 guard is REQUIRED — main_fn must NOT execute on all 32 lanes (heap alloc, spawn)
- cubin pre-compilation essential for kernel_std (6MB PTX → 37MB cubin, <1s load)
- Thread 1 sum(0..10)=45, Thread 2 5!=120, Combined=165 — all correct
- println! works from spawned GPU threads via hostcall

## Rejected Approaches
- bar.sync was NOT the cause (kernel_std.ptx has zero bar.sync)
- Stock std was NOT the current cause (sysroot already patched)

## Open Questions
- Should kernel_std.cubin be committed or generated at build time?

## Key Metrics
- real_std_thread_spawn: PASSED (45 + 120 = 165)
- std_thread_spawn_demo: PASSED (was hanging, now works)
- kernel_std.cubin: 37MB, loads <1s (vs >10min PTX JIT)

## Next Steps
- Epic Verification Gate for std-thread-gpu
- Clean up debug counter (SPAWN_RAW_COUNT)
