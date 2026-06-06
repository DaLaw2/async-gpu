# Feature Synthesis: panic-std-intercept
## Progress
- [x] panic-std-intercept.1: Investigation — full panic path traced (DONE)
- [x] panic-std-intercept.2: Experiment — GPU metadata in default_hook (DONE)
- [ ] panic-std-intercept.3: Hook set_warp_trapped/write_panic_to_result into abort path

## Verified Conclusions
1. The std panic path IS functional: panic!() → default_hook → Stderr → gpu_stdout_write → SERVICE_PRINT → host stdout → abort → trap
2. GPU panic output now includes block/warp/lane: `thread 'main' (block 0, warp 0, lane 3) panicked at ...`
3. Inline PTX asm (`%ctaid.x`, `%tid.x`) works directly in panicking.rs behind `cfg(target_os = "cuda")`
4. Abort path still missing set_warp_trapped() and write_panic_to_result() — needs extern C bridge from gpu-runtime
5. build-toolchain.sh missing copy_new for sys_thread_cuda.rs (pre-existing)

## Rejected Approaches: None
## Open Questions
1. Should std panics use SERVICE_PANIC (structured) or keep SERVICE_PRINT (plain text)?
2. Where to inject set_warp_trapped() — CUDA-specific abort_internal() vs panic hook?
3. no_threads RwLock on HOOK static is technically unsound with multi-warp — priority?

## Key Metrics
- Files in panic path: 7 (panicking.rs, stdio/cuda.rs, thread/cuda.rs, gpu_threads.rs, panic_abort/lib.rs, unsupported/common.rs, backtrace.rs)
- Hostcall round-trips per panic: 1+ (one SERVICE_PRINT per 56-byte chunk)

## Next Steps
1. Expose set_warp_trapped/write_panic_to_result as extern C from gpu-runtime
2. Hook them into std abort path (CUDA-specific abort_internal or pre-abort hook)
3. Test: verify full panic flow with kernel result buffer + warp status
