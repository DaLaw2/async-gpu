# Feature Synthesis: panic-std-intercept
## Progress
- [x] panic-std-intercept.1: Investigation — full panic path traced (DONE)
- [ ] Remaining tasks not yet defined

## Verified Conclusions
1. The std panic path IS functional: panic!() → panic_handler → default_hook → Stderr → gpu_stdout_write → SERVICE_PRINT hostcall → host stdout → process::abort() → trap;
2. Thread-local panic_count works correctly via ADR-17 gpu_tid()-indexed TLS (1024-slot array)
3. current_os_id() returns warp index; with_current_name() returns thread name or "<unnamed>"
4. Panic output format: `thread '<name>' (<warp_id>) panicked at <location>:\n<msg>`
5. Three things missing vs no_std path: SERVICE_PANIC hostcall, set_warp_trapped(), write_panic_to_result()
6. Injection point for GPU metadata: default_hook lines 262-273 in patched-std/src/panicking.rs

## Rejected Approaches: None yet
## Open Questions
1. Should std panics use SERVICE_PANIC (structured) or keep SERVICE_PRINT (plain text)?
2. Where to inject set_warp_trapped() — custom abort_internal() vs panic hook?
3. How to surface blockIdx (current_os_id only returns warp index)?
4. no_threads RwLock on HOOK static is technically unsound with multi-warp — priority?

## Key Metrics
- Files in panic path: 7 (panicking.rs, stdio/cuda.rs, thread/cuda.rs, gpu_threads.rs, panic_abort/lib.rs, unsupported/common.rs, backtrace.rs)
- Hostcall round-trips per panic: 1+ (one SERVICE_PRINT per 56-byte chunk of formatted message)

## Next Steps
1. Design: decide SERVICE_PANIC vs SERVICE_PRINT for std panics
2. Impl: add blockIdx/threadIdx/laneId to default_hook + hook set_warp_trapped()/write_panic_to_result() into abort path
3. Test: verify panic output matches CPU format with GPU metadata
