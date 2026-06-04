# std-thread-integration.2: Fix hang + verify std::thread::spawn

## Summary
Fixed the std::thread::spawn hang and verified real std::thread::spawn works on GPU with println!. The fix required three components: (1) the sysroot was already patched with cuda.rs routing, (2) gpu_main/gpu_main_poll needed lane-0-only guards for main_fn execution (SIMT safety), (3) sys_thread_cuda.rs needed lane masking in the trampoline. After rebuilding kernel_std.ptx and pre-compiling to cubin with ptxas, all tests pass.

## Findings

**Q: What was the actual root cause of the hang?**
Confidence: 95%
Multiple factors combined: (1) `gpu_main_poll` ran `main_fn()` on ALL 32 lanes of warp 0 — each lane independently called std::thread::spawn, creating 32× Box allocations and 32× warp assignments instead of 1. (2) The thread trampoline also ran on all 32 lanes of the assigned warp, causing duplicate Box::from_raw and use-after-free. (3) The previous PTX was built with stock std (before sysroot was patched), so std::thread::spawn routed to unsupported::Thread::new → panic.

**Q: Was the sysroot already patched?**
Confidence: 100%
YES. The sysroot at nightly-2026-06-03 already contains `std/src/sys/thread/cuda.rs`. The `thread/mod.rs` has `cfg_select! { target_os = "cuda" => { mod cuda; ... } }`. The investigation from .1 was incorrect — it tested for the file but may have run before the sysroot was patched (or a different toolchain).

**Q: What was the lane-0 fix?**
Confidence: 100%
In `gpu_main_poll`, `main_fn()` was moved inside the `if lane_id() == 0` block. Same for `gpu_main`. This ensures only one lane executes the user's main function, preventing SIMT-unsafe operations like heap allocation and std::thread::spawn from running 32× in parallel.

**Q: Does the cubin pre-compilation matter?**
Confidence: 100%
YES, critically. kernel_std.ptx is 6MB/160K lines. PTX JIT compilation takes >10 minutes on GTX 1660. Pre-compiling with `ptxas --gpu-name sm_75` produces a 37MB cubin that loads in <1 second.

## Unexpected Discoveries
- The sysroot WAS already patched — the .1 investigation's conclusion was wrong on this point
- println! works from spawned GPU threads via hostcall — each warp gets its own print output
- The ONLY_TEST env var (not CLI args) controls test selection in gpu-host

## Open Questions
- Should kernel_std.cubin be checked into the repo or generated at build time?
- The debug counter (SPAWN_RAW_COUNT) should be removed before shipping

## Impact on Downstream Tasks
- std-thread-gpu epic: 5/5 criteria now met (pending Epic Verification Gate)
- cooperative-compute (T1): can now start, T0 is clearing
