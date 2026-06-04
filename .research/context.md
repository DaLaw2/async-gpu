## Current Focus
T0 epics nearing completion. Both have all tasks done; ready for Epic Verification Gate.
- native-rust-dx: 3/4 criteria met (examples partially rewritten — 3/7 hostcall converted, 4 need flexible API)
- std-thread-gpu: 5/5 criteria likely met (std::thread::spawn verified working with println!)
tasks_since_brainstorm = 15 → brainstorm triggered.

## Recent Decisions
- 2026-06-04: SIMT lane-0 guard in gpu_main/gpu_main_poll — main_fn must run on lane 0 only
- 2026-06-04: sys_thread_cuda.rs trampoline needs lane masking (only lane 0 manages closure data)
- 2026-06-04: kernel_std.cubin pre-compilation required (6MB PTX → 37MB cubin, <1s vs >10min)
- 2026-06-04: Sysroot IS patched (cuda.rs routes std::thread → gpu_thread_spawn_raw)
- 2026-06-04: 3 hostcall examples simplified (hello-gpu, async-io, async-pipeline) — 4 need flexible API
- 2026-06-04: ONLY_TEST env var controls test selection (not CLI args)

## Tried & Rejected
- bar.sync removal for cross-launch fix: doesn't work (L1 cache coherence)
- Out-of-place elementwise_add: SLOWER than in-place (119 vs 160 GB/s)
- Cooperative closure captures: GPU local memory per-warp isolation → ILLEGAL_ADDRESS
- gpu::run_std() loading kernel_std.ptx at runtime: JIT too slow (>10min for 6MB PTX)

## Active Constraints
- GTX 1660 (sm_75): no tensor cores, 192 GB/s, 5 TFLOPS FP32
- kernel_std.cubin must be pre-compiled (ptxas --gpu-name sm_75)
- build.rs auto-rebuilds kernel PTX with stock nightly — use AUTO_BUILD_KERNEL=0
- CUDA module statics persist across launches — must load into separate modules

## Key Metrics
- Flash Attention V3: 559 GFLOPS causal @ seq=512
- Fused LN+residual: 2.01x speedup, 154 GB/s
- In-place elementwise_add: 160 GB/s (83% peak)
- std::thread::spawn: WORKS — spawn 2 threads, join, println! (45 + 120 = 165)
- kernel_std.cubin: 37MB, loads <1s

## Next
1. Run Epic Verification Gate on std-thread-gpu (5/5 criteria should pass)
2. Assess native-rust-dx criterion 4 (examples rewrite — partial, may need discussion)
3. Brainstorm triggered (tasks_since_brainstorm=15) — focus on remaining T0 gaps or T1 activation
4. If T0 clears → activate cooperative-compute (T1)
