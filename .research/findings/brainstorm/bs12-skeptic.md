# BS12 Skeptic Analysis — "What Comes Next?" Challenge
**Date**: 2026-03-12
**Brainstorm seq**: 12
**Role**: Skeptic

---

## 1. Is the Reproduction Actually Complete?

**Verdict: 85-90%, not 100%. The last 10% may be structurally unreachable without a rustc fork.**

### What we genuinely achieved
- Inline PTX system-scope atomics (arguably better than VectorWare's Relaxed)
- Lock-free hostcall protocol with multi-block scaling (512 threads)
- Embassy async executor on GPU (per-thread, Fat LTO, no fork)
- Slab allocator with concurrent deallocation (32 threads)
- Structured error propagation through hostcall
- Vendored std with PAL routing for stdout/stdin

### What we have NOT verified against VectorWare

**Thread scheduling model.** VectorWare's blog mentions running async tasks, but
never explains how tasks are scheduled across warps or blocks. We use per-thread
executors, which is the simplest model but may not match their design. If VectorWare
uses warp-cooperative scheduling (one executor per warp, tasks distributed across
lanes), their approach would have fundamentally different performance characteristics.
We have no way to verify this without their source code.

**Allocator sophistication.** VectorWare demonstrates `Vec` and `String` working on
GPU but never describes their allocator. Our slab allocator handles the demonstrated
cases, but VectorWare may have a more sophisticated design (e.g., multi-pool with
size classes tuned for common Rust allocations, or a GPU-specific arena allocator).
The slab allocator's 32-byte minimum block size wastes 50%+ on small allocations
like single-char Strings.

**Warp-level cooperative operations.** VectorWare's blog does not demonstrate
`__shfl_sync`, `__ballot_sync`, or other warp intrinsics combined with async/await.
We have `activemask` in gpu-atomics but have not explored cooperative operations.
This is likely not part of their current scope either, but it would be the natural
next step for performance-sensitive GPU async code.

**println!() vs writeln!(stdout()).** VectorWare uses `println!()` directly, which
implies they solved OnceLock for GPU. We use `writeln!(std::io::stdout(), ...)` as a
workaround. The oncelock theme completed, but the finding was a cfg-gate bypass, not
a proper OnceLock port. This distinction matters: VectorWare likely has a deeper
std integration that we may never replicate without their rustc modifications.

---

## 2. Fragility Assessment

**Verdict: The stack is held together by duct tape. Here are the failure modes.**

### Nightly Rust breakage

The entire project depends on:
- `#![feature(abi_ptx)]` — unstable ABI
- `-Zbuild-std=std` — unstable build flag
- `nvptx64-nvidia-cuda` — tier 3 target with no CI coverage in rustc
- Fat LTO with `llvm-bitcode-linker` — niche toolchain component
- Vendored std with manual patches

**Any nightly update can break any of these.** There is no regression test suite
that would catch a rustc update breaking the PTX backend, the bitcode linker, or
the `cfg_select` behavior we depend on. The LLVM NVPTX backend is maintained by a
small team and has a history of regressions (the circular dependency bug, the missing
scope qualifiers).

Concrete risk: When `gpu-kernel` ABI is stabilized and `ptx-kernel` is deprecated,
our entire kernel interface breaks. The migration path is unclear because
`gpu-kernel` may have different calling conventions or argument passing semantics
that we haven't tested.

### Unsafe audit

The project uses inline PTX assembly extensively in `gpu-atomics`. Every inline asm
block is an `unsafe` operation with no Miri verification possible (Miri doesn't
support nvptx64). The correctness of these operations depends on:
1. The PTX ISA specification being correctly interpreted
2. LLVM not reordering operations around inline asm blocks
3. The GPU hardware implementing the memory model as specified

None of these are formally verified. The atomics.5 task audited correctness at a
design level but did not stress-test edge cases like:
- What happens when a CAS succeeds on one lane but the warp diverges before the
  next instruction?
- Can the compiler hoist a load above an inline asm fence?
- Does `st.release.sys` guarantee visibility to the CPU before the next PCIe
  transaction completes?

### Hostcall protocol edge cases

The protocol was tested under "friendly" conditions. Untested scenarios:

1. **GPU thread killed mid-hostcall.** If a GPU thread crashes (trap, OOM, timeout)
   while holding a packet in CONTROL_FILLED state, the host will process it and set
   CONTROL_READY, but no GPU thread will ever read the response. That packet is
   leaked from the pool forever. With enough crashes, pool exhaustion becomes
   inevitable.

2. **Host process crash.** If the host crashes while the GPU is spinning in
   `sys_spin_load_acquire`, the GPU threads spin forever. There is no watchdog or
   timeout on the GPU side (the `GPU_MAX_SPIN` only applies to packet acquisition,
   not response waiting).

3. **Race between kernel completion and host processing.** If the kernel completes
   (returns to host) while the host listener is still processing packets, the
   `cuStreamSynchronize` call will succeed but packets may still be in-flight. The
   current code does not have a drain barrier.

---

## 3. Performance Reality Check

**Verdict: We have almost no meaningful performance data. What we have is concerning.**

### The numbers we have

| Metric | Value | Concern |
|--------|-------|---------|
| Single hostcall latency | 117-197us | This is PCIe round-trip dominated |
| 512-thread scaling | 88x slowdown for 16x threads | O(n^2) CAS contention |
| Duplicate message rate at 512 threads | 38% | Host listener design flaw |
| Slab allocator throughput | 362K ops/sec (32 threads) | No baseline comparison |
| Showcase kernel total time | 325.5us (1 thread) | Includes stdin wait |

### What we do NOT know

1. **Hardware register usage.** All register counts are PTX virtual registers, not
   hardware registers. The PTX assembler (ptxas) maps virtual to physical registers
   and may spill to local memory. We have NEVER run `cuobjdump --function-reg-count`
   or Nsight Compute to measure actual hardware register pressure. The "82 virtual
   registers" could map to 40 hardware registers (fine) or 200 (disaster for
   occupancy).

2. **Occupancy.** We have never measured occupancy. With unknown register counts,
   unknown shared memory usage, and variable thread counts, we cannot predict how
   many warps can be resident on an SM simultaneously. Low occupancy means the GPU
   cannot hide memory latency, which is the GPU's primary performance mechanism.

3. **Comparison with native CUDA.** The benchmark theme was parked. We have NO
   comparison with equivalent CUDA C++ code. The hostcall latency of 117-197us per
   operation is astronomical by GPU standards (a global memory access is ~400ns).
   For any workload where the compute-to-I/O ratio is less than ~500:1, this
   approach would be slower than just doing everything on the CPU.

4. **Host listener CPU usage.** The host listener busy-loops with `thread::sleep(100us)`
   between polls. This burns an entire CPU core for the lifetime of the kernel. For
   a system running multiple GPU kernels or multiple GPUs, this does not scale. No
   measurement of CPU overhead exists.

5. **Memory bandwidth impact.** System-scope atomics (`st.release.sys`,
   `ld.acquire.sys`) flush L2 cache and create PCIe traffic. With 512 threads issuing
   hostcalls, the GPU's memory subsystem is handling cross-device coherence traffic
   that competes with compute memory accesses. This is completely unmeasured.

### The fundamental performance question nobody has asked

**Is there ANY workload where Rust async on GPU is faster than the alternatives?**

- vs. CPU-only Rust: The GPU adds PCIe latency, kernel launch overhead, and hostcall
  round-trips. For I/O-bound work, the CPU is strictly faster.
- vs. native CUDA + host callbacks: CUDA streams with host function callbacks provide
  similar GPU-to-host communication with lower overhead (no CAS spin loops, kernel
  driver handles synchronization).
- vs. CUDA with separate compute/IO phases: Traditional approach of doing all compute
  on GPU, transferring results, then doing I/O on CPU. Simpler, proven, well-tooled.

The honest answer may be: "There is no workload where this is faster. The value is
ergonomic (write Rust instead of CUDA C++) not performance." That is a valid value
proposition, but it should be stated explicitly rather than avoided.

---

## 4. Scalability Concerns

### 512 threads is not "multi-block scaling"

The multiblock.2 test showed 512 threads work. But:
- Modern GPUs have 10,000-80,000 resident threads (e.g., RTX 4090: 128 SMs x 48
  warps/SM x 32 threads = 196,608 max threads)
- 512 threads is 0.26% of capacity
- The O(n^2) CAS contention already caused 88x slowdown at 512 threads
- At 8,192 threads, extrapolating quadratically: ~22,000x slowdown (hours per kernel)

**The current architecture fundamentally cannot scale to real GPU parallelism.** The
single global `free_stack_head` atomic is a serialization point. Per-block sharding
was identified as a solution but never implemented. Even with sharding, the
per-lane packet model means 8,192 threads need 8,192+ packets, consuming 8,192 x 64
bytes = 512 KB of pinned host memory just for the packet pool.

### Static array sizing

The multiblock.3 async test used `[ExecutorStorage; 4]` and `[TaskStorage; 4]`. For
N threads, you need `[ExecutorStorage; N]` known at compile time. This is:
- Inflexible: changing thread count requires recompilation
- Wasteful: allocating 8,192 static executor storages even if only 32 are used
- Fragile: buffer overflow if launched with more threads than the array size

### Slab allocator capacity

The slab allocator test used 32 threads x 5 cycles of small allocations. Real
concerns:
- **Heap size**: 1 MB shared across all threads. At 8,192 threads, that is 125
  bytes per thread. Even a single `Vec<u32>` with 8 elements (32 bytes data + 24
  bytes Vec struct + slab overhead) would consume most of it.
- **Size class coverage**: Only 32B, 64B, 128B, 256B, 512B, 1024B classes. A
  `Vec<u64>` with 200 elements needs 1600 bytes — no matching size class.
  Allocation falls back to... what? The findings don't say.
- **Bitmap contention at scale**: 8,192 threads competing for bitmap words would
  have much higher CAS retry rates than the tested 32.

### Multiple kernels and streams

Completely unexplored. Questions:
- Can two kernels share a hostcall buffer? (Probably not — packet pool state
  would be corrupt)
- Can two kernels use separate hostcall buffers? (Host needs multiple listeners)
- Can CUDA streams interleave kernels with hostcall? (Synchronization nightmare)
- What about CUDA graphs? (Hostcall breaks the static dependency model)

---

## 5. Direction Challenges

### AMD port is premature and dangerous

The NVIDIA-only stack is not stable. Porting to AMD would:
1. Require replacing ALL inline PTX with AMDGPU equivalents (different ISA,
   different memory model, different scope semantics)
2. Require a different hostcall transport (ROCm has its own mechanism, incompatible
   with our CUDA-based approach)
3. Require validating the gpu-kernel ABI on AMDGPU (untested in this project)
4. Double the maintenance surface while both backends are immature

**Risk**: An AMD port would consume 10+ sessions and produce a second unstable
platform instead of one stable one.

### "API ergonomics" is undefined

What API do users actually need? The project has demonstrated individual features
but has no coherent user-facing API. Questions:
- How does a user create a hostcall-enabled kernel? (Currently: manual buffer
  setup, manual PTX compilation, manual launch)
- How does a user define async tasks? (Currently: hand-written HostcallFuture
  implementations)
- How does a user handle errors? (Currently: check tuple return values manually)
- Where is the `#[gpu_kernel]` proc macro? Where is the build system integration?

Without answering these questions, "API ergonomics" is just a label for "we don't
know what to do next."

### Real workloads will expose design flaws

The showcase demo is a carefully constructed happy path:
- 1 thread (no contention)
- 8 data elements (trivial memory)
- 4 hostcall operations (no pool pressure)
- No error cases
- No long-running computation
- No interaction with CUDA libraries

A real workload would immediately hit:
- Heap exhaustion from non-trivial data sizes
- Pool starvation from many concurrent hostcalls
- Register spills from combining compute + hostcall code
- Warp divergence from heterogeneous async tasks
- OOM handling (the allocator returns null, but does the Rust code handle that?)

### Benchmarking is not meaningful yet

The benchmark theme was correctly parked, but the reasoning matters: benchmarking
a system with known architectural limitations (O(n^2) CAS contention, per-lane
packets, bump allocator at the time) would produce misleading numbers. However,
now that these are partially addressed, benchmarking is STILL premature because:
1. Hardware register usage is unknown (could invalidate all conclusions)
2. No workload has been identified where this approach makes sense
3. The architecture might change based on benchmark results (circular dependency)

---

## 6. Fundamental Design Questions

### Per-thread executors: wrong granularity?

Every GPU thread runs its own Embassy executor. On a warp of 32 threads, this means:
- 32 independent poll loops executing in SIMT lockstep
- When thread 0 polls its Future and gets Pending, threads 1-31 wait (divergence)
- When thread 15 polls and gets Ready, the other 31 threads wait
- Net effect: throughput approaches 1/32 of peak for heterogeneous workloads

**A warp-cooperative executor** (one logical executor per warp, tasks distributed
across lanes using shuffle instructions) would:
- Eliminate warp divergence for polling
- Reduce packet pool pressure by 32x (one packet per warp)
- Enable warp-level reduction for result aggregation
- Better match the GPU's SIMT execution model

This would be a fundamental redesign, not an incremental improvement. The question
is whether the per-thread model can ever be performant enough to justify NOT doing
the redesign.

### Hostcall is the bottleneck for everything

Every I/O operation goes through hostcall: PCIe round-trip + host processing +
PCIe return. At 117-197us per operation, a kernel doing 100 I/O operations spends
12-20ms just on hostcall latency. During that time, the GPU SM is mostly idle
(spinning in load loops or yielding in async poll loops).

**Is there a way to batch hostcalls?** Instead of one packet per operation, submit
N operations in one packet and get N results back. This would amortize the PCIe
latency across operations. But this requires:
- Larger packets (current 8 x u64 slots = 64 bytes is very tight)
- Vectorized host-side processing
- GPU-side batching logic

This is an architectural change, not an optimization.

### Fat LTO: permanent developer experience tax

Fat LTO is required for Embassy, std, and all cross-crate calls to resolve. This
means:
- **Compilation time**: Full-program LTO on every build. For a non-trivial project,
  this could be minutes per build.
- **Debugging**: LTO removes function boundaries, making debugging extremely
  difficult. No GPU debugger support exists for this setup.
- **Incremental compilation**: Impossible with Fat LTO — every change rebuilds
  everything.
- **Error messages**: LTO-phase errors are opaque LLVM errors, not Rust diagnostics.

Can this EVER be a good developer experience? The answer is probably no, unless
the LLVM NVPTX backend adds proper support for cross-module function calls without
LTO (i.e., proper device linking). This is not on any known LLVM roadmap.

---

## 7. Untested Assumptions

| Assumption | Risk | Consequence if Wrong |
|-----------|------|---------------------|
| Multi-GPU (same host) | High | Hostcall buffer is per-device. No mechanism for GPU-to-GPU communication. |
| Long-running kernels (>1 second) | Medium | Host listener CPU burn becomes significant. GPU watchdog timer may kill kernel. |
| Error recovery after hostcall failure | Medium | Failed hostcall leaves packet in limbo. No retry mechanism. |
| Memory pressure (OOM on GPU) | High | Slab allocator returns null. Rust's `alloc::alloc` path calls `handle_alloc_error` which likely panics on GPU. |
| Interaction with CUDA libraries (cuBLAS) | High | cuBLAS uses its own memory allocator, streams, and synchronization. Hostcall's system-scope atomics may interfere. |
| Multiple concurrent kernels | High | Single global hostcall buffer cannot be shared. No isolation mechanism. |
| GPU context migration | Medium | CUDA may migrate GPU context across SMs. Static variables with physical addresses could become invalid. |
| Power management / clock throttling | Low | Thermal throttling changes globaltimer frequency. Timing measurements become unreliable. |

---

## 8. Summary: What the Proposer Must Address

Any "what's next" proposal must answer these questions:

1. **Why continue?** The research mission is complete. VectorWare parity is achieved
   at 85-90%. Is the remaining 10-15% worth the effort, or should the project be
   declared done?

2. **Performance or ergonomics?** These are opposite directions. Performance requires
   warp-cooperative redesign, batched hostcalls, and register profiling. Ergonomics
   requires proc macros, build system integration, and documentation. Pick one.

3. **What is the first real workload?** Without a concrete workload (not a demo),
   all further development is speculative. Define one workload that benefits from
   Rust async on GPU, and let it drive the architecture.

4. **How do you handle the nightly breakage risk?** The project has zero CI, zero
   regression tests, and depends on unstable features. One nightly update could
   invalidate weeks of work. Is there a mitigation plan?

5. **What is the exit criterion?** "Make it better" is not a goal. Define what
   "done" looks like for the next phase, or acknowledge this is open-ended
   exploration with no defined endpoint.
