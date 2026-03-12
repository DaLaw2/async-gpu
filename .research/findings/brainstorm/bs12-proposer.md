# BS12 Proposer Analysis — Post-Completion Strategic Direction
**Date**: 2026-03-12
**Brainstorm seq**: 12
**Level**: deep (proposer role)
**Context**: All 12 active themes completed (47/47 tasks). Only `benchmark` parked. Full VectorWare parity achieved.

---

## Section 1: Systems Analysis

### 1.1 Next Logical Extensions

The system has three distinct extension axes:

**Axis A — Depth (hardening what exists):**
1. **Hostcall pool sharding.** The O(n^2) CAS contention measured in multiblock.2 (88x slowdown at 512 threads vs 32 threads) is the most concrete scaling bottleneck. Per-SM or per-block free stacks would reduce contention from O(n^2) to O(n^2/k) where k = shard count. The infrastructure for this exists: the two-stack protocol just needs to be replicated per shard, with a shard selection function (blockIdx.x % k or SM ID).

2. **Host listener throughput.** The 38% duplicate rate at 512 threads (multiblock.2) reveals a host-side weakness: the swap-and-iterate pattern on the ready stack has a race window where new packets arrive during processing. A per-packet "processed" bit or a monotonic drain cursor would eliminate duplicates and improve throughput.

3. **Allocator fragmentation under sustained workloads.** The slab allocator was tested for 20 alloc/dealloc cycles. Real workloads with mixed allocation sizes over thousands of cycles may exhibit fragmentation patterns where all blocks of one size class are exhausted while other classes remain mostly free. A defragmentation strategy or adaptive class sizing would address this.

4. **Heap lifecycle management.** The host must reset slab bitmaps between kernel launches. Currently this requires the host to zero all bitmap words. A more robust approach: version-stamp the heap metadata and have the allocator detect stale state on first use, enabling lazy initialization.

**Axis B — Breadth (new capabilities):**
1. **GPU-side async channels.** An mpsc or broadcast channel built on the slab allocator + CAS queues would enable inter-thread communication on GPU without going through the host. This is a natural extension of the async runtime and would enable producer-consumer patterns entirely on-device.

2. **Networking via hostcall.** Adding SERVICE_TCP_CONNECT, SERVICE_TCP_SEND, SERVICE_TCP_RECV opcodes would enable GPU code to make HTTP requests or send data over the network. The hostcall protocol already handles arbitrary request/response pairs — networking is "just another service."

3. **Dynamic task spawning.** Currently, Embassy tasks are statically allocated at compile time (TaskStorage arrays). Supporting runtime task creation via the slab allocator would enable spawn-style APIs, which are the expected async Rust programming model.

**Axis C — Portability:**
1. **AMDGPU backend.** Replace inline PTX with AMDGPU inline asm (s_atomic_*, buffer_atomic_*). The hostcall protocol design is already ROCm-inspired. The gpu-kernel ABI exists in Rust nightly for cross-vendor use.

2. **Multi-GPU.** The hostcall buffer is currently per-device. Supporting multiple GPUs would require per-device listener threads and device-indexed buffer allocation.

### 1.2 Unsafe Boundaries Requiring Hardening

Seven unsafe boundaries exist in the current system:

**U1: Hostcall packet lifetime (CRITICAL).**
GPU code acquires a packet from the free stack, writes payload, pushes to ready stack, and spin-waits for response. If a GPU thread is killed (timeout, trap) after acquiring but before releasing a packet, that packet is permanently leaked from the pool. At 512 threads with a 1024-packet pool, losing even a few packets per launch accumulates to pool exhaustion. Mitigation: host-side epoch-based reclamation — after kernel completion, scan all packets and return any still marked as in-use.

**U2: Slab allocator bitmap race window.**
The allocator's alloc path does: load bitmap → find zero bit → CAS to set bit. Between load and CAS, another thread could allocate the same bit. The CAS detects this (returns failure), but the retry loop must scan for a NEW zero bit, not retry the same one. The current implementation using `compare_exchange_weak` handles this correctly, but there is no bounds check on the bitmap word scan — if all words in a class are full, the scan falls through to the overflow bump allocator. If the bump region is also exhausted, the allocator returns null, which GlobalAlloc documents as UB for non-zero-sized allocations (should abort or loop instead).

**U3: Static executor array bounds.**
`MULTI_EXEC_STORAGE[global_tid]` and `MULTI_TASKS[global_tid]` are indexed by computed thread ID. If a kernel is launched with more threads than the array size N, this is an out-of-bounds access with no runtime check. Adding a bounds check (`if global_tid >= N { return; }`) at kernel entry would prevent this.

**U4: write_volatile for hostcall payload.**
Payload slots are written with `write_volatile` and read with `read_volatile`. This provides no ordering guarantees on GPU. The correct pattern is release-store for the writer and acquire-load for the reader. Currently, the protocol relies on the CONTROL field's sys-scope release-store to provide ordering for all preceding payload writes. This is correct per the CUDA memory model (release-store on the flag "publishes" all prior writes), but it's a subtle invariant that is not documented in code comments.

**U5: Embassy executor's raw pointer internals.**
Embassy uses `NonNull<TaskHeader>` and raw pointer manipulation for its task queue. On GPU, pointer provenance rules may differ (all pointers are effectively integers in PTX). Fat LTO merges Embassy's code into the kernel, where the LLVM PTX backend may make different optimization decisions about pointer aliasing. No issues have been observed, but this is an untested area.

**U6: Inline PTX asm clobber correctness.**
The gpu-atomics crate uses `asm!` with inline PTX. The clobber lists and constraint specifications must exactly match what the hardware actually does. For example, `atom.cas` reads and writes memory, so the asm block must be marked as having side effects (which `asm!` does by default). However, if LLVM decides to reorder or eliminate an asm block it considers side-effect-free due to an optimization bug, the atomics would break silently.

**U7: Cross-kernel-launch state persistence.**
Static variables in GPU code (`GPU_HEAP_STATE`, `MULTI_EXEC_STORAGE`, etc.) persist across kernel launches on the same device context. If the host does not reset these between launches, stale state from a previous launch could corrupt the next one. Currently, the host manually re-initializes the heap. A more robust pattern: use a launch-ID counter that the kernel checks on entry.

### 1.3 Unexplored Memory Model Issues

**M1: Weak ordering visibility windows.**
The CUDA memory model guarantees that a release-store on thread A is visible to an acquire-load on thread B, but does NOT guarantee WHEN. On architectures with deep store buffers (Hopper, Blackwell), the visibility window could be microseconds. Our spin-load loops use nanosleep(100ns) to avoid hammering the memory bus, but the actual visibility latency has never been measured. Under high contention, this could dominate hostcall latency.

**M2: L2 cache coherence for mapped memory.**
`cudaHostAllocMapped` memory is CPU-allocated but GPU-accessible. On discrete GPUs, this goes over PCIe. The L2 cache on the GPU may cache stale values of mapped memory even after the host writes new data, because the GPU's L2 does not snoop PCIe writes. System-scope atomics force L2 cache bypass, but ordinary loads/stores (even volatile) may read stale L2 data. This has not been tested because all our GPU-CPU communication uses sys-scope atomics. But if anyone adds a non-atomic fast path (e.g., bulk data transfer via mapped memory), L2 staleness would be a silent bug.

**M3: Memory ordering across warps within a block.**
Our no-op critical section assumes threads within a warp are lock-step (SIMT) and threads across warps are independent. But `__syncthreads()` (bar.sync in PTX) provides block-level synchronization with full memory ordering. We never use `bar.sync` because each thread runs independently. If a future feature requires inter-warp communication within a block (e.g., a shared executor or work-stealing), block-scope atomics and barriers would be needed — a different memory model from our current system-scope-only approach.

**M4: PTX memory space semantics.**
Our hostcall buffer is in "generic" address space (since it's a kernel argument pointer). PTX has distinct address spaces: `.global`, `.shared`, `.local`, `.const`, `.param`. The inline PTX in gpu-atomics specifies `.global` explicitly (`st.release.sys.global`). If the pointer actually refers to mapped (pinned) memory via a generic pointer, the hardware resolves the address space at runtime. This works on current hardware, but PTX spec does not guarantee that `.global` operations on mapped memory have system scope — only `.sys` scope guarantees cross-device visibility, which we use correctly.

---

## Section 2: Compiler Analysis

### 2.1 LLVM Atomics Scope Fix Tracking

**Current state (LLVM 21, Rust 1.90-1.94):**
- Loads/stores: FIXED (emit `.sys` scope + ordering)
- Fences: FIXED
- cmpxchg: FIXED (PR #140812, July 2025)
- atomicrmw (fetch_add, fetch_sub, exchange): BROKEN (Bug #173993, filed Dec 2025, still open)

**What to watch:**
- LLVM Bug #173993 (assigned to Artem-B, NVIDIA LLVM maintainer)
- Any PR touching `NVPTXIntrinsics.td` that adds scope/ordering to `ISD::ATOMIC_LOAD_ADD` patterns
- The fix is structurally straightforward — the infrastructure (`getAtomicScope()`, `getMemOrder()`) exists; it just needs to be called from the atomicrmw selection path, similar to how `tryLoad()`/`tryStore()` already call it

**Timeline estimate:** Given that Artem-B fixed loads/stores in PR #99709 (July 2024) and cmpxchg in PR #140812 (July 2025), the atomicrmw fix could land in LLVM 23 (late 2026) or LLVM 24 (2027). When it does, we should:
1. Empirically verify the fix with a test kernel
2. Gate the fix behind a cfg flag: `cfg(llvm_atomicrmw_scope_fixed)`
3. Keep inline PTX as fallback for older LLVM versions
4. Eventually deprecate the inline PTX path

**When can we drop inline PTX entirely?** Not until:
1. atomicrmw scope is fixed in LLVM
2. The fix lands in a stable Rust release
3. We drop support for Rust versions before that release
4. Estimated: late 2027 at earliest

### 2.2 Fat LTO Compilation Time

**The problem:** Fat LTO merges all crates (gpu-kernel, embassy-executor, gpu-atomics, gpu-protocol, patched std) into a single LLVM module before codegen. For the current codebase, this takes 10-30 seconds per build. As the kernel grows (more async tasks, more std coverage, more application code), this will increase linearly.

**Alternatives investigated:**

1. **Thin LTO:** Uses module-level summaries to guide cross-module optimizations without fully merging. On nvptx64, Thin LTO is NOT supported — the PTX backend requires a single merged module because PTX is a virtual ISA that gets assembled by ptxas (NVIDIA's assembler), which expects a single compilation unit. This is a hard constraint.

2. **Split kernel into multiple PTX files:** Each crate produces its own `.ptx`, and the host loads them separately. This works for independent kernels but NOT for kernels that share functions across crates (e.g., gpu-kernel calling gpu-atomics functions). CudarC's `cuModuleLoadDataEx` can load multiple PTX files and link them via `cuLinkCreate` + `cuLinkAddData`. However, this loses cross-module inlining, which is critical for performance (atomics must be inlined into the kernel).

3. **Incremental compilation:** Not supported for nvptx64. The PTX backend does not produce object files that can be incrementally linked — it produces a single PTX text module.

4. **Separate compilation for host and device:** Already done. Only device-side compilation uses Fat LTO. Host-side compilation is normal incremental Rust.

5. **CUDA separate compilation + device linking:** NVIDIA's toolchain supports `-dc` (device code) and `-dlink` (device linking) for CUDA C++. The Rust PTX backend does not support this — there is no equivalent of relocatable device code in PTX-via-LLVM.

**Verdict:** Fat LTO is unavoidable for nvptx64 compilation. The mitigation strategies are:
- Keep kernel crates small (only GPU code, no host code)
- Use `#[inline(always)]` judiciously to help LLVM skip unnecessary inlining analysis
- Invest in a development workflow with fast rebuild (e.g., `cargo watch` that only rebuilds the PTX when GPU code changes)
- Future: if LLVM adds nvptx64 relocatable device code support, this constraint lifts

### 2.3 nvptx64 Target Stability in Rustc

**Known upstream changes affecting us:**

1. **gpu-kernel ABI stabilization.** Currently `extern "gpu-kernel"` is nightly-only (feature gate `abi_gpu_kernel`). VectorWare is pushing for stabilization. When stabilized, we should migrate from `ptx-kernel` to `gpu-kernel` for forward compatibility and AMDGPU portability.

2. **LLVM 22 update in Rust.** No LLVM 22 update PR has landed in rust-lang/rust as of March 2026. When it does, we should verify:
   - All gpu-atomics inline PTX still compiles
   - Embassy still compiles and runs on nvptx64
   - No new LLVM PTX backend regressions

3. **nvptx64 target tier.** nvptx64-nvidia-cuda is a Tier 3 target in Rust, meaning no CI testing and no guarantees. Any rustc PR could break it without notice. We are currently insulated by pinning a specific nightly, but we should test against new nightlies monthly.

4. **core::arch::nvptx deprecation.** The `core::arch::nvptx` module provides intrinsics like `_nanosleep()`. If this module is deprecated or removed, our `sys_spin_load` functions would need to use raw inline PTX for nanosleep.

5. **build-std improvements.** The `-Zbuild-std` flag is evolving. Changes to how std is built could affect our vendored std patches. Specifically, if Cargo gains native vendored-std support (RFC 3073 follow-up), our patching workflow could simplify.

---

## Section 3: GPU Architecture Analysis

### 3.1 Warp-Cooperative Patterns for Performance

The single biggest performance opportunity is exploiting warp-level cooperation. Current design treats each lane independently, wasting the fundamental SIMT parallelism:

**Pattern 1: Warp-cooperative hostcall (batch I/O).**
Instead of 32 lanes each acquiring their own packet (32 CAS operations on the free stack), lane 0 acquires ONE packet, all 32 lanes fill their respective payload slots (already the packet layout), and lane 0 pushes to ready stack. This reduces CAS contention by 32x. Implementation:
- Lane 0: `packet = hc_pop_free()`
- All lanes: `packet.slots[lane_id] = my_data` (no sync needed, disjoint slots)
- `__syncwarp()` (bar.sync within warp)
- Lane 0: `hc_push_ready(packet)`
This requires all 32 lanes to reach the hostcall at the same time — only works for uniform code paths. For divergent async code, per-lane packets remain necessary.

**Estimated impact:** 32x reduction in CAS contention → O(n^2/32) instead of O(n^2). At 512 threads, this would reduce the 88x slowdown to approximately 3x.

**Pattern 2: Warp-level vote for common operations.**
Multiple lanes may want to print the same format string with different values. A warp vote (`__ballot_sync`) could detect this pattern and batch the operation:
- All lanes: `mask = __ballot_sync(0xFFFFFFFF, wants_print)`
- If all 32 lanes want print: use warp-cooperative packet
- If < 32 lanes: per-lane packets for wanting lanes only
This is an optimization that reduces contention without changing correctness.

**Pattern 3: Shuffle-based shared state.**
Instead of each lane reading shared state from global memory, one lane reads and broadcasts via `__shfl_sync`. For the hostcall protocol, the free_stack_head could be read by lane 0 and shuffled to all lanes, reducing global memory traffic by 32x during the CAS attempt. However, this doesn't help with contention — each lane still needs its own CAS.

**Pattern 4: Cooperative groups for multi-warp coordination.**
CUDA cooperative groups allow synchronization at arbitrary granularities (warp, block, grid, multi-GPU). If we could synchronize at the block level before issuing hostcalls, we could batch all of a block's hostcalls into fewer packets. This requires `cooperative_groups.h` equivalents in Rust, which don't exist — but `bar.sync` (block-level barrier) is available via inline PTX.

### 3.2 Shared Memory Usage Opportunities

The entire project currently uses only global memory. Shared memory (configurable 0-228 KB per SM on Ampere+) offers 10-100x lower latency than global memory:

**Opportunity 1: Per-block hostcall packet cache.**
Allocate a small number of hostcall packets in shared memory. Intra-block threads CAS on shared-memory atomics (`.cta` scope, much faster than `.sys` scope) for allocation. Only when a packet is ready for the host does it get copied to global mapped memory with system-scope stores. This decouples the fast path (packet allocation) from the slow path (GPU-CPU communication).

**Opportunity 2: Per-block executor shared state.**
For a block-level executor (rather than per-thread), the task queue and wake flags could live in shared memory. Shared-memory atomics are 5-10x faster than global-memory atomics. This would enable a shared executor for all threads in a block, reducing the N copies of executor state to 1 per block.

**Opportunity 3: Allocator fast path.**
The slab allocator's bitmap words could be cached in shared memory per block. Alloc operations CAS on the shared-memory copy, and periodically flush to global memory. This is complex (requires coherence between shared and global copies) but would dramatically reduce alloc contention for workloads where threads in a block have similar allocation patterns.

**Opportunity 4: String formatting buffer.**
`format!()` currently allocates on the heap for every call. A per-block shared-memory buffer (e.g., 1 KB per thread, 32 KB per block) could serve as a stack-like scratch space for formatting, avoiding heap allocation entirely for short strings.

### 3.3 Occupancy Impact of Per-Thread Executor Model

**Current state:** Each thread runs an independent Embassy executor. The executor's state includes:
- `ExecutorStorage` (~128 bytes, in static global memory)
- `TaskStorage<F>` per future type (~sizeof(F) + 32 bytes header, in static global memory)
- Stack-local variables for polling (~16-32 registers)

**Register pressure measured:**
- Sync kernel: ~13 PTX virtual registers
- Async 1-task kernel: ~31-39 PTX virtual registers
- Async 2-task kernel: ~57-82 PTX virtual registers

**Occupancy calculation (SM86, RTX 3060):**
- 65536 registers per SM
- Max 1536 threads per SM (48 warps)
- At 82 regs/thread: 65536/82 = 799 threads → 24 warps → 50% occupancy
- At 57 regs/thread: 65536/57 = 1149 threads → 35 warps → 72% occupancy
- At 39 regs/thread: 65536/39 = 1680 threads → capped at 48 warps → 100% occupancy

**Impact analysis:**
- Single async task (39 regs): Occupancy is near-maximal. No performance concern.
- Two async tasks (57-82 regs): Occupancy drops to 50-72%. For compute-bound kernels, this means 28-50% less latency hiding via thread-level parallelism. For I/O-bound kernels (which the hostcall model is designed for), occupancy matters less because threads spend most time waiting.
- Adding a real compute workload (additional 30-60 regs) to an async kernel: 87-142 regs/thread → 460-753 threads → 15-23 warps → 31-47% occupancy. This is where the per-thread executor model starts to hurt.

**Key insight:** The per-thread executor model is fine for I/O-focused kernels (where occupancy is irrelevant because threads are waiting). For kernels that mix compute and I/O, a shared executor (one per warp or per block) would dramatically improve occupancy. This is the strongest argument for a warp-cooperative or block-cooperative executor design.

---

## Section 4: Concrete Recommendations

### R1: Unpark `benchmark` theme — Quantify everything
- **Type**: Unpark existing theme
- **Priority**: P0
- **Rationale**: No performance data exists with statistical rigor. Every design decision going forward (warp-cooperative vs per-lane, shared memory usage, allocator optimization) requires baseline measurements. Without benchmarks, we are optimizing blind.
- **Goal**: Hostcall round-trip latency with percentiles (p50/p95/p99), register pressure via `cuobjdump --function-reg-count`, occupancy via `ncu` (Nsight Compute), CAS retry rate per contention level.
- **Success criteria**:
  1. Hostcall round-trip latency measured with 1000+ samples, p50/p95/p99 reported
  2. Hardware register count (not PTX virtual regs) measured for all kernel variants
  3. Occupancy measured with Nsight Compute for 1-task and 2-task async kernels
  4. At least one workload compared with equivalent CUDA C++
- **Dependencies**: None (all infrastructure exists)
- **Tasks**:
  - `benchmark.1` (investigation): Design benchmark methodology — what to measure, how to measure, statistical rigor requirements
  - `benchmark.2` (experiment): Hostcall microbenchmarks — round-trip latency, throughput at various thread counts, CAS retry rates
  - `benchmark.3` (experiment): Register and occupancy profiling with Nsight Compute
  - `benchmark.4` (experiment): Comparative benchmark — implement a simple workload (e.g., file-based key-value lookup) in both Rust+hostcall and CUDA C++ and compare

### R2: New theme `warp-coop` — Warp-cooperative hostcall and execution
- **Type**: New theme
- **Priority**: P1
- **Rationale**: The O(n^2) CAS contention at 512 threads (multiblock.2) is the most severe scaling bottleneck. Warp-cooperative patterns reduce contention by 32x with relatively localized changes to the protocol.
- **Goal**: Enable warp-level batching of hostcall operations for uniform workloads
- **Success criteria**:
  1. Warp-cooperative hostcall packet allocation (1 CAS per warp instead of 32)
  2. 512-thread benchmark shows < 4x slowdown vs 32-thread (currently 88x)
  3. Fallback to per-lane mode for divergent workloads
  4. Mixed-mode kernel where some warps use cooperative and some use per-lane
- **Dependencies**: benchmark.2 (need baseline measurements first)
- **Tasks**:
  - `warp-coop.1` (design): Protocol extension for warp-cooperative packet flow — lane 0 allocates, all lanes fill, lane 0 submits
  - `warp-coop.2` (experiment): Implement warp-cooperative alloc/submit with `__ballot_sync` and `__shfl_sync`
  - `warp-coop.3` (experiment): Benchmark cooperative vs per-lane at 128, 512, 2048 threads

### R3: New theme `api` — Library API and ergonomics
- **Type**: New theme
- **Priority**: P1
- **Rationale**: The current codebase is a collection of test crates. To be useful as a library, it needs a clean public API, builder patterns for kernel configuration, and host-side ergonomics (automatic PTX compilation, buffer management, listener lifecycle).
- **Goal**: A user can write a GPU kernel with async I/O by depending on 2-3 crates, not 12
- **Success criteria**:
  1. Single `gpu-runtime` crate that re-exports all necessary GPU-side APIs
  2. Host-side `GpuContext` builder with automatic hostcall buffer setup and listener management
  3. Proc macro or build script for automatic PTX compilation from Rust source
  4. At least one example that a Rust developer unfamiliar with the project can follow
- **Dependencies**: None (refactoring, not new research)
- **Tasks**:
  - `api.1` (design): API surface design — what is public, what is internal, naming conventions
  - `api.2` (experiment): Implement `gpu-runtime` facade crate aggregating gpu-kernel + gpu-atomics + gpu-protocol
  - `api.3` (experiment): Implement host-side `GpuContext` builder in gpu-host
  - `api.4` (experiment): End-to-end example with build script for PTX compilation

### R4: New theme `ci` — Continuous integration and toolchain management
- **Type**: New theme
- **Priority**: P1
- **Rationale**: The project depends on a specific Rust nightly and NVIDIA driver. Any nightly update could break the build. Without CI, regressions are discovered late. The nightly dependency is also the biggest barrier for external contributors.
- **Goal**: Automated build and test on every push, with toolchain version pinned and documented
- **Success criteria**:
  1. GitHub Actions workflow that compiles all PTX and runs host-side tests
  2. `rust-toolchain.toml` pinning specific nightly version
  3. Monthly nightly update test (manual or scheduled CI)
  4. Test matrix documenting which GPU architectures have been verified
- **Dependencies**: None
- **Tasks**:
  - `ci.1` (experiment): Set up GitHub Actions with CUDA toolkit (no GPU needed for PTX compilation)
  - `ci.2` (experiment): Add `rust-toolchain.toml` and document toolchain requirements
  - `ci.3` (investigation): Investigate self-hosted runner with GPU for end-to-end tests

### R5: New theme `upstream` — Contribute fixes to upstream projects
- **Type**: New theme
- **Priority**: P2
- **Rationale**: The project has generated concrete knowledge about LLVM NVPTX bugs, Embassy GPU compatibility, and std patching requirements. Contributing this upstream benefits the broader ecosystem and reduces our maintenance burden.
- **Goal**: At least 2 upstream contributions accepted
- **Success criteria**:
  1. LLVM bug #173993 monitored; if we can produce a minimal reproducer, file upstream
  2. Embassy: document nvptx64 compatibility findings (even if no code change needed)
  3. Rust: investigate whether our `_print`/`_eprint` nvptx64 bypass could be upstreamed as a PAL target
  4. gpu-atomics inline PTX patterns documented for community reference
- **Dependencies**: None
- **Tasks**:
  - `upstream.1` (investigation): Prepare minimal LLVM atomicrmw reproducer and contribute to Bug #173993 discussion
  - `upstream.2` (investigation): Assess feasibility of upstreaming nvptx64 PAL to rust-lang/rust
  - `upstream.3` (experiment): Create standalone Embassy nvptx64 example for Embassy docs/examples

### R6: New theme `networking` — TCP/UDP via hostcall
- **Type**: New theme
- **Priority**: P2
- **Rationale**: After file I/O, networking is the most natural I/O extension. The hostcall protocol already handles request/response; adding TCP opcodes is structurally similar to file opcodes. This would enable GPU code to make HTTP requests, send metrics, or communicate with external services.
- **Goal**: GPU kernel can open a TCP connection, send data, and receive response via hostcall
- **Success criteria**:
  1. `SERVICE_TCP_CONNECT(addr, port)` opens connection on host, returns socket fd
  2. `SERVICE_TCP_SEND(fd, data)` sends data, returns bytes sent
  3. `SERVICE_TCP_RECV(fd, len)` receives data, returns bytes + data
  4. End-to-end test: GPU kernel makes HTTP GET request and prints response
- **Dependencies**: error-handling (completed — error propagation works)
- **Tasks**:
  - `networking.1` (design): Protocol extension for TCP operations — opcodes, payload layout, fd management
  - `networking.2` (experiment): Implement TCP connect + send + recv hostcall handlers
  - `networking.3` (experiment): End-to-end HTTP GET from GPU kernel

### R7: New theme `async-channels` — GPU-side async channels
- **Type**: New theme
- **Priority**: P2
- **Rationale**: Inter-thread communication on GPU currently requires going through the host (hostcall). For patterns like producer-consumer or fan-out/fan-in, GPU-side channels would eliminate the PCIe round-trip entirely. This enables a new class of GPU programs: pipelines with intra-device communication.
- **Goal**: Lock-free async mpsc channel on GPU using slab allocator + CAS queues
- **Success criteria**:
  1. `GpuSender::send(value)` and `GpuReceiver::recv().await` work on GPU
  2. 32 producers, 1 consumer, all on GPU, no host involvement
  3. Channel backpressure: send returns Pending when buffer is full
  4. Integrates with Embassy executor for async recv
- **Dependencies**: allocator (completed — slab allocator available for channel buffers)
- **Tasks**:
  - `async-channels.1` (design): Channel data structure design — ring buffer vs linked list, capacity, backpressure
  - `async-channels.2` (experiment): Implement lock-free mpsc channel with atomic head/tail
  - `async-channels.3` (experiment): Async integration — ChannelRecvFuture with Embassy waker

### R8: New theme `real-workload` — Meaningful application beyond demos
- **Type**: New theme
- **Priority**: P2
- **Rationale**: All current tests are toy demos. A real workload would validate the architecture under realistic conditions and provide a compelling case for the project's value. Candidates: GPU-based log processing, parallel key-value store operations, distributed ML inference coordination.
- **Goal**: One non-trivial application that demonstrates clear value from async GPU I/O
- **Success criteria**:
  1. Workload processes real data (not synthetic)
  2. Demonstrates async I/O advantage (multiple concurrent host operations)
  3. Performance compared with CPU-only and GPU-only-compute baselines
  4. Results documented with methodology
- **Dependencies**: benchmark (need measurement framework), allocator (completed)
- **Tasks**:
  - `real-workload.1` (investigation): Evaluate candidate workloads — log processing, KV lookup, ML inference dispatch
  - `real-workload.2` (experiment): Implement chosen workload with async hostcall I/O
  - `real-workload.3` (experiment): Benchmark against CPU baseline and pure-compute GPU baseline

### R9: Host listener improvements (extend existing hostcall theme or new mini-theme)
- **Type**: Enhancement to existing infrastructure
- **Priority**: P1
- **Rationale**: The host listener polls in a busy loop and produces 38% duplicate messages at 512 threads. This is the weakest component in the current stack.
- **Goal**: Host listener with bounded CPU usage and zero duplicates
- **Success criteria**:
  1. Zero duplicate messages at 512 threads
  2. CPU usage < 5% when GPU is idle (currently 100% one core)
  3. Latency increase < 10% vs current busy-loop
- **Dependencies**: None
- **Approach**: Replace busy-loop poll with event-driven design. Options: (a) GPU atomically increments a doorbell counter, host uses futex/WaitOnAddress on the doorbell; (b) CUDA events for notification; (c) hybrid — poll for N microseconds, then sleep and retry. Option (c) is simplest and sufficient.
- **Tasks**:
  - `hostcall.5` (experiment): Add per-packet processed bit to eliminate duplicate reads
  - `hostcall.6` (experiment): Implement adaptive polling — spin for 10µs, then sleep 100µs, repeat

### R10: AMDGPU port (park for now)
- **Type**: New theme (parked)
- **Priority**: P2 (park until NVIDIA stack is production-ready)
- **Rationale**: The `gpu-kernel` ABI is cross-vendor. The hostcall protocol is ROCm-inspired. The architectural foundation supports a port. But the inline PTX in gpu-atomics, the vendored std PTX-specific patches, and the CUDA-specific host code (cudarc) all need AMDGPU equivalents. This is a major effort best deferred until the NVIDIA stack is stable and well-documented.
- **Goal**: Same async + std capabilities on AMDGPU/ROCm
- **Success criteria**:
  1. gpu-atomics equivalent using AMDGPU inline asm (s_atomic_*, buffer_atomic_*)
  2. Hostcall protocol working over ROCm shared memory
  3. println!() from AMDGPU kernel
  4. Embassy executor on AMDGPU
- **Dependencies**: api (clean API makes porting easier), benchmark (need NVIDIA baseline for comparison)

---

## Section 5: Priority Summary

| Rank | Theme | Priority | Rationale |
|------|-------|----------|-----------|
| 1 | `benchmark` (unpark) | P0 | Cannot make informed optimization decisions without measurements |
| 2 | `warp-coop` | P1 | 88x scaling overhead is the most concrete problem; 32x improvement possible |
| 3 | `api` | P1 | Transforms research artifact into usable library |
| 4 | `ci` | P1 | Prevents silent regressions, enables collaboration |
| 5 | Host listener fixes | P1 | 38% duplicates and 100% CPU busy-loop are production blockers |
| 6 | `upstream` | P2 | Community contribution, reduced maintenance burden |
| 7 | `networking` | P2 | Natural capability extension, moderate effort |
| 8 | `async-channels` | P2 | Enables new GPU programming patterns |
| 9 | `real-workload` | P2 | Validation and showcase, depends on benchmarks |
| 10 | `amdgpu` (park) | P2 | Strategic but massive effort; defer until NVIDIA stack is solid |

### Suggested Execution Order

**Phase 1 (immediate, parallel):**
- Unpark benchmark theme → benchmark.1 + benchmark.2
- ci.1 + ci.2 (GitHub Actions + toolchain pinning)
- hostcall.5 (eliminate duplicate messages)

**Phase 2 (after Phase 1 measurements):**
- benchmark.3 (Nsight Compute profiling)
- warp-coop.1 (design warp-cooperative protocol)
- api.1 (API surface design)
- hostcall.6 (adaptive polling)

**Phase 3 (implementation):**
- warp-coop.2 + warp-coop.3 (implement and benchmark)
- api.2 + api.3 (implement facade crate and host builder)
- benchmark.4 (comparative benchmark vs CUDA C++)

**Phase 4 (breadth):**
- upstream.1 + upstream.2 (community contributions)
- networking.1 + networking.2 (TCP hostcall)
- async-channels.1 + async-channels.2 (GPU-side channels)
- real-workload.1 + real-workload.2 (meaningful application)

---

## Section 6: Key Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Rust nightly update breaks nvptx64 | Medium | High | CI + toolchain pinning (ci theme) |
| LLVM atomicrmw fix introduces regressions | Low | Medium | Keep inline PTX as fallback, don't auto-migrate |
| Warp-cooperative protocol correctness | Medium | Medium | Formal warp-level reasoning + extensive testing |
| API design locks in suboptimal architecture | Medium | High | Design review before implementation; version API as unstable |
| Occupancy collapse with compute+async kernels | High | Medium | Shared executor design (warp-coop theme) as alternative |
| Host listener becomes bottleneck beyond 1024 threads | High | High | Thread pool or async host listener; sharded ready stacks |

---

## Conclusion

The project has achieved its research goal. The path forward is clear: measure (benchmark), scale (warp-coop), package (api + ci), and extend (networking, channels, real workloads). The most urgent action is unparking the benchmark theme — every other optimization decision depends on having real measurements. The second priority is warp-cooperative hostcall, which addresses the most severe measured bottleneck (88x scaling overhead). The third is packaging the research into a usable library with CI, which transforms this from a research artifact into a community resource.
