# GPU Architecture Analysis: Rust std + Async/Await on GPU
**Role:** CUDA/GPU Architecture Expert
**Sequence:** bs1
**Date:** 2026-03-11

---

## 1. SIMT Execution Model and Warp Divergence

### The Core Problem
CUDA executes threads in groups of 32 called warps. All threads in a warp execute the same instruction at the same time cycle (SIMT — Single Instruction, Multiple Threads). When threads take different branches, the warp serializes: it executes each branch in turn with a mask to disable non-participating threads.

Rust async state machines are, by construction, divergence bombs.

### How Futures Become Divergence
A Rust `Future` compiled to a state machine becomes an enum with one variant per `.await` point. The executor calls `poll()`, which dispatches on the current variant via a match statement. If 32 threads in a warp are polling 32 different futures, or the same future type but at different `.await` points, every poll invocation can become a different branch. In the worst case:

- 32 threads, 32 distinct states → 32 serialized branches → effective warp utilization: 1/32 (3.125%)
- Even with only 4 distinct states evenly distributed → 25% warp utilization

This is not hypothetical. Any async task graph with data-dependent yields will produce this pattern under real workloads.

### Implications for VectorWare's Design
VectorWare runs 3 concurrent tasks. With 3 tasks spread across a warp's 32 lanes:
- Best case: all 3 tasks are at the same state simultaneously → no divergence
- Realistic case: tasks diverge quickly due to different data, different hostcall timing → heavy serialization

The occupancy reduction from register pressure (Section 4) compounds this. Fewer active warps means less latency hiding, which means the divergence penalty is paid more directly.

### Mitigation Strategies
- **Task homogeneity**: Ensure all threads in a warp execute the same task type and reach `.await` points together. This is only feasible for data-parallel workloads where each thread processes an independent element through the same pipeline.
- **Warp-level tasks**: Assign one task per warp rather than one task per thread. Poll logic runs once per warp, all 32 lanes execute the same state. This wastes lane parallelism but eliminates divergence within the executor.
- **Occupancy-aware task batching**: Design futures so that all yields correspond to the same event type (e.g., all threads yield on hostcall), enabling synchronized re-entry.

---

## 2. Memory Hierarchy and Placement Strategy

### GPU Memory Tiers (A100/H100 reference)

| Level | Capacity | Latency | Bandwidth | Lifetime |
|-------|----------|---------|-----------|----------|
| Registers | 256KB/SM (65536 regs × 4B) | 0 cycles | ~20 TB/s | Thread |
| L1/Shared | 48–228KB/SM (configurable) | 20–30 cycles | ~19 TB/s | Block |
| L2 Cache | 40MB (A100) | 200 cycles | ~3.5 TB/s | Device |
| Global (HBM) | 40–80GB | 600–800 cycles | ~2 TB/s | Device |
| Unified Memory | Host-mapped | 10,000+ cycles | PCIe limited | Device |
| Host (pinned) | System RAM | 100,000+ cycles | PCIe ~32 GB/s | Application |

### Future State Machine Placement

Future state machines contain: the enum discriminant, all captured variables from each `.await` suspension point, and padding for alignment. This state **must live in registers or spill to local memory** (which maps to L1/L2/global via the register file spill mechanism).

- **Registers**: Zero-overhead access. Target for hot state (current enum variant fields).
- **Local memory spill**: Compiler spills when register budget is exceeded. Spilled state goes to global memory through L1 — 200–800 cycle penalty per access.
- **Shared memory**: Cannot directly hold future state (it is per-block, not per-thread). Could be used for a block-level task queue, but task state itself stays per-thread.

### Hostcall Buffer Placement

The hostcall mechanism requires a shared memory region visible to both GPU and CPU:

- **Unified Memory (managed)**: Simplest API, but page-migration overhead. On access from GPU, triggers page fault if page is on host → catastrophic latency spike. Acceptable only if pages are pre-faulted or pinned.
- **Pinned host memory (cudaHostAlloc)**: GPU accesses via PCIe/NVLink. Latency: ~6µs minimum per access. Bandwidth: limited to PCIe gen4 ×16 = 32 GB/s bidirectional. For a ring buffer with small messages, latency dominates, not bandwidth.
- **Device global memory + cuMemcpy async**: GPU writes request to device VRAM, CPU reads via DMA. Adds a copy step but enables better pipelining and avoids GPU-side page-fault stalls.

**Recommendation**: Double-buffer design in pinned host memory. GPU writes to one buffer slot, CPU reads and responds to the other. Avoids page faults, predictable latency, compatible with CUDA streams.

### Executor State Placement

The Embassy-adapted executor's ready queue and waker structures should live in:
- **Shared memory** (if the executor is block-scoped and tasks are thread-parallel within a block): Fast access, 20–30 cycles, 48–228KB available.
- **Global memory** (if the executor is device-scoped across blocks): Accessible everywhere, 600–800 cycle access, requires atomic operations for thread-safety.

A block-scoped executor with shared memory backing is architecturally superior for latency, but limits parallelism to one block.

---

## 3. Atomics and Synchronization

### GPU Atomic Scope Hierarchy (PTX)

PTX provides atomic operations with explicit scope:

| Scope | PTX qualifier | Visibility |
|-------|---------------|------------|
| CTA (block) | `.cta` | Within one thread block |
| GPU | `.gpu` | All SMs on device |
| System | `.sys` | GPU + all CPUs (via coherence fabric) |

System-scope atomics are required for hostcall synchronization because the CPU must observe the GPU's writes and vice versa. On pre-Volta hardware (before SM70), system-scope atomics did not exist — they were added with the memory consistency model revision in CUDA 9/Volta.

**Critical requirement**: Minimum GPU target for this project is SM70 (Volta) for correct system-scope atomic behavior. Older GPUs lack the hardware memory model guarantees needed for lock-free GPU-CPU communication.

### Memory Ordering on GPU

GPU memory ordering follows the CUDA memory consistency model (post-Volta), which is weaker than x86's TSO but stronger than pure relaxed:

- Loads/stores within a thread: program order guaranteed
- Across threads: requires explicit fence (`membar` / `fence.sc.sys`) for sequential consistency
- `atomic.acquire` / `atomic.release` semantics are available in PTX and map to `cuda::atomic` in C++ (and `core::sync::atomic` in Rust with nvptx backend, with caveats)

For the hostcall ring buffer:
- GPU thread writes request: use `store.release.sys` to ensure all prior stores are visible to CPU before the flag is set
- CPU polls flag: use `load.acquire.sys` on CPU side to ensure the request body is visible after reading the flag
- GPU thread polls response: use `load.acquire.gpu` or `.sys` depending on whether CPU shares a cache domain

**Rust Caveat**: `core::sync::atomic` on nvptx64 targets may not emit the correct PTX scope qualifiers. VectorWave likely uses inline PTX or a CUDA intrinsic wrapper. This needs validation — a `SeqCst` fence in Rust on nvptx64 may compile to a `.gpu`-scoped fence, which is insufficient for CPU visibility.

### GPU-Side Executor Synchronization

For the Embassy executor on GPU:
- Waker registration: atomic bitfield per task (ready flags) — fits in registers or shared memory
- If executor is per-warp: no atomics needed (single-threaded within warp scheduling)
- If executor is per-block with multiple warps sharing tasks: need `atom.shared.cta` operations
- `__syncthreads()` (barrier) is available but blocks all threads in a block — too coarse for an async executor

The Embassy executor in embedded contexts uses a bitmask waker. On GPU, this bitmask can be stored in a register (if ≤32 tasks per warp) or in shared memory (if block-scoped). Atomic OR for waking, atomic AND for consuming.

---

## 4. Occupancy and Register Pressure

### Quantitative Framework

Each SM has 65,536 registers (on Ampere/Ada). These are shared among all active warps on the SM. The maximum number of active warps per SM on Ampere is 64 (16 warps × 4 blocks, or other combinations). Each warp has 32 threads.

Registers per thread for 100% occupancy:
- 65,536 registers / (64 warps × 32 threads) = **32 registers per thread** for 100% occupancy

### Future State Machine Register Cost

A Rust async function that awaits on 3 operations and captures 4 variables of 8 bytes each:
- Enum discriminant: 1 register (4 bytes, round up)
- Live variables per state: varies, but if all 4 vars must survive across all awaits: 8 registers (4 vars × 2 regs for 64-bit)
- Return value staging: 2–4 registers
- Compiler temporaries for poll logic: 4–8 registers
- Stack frame overhead: 2–4 registers
- **Rough estimate: 20–30 registers for a minimal future**

Adding a Future to a kernel that already uses 32 registers:
- Original kernel: 32 regs/thread → 64 warps/SM → 100% occupancy
- With future (+20 regs): 52 regs/thread → 65,536 / (52 × 32) = 39.3 warps → **61% occupancy**
- With future (+30 regs): 62 regs/thread → 65,536 / (62 × 32) = 33.0 warps → **52% occupancy**

With VectorWave's 3 concurrent tasks (3 futures stacked):
- 3 × 25 regs = 75 regs for futures alone + base kernel regs
- If base is 20 regs: 95 total → 65,536 / (96 × 32) ≈ 21 active warps → **33% occupancy**

This is the fundamental tradeoff VectorWare identified. It is not a bug — it is a consequence of futures being fat state machines.

### Mitigation

- **`#[inline(never)]` on future poll functions**: Forces register save/restore at poll call boundaries. Reduces register pressure during the call graph, at cost of call overhead (few cycles). Net win when the function complexity is high.
- **Smaller captured state**: Design futures to capture references/indices rather than values. Tradeoff: more memory traffic.
- **`maxrregcount` PTX pragma**: Hard-cap registers per thread. Compiler will spill to local (global) memory. Explicit control over the occupancy/spill tradeoff.
- **Warp-level tasks** (from Section 1): One future per warp. Register pressure is 1/32 of the per-thread cost in terms of SM-wide register consumption per task.

---

## 5. Hostcall Latency and Hiding Strategies

### PCIe Round-Trip Latency Breakdown

For a GPU-to-host-to-GPU round trip:

| Segment | Latency |
|---------|---------|
| GPU atomic write (global) | ~1µs |
| PCIe propagation (GPU→CPU) | ~1–2µs |
| CPU cache coherence + interrupt/polling detection | ~1–10µs |
| CPU-side work (syscall, etc.) | ~1–100µs |
| PCIe propagation (CPU→GPU) | ~1–2µs |
| GPU atomic read | ~1µs |
| **Total (minimal, polling CPU)** | **~6–20µs** |
| **Total (CPU under load)** | **~50–500µs** |

These are real-world figures. NVLink reduces PCIe segments to ~1µs each but does not eliminate them.

### GPU Idle Time During Hostcall

During a 100µs hostcall, how much compute is lost?

- A100 at 1GHz effective compute rate: 100,000 cycles wasted per spinning thread
- If 1024 threads spin on a hostcall: 1024 × 100,000 = 102.4M wasted thread-cycles
- A100 has 6912 CUDA cores: that is ~14,800 cycles of full-SM compute wasted per spinning thread

The nanosleep approach (VectorWare's method) reduces power but does not recover compute. The SM scheduler can switch to other warps — but only if other warps are ready. This is the latency-hiding mechanism.

### Latency Hiding via Oversubscription

Standard CUDA latency hiding: issue many warps per SM so memory latency (600–800 cycles) is hidden behind other warps executing. For hostcall at 100µs = 100,000 cycles:

- To hide 100,000 cycles, need enough warps with enough independent work to fill 100,000 cycles
- At 32 instructions/warp/cycle (rough), need ~3,125 independent warp-instructions
- This requires extreme SM oversubscription — typically impossible when register pressure is already reducing occupancy

**Practical conclusion**: Hostcall latency cannot be hidden by standard warp switching when the calling thread is the only work available. Other thread blocks on the same SM must have independent work. This argues for:
1. High SM occupancy via many independent blocks
2. Coarse-grained hostcalls (batch requests, reduce frequency)
3. Async hostcall: fire-and-forget style with callback futures

### CUDA Streams for Blocking Prevention

VectorWare uses CUDA streams to prevent GPU blocking. The mechanism: if the hostcall blocks the GPU kernel (via `__nanosleep` or spinning), other streams can continue on other SMs. Within one SM, only other thread blocks can run. Streams do not help intra-SM, only inter-SM.

---

## 6. Spin Loops and nanosleep

### `__nanosleep` Behavior

PTX `nanosleep` (SM70+) hints to the warp scheduler to de-schedule the warp for approximately N nanoseconds. It does not:
- Release registers (they remain allocated, blocking other warps from using them)
- Consume power (the warp is not issuing instructions)
- Guarantee exact timing (it is a hint, not a precise timer)

The SM can schedule other warps during the nanosleep period, but only if those warps exist and their register requirements fit within the remaining register budget.

### Power and Thermal Impact

Spin loops without nanosleep:
- Warp issues instructions every cycle → full SM power draw for zero useful work
- On a high-end GPU, this can draw 300–400W in a spin-heavy kernel
- Thermal throttling may reduce clock frequency, hurting other workloads on the same device

Spin loops with nanosleep (VectorWare's approach):
- Power draw during sleep: ~10–20% of peak (register file and SM logic still powered)
- Thermal impact significantly reduced
- SM can schedule other warps: occupancy-limited benefit

### Alternatives to Spin Loops

| Approach | Mechanism | Feasibility |
|----------|-----------|-------------|
| Spin + nanosleep | PTX nanosleep between polls | Feasible, VectorWave's approach |
| Persistent threads + work queues | Never exit kernel, pull work from queue | Feasible, standard CUDA pattern |
| Kernel re-launch | Exit kernel, CPU re-launches with new state | High overhead per task switch, ~10–50µs |
| Cooperative Groups barriers | Block-level cooperative wait | Limited to intra-block sync |
| Tensor Memory Accelerator (TMA) | Async memory copy with completion | Only for memory, not arbitrary host calls |
| CUDA Graph with conditional nodes | Dynamic kernel dispatch based on conditions | CUDA 12.4+, limited expressiveness |

For the async executor use case, persistent threads with a spin+nanosleep poll loop is the correct approach. There is no interrupt mechanism on GPU that would allow a warp to be woken by an external event without polling.

---

## 7. Shared Memory Design for Hostcall Ring Buffers

### Capacity Constraints

Shared memory is per-block, not per-device. Total shared memory per SM: 164KB (Ampere), configurable up to 228KB with reduced L1 cache. Per-block allocation: up to 48KB by default, 99KB or 164KB with explicit `cudaFuncSetAttribute`.

For a hostcall ring buffer in shared memory:
- Buffer must be accessible by all threads in the block that might make hostcalls
- A 32-entry ring buffer with 64-byte messages: 32 × 64 = 2KB — trivially fits
- A 256-entry ring buffer with 256-byte messages: 256 × 256 = 64KB — requires extended shared memory request

However: shared memory is only visible within one block. The CPU cannot directly read shared memory. The ring buffer for GPU-CPU communication must be in global memory or unified/pinned memory.

**Correct design**: Ring buffer in pinned host memory. Use shared memory only for per-block staging (gather requests from threads within a block before sending to the pinned buffer). This reduces PCIe transaction frequency.

### Bank Conflicts

Shared memory has 32 banks (on Ampere), each 4 bytes wide, with stride-32 access pattern. A bank conflict occurs when multiple threads in a warp access different addresses that map to the same bank (same bank = address % 32 same).

For a per-thread slot in a ring buffer:
- If slot size is 64 bytes and slots are laid out consecutively: thread 0 accesses bytes 0–63, thread 1 accesses bytes 64–127, etc.
- Bank of byte `b` in shared memory: `(b / 4) % 32`
- Thread 0, slot byte 0: bank 0. Thread 1, slot byte 64: bank 16. Thread 2, slot byte 128: bank 0 — **conflict with thread 0**

For 64-byte slots accessed by 32 threads (a full warp):
- Threads 0 and 2 conflict (both hit bank 0), threads 1 and 3 conflict (both hit bank 16), etc.
- This creates 2-way bank conflicts for every warp access to the buffer

**Fix**: Pad slots to 128 bytes (32 banks × 4 bytes) to ensure each thread hits a unique bank. Or use a broadcast pattern where one thread per warp manages the buffer.

### Multi-Warp Access Patterns

If multiple warps in a block make concurrent hostcall requests:
- Use a shared-memory atomic counter to claim ring buffer slots: `atomicAdd(&head, 1) % RING_SIZE`
- One warp's `atomicAdd` will serialize with other warps' — this is unavoidable but the critical section is tiny (1 atomic instruction)
- Warp-level cooperation: use `__shfl_sync` to let one elected thread (lane 0) perform the atomic and broadcast the slot index to all 32 lanes

For the response path:
- CPU writes response to designated slot in pinned memory
- GPU thread polls its specific slot — no inter-warp contention on the read path
- Slot ownership is per-thread or per-warp; release via atomic store to make slot available for reuse

---

## 8. Critical Hardware Limitations

### What Cannot Work on Current GPU Hardware

**1. Interrupts and Signal Delivery**
CPU async/await and embedded executor designs often rely on hardware interrupts or OS signals to wake a sleeping task. GPUs have no interrupt mechanism visible to shader code. A GPU kernel cannot be woken by an external event without polling. There is no equivalent of `epoll`, `io_uring`, or ARM WFI (Wait For Interrupt).

**2. Dynamic Memory Allocation (heap)**
`malloc`/`free` on GPU exist (`cudaMalloc` device-side in CUDA 9+), but they use a device heap backed by global memory with a mutex-protected allocator. This is:
- Slow: ~µs per allocation vs. ns on CPU
- Capacity-limited: device heap defaults to 8MB, configurable to a few GB
- Deadlock-prone: if all warps on an SM simultaneously allocate, the mutex serializes them
`Box<T>`, `Vec<T>`, and any allocating Rust type will use this path. It will work but will be slow.

**3. Thread-Local Storage (TLS)**
GPU threads do not have OS-level TLS. The GPU equivalent is per-thread registers and local memory (a private global memory region per thread). Rust `thread_local!` will not compile or behave correctly on nvptx64 without explicit shim support.

**4. Panic Unwinding**
Stack unwinding via Rust's standard panic mechanism is not feasible on GPU. There is no `libunwind` for CUDA targets. VectorWare must use `panic = "abort"` and likely trap (`__assertfail` or `__trap()`) on panics. Error propagation must use `Result<T, E>` throughout with no unwinding.

**5. Floating-Point and IEEE Compliance**
GPU default floating-point behavior deviates from IEEE 754 in performance-oriented kernels (FTZ — flush to zero for denormals, FMA contraction). Code relying on exact IEEE behavior may produce different results. This affects any `f32`/`f64` arithmetic in futures.

**6. POSIX System Call Interface**
Any code path that reaches a libc function expecting kernel system calls (`open`, `read`, `mmap`, etc.) will fail. VectorWare's hostcall framework proxies these through the host, but the shim layer must intercept every such call before it reaches the PTX code generator. There is no POSIX kernel on the GPU.

**7. Preemption and Context Switching**
GPU compute contexts are not preempted by the scheduler in the same way OS threads are. CUDA supports compute preemption at instruction level (Pascal+), but this is for multi-process sharing, not intra-kernel task switching. The async executor must be cooperative — there is no way to forcibly preempt a running task.

**8. Stack Size Limitations**
GPU local memory (per-thread stack) is limited by hardware. Deeply nested async call chains, each capturing state, can exceed the per-thread local memory allocation. The CUDA driver allocates local memory statically based on worst-case analysis. Deep future chains may cause compilation failure or silent stack overflow.

---

## 9. Recommendations by Theme

### toolchain
- Target `nvptx64-nvidia-cuda` with `opt-level = 3` and `panic = "abort"` in all crates
- Use `#[no_std]` with careful selection of core features; avoid any crate pulling in `std::thread`, `std::sync` (beyond `core::sync::atomic`), or heap allocators by default
- Validate that `core::sync::atomic` emits correct PTX scope qualifiers for system-scope operations; if not, wrap with inline PTX intrinsics
- Enable PTX feature flags for SM70+ to access `nanosleep` and correct memory model
- Use `nvcc` or `clang` as the linker backend for PTX → CUBIN compilation; `rustc` alone cannot produce a loadable CUDA module

### hostcall
- Allocate the ring buffer in pinned host memory (`cudaHostAlloc` with `cudaHostAllocMapped | cudaHostAllocWriteCombined` for write-combining on the CPU write path)
- Use system-scope atomics (PTX `.sys` scope) for the ring buffer head/tail and ready flags — `.gpu` scope is insufficient for CPU visibility
- Slot size: pad to 128 bytes to eliminate shared memory bank conflicts if staging through shared memory; for pinned memory, align to cache line size (64 bytes on x86)
- Implement per-warp batching: elect lane 0 to manage PCIe traffic, broadcast results to the warp. This reduces PCIe transactions by 32×
- CPU poller: use a dedicated thread with tight spin on the ring buffer, not an interrupt-based handler. Polling latency (~1µs) beats interrupt latency (~5–10µs) for high-frequency hostcalls
- Design for **coarse-grained** hostcalls: batch multiple GPU requests into one PCIe round trip where possible (e.g., one `write(fd, buf, N)` vs. N single-byte writes)

### gpu-std
- Implement a libc shim layer that intercepts all syscall-equivalent functions at link time and routes to the hostcall framework
- Priority order for native vs. hostcall implementation:
  - **Native on GPU**: `Instant` (via `%globaltimer`), basic memory allocation, `AtomicXxx`, panic handling, `core::fmt` (with output buffering)
  - **Via hostcall**: `SystemTime`, file I/O, network I/O, environment variables, process exit
  - **Not feasible**: thread spawning, signal handlers, mmap, shared memory IPC
- Use `#[global_allocator]` to install a custom allocator backed by the CUDA device heap, but impose a hard size limit and prefer stack allocation in future state machines

### async-runtime
- Port Embassy's executor with these GPU-specific modifications:
  - Replace `core::task::Waker` vtable dispatch with a direct bitfield waker (vtable dispatch on GPU means register spill for the function pointer + indirect branch → SM front-end pipeline stall)
  - Replace interrupt-based wake mechanism with a polling loop using `__nanosleep(200)` between polls (200ns sleep, tunable)
  - Use one executor instance per warp (not per thread): poll is called from lane 0, result broadcast via `__shfl_sync`. This eliminates intra-warp divergence in the poll dispatch path
  - Cap the number of concurrent tasks per executor to ≤32 (fits in a 32-bit bitmask waker, stored in one register)
- **Register pressure mitigation**: annotate all `Future::poll` implementations with `#[inline(never)]` as a first experiment; measure register count with `ptxas -v`; tune with `#[target_feature(enable = "...")]` or manual `maxrregcount`
- Design for **cooperative yielding at natural boundaries**: yield only at hostcall points, not arbitrary computation points. This matches GPU SIMT better (all threads in a warp are likely making a hostcall at the same time if they're processing the same pipeline stage)

### integration
- Start with a single-block design: one block, one executor, N threads each running the same task type. Validates the system before scaling.
- Measure occupancy first using Nsight Compute before claiming success — a working but 10% occupancy kernel is a known footgun
- Profile with CUDA Nsight Systems to identify whether hostcall latency or warp divergence is the primary bottleneck
- Consider a two-level design: outer level is standard data-parallel CUDA (high occupancy, many warps), inner level is async only for I/O-bound tasks (few warps, low occupancy acceptable)
- End-to-end correctness test: `async fn` that reads a file via hostcall, processes data, writes result via hostcall — validates all three subsystems together

---

## Summary: Key Risk Areas

| Risk | Severity | Mitigation |
|------|----------|------------|
| Warp divergence in async state machines | High | Warp-level tasks, homogeneous task types |
| Register pressure killing occupancy | High | `#[inline(never)]`, cap future complexity, measure with ptxas |
| System-scope atomic correctness | Critical | Validate PTX output, may need inline PTX |
| Hostcall latency hiding | Medium | Batching, CPU spin-poll, async fire-and-forget |
| Stack overflow from deep future chains | Medium | Monitor local memory usage per ptxas report |
| Heap allocator contention | Medium | Prefer stack allocation, limit dynamic allocation in hot paths |
| Panic unwinding | Low | `panic = "abort"` enforced at toolchain level |
| Missing POSIX primitives | Low | Hostcall shim covers most; identify gaps early |

The fundamental challenge is that GPU hardware is optimized for data-parallel SIMT execution, while async/await is optimized for I/O-concurrent, control-flow-divergent execution. VectorWare's approach is technically valid but operates against the hardware grain. The designs that work best will be those that impose data-parallel structure on the async tasks — same task type per warp, same yield points per cycle, batched I/O — rather than general-purpose async concurrency.
