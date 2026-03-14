# BS54 — Proposer Analysis: gpu-autonomous v2

**Epic**: gpu-autonomous v2 — GPU as Autonomous Compute Environment
**Date**: 2026-03-14
**Cycle**: 214

---

## Active Epics Assessment

### All Active Epics

| Epic ID | Title | Status | Progress |
|---------|-------|--------|----------|
| gpu-perf | GPU Inference Performance Optimization | active | Criterion 1 DONE (KV cache). Criteria 2-4 PARKED (out of scope) |
| real-std | Real std on GPU | active | Surface-level 5/5, but ThreadLocal uses `no_threads` — only single-thread safe |
| gpu-debug | GPU Debugging & Observability | active | Criteria 1-2 done (gpu_trace!, gpu_assert!). Criteria 3-4 pending (flight recorder, conditional compilation) |
| public-api | Public API — Host SDK | active (evergreen) | 3-4/4 criteria met |
| codebase-health | Codebase Health | active (evergreen) | Ongoing |
| gpu-autonomous | GPU Autonomous Compute v2 | active | Reopened with new criteria. v1 completed (GPT-2 inference, basic autonomous). v2 is 0/4 |

### gpu-autonomous v2 — Infrastructure Inventory

**What already exists** (reusable for v2):

1. **Hostcall RPC** (`gpu-host/src/hostcall.rs`, `gpu-runtime/src/lib.rs`):
   - Full GPU→host packet protocol with free/ready stacks, ABA-tagged lock-free CAS
   - Services: PRINT, FILE (open/read/write/close), STDIN, TIME, PANIC, TRACE, ASSERT, BULK_READ/WRITE
   - Sharded buffers for multi-block scaling
   - Adaptive polling listener with spin→sleep phases
   - I/O thread separation (fast path inline, slow path offloaded)

2. **Mapped memory** (`gpu-host/src/memory.rs`):
   - `MappedBuffer<T>` — RAII pinned device-mapped memory
   - `cuMemHostAlloc(DEVICEMAP|PORTABLE)` — visible from both GPU and CPU
   - Already used for hostcall buffer and sideband buffer

3. **WarpFuture state machines** (`gpu-kernel/src/pipeline.rs`):
   - `FileTransformFuture` — 16-state GPU-autonomous file transform pipeline
   - `BranchingPipelineFuture` — conditional state transitions based on hostcall results
   - `PipelinedComputeFuture` — overlapped I/O + compute
   - All demonstrate GPU driving multi-step workflows autonomously

4. **Sideband bulk transfer** — 1MB mapped buffer for >56-byte data transfers

5. **GpuRuntime** (`gpu-host/src/runtime.rs`):
   - `CudaDevice` wrapper with PTX loading, kernel launch, memory allocation
   - `alloc_zeros`, `htod_sync_copy`, `dtoh_sync_copy`

**What's missing** for v2 criteria:

1. **No persistent kernel pattern** — all kernels are launch-once-and-exit
2. **No command buffer** — GPU cannot receive commands after launch
3. **No session-mode hostcall** — `HostcallBuffer` + listener are per-launch
4. **No cross-launch device memory** — `CudaSlice` survives across launches (cudarc handles this), but never demonstrated
5. **No host→GPU signaling** — current protocol is GPU→host only (GPU pushes to ready stack, host polls)

---

## Systems Analysis

### Memory Model

**CUDA mapped memory lifetime:**
- `cuMemHostAlloc(DEVICEMAP|PORTABLE)` allocates pinned host memory visible from GPU
- Lifetime: from `cuMemHostAlloc` to `cuMemFreeHost` — survives across kernel launches
- The `HostcallBuffer` wraps this and drops in its `Drop` impl
- For session mode: simply don't drop the buffer between launches

**Device memory persistence across launches:**
- cudarc's `CudaSlice<T>` uses `cuMemAlloc` for device-only memory
- This memory persists until `CudaSlice` is dropped — it is NOT tied to kernel lifetime
- Kernel A can write to a `CudaSlice`, return, and Kernel B can read from the same `CudaSlice`
- The device pointer (`CUdeviceptr`) is just a u64 — pass it to both kernels
- **Key insight**: cross-launch persistent state is trivially achievable with cudarc's existing API

**Memory coherence for command buffer:**
- Mapped memory (`cuMemHostAlloc DEVICEMAP`) provides CPU↔GPU coherence
- System-scope atomics (`gpu_atomics::sys_*`) provide visibility guarantees
- GPU reads see CPU writes after `sys_spin_load_acquire_u32`
- CPU writes are visible to GPU through volatile/atomic stores
- This is exactly the mechanism needed for host→GPU command submission

### Hostcall Buffer Lifecycle — Current vs Session Mode

**Current pattern** (per-test in `tests_hostcall.rs`):
```
let hc_buf = HostcallBuffer::new(4);       // Allocate
let listener = spawn(|| hc_buf.listen()); // Start listener
launch_kernel(hc_buf.dev_ptr);            // Launch
dev.synchronize();                         // Wait
hc_buf.signal_shutdown();                  // Kill listener
listener.join();                           // Wait for thread
// hc_buf dropped → cuMemFreeHost
```

**Session mode** requires:
1. `HostcallBuffer` lives across multiple launches
2. Listener thread stays alive between launches
3. File descriptor table (`fd_table` in `io_thread_loop`) persists — files opened by Kernel A are still valid for Kernel B
4. Sideband buffer allocator needs reset between launches (or managed explicitly)

**Changes needed for session mode:**
- Factor `listen_unified` into `start_listening()` (returns handle) + `stop_listening()`
- OR: Add a `session()` method that returns a `HostcallSession` owning the buffer + listener
- The `fd_table` already uses `HashMap<u64, File>` — it naturally persists while the I/O thread lives
- Add `reset_sideband()` method to reset the bump allocator between launches
- Add `reinit_free_stack()` to reset packet pool without reallocating memory

### Command Buffer Design

For a persistent kernel that receives commands from the host, we need a **host→GPU command channel**. Two design options:

**Option A: Mapped memory ring buffer**
```
Header (64 bytes):
  write_idx: u64  — host increments after writing a command
  read_idx: u64   — GPU increments after processing a command
  capacity: u32   — number of command slots
  shutdown: u32   — set by host to signal kernel exit

Command slot (64 bytes each):
  cmd_type: u32   — COMPUTE, PRINT, EXIT, etc.
  payload: [u8; 60] — command-specific data
```
- Host writes command at `write_idx % capacity`, then atomically increments `write_idx`
- GPU polls `write_idx`, processes commands from `read_idx` up to `write_idx`
- Natural FIFO ordering, bounded buffer with backpressure

**Option B: Repurpose hostcall buffer as bidirectional**
- Add a "host→GPU ready stack" alongside the existing "GPU→host ready stack"
- Host pushes command packets; GPU pops and processes
- More complex but reuses existing tagged-pointer infrastructure

**Recommendation: Option A** — simpler, purpose-built, avoids overloading the hostcall protocol. The command buffer is a separate mapped allocation, small (a few KB), and orthogonal to the hostcall buffer.

### Concurrency: Persistent Kernel + Host Command Submission

A persistent kernel runs a main loop:
```
loop {
    // 1. Check for new commands (poll command buffer)
    // 2. If command available, dispatch:
    //    - COMPUTE: run computation on shared data
    //    - PRINT: use hostcall to print a message
    //    - EXIT: break the loop
    // 3. If no command, nanosleep and loop
}
```

**Host-side command submission:**
```rust
session.submit_command(Command::Compute { ... });  // Write to mapped memory
session.submit_command(Command::Print { msg: "status" });
session.submit_command(Command::Exit);
session.wait_for_kernel();  // cuCtxSynchronize
```

The persistent kernel and hostcall listener coexist:
- Listener thread polls hostcall ready stack (GPU→host)
- Main thread writes commands to command buffer (host→GPU)
- No conflict — different memory regions, different directions

---

## Compiler & Runtime Analysis

### Kernel Launch Model — Can CUDA Kernels Run Indefinitely?

**Yes, with caveats:**

1. **No inherent time limit in CUDA itself** — a kernel can loop forever as long as:
   - The GPU is not the display adapter (no watchdog), OR
   - The Windows TDR is disabled/extended (see GPU Architecture section)
   - No other CUDA context needs the GPU

2. **cudarc's `dev.synchronize()`** calls `cuCtxSynchronize()` which blocks until the kernel finishes — for a persistent kernel, the host must NOT call synchronize until it's ready for the kernel to exit.

3. **CUDA streams**: The default stream is synchronous. Using a non-default stream allows the host to continue operations (including launching other kernels) while the persistent kernel runs. cudarc supports this via `CudaStream`.

4. **Timeout behavior**: `cuLaunchKernel` returns immediately. The kernel runs asynchronously. The host can continue to interact with mapped memory while the kernel runs.

### CUDA Cooperative Groups

**Limited relevance for our use case:**
- Cooperative groups enable grid-wide synchronization (`cudaLaunchCooperativeKernel`)
- Useful if the persistent kernel uses multiple blocks that need to synchronize
- For a single-block persistent kernel (our v2 starting point), not needed
- Could be relevant in future for multi-block persistent kernels
- cudarc does NOT expose `cuLaunchCooperativeKernel` — would need raw CUDA driver API

### cudarc API — Persistent State and Multi-Launch

Key cudarc functions for v2:

1. **`CudaDevice::alloc_zeros::<T>(n)`** → `CudaSlice<T>` — device memory that persists until dropped
2. **`CudaSlice` device pointer** — accessible via `CudaSlice.device_ptr()` — pass as kernel arg
3. **`LaunchAsync::launch()`** — returns immediately, kernel runs on default stream
4. **`CudaDevice::synchronize()`** — blocks until all kernels complete
5. **`CudaDevice::htod_sync_copy()` / `dtoh_sync_copy()`** — can be called while a kernel on a different stream is running (with proper stream management)

**Missing from cudarc** (may need raw API):
- `cuStreamCreate` / `cuStreamSynchronize` for non-default streams (cudarc has `CudaStream` but limited API)
- `cuLaunchCooperativeKernel` for grid-wide sync
- TDR configuration is OS-level, not API-level

---

## GPU Architecture Analysis

### Persistent Kernel Patterns in CUDA

Production CUDA persistent kernels follow well-established patterns:

1. **NVIDIA's own persistent kernel** (CUDA Samples `persistentKernels`):
   - One block runs indefinitely, polling a work queue in global memory
   - Host fills the queue, kernel drains it
   - Used in: CUDA Graphs, NCCL, cuDNN internal implementations

2. **GPU task schedulers** (research/production):
   - A "scheduler block" runs persistently, dispatching tasks to worker warps
   - Used in ray tracing (OptiX) and graph processing (Gunrock)
   - Pattern: scheduler polls a task queue, assigns tasks, collects results

3. **Communication pattern for persistent kernels:**
   - **Global memory ring buffer** — most common, simplest
   - `__threadfence_system()` + volatile loads — our `sys_*` atomics are equivalent
   - Host writes command, GPU reads; GPU writes result, host reads

4. **Key insight from production usage:** Persistent kernels are almost always single-block or few-block to avoid SM starvation. They are a dispatcher pattern, not a bulk compute pattern.

### SM Occupancy

- A persistent kernel blocks at least 1 SM (32 threads minimum for 1 warp)
- Our GPU (likely RTX 3080/3090 from sm_86) has 68-82 SMs
- Blocking 1 SM is <2% capacity — negligible for single-GPU systems
- **Risk**: If the persistent kernel uses a full block (256+ threads), it wastes SM resources while idle (polling)
- **Recommendation**: Use minimal thread count (1 warp = 32 threads) for the persistent kernel

### Windows TDR (Timeout Detection and Recovery)

**This is the #1 technical risk for persistent kernels on Windows.**

- Windows has a Timeout Detection and Recovery (TDR) mechanism
- Default timeout: **2 seconds** — if a GPU kernel doesn't complete within 2 seconds, the driver kills it and resets the GPU
- This applies to the display GPU — if the GPU is also rendering the desktop, TDR WILL trigger
- **The project is running on Windows 11 Pro** (per environment info)

**Mitigations:**
1. **Registry setting**: `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\GraphicsDrivers`
   - `TdrDelay` (DWORD) — seconds before TDR triggers (default: 2)
   - `TdrLevel` (DWORD) — 0 = TDR disabled, 3 = default
   - Set `TdrDelay` to 300 (5 minutes) or `TdrLevel` to 0
   - **Requires system reboot**

2. **Use a non-display GPU**: If the system has 2 GPUs, use the non-display one for compute — TDR doesn't apply to compute-only GPUs

3. **Periodic kernel exit and relaunch**: The "soft persistent" pattern — kernel runs for 1 second, exits, host relaunches immediately. Combined with session mode, this avoids TDR while appearing persistent.

4. **WDDM→TCC driver mode**: Tesla/A100 cards can switch to TCC mode (no display). Consumer GPUs cannot.

**Recommendation for v2**: Start with the "soft persistent" pattern (criterion 1 variant) — kernel processes a batch of commands, then exits. Host relaunches if more commands pending. This avoids TDR entirely and naturally combines with criterion 4 (session persistence). True persistent kernels can be added later with TDR mitigation.

### Memory Hierarchy for Command Buffers

| Memory Type | Host Visible | GPU Visible | Atomic Support | Latency | Best For |
|-------------|-------------|-------------|----------------|---------|----------|
| Mapped (cuMemHostAlloc DEVICEMAP) | Yes (direct) | Yes (PCIe) | System-scope | ~5-10us | Command buffer, small messages |
| Device (cuMemAlloc) | No | Yes | Device-scope | ~200ns | Bulk compute data, persistent state |
| Unified (cuMemAllocManaged) | Yes (paged) | Yes (paged) | System-scope | Variable | Large shared datasets |
| Pinned host (cuMemHostAlloc) | Yes | No | N/A | N/A | DMA staging |

**For command buffer**: Mapped memory is the clear choice — same mechanism as the hostcall buffer, proven to work with system-scope atomics.

**For persistent state (cross-launch)**: Device memory (`CudaSlice`) is ideal — lowest latency for GPU access, no PCIe overhead, naturally persists across launches.

---

## Concrete Recommendations

### Theme 1: `cmd-buffer` — Persistent Kernel with Command Buffer
**Serves criterion 1**: Persistent kernel receives commands via shared memory

**Tasks:**
1. **`cmd-buffer.1`** (design): Define command buffer protocol in `gpu-protocol`
   - Ring buffer layout (header + command slots) in mapped memory
   - Command types: COMPUTE, PRINT, EXIT (minimum 3 per criterion)
   - Host→GPU signaling via write_idx atomic increment

2. **`cmd-buffer.2`** (experiment): Implement host-side `CommandBuffer` in `gpu-host`
   - `CommandBuffer::new(capacity)` → allocates mapped memory
   - `submit(&self, cmd: Command)` → writes command slot, increments write_idx
   - Integration with `HostcallSession` (theme 3)

3. **`cmd-buffer.3`** (experiment): Implement GPU-side command polling kernel
   - Kernel main loop: poll `write_idx`, dispatch commands, update `read_idx`
   - TDR-safe: process batch then exit if idle too long (soft persistent)
   - Use hostcall for PRINT command (reuse existing infrastructure)

4. **`cmd-buffer.4`** (experiment): Integration test — host submits COMPUTE → PRINT → EXIT, kernel processes all three
   - Verify command ordering, result correctness
   - Measure command latency (host submit → GPU acknowledge)

### Theme 2: `cross-launch` — Cross-Launch Persistent State
**Serves criterion 2**: Device buffer survives across kernel launches

**Tasks:**
1. **`cross-launch.1`** (investigation): Verify cudarc `CudaSlice` persistence across launches
   - Allocate `CudaSlice<u32>`, launch Kernel A to write pattern, synchronize
   - Launch Kernel B with same device pointer, read and verify pattern
   - Confirm zero host copy occurred (only device pointer passed)

2. **`cross-launch.2`** (experiment): Two-stage pipeline demonstration
   - Kernel A: compute partial results (e.g., prefix sum of first half), write to persistent buffer
   - Kernel B: read partial results, complete computation (e.g., reduce), write final answer
   - Host reads final answer via `dtoh_sync_copy`
   - Demonstrate with non-trivial computation (not just copy)

### Theme 3: `hc-session` — Hostcall Session Persistence
**Serves criterion 4**: Hostcall listener stays alive across launches

**Tasks:**
1. **`hc-session.1`** (design): Design `HostcallSession` API
   - `HostcallSession::start(config)` → creates buffer + starts listener thread
   - `session.buffer()` → returns buffer device pointer for kernel launch
   - `session.shutdown()` → signals listener, joins thread
   - File descriptor table persists across launches

2. **`hc-session.2`** (experiment): Implement `HostcallSession` in `gpu-host`
   - Factor existing `HostcallBuffer::listen_unified` into persistent listener
   - Add `reinit_packets()` to reset free/ready stacks without reallocation
   - Sideband bump allocator reset between launches

3. **`hc-session.3`** (experiment): Multi-launch session test
   - Session starts, Kernel A opens a file and writes data, synchronize
   - Reset packet stacks, Kernel B reads from same fd, synchronize
   - Verify data read matches data written — fd is valid across launches
   - Session shutdown, all files closed

### Theme 4: `auto-workflow` — GPU-Driven Autonomous Workflow
**Serves criterion 3**: Kernel autonomously decides operations based on intermediate results

**Tasks:**
1. **`auto-workflow.1`** (experiment): Full autonomous file processing kernel
   - Kernel receives input filename via mapped memory parameter
   - Autonomously: open file → read content → process (e.g., word count/transform) → decide output action → write result file → print summary
   - Decision point: if content size > threshold, write summary; else write full content
   - Host only provides: kernel launch + hostcall services
   - Uses existing WarpFuture state machine pattern from `pipeline.rs`

2. **`auto-workflow.2`** (experiment): Multi-stage autonomous pipeline with branching
   - Extend auto-workflow.1: read config file → based on config, choose processing algorithm → apply → write output
   - Demonstrates GPU making runtime decisions that affect program flow
   - At least 2 branch points based on data content

### Priority Ordering

1. **`cross-launch` (criterion 2)** — Start here. Lowest risk, simplest to implement, validates a fundamental CUDA property. Builds confidence. 1-2 tasks, fast to complete.

2. **`hc-session` (criterion 4)** — Second priority. Required by criteria 1 and 3. Session persistence is the foundation for persistent kernels and autonomous workflows. Medium complexity, well-understood refactoring of existing code.

3. **`cmd-buffer` (criterion 1)** — Third priority. Depends on session persistence (theme 3). The most novel component — command buffer protocol is new infrastructure. Medium-high complexity. TDR is a real risk on Windows.

4. **`auto-workflow` (criterion 3)** — Last priority. Largely already demonstrated by `FileTransformFuture` and `BranchingPipelineFuture` in `pipeline.rs`. The existing pipelines already show GPU-driven autonomous workflows with branching. The new element is making decisions based on intermediate results (data-dependent branching), which `BranchingPipelineFuture` already does (checks if file exists → branches). This criterion may be partially satisfiable by combining existing patterns with session persistence.

**Rationale**: The ordering follows dependency chains (session before persistent kernel), risk gradient (low→high), and novelty (validate known properties first, then build new infrastructure).

---

## Risk Assessment

### Criterion 1: Persistent Kernel
**Risk: HIGH**

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Windows TDR kills persistent kernel | Critical | Very High (default 2s timeout) | Soft-persistent pattern (batch + relaunch), or user modifies TDR registry |
| Persistent kernel wastes SM resources while idle | Medium | Certain | Use minimal thread count (1 warp), implement nanosleep in poll loop |
| Kernel hang → requires GPU reset | High | Medium | Heartbeat mechanism: kernel writes timestamp, host monitors |
| Command buffer race conditions | Medium | Medium | Use proven system-scope CAS pattern from hostcall protocol |

**Unknown unknowns**: How does cudarc handle context state when a kernel is still running? Can we query kernel completion status without blocking? What happens if the host tries to free memory while a persistent kernel is reading it?

### Criterion 2: Cross-Launch Persistent State
**Risk: LOW**

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| cudarc drops CudaSlice unexpectedly | Low | Very Low | Hold reference in host scope |
| Device memory not coherent between launches | Low | Very Low | cuCtxSynchronize between launches ensures visibility |
| cudarc API doesn't expose raw device pointer | Low | Low | CudaSlice has `device_ptr()` or `&CudaSlice` passed to launch |

This is essentially verifying a well-known CUDA property. The main risk is API ergonomics in cudarc, not fundamental capability.

### Criterion 3: GPU-Driven Autonomous Workflow
**Risk: LOW-MEDIUM**

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Already demonstrated by existing pipeline.rs | N/A | Certain | Reuse patterns, extend with data-dependent decisions |
| Data-dependent branching causes warp divergence | Medium | Low | All autonomous decisions are lane-0-decides-then-broadcast (existing pattern) |
| Complex state machines are hard to debug | Medium | Medium | Use gpu_trace! for state transition logging |

Most of the infrastructure already exists. This is more of a demonstration/integration task than new capability development.

### Criterion 4: Hostcall Session Persistence
**Risk: MEDIUM**

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Packet pool state corruption across launches | High | Medium | `reinit_packets()` must carefully reset without touching in-flight state |
| File descriptor table grows unbounded | Low | Low | Add cleanup/GC for closed fds |
| Listener thread termination/restart | Medium | Low | Session owns thread lifecycle with proper shutdown signaling |
| Sideband allocator fragmentation | Low | Low | Reset allocator between launches (already done in tests) |

The main risk is getting the packet pool reinitialization right — all packets must be returned to the free stack, ready stack cleared, but the shutdown flag NOT set.

### Overall Assessment

**Hardest criterion**: Persistent kernel (criterion 1) due to Windows TDR. This is the most likely to require user intervention (registry edit or TDR disable). The soft-persistent workaround reduces this to medium difficulty.

**Most unknown unknowns**: Persistent kernel behavior under Windows WDDM. Specific behaviors when host interacts with CUDA context while a kernel is running are underspecified in documentation and may vary between driver versions.

**Easiest criterion**: Cross-launch persistent state (criterion 2). This is a verification task — the CUDA memory model guarantees this behavior. The only work is writing a test that demonstrates it.

**Best starting point**: Criterion 2 (cross-launch) → Criterion 4 (session) → Criterion 1 (persistent kernel) → Criterion 3 (autonomous workflow). This order minimizes risk exposure and builds infrastructure incrementally.
