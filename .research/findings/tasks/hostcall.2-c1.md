# hostcall.2: Existing Hostcall Implementations
**Date**: 2026-03-11
**Cycle**: 1
**Theme**: hostcall
**Kind**: investigation
**Status**: done

## Summary

AMD ROCm implements a production-grade hostcall protocol using a lock-free packet ring buffer with doorbell signaling, where GPU warps push service requests onto an atomic "ready stack" and a host listener thread drains it via adaptive-timeout polling. NVIDIA CUDA implements device-side printf via a pre-allocated FIFO circular buffer (configurable via `cudaLimitPrintfFifoSize`) that the host drains at kernel completion or explicit synchronization points. The ROCm ring-buffer approach — with per-warp packet slots, tagged-pointer stacks, and multi-packet message reassembly — is the superior pattern for a general-purpose hostcall RPC system and should directly inform our `hostcall.3` design.

---

## Detailed Findings

### Q1: AMD ROCm Hostcall Protocol

**Source**: `ROCm/clr` (Compute Language Runtime) repository, `rocclr/device/` directory.
The ROCm repository was originally at `ROCm/ROCm-Device-Libs` but has moved to `ROCm/llvm-project` (AMD LLVM fork) for device-side code and to `ROCm/clr` for the runtime host-side code. The host-side implementation lives in:

- `rocclr/device/devhostcall.hpp` — data structures and interface
- `rocclr/device/devhostcall.cpp` — listener thread and packet dispatch
- `rocclr/device/devhcmessages.hpp` / `devhcmessages.cpp` — multi-packet message reassembly
- `rocclr/device/devhcprintf.cpp` — printf service handler

#### Buffer Structure

```
HostcallBuffer {
    PacketHeader*  headers_       // array of packet headers
    Payload*       payloads_      // array of payloads, parallel to headers_
    void*          doorbell_      // device signal object for GPU->host notification
    uint64_t       free_stack_    // tagged pointer: head of free packet stack
    atomic<uint64_t> ready_stack_ // tagged pointer: head of ready packet stack
    uint64_t       index_mask_    // bitmask for extracting packet index from tagged ptr
    const amd::Device* device_
}

PacketHeader {
    uint64_t          next_        // tagged pointer to next packet in stack
    uint64_t          activemask_  // bitmask: which of the 64 work-items are valid
    uint32_t          service_     // service ID (printf=2, function_call=1, devmem=3)
    atomic<uint32_t>  control_     // bit 0 = READY flag
}

Payload {
    uint64_t slots[64][8]  // 64 work-items, 8 uint64_t args each
}
```

The key property: one `Payload` holds data for an entire SIMT wave (64 work-items). Each work-item gets 8 `uint64_t` argument slots. The first slot (index 0) is always the **message descriptor**.

#### Service IDs

```
SERVICE_RESERVED      = 0
SERVICE_FUNCTION_CALL = 1
SERVICE_PRINTF        = 2
SERVICE_DEVMEM        = 3
SERVICE_SANITIZER     = 4  (conditional, ASan builds only)
```

#### Tagged Pointer Design

Both `free_stack_` and `ready_stack_` use **tagged pointers** to implement lock-free stacks using `atomic<uint64_t>` CAS. A tagged pointer encodes `(ABA_tag, packet_index)` in a single 64-bit word. The constraint is: "the tag and the index part must never both be zero simultaneously." This prevents ABA hazards that would cause packet corruption under concurrent warp access.

#### GPU-Side Protocol (device code in LLVM amd/device-libs)

1. GPU warp pops a free packet from `free_stack_` via CAS loop.
2. Writes service ID, active mask, and per-work-item arguments into the packet.
3. Sets `control_.READY = 0` (packet being prepared).
4. Pushes the packet onto `ready_stack_` via atomic exchange.
5. Rings the doorbell: writes a new value to the signal object.

#### Host-Side Protocol (listener thread in devhostcall.cpp)

1. The listener thread calls `doorbell_->Wait(signal_value, Condition::Ne, timeout)`.
2. On wakeup: atomically exchanges `ready_stack_` with `nullptr` (grabs all pending packets).
3. Walks the resulting linked list (LIFO order), for each packet:
   a. Reads `service_` and `activemask_`.
   b. Dispatches to the appropriate service handler.
   c. For each active work-item in the mask, invokes the handler with `payload.slots[i]`.
   d. Clears `control_.READY` with `memory_order_release` (signals GPU the response is ready).
4. GPU side polls `control_.READY` with `memory_order_acquire` to detect completion.
5. Pushes the packet back onto `free_stack_`.

#### Adaptive Timeout Algorithm

The listener uses a self-tuning backoff rather than fixed polling:

```
kTimeoutFloor = K * K * 4    (fast path)
kTimeoutCeil  = K * K * 16   (slow path, K is a hardware-dependent constant)

on signal detected: timeout = max(timeout >> 1, kTimeoutFloor)
on timeout expired: timeout = min(timeout << 1, kTimeoutCeil)
```

This halves latency during active bursts and doubles sleep time during idle periods, balancing host CPU utilization against hostcall response latency.

#### Multi-Packet Message Reassembly (devhcmessages)

For services like printf whose data exceeds one packet's 7 data slots, the protocol uses **message fragmentation**:

The **message descriptor** (slots[i][0]) encodes:
```
Bit 0:     BEGIN flag (first packet of a message)
Bit 1:     END flag   (last packet of a message)
Bits 2-4:  reserved, must be zero
Bits 5-7:  length (number of valid data qwords: 1-7, from slots[i][1..7])
Bits 8-63: 56-bit message ID
```

The host-side `MessageHandler` maintains a pool of `Message` objects (slot reuse). On BEGIN: allocates a new message slot and updates the ID. On continuation packets: looks up by ID and appends data. On END: dispatches the fully-assembled message to the printf handler, then frees the slot.

This allows arbitrarily large format strings and argument lists without GPU heap allocation — all fragmentation is handled with the pre-allocated packet pool.

#### Printf Argument Encoding (devhcprintf.cpp)

Format strings are transmitted as a null-terminated char array packed into `uint64_t` slots. Arguments follow:
- Integer types (`d`, `i`, `o`, `u`, `x`, `X`, `c`): raw `uint64_t`.
- Floating-point (`f`, `F`, `e`, `E`, `g`, `G`, `a`, `A`): stored as `double` via `memcpy` into `uint64_t`.
- String (`s`): device pointer; padding = `(strlen + 1 + 7) / 8` qwords.
- Pointer (`p`): device pointer as `void*`.
- Dynamic width/precision (`*`): preceding argument slots supply the width values.

The host reconstructs the format string and calls system `printf()` per specifier. The LSB of the control word selects stdout vs stderr.

---

### Q2: CUDA printf Implementation

**Source**: CUDA C++ Programming Guide section 10.35 (Formatted Output), CUDA Runtime API `cudaDeviceSetLimit`.

#### Mechanism

CUDA device-side `printf()` uses a **pre-allocated circular FIFO buffer** in device-accessible pinned memory (or unified memory):

1. At kernel launch, the CUDA runtime has reserved a FIFO of configurable size for printf output.
2. Each GPU thread calling `printf()` atomically writes its format string (as a pointer or inline bytes) and binary argument data into the FIFO. Each entry is a variable-length record containing: the format string reference, argument count, and argument values.
3. The FIFO is **not drained during kernel execution** — the GPU does not signal the host per-call.
4. After kernel completion (or at an explicit `cudaDeviceSynchronize()`), the CUDA runtime reads all entries from the FIFO and calls the host-side libc `printf()` for each entry.

#### Buffer Size Control

```c
// Must be called before launching any printf-using kernel
cudaDeviceSetLimit(cudaLimitPrintfFifoSize, size_in_bytes);

// Verify actual size (runtime may adjust for alignment)
size_t actual_size;
cudaDeviceGetLimit(&actual_size, cudaLimitPrintfFifoSize);
```

The default size is **1 MB** (implementation-defined but documented as 1 MB for most CUDA versions). If the FIFO overflows, excess printf output is silently dropped.

#### Key Limitations

- Output ordering is not guaranteed across threads (SIMT execution means no deterministic order).
- No response/return path: the GPU thread does not receive the return value of printf.
- No synchronization during kernel execution: all output appears only after the kernel finishes or `cudaDeviceSynchronize()` is called.
- The FIFO size limit means high-throughput printf from thousands of threads can overflow quickly.
- `cudaLimitPrintfFifoSize` cannot be changed after any printf-capable kernel has launched.

#### Architectural Difference from ROCm

CUDA printf is fundamentally a **write-only unidirectional log buffer**. It is not a general RPC mechanism — there is no way for the GPU thread to receive a response from the host. The ROCm hostcall protocol is bidirectional: the GPU waits on the `control_.READY` flag for the host's response, enabling true remote procedure calls.

---

### Q3: Ring Buffer vs Double Buffer

Based on the ROCm and CUDA implementations, plus general GPU communication literature:

#### ROCm's Choice: Lock-Free Stack (Not a Ring Buffer)

Strictly speaking, ROCm does not use a traditional ring buffer. It uses **two lock-free tagged-pointer stacks** (free list + ready list) over a fixed-size packet array. This is a pool allocator with work-stealing semantics.

**Advantages of the ROCm pool/stack approach:**
- No head/tail pointer contention — multiple warps can simultaneously push to `ready_stack_` via CAS.
- No ordering guarantee needed (LIFO is fine for independent service requests).
- No wrap-around index arithmetic — tagged pointers handle reuse safely.
- Pool exhaustion is detectable: CAS on `free_stack_` returns null when no packets are available.

#### CUDA's Choice: FIFO Ring Buffer

CUDA printf uses a classic FIFO ring buffer because:
- Output ordering matters (producers want FIFO semantics for log-like output).
- Single-producer-per-thread model with a global ring pointer.
- No response needed, so no need for per-slot "done" flags.

#### Double Buffer Analysis

A **double buffer** (ping-pong buffer) is simpler but has critical drawbacks for GPU hostcall:
- Only one buffer active at a time — while the host processes buffer A, all GPU threads must wait before writing to buffer B. This creates a barrier that stalls the entire kernel.
- No concurrent warp-level parallelism: all warps across all blocks share the two buffers.
- Works acceptably for bulk transfer (like compute output staging) but poorly for per-warp RPC.

#### Recommendation for hostcall.3

For our use case (general GPU-to-host RPC where GPU threads wait for responses), the **ROCm pool/stack model** is clearly superior:

| Criterion | ROCm Pool+Stack | CUDA FIFO | Double Buffer |
|---|---|---|---|
| Multi-warp concurrency | Excellent (lock-free CAS) | Good (atomic ring ptr) | Poor (barrier) |
| Bidirectional RPC | Yes (READY flag handshake) | No (write-only) | Possible but awkward |
| Pool exhaustion detection | Yes (null from free stack) | Implicit (overflow drops) | N/A |
| Implementation complexity | Medium | Low | Low |
| Latency per call | Low (doorbell + adaptive wait) | Zero (batch at sync) | High (stall all warps) |

We should adopt the ROCm two-stack pool design with the doorbell signaling mechanism.

---

### Q4: Error Handling and Timeout Mechanisms

#### ROCm Error Handling

From the source code analysis:

1. **Pool exhaustion**: If `free_stack_` is empty, the GPU-side code must spin until a packet is returned. In practice, the packet pool is sized to `num_warps * padding_factor` to make exhaustion unlikely.

2. **Invalid service ID**: The host handler calls `guarantee(false, "Unknown service: %d", service)`, which logs an error and may abort — no silent failure.

3. **Bad device pointer in SERVICE_DEVMEM**: The handler logs an error: "Unknown pointer in devmem hostcall" and continues processing other packets.

4. **ASan integration**: The `SERVICE_SANITIZER` service routes to the address sanitizer handler, which has its own error reporting path.

5. **Graceful shutdown**: The listener thread exits its main loop when the doorbell signal equals `SIGNAL_DONE` (value 0). The `disableHostcalls()` function writes `SIGNAL_DONE` to the doorbell, causing the thread to drain remaining packets and exit.

#### ROCm Timeout Mechanism

The adaptive timeout described in Q1 serves dual purpose:
- **Prevents CPU busy-spin**: When no hostcalls are occurring, the listener sleeps up to `kTimeoutCeil` cycles between checks.
- **Handles GPU stalls**: If a GPU kernel deadlocks or crashes without signaling `SIGNAL_DONE`, the timeout prevents the listener from hanging forever — it will eventually wake up, find no packets, and continue (though it cannot detect a kernel crash independently).

There is **no explicit RPC timeout** on the GPU side. A GPU warp that issues a hostcall and spins on `control_.READY` will spin indefinitely if the host listener crashes or becomes stuck. This is a known limitation.

#### CUDA Error Handling

- Printf FIFO overflow: silently drops excess output. No error is reported to the GPU thread.
- No timeout mechanism: the GPU thread never waits for a response.
- Buffer size misconfiguration: `cudaErrorInvalidValue` returned if `cudaDeviceSetLimit` is called after kernel launch.

#### Lessons for Our Design

1. The ROCm approach has no per-RPC timeout on the GPU side — this is a significant gap for a robust system. We should add a bounded spin-count before returning an error code.
2. Pool exhaustion should return a sentinel error (e.g., `u64::MAX`) rather than spinning indefinitely.
3. A separate "error channel" bit in the response payload should signal host-side failures back to the GPU.
4. The listener thread should set a flag on orderly shutdown so in-flight RPCs can detect it.

---

## Unexpected Discoveries

1. **ROCm does NOT use a ring buffer** — the two-stack lock-free pool is architecturally cleaner and more appropriate for GPU workloads where arrival order does not matter.

2. **CUDA printf is fundamentally write-only** — it is not a hostcall RPC system at all. It has no response mechanism. VectorWare's design for a bidirectional hostcall is entirely custom (not based on CUDA printf internals).

3. **The ROCm packet pool is warp-granular** — a single packet slot covers all 64 work-items in a wave via the `activemask_` field and the `slots[64][8]` payload array. This means we need to think carefully about NVIDIA warps (32 threads) vs AMD waves (64 threads) when adapting the design.

4. **Multi-packet message reassembly is non-trivial** — the BEGIN/END/message-ID protocol in ROCm devhcmessages is more complex than expected. Our printf-equivalent will need something similar, or we can simplify by enforcing a maximum message size that fits in one packet.

5. **ROCm's SERVICE_FUNCTION_CALL** — this is a general-purpose function pointer invocation service. The GPU can pass a function pointer from the host's address space, and the host will call it with the provided arguments. This is a powerful but dangerous primitive — essentially arbitrary code execution on the host triggered by GPU code.

6. **VectorWare uses CUDA streams** to prevent GPU blocking during host request processing (mentioned in the rust-std-on-gpu blog). This suggests they use asynchronous hostcall dispatch rather than synchronous blocking, which is different from the ROCm model where the GPU warp busy-waits.

---

## Key Conclusions

1. **Adopt the ROCm two-stack pool design** for `hostcall.3`. The architecture is well-tested, handles multi-warp concurrency correctly, and supports bidirectional RPC.

2. **Key parameters to tune**: packet pool size (suggest: `num_SMs * max_warps_per_SM * 2` as starting point), slots per work-item (ROCm uses 8 × u64; start with the same), wave size (32 for NVIDIA vs 64 for AMD — use a compile-time constant).

3. **Doorbell mechanism**: On NVIDIA, this maps to a pinned-memory `AtomicU64` that the GPU writes and the host polls. We cannot use AMD's HSA signal objects. We need to verify that host-side `AtomicU64::load(SeqCst)` on pinned memory is visible to GPU writes (requires `.sys`-scoped atomics — see `atomics.1`).

4. **No CUDA printf reliance**: We should not build on `printf` as a communication primitive. Our hostcall must be an independent protocol built on pinned shared memory.

5. **Avoid double-buffering**: The stall-all-warps semantics make it unsuitable for concurrent RPC from multiple warps.

6. **Add what ROCm lacks**: GPU-side RPC timeout (bounded spin count), pool-exhaustion error codes, and host-side shutdown signaling.

---

## Open Questions

1. **Doorbell on NVIDIA**: What is the correct atomic ordering for a GPU-side write and host-side read on pinned memory? Does Rust's `AtomicU64` compile to `.sys`-scoped PTX atomics on `nvptx64`? (Blocked on `atomics.1`)

2. **Wave size**: Should we hard-code 32 (NVIDIA warp) or make it a const generic? ROCm hard-codes 64. A warp-generic design adds complexity but supports both platforms.

3. **Packet pool sizing**: The right number of packets depends on the kernel's warp count and hostcall frequency. ROCm computes it from `numPackets` passed at runtime. Should we use a static pool or dynamic allocation?

4. **VectorWare's CUDA streams approach**: How exactly does VectorWare use CUDA streams to avoid GPU blocking? Is this a fundamentally different protocol from ROCm's spin-wait model? This warrants follow-up investigation.

5. **SERVICE_FUNCTION_CALL security**: A general function pointer call service is powerful but requires trust that GPU code is not adversarial. For our use case (we control both sides), this is acceptable — but should we implement this, or limit to specific service IDs?

6. **Response size**: ROCm returns responses in the same 8-slot payload (in-place overwrite). Is 7 × u64 = 56 bytes sufficient for all our use cases (e.g., returning a file descriptor + error code is trivially fine; returning read bytes might need more)?

---

## Impact on Downstream Tasks

- **hostcall.1** (shared memory mechanisms): The ROCm design confirms pinned mapped memory is the right approach. The doorbell should be a single `AtomicU64` in pinned memory. The packet buffer itself should be a `cudaHostAlloc`-allocated region.

- **hostcall.3** (design and implement protocol): Directly implementable based on this research. Key decisions resolved: two-stack lock-free pool, warp-granular packets (32 slots for NVIDIA), 8 × u64 per slot, doorbell signal, adaptive-timeout host listener thread. Remaining decisions: wave size, pool sizing formula, timeout constants.

- **atomics.1** (PTX scope verification): The correctness of the entire hostcall protocol depends on GPU-side atomic writes being visible to the host. This must be confirmed before `hostcall.3` experiments begin.

- **gpu-std.1** (std dependency graph): The ROCm `SERVICE_PRINTF` implementation reveals the complete printf argument encoding scheme. We can reuse this encoding for our own printf service, reducing design effort.

- **hostcall.4** (GPU println experiment): The multi-packet message fragmentation protocol from ROCm devhcmessages is the right model. However, for a first experiment, we can start with a simplified version: enforce max one packet per printf call (max ~42 bytes of format string + args), avoiding the reassembly complexity.

---

## Theme Progress

**hostcall** theme: `hostcall.2` done. The investigation resolves the core design question (ring buffer vs pool: use pool). `hostcall.3` (design) can now begin in parallel with `atomics.1` (which is a prerequisite for the experiment phase `hostcall.4`). The ROCm source code provides a complete reference implementation that we can directly translate into Rust for `hostcall.3`.
