# BS14 Proposer — Next Research Directions
**Date**: 2026-03-12
**Brainstorm seq**: 14
**Role**: Proposer
**Level**: deep (proposer + skeptic)
**Trigger**: Project product-ready (59/59 tasks, 15 themes completed, 3 parked). Assess next frontiers.

---

## Systems Analysis (memory models, ABI, unsafe boundaries)

### New capabilities that could be unlocked

**1. Large-payload hostcall (streaming protocol)**
The current protocol caps payloads at 56 bytes per slot (7 slots × 8 bytes for PRINT/READ, 6 slots for WRITE). This is a hard architectural limit — any operation needing more data (e.g., reading a 4KB file block) requires multiple round-trips at 13µs each. A streaming extension could:
- Allow multi-packet transfers for a single logical operation
- Use a dedicated bulk-transfer region in shared memory (ring buffer)
- Keep the existing packet protocol for control, add a sideband data channel

**2. GPU-side memory allocator hardening**
The current `SERVICE_MALLOC` / `SERVICE_FREE` services exist in the protocol but their GPU-side integration is minimal. A proper GPU-side bump allocator (backed by a pre-allocated CUDA device memory region) would enable:
- Dynamic string formatting on GPU (currently limited to 56-byte fixed buffers)
- Arbitrary-sized hostcall payloads via pointer indirection
- More complex async task state without stack spilling

**3. Unsafe boundary audit**
The project has significant `unsafe` surface area across 7 crates. Key boundaries:
- `gpu-atomics`: All inline PTX asm — correct by inspection but no Miri/sanitizer coverage
- `gpu-host/hostcall.rs`: Raw pointer arithmetic for packet field access — offset constants must match between GPU and host
- `HostcallBuffer` is `Send + Sync` via manual impl — safe only if the protocol invariants hold
- `gpu-libc`: C ABI shim with `#[no_mangle]` — no type checking at the boundary

A systematic unsafe audit (perhaps via `cargo-careful` on host-side, manual review on GPU-side) would increase confidence but is low priority for a research project.

### Memory model improvements

**4. Relaxed-ordering fast path**
Currently all GPU-CPU communication uses `sys` scope atomics (system-scope acquire/release). For intra-GPU communication (e.g., between warps on the same SM), `gpu` scope atomics would be sufficient and faster. The `gpu-atomics` crate could expose a `gpu_store_release_u32` / `gpu_load_acquire_u32` family for intra-GPU synchronization, reserving `sys` scope for cross-domain communication only.

**5. Memory-mapped I/O region**
Instead of encoding data in packet payload slots, a shared memory-mapped region could allow the GPU to write arbitrary-length data to a known address and signal the host with a single hostcall containing just an offset + length. This would decouple control flow (hostcall protocol) from data transfer (shared memory DMA).

---

## Compiler Analysis (rustc, LLVM, codegen, PTX backend)

### Nightly features to test

**6. `gpu-kernel` ABI migration**
The `extern "gpu-kernel"` ABI (RFC #3637) is available in nightly and intended to replace `extern "ptx-kernel"` for cross-vendor GPU support. While ADR-1 chose `ptx-kernel` (correct at the time), the `gpu-kernel` ABI is now more mature. Testing the migration would:
- Validate forward compatibility with future Rust stable GPU support
- Potentially unlock AMDGPU support via `amdgcn-amd-amdhsa` target
- Require zero code changes (same LLVM calling convention on NVPTX)

**7. Nightly unpin from 2025-08-25**
The project is pinned to `nightly-2025-08-25` due to a PTX header bug in `llvm-bitcode-linker`. Testing newer nightlies (especially post-LLVM 20 merge) could:
- Fix the PTX header bug upstream (eliminating the build.rs workaround)
- Bring improved PTX codegen (LLVM 20 has better nvptx register allocation)
- Enable new features (`asm_goto`, improved inline asm constraints)

Risk: any nightly update can break the build. CI pipeline mitigates this.

### Compile time reduction

**8. Incremental PTX compilation**
Currently `lto = "fat"` is required for Embassy cross-crate inlining. This makes every rebuild a full LTO pass. Potential mitigations:
- Split gpu-kernel into multiple cdylib targets (one per kernel family)
- Investigate thin LTO (`lto = "thin"`) — may not resolve Embassy calls
- Pre-compiled PTX caching in build.rs (skip recompilation if sources unchanged)

**9. PTX optimization opportunities**
- `__launch_bounds__` equivalent via `#[nvptx::maxntid]` could constrain register allocation for high-occupancy kernels
- The `hostcall_file_test` kernel (888 virtual registers) could be split into smaller kernels (one per file operation) to reduce register pressure
- `nanosleep.u32 64` in spin loops could be tuned — 64ns may be too short for high-contention scenarios (increase to 256ns or 1µs)

---

## GPU Architecture Analysis (warp model, memory hierarchy, occupancy)

### Warp-cooperative execution

**10. Warp-cooperative hostcall (unpark warp-coop theme?)**

Current state: each lane independently does CAS on the free stack → 32 CAS operations per warp. At 128 threads (4 warps), CAS retry rate is 49 retries/call.

Warp-cooperative approach: elect one lane per warp to do CAS, broadcast packet pointer via `shfl.sync`. This reduces CAS operations from 32 per warp to 1 per warp — a 32x reduction in atomic contention.

**Should we unpark?** Analysis:

Arguments FOR:
- The benchmark data now exists (benchmark.2 showed CAS contention is the GPU-side bottleneck)
- 49 CAS retries/call at 128 threads × 100ns/CAS = ~5µs wasted per call per thread
- At 512 threads, 75% starvation is directly caused by free-stack contention
- Warp-cooperative would reduce effective contenders from 512 to 16 (warps), making the protocol viable at higher thread counts

Arguments AGAINST:
- The host listener (single-threaded, ~28K calls/s) is still the throughput ceiling — warp-coop reduces GPU-side contention but doesn't increase host throughput
- Per-lane async hostcall (ADR-4) is fundamentally incompatible with warp-cooperative allocation — lanes reach hostcall at different times during async execution
- Implementation requires `bar.sync` or `match_sync` within warp, adding complexity

**Verdict: Conditional unpark.** Unpark ONLY if paired with multi-threaded host listener. Without that, warp-coop reduces contention but doesn't increase throughput.

### Memory hierarchy optimizations

**11. Shared memory packet cache**
On SM_86, each SM has 100KB of shared memory (configurable vs L1). A per-block packet cache in shared memory could:
- Pre-fetch free packets into shared memory (one CAS per block instead of per thread)
- Distribute packets to threads within the block via `__syncthreads()` + shared memory reads
- Reduce global memory traffic for the free stack

This is architecturally similar to warp-coop but at block granularity. More impactful but more complex.

**12. Local memory spilling mitigation**
All async kernels spill to local memory (confirmed by benchmark.3 SP/SPL analysis). Mitigations:
- Reduce Embassy executor state size (custom stripped executor)
- Use `maxrregcount` to force ptxas to use fewer registers (trades occupancy for spilling)
- Investigate whether the spilling is from the executor state machine or from hostcall protocol code

### Multi-GPU support

**13. Multi-GPU feasibility**
The current architecture is single-GPU:
- One `HostcallBuffer` per kernel launch
- One host listener thread per buffer
- `cuMemHostAlloc(DEVICEMAP)` is per-context

Multi-GPU would require:
- Per-device hostcall buffers
- Per-device listener threads (or a shared thread pool)
- Cross-device synchronization for shared file descriptors

Feasibility: straightforward engineering, not research. The protocol is already device-independent (uses system-scope atomics on the GPU side, host-visible mapped memory). The main work is plumbing multiple CUDA contexts.

---

## New Research Directions

### High-value new capabilities

**14. Multi-threaded host listener (P0)**
This is the single highest-impact improvement. Current bottleneck: 28K calls/s max throughput, 75% starvation at 512 threads. A multi-threaded listener could:
- Partition the ready stack into N segments (one per listener thread)
- Or use a work-stealing pool (N threads compete on the single ready stack)
- Target: linear throughput scaling up to N threads

This directly addresses the #1 known limitation ("throughput does not scale").

**15. GPU-side computation offload patterns (P1)**
The project proves GPU can do I/O. The next frontier: GPU doing I/O AND compute. Patterns:
- **Map-reduce with I/O**: GPU reads file chunks, processes in parallel, writes results
- **Async data pipeline**: GPU streams data from host → processes → streams back
- **Heterogeneous workload**: some warps do compute, others do I/O (warp specialization)

This would demonstrate that the technology is useful beyond proof-of-concept.

**16. GPU-to-GPU communication (P2)**
Two GPUs communicating via hostcall relay:
- GPU-A sends hostcall → host relays → GPU-B receives via reverse hostcall
- Or use NVLink/PCIe peer-to-peer with system-scope atomics directly
- Enables distributed GPU computing with Rust async

Ambitious but would be a unique contribution.

**17. Developer tooling: GPU panic with backtrace (P1)**
Current panic handler is `loop {}` — a hard hang. A better panic handler could:
- Format the panic message into a hostcall packet
- Include thread ID, block ID, warp ID for diagnosis
- Signal the host to abort the kernel
- Print the panic message on the host side

This is high value for anyone trying to use the system for real development.

**18. GPU-side logging/tracing (P1)**
Beyond `println!()`, structured logging:
- `gpu_trace!("processing element {}", idx)` with thread/warp/block metadata
- Host-side log aggregation with deduplication (many threads print same message)
- Severity levels (trace/debug/info/warn/error)
- Compile-time log level filtering to eliminate dead code

### What would make this uniquely valuable beyond VectorWare?

VectorWare demonstrated the concept. This project could differentiate by:
1. **Scalability**: Multi-threaded host listener + warp-coop → handle 1000+ GPU threads
2. **Developer experience**: Panic backtraces, structured logging, better error messages
3. **Real workloads**: End-to-end examples that actually compute something useful
4. **Portability**: `gpu-kernel` ABI migration → path to AMD GPU support

---

## Concrete Recommendations

### New themes and tasks

| # | Theme | Priority | Rationale |
|---|-------|----------|-----------|
| 1 | `host-scaling` | **P0** | Multi-threaded host listener — directly addresses #1 bottleneck |
| 2 | `gpu-panic` | **P1** | GPU panic handler with hostcall reporting — critical for DX |
| 3 | `large-payload` | **P1** | Streaming/bulk data transfer — enables real workloads |
| 4 | `nightly-update` | **P1** | Unpin from 2025-08-25, test gpu-kernel ABI — future-proofing |
| 5 | `gpu-logging` | **P2** | Structured logging with metadata — nice-to-have DX |
| 6 | `workload-demo` | **P2** | Real computation example (e.g., parallel grep) — showcase value |

### Detailed task breakdown

**host-scaling (P0)**:
- `host-scaling.1` (design): Multi-threaded listener architecture — partitioned vs work-stealing vs thread-per-packet
- `host-scaling.2` (experiment): Implement multi-threaded listener with thread pool
- `host-scaling.3` (experiment): Benchmark throughput scaling (1/2/4/8 listener threads × 32/128/512 GPU threads)

**gpu-panic (P1)**:
- `gpu-panic.1` (design): GPU panic handler design — message encoding, hostcall service, kernel abort mechanism
- `gpu-panic.2` (experiment): Implement panic handler that sends formatted message via hostcall + triggers kernel exit

**large-payload (P1)**:
- `large-payload.1` (design): Streaming protocol design — ring buffer vs scatter-gather vs multi-packet chain
- `large-payload.2` (experiment): Implement bulk read/write (4KB+ transfers) via shared memory sideband

**nightly-update (P1)**:
- `nightly-update.1` (investigation): Test latest nightly — does PTX header bug persist? Does gpu-kernel ABI work?
- `nightly-update.2` (experiment): If nightly works, update toolchain pin and remove build.rs header workaround

**gpu-logging (P2)**:
- `gpu-logging.1` (design): Structured logging API design — macros, severity levels, compile-time filtering
- `gpu-logging.2` (experiment): Implement gpu_trace!/gpu_warn!/gpu_error! with host-side aggregation

**workload-demo (P2)**:
- `workload-demo.1` (experiment): Implement parallel file grep — GPU reads file chunks via hostcall, searches in parallel, reports matches

### Which parked themes to unpark?

| Theme | Recommendation | Condition |
|-------|---------------|-----------|
| `warp-coop` | **Conditional unpark** | Only after host-scaling is done. Without multi-threaded host, warp-coop reduces contention but doesn't increase throughput. |
| `networking` | **Keep parked** | Requires large-payload first (HTTP responses > 56 bytes). Premature without real workload demand. |
| `upstream` | **Keep parked** | Community contribution is valuable but does not advance research. |

### Dependency ordering

```
Phase 3A (parallel, immediate):
  host-scaling.1 + gpu-panic.1 + nightly-update.1

Phase 3B (after 3A):
  host-scaling.2 + gpu-panic.2 + nightly-update.2

Phase 3C (after host-scaling.2):
  host-scaling.3 (benchmark) + large-payload.1

Phase 3D (after host-scaling.3):
  → If throughput scales: unpark warp-coop
  → If throughput doesn't scale: investigate host-side bottleneck further
  large-payload.2 + gpu-logging.1

Phase 3E (after large-payload.2):
  workload-demo.1 + gpu-logging.2
```

### What NOT to pursue and why

| Direction | Why NOT |
|-----------|---------|
| **AMD GPU port** | Requires `amdgcn-amd-amdhsa` target, ROCm toolchain, entirely different ISA. Not feasible without dedicated AMD hardware and toolchain investment. Defer until Rust GPU ecosystem matures. |
| **Custom rustc fork** | VectorWare likely uses a custom fork for deeper std integration. Reproducing this diverges from upstream Rust and creates maintenance burden. Our approach (vendored std + gpu-libc shim) is sufficient. |
| **CUDA library interop (cuBLAS, cuDNN)** | Calling CUDA libraries from GPU kernels requires device-side linking, which ptxas handles but the Rust PTX pipeline does not support. Would require significant toolchain work. |
| **Full std coverage** | Porting all of std (networking, threads, process, etc.) is massive effort with diminishing returns. Focus on I/O primitives (file, print, time) that demonstrate the concept. |
| **Register optimization** | The 888-register file kernel is an artifact of including all file operations in one kernel. Splitting it is trivial engineering, not research. Don't create a theme for it. |
| **WGSL/Vulkan compute** | Different ecosystem entirely. NVIDIA CUDA is the right target for this research. |

---

## Summary

The project has successfully reproduced VectorWare's proof-of-concept. The next phase should focus on making the system **practical**: scale the host listener to handle real GPU thread counts (P0), improve developer experience with panic handling (P1), and enable real workloads via larger data transfers (P1). Warp-cooperative execution should be conditionally unparked only after host-side scaling proves that GPU-side contention is the remaining bottleneck. AMD support, full std coverage, and CUDA library interop are explicitly out of scope.

The critical insight: the current system is I/O-throughput-limited by the single host listener thread, not by GPU-side contention. Fixing the host side first gives the most return on research investment.
