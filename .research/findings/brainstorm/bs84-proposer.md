# Brainstorm 84 — Proposer: Post-v0.1.0 Strategic Directions
**Cycle**: 294 | **Date**: 2026-03-15 | **Level**: Deep

## Situation Assessment

The project has reached a significant milestone: v0.1.0 released, 311 tasks completed across 10 closed epics, all documentation polished, CI green, 6 working examples. The core research thesis — Rust async/await on NVIDIA GPUs — is proven. Both evergreen epics are satisfied.

The question is no longer "can we do it?" but "where do we go from here to make this consequential?"

### Current Technical Capabilities
- Lock-free hostcall protocol with per-block sharding (proven scalable in design)
- `#[warp_cooperative]` MIR pass in patched rustc (auto-inserts `bar.warp.sync` at yield points)
- Patched std with PAL layer (std::fs, println!, Vec, String on GPU)
- Async Futures for all I/O operations (open/read/write/close, bulk read/write)
- `block_on()` executor for GPU-side async
- Sideband bulk I/O for large transfers (up to 1MB)

### Current Limitations (identified from code review)
1. **Single-thread I/O**: All examples use `if tid != 0 { return; }` — only thread 0 does hostcall I/O
2. **Single warp**: All kernels launch 1 block x 32 threads — no multi-warp concurrent async
3. **No performance data**: Zero benchmarks, no latency measurements, no throughput numbers
4. **Patched toolchain required**: `#[warp_cooperative]` needs custom rustc — high adoption barrier
5. **No multi-GPU**: Single device only
6. **No networking**: File I/O only, no TCP/HTTP/socket hostcall services

---

## Direction 1: Multi-Warp Concurrent Async

### What
Launch kernels with multiple warps (e.g., 4 warps = 128 threads, or multiple blocks) where each warp runs its own async pipeline concurrently. While warp 0 waits for I/O, warp 1 runs compute.

### Technical Analysis
The infrastructure is already designed for this:
- Per-block sharding exists in the hostcall protocol (`blockIdx.x % num_shards`)
- `HostcallBuffer::new_sharded(num_shards, pkts_per_shard)` API exists on host
- The packet pool supports concurrent access from multiple warps via CAS
- `block_on()` executor uses `nanosleep` for yield, which is warp-local

The gap is **kernel architecture**: current kernels do `if tid != 0 { return; }` or have a single warp doing everything. A multi-warp kernel would need:
- Each warp independently runs `block_on(pipeline())` with warp-local state
- The host listener handles concurrent requests from multiple warps (already supported — it pops from ready stack in a loop)
- Demonstration that warp 1 runs compute while warp 0 blocks on I/O

### Feasibility
HIGH. The protocol and host infrastructure support this today. The work is primarily kernel architecture and a demonstration example. No toolchain or host system changes needed.

### Impact
HIGH. This is the key differentiator of GPU async — showing that I/O latency is hidden by warp-level parallelism, which is the entire point of the `#[warp_cooperative]` design. Without this demo, the project's value proposition remains theoretical.

### Effort
MEDIUM (3-5 tasks, 1-2 sessions).
- Task 1: Design multi-warp kernel architecture (which warp does what, state isolation)
- Task 2: Implement a multi-warp async pipeline kernel
- Task 3: Host-side launch with multiple warps, verify concurrent hostcall handling
- Task 4: (optional) Measure overlap — compute throughput during I/O wait

### Dependencies
- Patched toolchain (for `#[warp_cooperative]`)
- GPU hardware for end-to-end test
- No new crates, no host system changes

### Priority: **P1 — Must Do**
This is the central thesis demo. Without it, the project proved async compiles but not that it is useful.

---

## Direction 2: Performance Benchmarks

### What
Quantitative measurements: hostcall round-trip latency, I/O throughput, warp scheduling overhead, comparison vs. CPU-driven approaches.

### Technical Analysis
The project has zero performance numbers. Key metrics to measure:
1. **Hostcall round-trip latency**: Time from GPU `gpu_hostcall_request()` to response ready. Expected: 5-50us (dominated by PCIe latency + host polling interval).
2. **I/O throughput**: Bytes/sec for file read/write via hostcall vs. host-side I/O.
3. **Warp synchronization overhead**: Cost of `bar.warp.sync` per yield point. Expected: negligible (<100 cycles).
4. **Multi-warp overlap efficiency**: With N warps, what fraction of time is spent in useful compute vs. I/O wait?
5. **Comparison**: GPU kernel doing file I/O via hostcall vs. host program doing the same work — when does GPU win?

### Feasibility
HIGH. The `SERVICE_TIME` hostcall already exists for timestamps. GPU-side `clock()` PTX instruction gives cycle counts. Host-side timing is trivial.

### Impact
MEDIUM-HIGH. Numbers make the project credible to systems researchers and potential adopters. "5us hostcall latency" is a concrete claim; "it works" is not.

### Effort
MEDIUM (3-4 tasks, 1-2 sessions).
- Task 1: Design benchmark suite (what to measure, methodology)
- Task 2: Implement hostcall latency microbenchmark (GPU measures RTT for N iterations)
- Task 3: Implement I/O throughput benchmark (bulk read/write varying sizes)
- Task 4: Write results into a BENCHMARKS.md or similar

### Dependencies
- GPU hardware (required for any measurement)
- No toolchain changes, no new crates

### Priority: **P2 — High Value**
Numbers give the project credibility. Should follow multi-warp demo since multi-warp benchmarks are more interesting than single-warp ones.

---

## Direction 3: Network I/O from GPU

### What
Add TCP and/or HTTP hostcall services so GPU kernels can make network requests.

### Technical Analysis
The hostcall protocol is service-agnostic — adding a new service means:
1. Define a new `SERVICE_TCP_CONNECT` / `SERVICE_TCP_SEND` / `SERVICE_TCP_RECV` in gpu-protocol
2. Implement the handler in the host listener (trivial — just call std::net)
3. Create GPU-side Future types (`GpuTcpConnectFuture`, etc.)
4. Sideband buffer already handles arbitrary byte payloads

The design maps cleanly: TCP socket = file descriptor. Connect returns fd, send/recv use fd + data buffer. The sideband buffer provides sufficient capacity for typical HTTP responses.

A compelling demo: GPU kernel that fetches data from a REST API, processes it, and writes results. This would be genuinely novel — no other GPU framework does this.

### Feasibility
HIGH. The architecture naturally supports it. Implementation is ~200 lines of host-side service handlers + ~200 lines of GPU-side Futures.

### Impact
MEDIUM. Impressive as a demo, but limited practical use. GPU kernels doing HTTP requests is a party trick unless there is a real workload that benefits. ETL pipelines that fetch + transform could be a use case, but the I/O latency would dominate.

### Effort
MEDIUM (4-5 tasks).
- Task 1: Design network hostcall protocol (service IDs, fd model, error handling)
- Task 2: Implement TCP connect/send/recv host handlers
- Task 3: Implement GPU-side `GpuTcpConnectFuture` / `GpuTcpSendFuture` / `GpuTcpRecvFuture`
- Task 4: Demo: GPU kernel fetches URL via hostcall, processes response
- Task 5: (optional) HTTP client helper on GPU side

### Dependencies
- No new crates needed (std::net on host side)
- GPU hardware for demo
- No toolchain changes

### Priority: **P4 — Nice to Have**
Impressive but not essential. The skeptic in bs78 correctly called this a "party trick." Revisit after multi-warp and benchmarks prove the core value.

---

## Direction 4: GPU-to-GPU Communication

### What
Multi-GPU hostcall, NVLINK direct memory access, or multi-device kernel coordination.

### Technical Analysis
This requires significant infrastructure:
- Each GPU needs its own hostcall buffer and listener thread
- Cross-GPU communication options:
  a. Via host: GPU-A sends hostcall → host forwards to GPU-B's buffer. Latency: 2x PCIe RTT.
  b. Direct NVLINK: Requires `cuMemcpyPeer` or unified virtual addressing. Bypasses hostcall entirely.
  c. NCCL integration: For collective operations (allreduce, broadcast). Overkill for hostcall-level messaging.

Option (a) is straightforward but slow. Option (b) is fast but requires NVLINK hardware and significant CUDA API work. Option (c) is irrelevant to this project's goals.

### Feasibility
MEDIUM. Option (a) is feasible without hardware changes but adds marginal value. Options (b) and (c) require multi-GPU hardware and significant effort.

### Impact
LOW for this project. Multi-GPU is relevant for distributed training/inference, but this project is about autonomous GPU kernels doing I/O. The motivating use case for multi-GPU is unclear.

### Effort
HIGH (6-10 tasks).

### Dependencies
- Multi-GPU hardware
- Significant host-side changes (device management, cross-device buffers)
- Possibly CUDA peer access APIs

### Priority: **P6 — Defer**
No motivating use case. Defer until someone asks for it.

---

## Direction 5: Upstream MIR Pass to Rustc

### What
Propose the `#[warp_cooperative]` MIR pass for inclusion in the official Rust compiler.

### Technical Analysis
Current state:
- The MIR pass works: 40+ tests confirm warp convergence is correct
- It modifies `StateTransform` output (the async state machine generator)
- It is specific to `nvptx64` target

Upstream challenges:
- Rust's MIR pass pipeline is not plugin-extensible — this would need to be in-tree
- The pass is nvptx64-specific, which is a tier 3 target with minimal maintenance
- The Rust compiler team has limited bandwidth for GPU-specific features
- `#[register_tool(warp_cooperative)]` is a workaround; a proper attribute would need an RFC
- The pass assumes warp size = 32, which is NVIDIA-specific (AMD wavefront = 32 or 64)

### Feasibility
LOW-MEDIUM. Technically possible but politically difficult. An RFC would need to justify a GPU-specific compiler pass in rustc, which is a hard sell given the small user base.

### Impact
HIGH if successful — it eliminates the patched toolchain barrier entirely. But the probability of acceptance is low in the near term.

### Effort
VERY HIGH (many sessions, plus RFC process, plus compiler team engagement).

### Dependencies
- Active engagement with the Rust compiler team
- RFC process (months)
- May require generalizing beyond NVIDIA (e.g., supporting AMD wavefronts)

### Priority: **P5 — Long-Term**
Important for adoption but not actionable in the near term. The project should first prove value via demos and benchmarks. An RFC is worth pursuing after there are users who want stable toolchain support.

---

## Direction 6: Ecosystem Integration

### What
Publish to crates.io, create a cargo-gpu plugin for build automation, interop with rust-gpu.

### Technical Analysis

**crates.io publish**: Partially feasible. `gpu-protocol` and `gpu-atomics` could publish today (pure Rust, no special requirements). `gpu-host` depends on `cudarc`. `gpu-runtime` requires `nvptx64` target. Publishing is possible but the build requirements are unusual enough that it would confuse most users.

**cargo-gpu plugin**: A `cargo gpu build` command that automates kernel compilation, PTX post-processing, and sysroot setup. Currently, each example's `build.rs` does this. A plugin would centralize it.

**rust-gpu interop**: rust-gpu uses SPIR-V backend, this project uses PTX/LLVM nvptx. Fundamental incompatibility at the IR level. However, rust-gpu's development model (proc macros for GPU code) could inform API design.

### Feasibility
MEDIUM. crates.io publish is possible but premature. cargo-gpu is useful but significant effort. rust-gpu interop is architecturally incompatible.

### Impact
MEDIUM. crates.io publish would increase discoverability. cargo-gpu would reduce friction. Neither adds new capability.

### Effort
MEDIUM-HIGH.
- crates.io: 2-3 tasks (workspace reorganization, CI for publish, documentation)
- cargo-gpu: 5-8 tasks (new crate, PTX build logic, sysroot management, error handling)
- rust-gpu: Not feasible

### Dependencies
- crates.io: Stable APIs (currently evolving)
- cargo-gpu: Deep understanding of rustc build pipeline for nvptx64

### Priority: **P5 — Long-Term**
Premature. The project has <10 known users. Publish when there is demand.

---

## Direction 7: Real-World Workloads

### What
Implement non-trivial applications: ETL pipeline, log parsing, data transformation, JSON processing.

### Technical Analysis
Current examples are proof-of-concept demos (read file, uppercase, write file). A real workload would:
- Process meaningful data volumes (MB, not bytes)
- Perform compute that benefits from GPU parallelism
- Show a clear performance or capability advantage over CPU-only approaches

Candidate workloads:
1. **Log parsing**: Read log files, parse structured fields, filter/aggregate on GPU. Good for parallelism (each warp processes a different log chunk).
2. **CSV/JSON transformation**: Parse structured data, apply transformations, output results. Limited by parsing being inherently sequential per-record.
3. **Image processing**: Read image, apply filters across warps, write result. Good parallelism but requires understanding pixel layout.
4. **Encryption/hashing**: Read data, compute SHA256/AES per block. Embarrassingly parallel, well-suited to GPU.
5. **Regex matching**: Extension of parallel-search to regex patterns. Already close to existing capability.

### Feasibility
MEDIUM. The I/O infrastructure supports it. The challenge is that useful workloads need libraries (JSON parser, CSV parser, regex engine) that do not exist in `#![no_std]` for nvptx64.

### Impact
HIGH if done well. A compelling real-world demo is the difference between "research project" and "useful tool."

### Effort
HIGH per workload (5-8 tasks each). Building a no_std JSON parser for GPU is a project in itself.

### Dependencies
- GPU hardware
- Possibly new GPU-side library crates (parser, data format handling)
- Large test data files

### Priority: **P3 — Important**
One well-chosen workload (log parsing or regex matching, which build on existing parallel-search) would demonstrate real value. Start small.

---

## Direction 8: Safety Improvements

### What
Audit unsafe code, add SAFETY comments, develop soundness analysis, explore miri-like tools for GPU.

### Technical Analysis
Current state: 822 `unsafe` blocks in gpu-host, many without SAFETY comments. GPU-side code is inherently unsafe (raw pointer manipulation, inline PTX assembly, CUDA API calls).

Possible improvements:
1. **SAFETY comments**: Document invariants for each unsafe block. Large effort but high value for code review.
2. **Safe wrapper APIs**: Replace raw pointer patterns with safe abstractions (e.g., `HostcallPacket` type with safe accessors instead of offset arithmetic).
3. **Soundness analysis**: Document which invariants the protocol relies on (e.g., single-writer per packet, CAS ordering guarantees).
4. **GPU miri**: Not feasible — miri does not support nvptx64 and cannot execute PTX.

### Feasibility
MEDIUM. SAFETY comments and safe wrappers are feasible. Soundness analysis is feasible as documentation. GPU miri is not feasible.

### Impact
MEDIUM. Improves code quality and contributor experience. Does not add new capability. Important for long-term maintainability.

### Effort
HIGH (10+ tasks for full unsafe audit).

### Dependencies
- No external dependencies, but requires careful analysis of each unsafe block

### Priority: **P4 — Important but Large**
bs82 already identified this as deferred (large effort). Worth doing incrementally when touching affected code.

---

## Direction 9: Dynamic Dispatch on GPU

### What
Support trait objects (`dyn Trait`) and vtables on GPU, enabling runtime polymorphism.

### Technical Analysis
Current limitation: GPU code is `#![no_std]` and all dispatch is static (generics, concrete types). Dynamic dispatch requires:
1. **vtables**: Must be in GPU-accessible memory. Currently, vtables are in `.rodata` which maps to GPU constant memory — this should work.
2. **fat pointers**: `&dyn Trait` is a `(data_ptr, vtable_ptr)` pair. LLVM nvptx should handle this as it is standard Rust codegen.
3. **Allocation**: `Box<dyn Trait>` requires heap allocation, which exists (bump allocator).
4. **Testing**: Has anyone tried `&dyn Trait` on nvptx64? Possibly just works with `alloc` crate.

This might already work and just needs verification. Or it might hit LLVM nvptx limitations with indirect calls (nvptx does not support function pointers in all contexts — `call.uni` vs `call` instructions).

### Feasibility
UNCERTAIN. Might work out of the box, might hit fundamental LLVM nvptx limitations with indirect calls. An investigation task would clarify.

### Impact
MEDIUM. Enables more flexible kernel architectures (plugin systems, strategy patterns). Not critical for current use cases.

### Effort
LOW to investigate (1-2 tasks), MEDIUM-HIGH to fix if broken.

### Dependencies
- GPU hardware for testing
- May require LLVM patches if indirect calls are broken on nvptx

### Priority: **P5 — Investigate Only**
Worth a quick investigation to see if it just works. Low priority for fixing if it does not.

---

## Direction 10: Interop with CUDA Libraries

### What
Call cuBLAS, cuDNN, cuFFT from GPU kernels via hostcall or direct linking.

### Technical Analysis
Two approaches:
1. **Via hostcall**: GPU sends "run cuBLAS GEMM" request, host calls cuBLAS, result available in device memory. Simple but adds PCIe latency for every library call.
2. **Direct linking**: Link against cuBLAS device-side library (libcublas_device.a). This requires CUDA compilation toolchain (nvcc or device-link), not just rustc.

Approach 1 is simpler but defeats the purpose — if you are calling cuBLAS from host anyway, why run the orchestration on GPU? Approach 2 is technically challenging and may not be compatible with the rustc-based PTX pipeline.

### Feasibility
LOW-MEDIUM. Approach 1 is easy but uncompelling. Approach 2 requires deep integration with CUDA device linking which is orthogonal to the rustc PTX pipeline.

### Impact
LOW. The project's value is Rust async on GPU, not wrapping CUDA libraries. Users who need cuBLAS already use it from C++/Python.

### Effort
MEDIUM for approach 1, HIGH for approach 2.

### Dependencies
- CUDA toolkit installation
- Approach 2: CUDA device linking, which may conflict with rustc PTX output

### Priority: **P6 — Defer**
Orthogonal to the project's thesis. No compelling use case.

---

## Summary: Priority Ranking

| Rank | Direction | Feasibility | Impact | Effort | Verdict |
|------|-----------|-------------|--------|--------|---------|
| **P1** | Multi-warp concurrent async | HIGH | HIGH | MEDIUM | **Must do** — proves the thesis |
| **P2** | Performance benchmarks | HIGH | MEDIUM-HIGH | MEDIUM | **High value** — makes project credible |
| **P3** | Real-world workloads | MEDIUM | HIGH | HIGH | **Important** — start with log/regex |
| **P4** | Network I/O from GPU | HIGH | MEDIUM | MEDIUM | Nice demo, defer until P1-P3 done |
| **P4** | Safety improvements | MEDIUM | MEDIUM | HIGH | Incremental, do alongside other work |
| **P5** | Upstream MIR pass | LOW-MEDIUM | HIGH | VERY HIGH | Long-term, needs ecosystem traction |
| **P5** | Ecosystem integration | MEDIUM | MEDIUM | MEDIUM-HIGH | Premature, publish when demand exists |
| **P5** | Dynamic dispatch on GPU | UNCERTAIN | MEDIUM | LOW-HIGH | Worth investigating, not worth fixing |
| **P6** | GPU-to-GPU communication | MEDIUM | LOW | HIGH | No motivating use case |
| **P6** | CUDA library interop | LOW-MEDIUM | LOW | MEDIUM-HIGH | Orthogonal to project thesis |

## Recommended Epic

### Epic: "multi-warp-async" (HIGH priority)
**Goal**: Demonstrate that multi-warp GPU kernels can overlap I/O wait with compute via warp-cooperative async, with quantitative evidence.

**Success criteria**:
1. A kernel with 2+ warps where warp N does async I/O while warp M does compute
2. PTX shows independent warp scheduling (no cross-warp barriers)
3. Hostcall protocol handles concurrent requests from multiple warps without deadlock
4. At least one benchmark with concrete numbers (hostcall RTT, multi-warp overlap ratio)

**Themes**:
1. `multi-warp-kernel` — design and implement multi-warp async kernel
2. `benchmark-suite` — hostcall latency, I/O throughput, warp overlap measurements
3. `workload-demo` — one real-world-ish workload (log search or regex, building on parallel-search)

This epic addresses P1, P2, and the beginning of P3 in a coherent narrative: "multi-warp async is fast, here are the numbers, and here is a real workload."

## Key Insight

The project proved the concept. The next phase must prove the value. Multi-warp overlap is the mechanism, benchmarks are the evidence, and a real workload is the motivation. All three are needed to make the project consequential rather than merely interesting.
