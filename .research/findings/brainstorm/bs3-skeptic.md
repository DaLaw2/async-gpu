# Brainstorm BS3 — Devil's Advocate / Skeptic Analysis
**Date:** 2026-03-11
**Role:** Skeptic
**Seq:** 3

---

## Preamble: The Danger of Confirmed Success

BS1 challenged unverified assumptions. BS2 challenged whether our workarounds were safe. In
BS3, the project is further along: we have an atomics crate that compiled and passed a basic
test, a hostcall protocol design, and a confirmed Embassy compatibility assessment. The
danger now is different — it is not that we are building on sand, it is that small, clean
victories over tractable problems have created misplaced confidence about the much harder
problems ahead.

The skeptic's role this cycle is to identify where our "confirmed" successes have hidden
assumptions that no one has stress-tested, and where the next stage of the project will
run into walls that nobody has designed around.

---

## 1. "Inline PTX asm works" — On ONE nightly, for SIMPLE instructions

### Claim
`atomics.3-c1` confirmed: `core::arch::asm!` works on nightly built 2025-08-25 with
`#![feature(asm_experimental_arch)]`.

### The hole

**The feature is called `asm_experimental_arch` for a reason.** It is not scheduled for
stabilization. There is no tracking issue promising a stable future. The feature was
added to unblock NVPTX development in LLVM, but the LLVM PTX inline assembly backend
has a documented history of supporting a subset of PTX instructions and rejecting others.

What has been tested: `membar.sys`, `st.release.sys.global.u32`, `ld.acquire.sys.global.u32`,
`atom.cas.sys.global.b32`, `atom.add.sys.global.u32`. That is five instruction forms.

What has NOT been tested:
- `atom.cas.sys.global.b64` (64-bit CAS — required for tagged pointer operations in hostcall)
- `atom.add.sys.global.u64` (64-bit fetch-add — required for doorbell counter)
- `atom.exch.sys.global.b64` (64-bit exchange — required for `ready_stack` swap)
- `vote.ballot.sync` (required for `__activemask()`)
- `shfl.sync` (required for warp-cooperative slot filling in hostcall)
- `mov.u32 %r, %tid.x` — special register reads used in the integration test kernel

The hostcall protocol (ADR-3) depends on all the u64 variants. These require LLVM's
NVPTX backend to correctly parse and emit 64-bit scoped atomic PTX instructions via
inline asm. There is nothing in the PTX ISA that prevents this — but the atomics.3.1
findings explicitly defer these: "u64 CAS/fetch_add + exchange primitive → defer to
hostcall.3 design phase."

**The u64 variants have never been compiled or executed.** hostcall.4 is blocked on
implementing them. If the LLVM PTX backend rejects `atom.cas.sys.global.b64` in an
inline asm block (which has happened in prior LLVM versions for different PTX constructs),
the entire hostcall protocol collapses and ADR-3 must be redesigned.

This is not a theoretical concern. `atomics.3-c1` itself notes that its nightly is from
a specific build date (2025-08-25). When the team updates to a newer nightly to get other
fixes or features, the inline asm behavior may change. The feature's experimental status
means there is no stability guarantee between nightly builds.

**Recommendation for other teammates**: Do not assume the u64 variants work. Treat this
as an unverified blocker for hostcall.4. The first action in hostcall.4 should be to
verify u64 CAS/fetch-add/exchange inline PTX compiles and executes correctly.

---

## 2. "ROCm hostcall design is proven" — Proven on HSA, not on PCIe polling

### Claim
ADR-3 adopts the ROCm two-stack lock-free pool design. `hostcall.2-c1` confirms it is
"proven in production."

### The hole

ROCm's hostcall uses an HSA signal object as its doorbell. The `doorbell_->Wait(signal_value,
Condition::Ne, timeout)` call in the listener is NOT a polling loop — it is a hardware-assisted
wait. AMD's HSA signals leverage the GPU's SDMA engine and interrupt infrastructure to wake the
host listener when the doorbell is updated. The CPU does not spin; it sleeps in a kernel wait
queue until the GPU signals it.

Our "doorbell" is a `u64` counter in pinned memory that the host polls with `AtomicU64::load`.
The host spins, checking the counter in a loop. This is fundamentally different:

1. **CPU utilization**: ROCm's host listener uses 0% CPU when no hostcalls are pending.
   Our listener burns 100% of a CPU core in the spin path.

2. **Latency under host load**: ROCm's listener is woken by hardware interrupt. Ours
   competes with all other threads for CPU time. Under host load (which is the normal
   state — the host CPU is running other work while the GPU executes), our listener may
   not get scheduled for milliseconds. Meanwhile GPU warps are spinning on the response.
   A 1ms host scheduling delay × 448 warps that could be blocked = 448 GPU-milliseconds
   of wasted compute.

3. **Scale**: ROCm's production use is for HPC workloads where GPUs with thousands of
   wavefronts issue hostcalls. Our design is validated on a single GPU thread (the
   `atomics.3` integration test). At 448 warps all issuing concurrent hostcalls, the
   single host thread must process them one at a time in sequence. At 10µs per hostcall
   processing (optimistic), serving 448 packets = 4.5ms. If GPU warps time out at 1M
   spin iterations, what is the actual spin count → milliseconds conversion on RTX 3060?
   This has not been measured.

The `hostcall.3-c10` design document mentions "adaptive-timeout host polling" as a
mitigation for CPU utilization, but this adds latency in the wrong direction: when
backoff is active, hostcall round-trip latency increases. There is no mechanism in the
design to both reduce CPU utilization AND minimize latency simultaneously — these are
in direct tension.

**The doorbell mechanism is not equivalent to HSA signals.** The claim that "ROCm's
design translates directly to NVIDIA" (hostcall.3-c10, Key Conclusion 1) is overstated.
The data path (two-stack lock-free packets) translates. The notification mechanism does not.

CUDA provides `cuStreamWaitValue64` and `cuStreamWriteValue64` (CUDA 12+) which use
hardware-assisted wait semantics similar to HSA signals. These are mentioned as an option
in `hostcall.3-c10` (Open Question 3) but have been deferred. If the spinning doorbell
approach produces unacceptable CPU utilization or latency, this deferral will need to be
revisited urgently before the integration theme can proceed.

---

## 3. "Tagged pointers prevent ABA" — The wraparound math has not been done

### Claim
ADR-3: "Tagged pointers for ABA prevention... This allows up to 65534 packets and 2^32
tag generations before wraparound."

### The hole

The 32-bit tag wraps after 2^32 = 4,294,967,296 tag generations. On the surface this
sounds enormous. Let's do the actual math.

The RTX 3060 has 28 SMs × 48 warps/SM = 1344 maximum concurrent warps. Each hostcall
involves at minimum 2 CAS operations on the `free_stack` (pop) and 2 on the `ready_stack`
(push + host swap). That is 4 tag increments per hostcall cycle per warp.

If all 1344 warps are issuing hostcalls as fast as possible:
- Each warp CAS loop iteration takes roughly 1 PCIe round trip ≈ 1-2µs
- Minimum cycle time per warp: ~2µs (pop + push + wait + response + return)
- At 1344 concurrent warps: 1344 / 2µs = 672 million CAS ops/second at peak

At 672M CAS/s × 4 tag increments/hostcall: ~2.7 billion tag increments per second.

2^32 / (2.7 × 10^9) ≈ **1.6 seconds to tag wraparound under peak load**.

This is not safe. ABA in a lock-free stack after 1.6 seconds of peak operation would
corrupt the packet pool silently — the GPU would write into a packet that the host is
still processing, or the host would dispatch a packet that the GPU is already reusing.

The ROCm design uses the same tagging approach, but ROCm's context is typically short
kernel bursts, not sustained tight-loop hostcall workloads. For our async runtime use
case — where async tasks may be continuously awaiting hostcalls as their primary compute
pattern — sustained hostcall throughput is exactly the expected workload.

**The 2^32 tag is insufficient for sustained high-throughput hostcall.** A 64-bit tag
(using the full 64 bits of the stack pointer, with packet index encoded differently)
would be safe, but requires restructuring the tagged pointer format. This should be
caught before hostcall.4 commits to the 32-bit tag design.

---

## 4. "Warp-granular packets are the right size" — 95% payload waste on printf

### Claim
ADR-3 / hostcall.3-c10: Warp-granular packets (2048 byte payload) match NVIDIA's SIMT
model and minimize atomic contention.

### The hole

The packet design requires one packet per warp. Each packet's payload is
`slots[32][8]` = 32 lanes × 8 u64 × 8 bytes = 2048 bytes.

This is correct when all 32 lanes in a warp issue the same hostcall (e.g., all 32 threads
print a string). But consider the real async use case: an async task running on one lane
awaits a `File::open("config.json")`. The other 31 lanes are either:
a) executing different code paths (warp divergence — some may be at other await points), or
b) idle / masked out by the active mask.

In case (a), the `active_mask` will have 1 bit set. The packet is allocated (removing it from
the free pool), 2080 bytes are written to pinned memory (PCIe transfer), and the host processes
the response. Only 64 bytes of the 2048-byte payload are used (the slot for lane 0).

**Payload utilization: 64 / 2048 = 3.1%.** In the realistic async scenario where
different lanes diverge into different `await` points, each packet carries one meaningful
request and 31 zeroed slots.

The consequences:
1. PCIe bandwidth: every hostcall transfers ~2KB across PCIe even for a 1-byte read result.
2. Memory pressure: 64 packets × 2112 bytes = 132KB of pinned memory, all of it for
   per-warp granularity. If 448 warp-slots, that's ~922KB of pinned PCIe-visible memory.
3. The free pool is sized per warp, not per active lane. If 1344 warps all have divergent
   async tasks, 1344 packets are needed simultaneously — the "Conservative: 448 packets"
   pool sizing in hostcall.3-c10 is too small by 3×.

The ROCm design for AMD GPUs has the same structure but with 64 work-items per packet
(matching the wave64 model). AMD workloads are typically compute-heavy with all lanes
active. Rust async/await workloads on GPU are explicitly divergence-heavy — that is the
whole point of independent async tasks per lane.

**The warp-granular packet design is correctly sized for uniform SIMT workloads, but
poorly matched to divergent async workloads.** A per-lane packet design would use 64 bytes
instead of 2048 bytes per hostcall, at the cost of 32× more CAS operations (one per lane
instead of one per warp). This tradeoff should be evaluated before the async-runtime and
integration themes commit to the hostcall interface.

---

## 5. "Embassy is 90% GPU-compatible" — This is a desk analysis, not a compile test

### Claim
ADR-2 / `async-runtime.1-c1`: "Embassy is ≈90% GPU-compatible out of the box for nvptx64
SM 7.0+ targets. Only 3 concrete changes needed."

### The hole

The `async-runtime.1` investigation was a code reading exercise. Nobody has attempted
to compile `embassy-executor` for `nvptx64-nvidia-cuda`. The "90% compatible" figure is
an assessment, not a measurement.

The investigation itself lists several "Potentially Problematic" items that were not
verified:
- `AtomicU8` on nvptx64: PTX natively supports 32-bit and 64-bit atomics. 8-bit atomics
  are NOT in the PTX ISA. If LLVM cannot emit 8-bit atomics for PTX, Embassy's `state_atomics.rs`
  path will fail to compile because `AtomicU8` is used for task state.
- `cordyceps` intrusive list with `AtomicPtr` (pointer-width = 64-bit): unverified.
- `embassy-executor-macros` proc-macro generating code that compiles for nvptx64: unverified.
- `static TaskStorage<F>` in GPU global memory address space vs stack: Embassy's `TaskStorage`
  is declared `static`, which in GPU code is in `.global` memory (low-bandwidth DRAM). Embassy
  assumes static allocation in embedded contexts where static = fast SRAM. On GPU, static = slow
  DRAM. This is not a correctness issue but an occupancy/performance issue that was not flagged.

The three "Required Changes" (no-op pender, disable timers, SM70+ target) are all correct
and clearly small. The issue is that the 10% incompatibility may not be small:

**Function pointers in Embassy's waker vtable.** The `RawWakerVTable` contains four function
pointers: `clone`, `wake`, `wake_by_ref`, `drop`. All four point to the same `wake` function
in Embassy's implementation. On NVIDIA GPU, static function pointers that are monomorphizable
at compile time produce direct calls. Dynamic function pointers (where the pointer value is
not known at compile time) produce indirect calls via PTX `call.uni` or `call.i`, which the
PTX backend has historically had trouble with.

The waker vtable pointer is stored in `TaskHeader` as `*const RawWakerVTable`. When
`Waker::wake()` is called, the vtable pointer is dereferenced and the function pointer is
called indirectly. Whether LLVM's NVPTX backend can correctly lower this indirect call
to valid PTX is not documented and not tested. This is the exact scenario that BS2-skeptic
identified as "the 10% incompatibility may contain precisely these dynamic dispatch cases."

None of this has been resolved. The Embassy port remains unstarted.

---

## 6. "Host polling with adaptive timeout is sufficient" — 100 simultaneous warps

### Claim
hostcall.3-c10: The host listener uses adaptive-timeout polling on the doorbell counter.
GPU-side timeout at 1,000,000 spin iterations prevents infinite hangs.

### The hole

The design assumes the host listener can process packets faster than they arrive.
Consider 100 warps issuing simultaneous hostcalls:

1. All 100 warps pop a free packet (100 concurrent CAS loops on free_stack)
2. All 100 warps fill their packets (100 × 2KB = 200KB of PCIe writes)
3. All 100 warps push to ready_stack (100 concurrent CAS loops)
4. All 100 warps atomically increment the doorbell
5. Host wakes up, swaps ready_stack to get all 100 packets
6. Host processes 100 packets sequentially (each hostcall is a syscall or I/O operation)
7. For each packet: host writes response, sets READY bit

Step 6 is the critical path. If each packet is a `write(fd, buf, len)` syscall, processing
time is bounded by I/O throughput. A typical file write takes 1-100µs per call. Even at
1µs per packet: 100 packets = 100µs. The last warp to get its response has waited 100µs.

But the GPU spin timeout is 1,000,000 iterations. What is one iteration in time? On
a GPU spinning on a PCIe-visible memory location, each iteration is roughly:
- `ld.acquire.sys.global.u32` across PCIe ≈ 1-2µs
- So 1,000,000 iterations ≈ 1-2 **seconds**

The timeout is effectively never reached in normal operation. The GPU spins for up to
2 seconds waiting for the host. This is fine for simple hostcalls. It is terrible for
I/O-bound operations where the host blocks on disk or network. If the host is waiting
for a file to flush and the GPU is spinning, the entire kernel is blocked for the
duration of that I/O.

More critically: **100 warps issuing concurrent file writes will serialize on the host**.
The host processes packets one at a time. Warp 1 gets its response at T+1µs. Warp 100
gets its response at T+100µs. But warps 1-99 must wait until the host finishes processing
all packets before returning their specific packet to the free pool (step 8 in the GPU
protocol). This means the free pool shrinks by 100 during the burst, potentially
exhausting it.

With a 64-packet pool and 448 warps, pool exhaustion happens immediately at any significant
concurrency. The "Pool Exhausted" error returns an error code to the GPU, but what does
the GPU-side async executor do with a hostcall error? The design doesn't say. If the
async task retries the hostcall, it enters a busy-retry loop that consumes GPU cycles.
If it propagates the error, the async task fails — which may not be the semantics the
caller expected (a `File::open` that returns `PoolExhausted` is not a valid IO error).

---

## 7. "atomics.2 stress test can wait" — The foundation is still unverified

### Claim
`atomics.2` (stress-test GPU-CPU atomic communication) is `pending` and depends on
`atomics.3.1` (done). It has not been prioritized.

### The hole

The entire hostcall protocol correctness depends on GPU-CPU atomics being correct under
concurrent multi-warp access. The `atomics.3` integration test used **a single GPU thread**.
The fix in `atomics.3.1` eliminated redundant membar.sys but was also tested by inspection
of PTX output, not by running concurrent multi-warp tests.

Specifically what is unverified:
1. `sys_cas_u32` under concurrent warp access: does the CAS loop converge correctly when
   all 1344 warps simultaneously execute it? Or does warp scheduling under high contention
   produce livelock?
2. WARP DIVERGENCE in CAS loops: A CAS loop (`loop { if cas() == expected { break } }`)
   diverges at the branch. On NVIDIA SIMT, diverged lanes are serialized. This means that
   in a 32-lane warp where all lanes execute the CAS loop, only the lanes that succeed on
   the SAME cycle execute the `break` path together. The lanes that fail re-enter the loop.
   This is correct behavior, but it reduces the effective parallelism to near-sequential
   for the CAS path. Measured throughput of `sys_cas_u32` under 1344-warp contention is
   completely unknown.
3. `ld.acquire.sys` + `st.release.sys` memory ordering under concurrent access: the
   single-thread test confirms the GPU can signal the CPU. It does NOT confirm that
   when 32 threads simultaneously write release stores and one CPU reads acquire loads,
   the CPU sees a consistent view.

The `atomics.2` task has been in `pending` state since BS1. It directly blocks being able
to claim the atomics theme's second success criterion: "Stress-test passes for GPU-CPU atomic
communication." Yet the project is proceeding to design hostcall.3 (which depends on
multi-warp CAS correctness) and planning async-runtime.2 (which depends on atomics for
waker state transitions).

**Building hostcall.4 on an untested multi-warp atomics foundation is the exact "False Progress
via Trivial Success" failure mode** identified in BS2. The single-thread test gives confidence
for the single-thread case. It says nothing about the multi-warp case that the actual
production workload requires.

---

## 8. "gpu-std will work via libc shim" — `-Zbuild-std=std` on nvptx64 has zero precedent

### Claim
The `gpu-std` theme assumes compiling Rust's full `std` for `nvptx64-nvidia-cuda` via
`-Zbuild-std=std` with a libc shim layer.

### The hole

`-Zbuild-std=std` for `nvptx64-nvidia-cuda` has **never been demonstrated** by anyone in
the public Rust ecosystem, including VectorWare. VectorWare's blog post describes the
approach at a high level but has not open-sourced the implementation. The Rust standard
library contains code that has never been designed for or tested on a GPU target.

The specific obstacles that have not been analyzed:

**`std::io::Write` and buffered I/O**: `BufWriter<W>` allocates a heap buffer (`Vec<u8>`).
This requires a working global allocator on GPU. The `alloc` crate path requires a custom
`GlobalAlloc` (confirmed doable but not implemented). But more importantly, `BufWriter`
assumes that `flush()` can happen eagerly. In our design, `flush()` must become a hostcall.
But `std::io::BufWriter::flush()` is not async — it is a synchronous blocking call. The
entire I/O stack in std is synchronous. Making it work on GPU requires either making it
blocking (warp spins until the host responds) or making it async (requires rewriting the
std I/O traits).

**`std::fs::File::open`**: This calls `open(path, flags)` which is a hostcall in our
design. But the `path` argument is a Rust `&Path` which is a reference to a `OsStr`.
In the hostcall protocol, string arguments must be serialized into the 7-slot payload
(7 × u64 = 56 bytes). A path like `/home/user/.config/myapp/settings.json` is 43 bytes
— it fits, barely. But a path longer than 56 bytes requires multi-packet message
fragmentation, which is not yet designed for `std::fs` operations.

**`std::env` and `std::process`**: These modules have deep system call dependencies
that are unlikely to be implementable via hostcall without completely reimplementing
them. `std::process::exit()` on GPU should map to kernel abort — but std calls into
platform-specific code for this.

**Thread-local storage in std internals**: ADR-1 amendment notes that `core::sync::atomic`
is broken for cross-device use. But std also uses `thread_local!` extensively. On nvptx64,
TLS is not supported. If any code path in `std` that gets compiled for GPU reaches a
`thread_local!` macro, the compilation will fail. The `gpu-std.1` task is supposed to
audit this, but the scope of the problem is significantly larger than acknowledged — it
is not just libc function calls, it is the entire std internal structure.

**The `std::alloc` module**: `std::alloc::System` is the default global allocator. It
calls `malloc()`/`free()` from libc. In the GPU context, our libc shim must redirect these
to GPU device heap (`core::arch::nvptx::malloc`/`free`). But `core::arch::nvptx::malloc`
is limited by the device heap size configured by `cudaDeviceSetLimit(cudaLimitMallocHeapSize)`.
The default is 8MB for the entire device. With 1344 warps each potentially allocating, the
heap exhausts trivially for any real workload.

The gpu-std theme has no completed investigation tasks. `gpu-std.1` (libc dependency graph)
is still `pending` behind `hostcall.4`. The entire theme is one of the most speculative in
the project with the least empirical grounding. It should not be treated as a peer to the
atomics and hostcall themes in terms of readiness.

---

## 9. "cudarc's default context supports mapped memory" — Documented as undocumented

### Claim
`hostcall.1-c1` notes: "In practice on modern CUDA (>= 4.0), cuCtxCreate without
CU_CTX_MAP_HOST may still allow cuMemHostAlloc with CU_MEMHOSTALLOC_DEVICEMAP if UVA
is enabled (behavior is driver-version-dependent and not guaranteed by the spec)."

### The hole

The investigation explicitly acknowledges this is "behavior is driver-version-dependent and
not guaranteed by the spec." And yet ADR-3 was accepted with this as the memory allocation
mechanism, and `atomics.3` was successfully run on top of it.

The `atomics.3` test was run on a specific machine (Windows, CUDA 13.0 driver, RTX 3060).
If a team member attempts to reproduce this on a different machine (Linux, CUDA 12.3 driver,
RTX 2080) and the `cudarc` default context does NOT have `CU_CTX_MAP_HOST`, the
`cuMemHostAlloc` call will silently succeed (no error) but the resulting memory may not be
accessible from the GPU via the device pointer. The kernel would read garbage or segfault.

This is a reproducibility hazard. The fact that it "worked" on the test machine does not
mean it works reliably across CUDA versions and driver configurations.

The fix is simple: explicitly set `CU_CTX_MAP_HOST` when creating the CUDA context, rather
than relying on UVA implicit behavior. But this requires creating the CUDA context via
`cuCtxCreate_v2` with the explicit flag, BEFORE any cudarc safe API is initialized (since
cudarc creates its own context internally). This is a non-trivial integration concern for
the `gpu-host` crate.

Until this is explicitly handled and tested on multiple CUDA configurations, the hostcall
foundation is fragile in a way that is difficult to diagnose (silent corruption rather
than a clear error).

---

## 10. "The project can proceed without Rust-CUDA" — For simple kernels, yes. For async state machines, unknown.

### Claim
ADR-1 (amended in BS2): nvptx64 is confirmed valid. The `atomics.3` experiment resolved
the critical atomics blocker. `toolchain.2` (Rust-CUDA investigation) is now lower priority.

### The hole

`atomics.3` confirmed that inline PTX assembly for a small library of 5 instruction forms
works on the current nightly. It did NOT confirm that complex Rust programs compile
correctly to PTX.

The specific concerns that remain entirely unaddressed:

**Indirect calls and function pointers**: Embassy's waker vtable requires dynamic dispatch.
The `#[embassy_executor::task]` macro generates `TaskStorage<F>` which includes a
`poll_fn` stored as `unsafe fn(TaskRef)`. This is a function pointer stored in a struct
in global memory, dereferenced at runtime. LLVM's NVPTX backend is known to have
difficulty with indirect calls that are not statically resolvable. `toolchain.2` was the
task to investigate this — it remains `pending`.

**Async state machine register pressure**: An `async fn` that awaits a hostcall compiles
to a state machine struct containing all live variables at each suspend point. For a complex
task (e.g., `async fn transfer_file(path: &str) -> Result<usize, Error>`) the state machine
may be hundreds of bytes. On GPU:
- If it fits in registers: no problem, fast access.
- If it spills to `.local` memory (per-thread DRAM): 5-10× slower access, reduced occupancy.
- If it exceeds `.local` budget: compiler error or silent stack overflow.

There is no measurement of async state machine register pressure on nvptx64. Zero. This
is listed as an open question in `async-runtime.1-c1` and in BS2's skeptic analysis, and it
has not moved since BS1. If the async state machine for a realistic task spills heavily to
`.local` memory, the performance overhead will make the project unviable — you would need
hundreds of GPU cores to do the work of one CPU core just to compensate for the spill penalty.

**Complex control flow and LLVM PTX codegen**: The BS2 skeptic raised the concern that
LLVM's NVPTX backend has problems with `alloca` in complex control flow. Async state
machine code is precisely "complex control flow" — it is a large `match` on a discriminant
with many arms, each arm executing a different code path. The `alloca` for the state machine
struct is a primary allocation that feeds into all arms. This is the exact pattern that
triggers LLVM PTX backend issues with address space inference.

**The `toolchain.2` task has been deprioritized but not eliminated.** Rust-CUDA's NVVM IR
path handles async state machines, function pointers, and complex control flow more reliably
than the upstream NVPTX backend — not because NVVM IR is fundamentally different, but
because Rust-CUDA's team has put significant engineering effort into working around LLVM
NVPTX backend bugs specific to these patterns.

If the project reaches async-runtime.3 (minimal async/await execution experiment) and the
async state machine produces invalid PTX, the correct response at that point will be: "we
need Rust-CUDA." But at that point, significant work will need to be redone against the
Rust-CUDA toolchain.

The risk-adjusted strategy would be to run `toolchain.2` in parallel with hostcall
experiments rather than deferring it to be lowest priority.

---

## Summary: Prioritized Risk Table

| Claim | Actual Risk | Severity | Evidence |
|-------|------------|----------|---------|
| Inline PTX asm works (u64 variants) | UNVERIFIED — never compiled | HIGH | atomics.3 deferred u64 to hostcall.4 |
| ROCm design translates | PARTIAL — doorbell mechanism is fundamentally different | MEDIUM | PCIe polling vs HSA signals |
| Tagged pointer ABA safety | BROKEN at peak — tag wraps in ~1.6s | HIGH | Math in §3 above |
| Warp-granular packets | INEFFICIENT for async — 97% waste on divergent warps | MEDIUM | SIMT divergence analysis |
| Embassy 90% compatible | UNTESTED — no compilation attempt | HIGH | async-runtime.1 is desk analysis only |
| Host polling scalability | UNSCALED — 100-warp test never run | HIGH | hostcall.3 design only, no experiment |
| atomics.2 can wait | DANGEROUS — multi-warp atomics untested | HIGH | Only single-thread test done |
| gpu-std via libc shim | SPECULATIVE — zero precedent | MEDIUM | No public implementation exists |
| cudarc context DEVICEMAP | FRAGILE — relies on undocumented behavior | MEDIUM | Explicit in hostcall.1 findings |
| No Rust-CUDA needed | PREMATURE — async state machines untested | HIGH | toolchain.2 deprioritized too early |

---

## Recommendations

1. **BLOCKING: Run atomics.2 before any further hostcall work.** Multi-warp CAS correctness
   under contention must be confirmed before building the hostcall protocol on top of it.

2. **BLOCKING: Verify u64 inline PTX variants before committing to ADR-3.** The entire
   hostcall tagged-pointer design requires u64 CAS/fetch-add/exchange. If these fail,
   ADR-3 must be redesigned.

3. **HIGH: Fix the ABA tag.** Either switch to 64-bit tags (requires redesigning the
   tagged pointer format) or document a maximum safe throughput bound for the current design.

4. **HIGH: Prioritize toolchain.2 (Rust-CUDA).** The decision to deprioritize it was based
   on "atomics workaround found." That is insufficient. Async state machine compilation
   correctness on nvptx64 is the real unknown, and Rust-CUDA is the fallback. Understanding
   the fallback NOW is cheaper than discovering the blocker at async-runtime.3.

5. **MEDIUM: Attempt Embassy compilation for nvptx64 as an early experiment.** Replace the
   desk analysis with an actual compilation test before investing further in the Embassy port
   design.

6. **MEDIUM: Explicitly set CU_CTX_MAP_HOST.** Remove reliance on undocumented UVA behavior
   in the cudarc context initialization.

7. **LOW: Evaluate per-lane vs per-warp packet design** before gpu-std integration commits
   to the hostcall interface.
