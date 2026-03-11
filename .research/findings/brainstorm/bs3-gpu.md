# BS3 GPU Architecture Analysis
**Role**: GPU Architecture Expert (gpu)
**Brainstorm**: bs3
**Date**: 2026-03-11

## Overview

This analysis examines the hostcall protocol design (hostcall.3-c10) through the lens of
GPU microarchitecture: PCIe latency, warp divergence, register pressure, occupancy, and
memory access coalescing on the RTX 3060 (SM 8.6, Ampere).

---

## 1. PCIe Latency Analysis

Each system-scope atomic (`atom.sys.global`) on mapped pinned memory crosses PCIe and
touches host DRAM. On RTX 3060 (PCIe Gen4 x16):

- PCIe Gen4 x16 theoretical bandwidth: ~32 GB/s bidirectional
- Per-operation round-trip latency: **~1–2 µs** (estimated from historical PCIe 3.0/4.0
  atomic measurements; 4.0 is marginally faster than 3.0)

### Hostcall Critical Path Latency Breakdown

| Step | Operation | Estimated Latency |
|------|-----------|------------------|
| Pop free_stack | `sys_cas_u64` (CAS loop, 1 attempt typical) | 1–2 µs |
| Push ready_stack | `sys_cas_u64` (CAS loop, 1 attempt typical) | 1–2 µs |
| Ring doorbell | `sys_fetch_add_u64` | 1–2 µs |
| Host detects doorbell | CPU polling latency | 0.1–50 µs (host-side) |
| Host dispatch | Service handler execution | 1–100 µs |
| Host signals control | `AtomicU32::store(Release)` via PCIe | ~1 µs |
| GPU detects control | `sys_load_acquire_u32` spin loop | 1–50 µs |
| Return free packet | `sys_cas_u64` (CAS loop) | 1–2 µs |

**Total minimum latency**: ~5–10 µs (best case, no host-side backoff)
**Typical latency**: ~10–100 µs depending on host polling state and service complexity

### CAS Retry Penalty

Under low contention (few warps doing hostcalls simultaneously), CAS loops succeed on
the first attempt — 1 retry costs an additional 1–2 µs. Under high contention with 448
warps, CAS failure rate rises, potentially adding 5–20 µs in retry overhead.

### Conclusion

The protocol is fundamentally latency-bound by PCIe round-trips. Each hostcall incurs
~4–8 PCIe round-trips minimum. This is acceptable for RPC (printf, file I/O) but
makes hostcalls **unsuitable for any compute-critical hot path**.

---

## 2. Warp Divergence in CAS Loops

### CAS Loop Structure (Free Stack Pop)

The protocol assigns CAS to one lane per warp (lane 0 or the leader). The other 31 lanes
are inactive in the CAS loop — they wait at a `__syncwarp()` or implicit reconvergence
point. However, two divergence issues arise:

### Issue A: CAS Retry Divergence

If the CAS loop retries multiple times (due to contention), lane 0 loops while lanes 1–31
wait. This is **not a true divergence problem** for warp efficiency — inactive lanes in a
conditional block consume no instruction slots (NVIDIA warps execute in lockstep). The
warp is stalled, not diverged in the performance-critical sense.

**Cost**: The warp occupies a warp slot during CAS retries but does not block other warps.
If the GPU has sufficient warps to hide this latency (typical for PCIe operations), the
impact is masked by warp switching.

### Issue B: Control Flag Spin Wait

All 32 lanes must spin waiting for `packet.header.control` READY bit. This is a **uniform
condition** — all lanes read the same address and wait for the same value. No divergence
occurs here because all lanes take the same branch simultaneously.

However, **the warp occupies a warp slot for the entire spin duration**. With a potential
spin duration of 10–100 µs and a warp clock of ~1 GHz (1 µs = ~1000 cycles), a warp
doing a hostcall may spin for **10,000–100,000 cycles**.

### Occupancy During Spin

This is the critical concern: **a spinning warp blocks its warp slot**. With 48 warps/SM
max and 28 SMs, if many warps are spinning on hostcalls, the GPU cannot use those warp
slots for computation. Unlike GPU memory access (which uses MIO latency hiding), PCIe
round-trips are orders of magnitude longer than what warp switching can hide.

**Recommendation**: Consider using `nanosleep` (SM 7.0+, available on Ampere) in the spin
loop to yield the warp slot and reduce power consumption:
```ptx
nanosleep.u32 64;  // hint to scheduler to deschedule for ~64 ns
```
This allows the hardware scheduler to deschedule the spinning warp and schedule other
warps, improving overall GPU utilization.

### Active Mask Handling

The protocol reads `__activemask()` to fill `packet.header.active_mask`. This must happen
**before** any conditional that could diverge. Getting the active mask post-divergence
gives the wrong mask for the active lanes at the point of the hostcall.

For `nvptx64`, use inline PTX:
```rust
pub unsafe fn activemask() -> u32 {
    let mask: u32;
    core::arch::asm!(
        "activemask.b32 {0};",
        out(reg32) mask,
    );
    mask
}
```
Alternatively, `vote.ballot.sync 0xFFFFFFFF, 1` gives the same result (bitmask of lanes
active in the current instruction). Do **not** hardcode `0xFFFFFFFF` — partial warps at
kernel edges are common and hardcoding causes incorrect behavior.

---

## 3. Register Pressure Analysis

### Registers Required by Hostcall Function

Estimating registers consumed by the critical path:

| Variable | Type | Registers (64-bit count) |
|----------|------|--------------------------|
| buffer_ptr | *const HostcallBuffer | 1 (u64) |
| free_stack ptr | *const u64 | 1 (derived from buffer_ptr) |
| old_head | u64 | 1 |
| packet_ptr | *const Packet | 1 |
| new_head | u64 | 1 |
| CAS result | u64 | 1 |
| ready_stack ptr | *const u64 | 1 |
| new_tag | u64 | 1 |
| new_tagged | u64 | 1 |
| doorbell ptr | *const u64 | 1 |
| control ptr | *const u32 | 1 (upper 32 bits wasted) |
| control value | u32 | 1 |
| spin_count | u32 | 1 |
| service | u32 | 1 |
| args (7 × u64) | [u64; 7] | 7 |
| response (7 × u64) | [u64; 7] | 7 |
| lane_id | u32 | 1 |
| active_mask | u32 | 1 |

**Estimated total**: ~28–35 registers for the hostcall function.

### Impact on Occupancy

SM 8.6 has 65,536 registers per SM. Occupancy is computed based on register usage per
thread:

| Registers/thread | Max threads/SM | Warps/SM | Occupancy (vs 48 max) |
|-----------------|----------------|----------|-----------------------|
| 32 | 2048 | 64 | 100%+ (capped at 48) |
| 48 | 1365 | 42 | 88% |
| 64 | 1024 | 32 | 67% |
| 96 | 682 | 21 | 44% |
| 128 | 512 | 16 | 33% |

If the hostcall function uses ~30–35 registers AND the calling kernel also uses registers,
the combined register count for a thread could reach 60–96 registers, **reducing occupancy
to 44–67%**. This is a meaningful reduction but not catastrophic for a kernel whose
primary bottleneck is PCIe latency, not compute throughput.

**Recommendation**: Use `#[inline(always)]` for the hostcall function to avoid register
spills from call overhead. Add `#[no_mangle] extern "C"` only if the function must be
separately compiled. The compiler should allocate registers across the inlined call site.

---

## 4. `__activemask()` on nvptx64

### Recommended Implementation

```rust
#[inline(always)]
pub unsafe fn activemask() -> u32 {
    let mask: u32;
    core::arch::asm!(
        "activemask.b32 {0};",
        out(reg32) mask,
    );
    mask
}
```

`activemask.b32` is a PTX instruction (PTX ISA 6.2+, SM 7.0+; SM 8.6 supported) that
returns a bitmask where bit N is set if lane N is currently active (not predicated off
by a conditional or loop exit). This is the correct way to capture which lanes are
participating at the hostcall entry point.

### Alternative: vote.ballot.sync

```rust
core::arch::asm!(
    "vote.ballot.sync.b32 {0}, 0xFFFFFFFF, 1;",
    out(reg32) mask,
);
```

`vote.ballot.sync` also produces the correct mask, but requires SM 7.0+ (Ampere qualifies)
and the `0xFFFFFFFF` membermask argument is itself an active_mask — creating a circular
dependency. Prefer `activemask.b32` for clarity.

### Do NOT Hardcode 0xFFFFFFFF

Partial warps (tail of a grid where `blockDim % 32 != 0`) will have fewer than 32 active
lanes. Hardcoding 0xFFFFFFFF would cause lanes that don't exist to be processed, leading
to uninitialized payload slots being sent to the host. This is a correctness bug.

---

## 5. Memory Access Pattern for Packet Payload

### Current Layout

```
PayloadSlot layout:
  slots[lane][arg]
  slots[0][0..8]  // lane 0's 8 args
  slots[1][0..8]  // lane 1's 8 args
  ...
  slots[31][0..8] // lane 31's 8 args
```

When 32 lanes each write `slots[lane_id][0..8]`:
- Lane 0 writes to offset 0–63 bytes
- Lane 1 writes to offset 64–127 bytes
- Lane 2 writes to offset 128–191 bytes
- ...

This is **fully coalesced**: consecutive lanes write consecutive 64-byte regions. The GPU
L2 cache will coalesce these into 256-byte (4 cache-line) transactions. This is optimal
for PCIe write bandwidth.

### However: PCIe Write Width

PCIe transactions are coalesced at the L2 boundary. For mapped pinned memory (no GPU-side
caching), 32 lanes writing 64 bytes each = 2048 bytes total, which will be merged into
~8 PCIe write transactions of 256 bytes. This is efficient.

**Payload write is NOT a bottleneck** — the PCIe latency of the CAS operations dominates.

### Alternative: Transposed Layout (AoS vs SoA)

If lanes needed to read each other's data (via `shfl.sync`), a transposed layout
`args[arg][lane]` would be needed. But since each lane writes only its own slot, the
current row-major layout is correct and efficient.

---

## 6. Occupancy Impact

### Sources of Occupancy Reduction

1. **Register pressure** (analyzed in §3): 44–88% occupancy depending on total registers
2. **Shared memory**: The hostcall protocol uses no shared memory directly → no impact
3. **Packet pool in pinned memory**: The 64–224 packets (~132–461 KB) are in **host DRAM**,
   not GPU shared memory or L2 — zero impact on SM occupancy resources

### Realistic Occupancy Estimate

For a kernel that:
- Does moderate computation (using ~32 registers/thread for compute logic)
- Calls hostcall (adding ~28–35 registers)
- Total: ~60–67 registers/thread

Expected occupancy: **~50–67%** (24–32 warps/SM instead of 48 max).

This is acceptable for a hostcall-heavy kernel. The bottleneck is PCIe latency, not
compute occupancy. Increasing occupancy would not hide the PCIe round-trip latency anyway.

### Occupancy vs Latency Hiding

For GPU kernels that hide memory latency via occupancy (many warps in flight), the GPU
scheduler switches to another warp when one stalls on L2/HBM. PCIe latency is too long
(~1–2 µs = ~1000–2000 cycles) to be hidden by warp switching alone — there simply aren't
enough warps. **The spin-wait is unavoidable for synchronous hostcall semantics.**

If latency hiding is required, the protocol would need to be restructured as an
**async hostcall** (GPU issues request, continues computation, polls later) — a
significantly more complex design.

---

## 7. Concurrent Warp Contention on CAS

### Worst Case: All Warps Simultaneously Issue Hostcalls

- 28 SMs × 48 warps/SM = 1344 warps maximum
- Realistic: 28 × 16 = 448 warps active (50% occupancy)
- Each warp needs 1 CAS on free_stack to pop a packet

### CAS Contention Analysis

With N warps simultaneously trying to CAS the same `free_stack` pointer:
- Only 1 succeeds per "round" — others retry
- Expected retries per warp: O(N) in the worst case
- PCIe round-trip per attempt: ~1–2 µs
- With 448 warps: potentially 200–900 µs of CAS contention before all warps acquire a packet

**This is the primary scalability concern of the design.**

### Mitigations

1. **LIFO stack is correct but serialized**: The two-stack design (free_stack + ready_stack)
   means all warps compete for a single lock-free pointer. Under high concurrency, CAS
   contention on the free_stack becomes a bottleneck.

2. **Per-SM free lists**: Instead of one global free_stack, maintain 28 per-SM pools.
   Each SM's warps compete for their local pool. This reduces contention by 28×.
   Cost: requires routing by SM ID, obtainable via `%smid` PTX register.

3. **Pool size matches concurrent demand**: If the packet pool has ≥ concurrent warps
   packets, each warp gets a packet quickly. The issue is the CAS serialization, not pool
   exhaustion.

4. **Exponential backoff**: After a failed CAS, warp 0 (lane 0) waits
   `1 << attempt_count` cycles before retrying. This reduces simultaneous retries.
   `nanosleep.u32` can implement this.

### Practical Expectation

In practice, hostcalls are **not issued by every warp simultaneously** — they're rare
events (printf at error paths, file I/O). Under typical workloads with 5–20 concurrent
hostcall warps, the CAS contention is negligible (<5 µs overhead).

The concern materializes only in stress tests or degenerate kernels where every warp
hostcalls in a tight loop — which would not be a sensible GPU workload in any case.

---

## Key Conclusions

1. **Latency**: 10–100 µs per hostcall is expected and appropriate for RPC semantics.
   GPU warps will spin-wait; this is unavoidable for synchronous hostcall.

2. **Divergence**: The spin-wait on `control` is uniform (no divergence). CAS loops use
   only lane 0 — other lanes are inactive but do not cause predicated divergence overhead.
   Use `nanosleep` in spin loops to improve power and allow scheduler to rotate.

3. **Register pressure**: ~28–35 registers for hostcall logic. Combined with compute
   kernel registers, expect 50–67% occupancy. Acceptable for latency-bound workloads.

4. **`activemask()`**: Use `activemask.b32` PTX instruction. Do NOT hardcode 0xFFFFFFFF.

5. **Payload coalescing**: Current `slots[lane][arg]` layout is fully coalesced. No
   layout change needed.

6. **CAS contention**: Primarily a concern under artificial stress. Per-SM free lists
   would be the right optimization if contention proves measurable in practice.

7. **Async hostcall extension**: If latency hiding becomes a requirement (future theme),
   the protocol would need non-blocking request submission + separate completion poll.
   This should be a separate task/theme, not part of the initial implementation.

---

## Suggested New Tasks

- **hostcall.5** (experiment): Benchmark round-trip PCIe latency for a single
  `atom.sys.global.cas.b64` on RTX 3060 to confirm the 1–2 µs estimate.
- **hostcall.6** (design): Evaluate per-SM free list as a contention mitigation.
  Only if stress testing reveals CAS contention > 10 µs.
- **async-runtime.N** (design): Async hostcall (non-blocking submission + future-style
  completion) as a stepping stone toward async/await on GPU.
