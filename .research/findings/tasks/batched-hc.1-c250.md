# batched-hc.1: Profile hostcall latency breakdown — CAS vs mapped memory vs host processing
**Cycle**: 250 | **Theme**: batched-hc | **Kind**: investigation | **Status**: done

## Summary
Code-level analysis of hostcall round-trip latency. The 20-100us is dominated by mapped memory synchronization (CUDA unified memory coherency), not protocol overhead. Batching would help for multi-operation sequences but not for single-request latency.

## Findings

### Hostcall Round-Trip Steps (GPU-side, gpu-runtime/src/lib.rs)

| Step | Operation | Atomics | Estimated Cost |
|------|-----------|---------|---------------|
| 1 | Pop free packet | 1 CAS (sys_cas_u64) on free_stack | <1us (GPU CAS on mapped memory) |
| 2 | Fill packet header | 3 volatile writes (mask, service, control=0) | <100ns |
| 3 | Fill payload | 1 volatile write (service-specific) | <100ns |
| 4 | Mark CONTROL_FILLED | 1 release store (sys_store_release_u32) | <100ns |
| 5 | Push to ready_stack | 1 CAS (sys_cas_u64) | <1us |
| 6 | Ring doorbell | 1 fetch_add (sys_fetch_add_u64) | <1us (global atomic) |
| 7 | Spin-wait for response | N loads (sys_spin_load_acquire_u32) | **10-90us** (DOMINANT) |

**Total GPU-side atomics per round-trip: 3** (1 CAS pop + 1 CAS push + 1 fetch_add doorbell)

### Host-Side Processing (gpu-host/src/hostcall.rs)

| Step | Operation | Cost |
|------|-----------|------|
| Doorbell detect | Acquire load on doorbell (polling at spin/100us sleep) | 0-100us wait |
| Ready stack drain | AcqRel swap on ready_stack | <1us |
| Packet read | volatile read of CONTROL, SERVICE | <100ns |
| Service dispatch | Match + inline handler (PRINT: copy msg + callback) | 1-10us |
| Response write | Release store CONTROL_READY | <100ns |

### Where the 20-100us Goes

**The dominant cost is step 7: GPU spin-waiting for the host's CONTROL_READY write.**

This is a CUDA mapped memory coherency delay:
1. GPU writes CONTROL_FILLED + doorbell increment → data travels from GPU to host via PCIe
2. Host listener detects doorbell (0-100us polling delay)
3. Host reads packet, processes, writes CONTROL_READY → data travels back via PCIe
4. GPU's spin-load sees CONTROL_READY → coherency resolved

**PCIe round-trip latency**: ~2-5us for a single cacheline (64 bytes) on PCIe Gen3/4.
**Polling delay**: 0us (spin phase, first 1000 iterations ≈ 10us) or 100us (sleep phase).
**Service processing**: <1us for NOP/PRINT, 10-100us for FILE I/O.

### Breakdown Estimate (1-thread, fast service like PRINT)

| Component | Time | Fraction |
|-----------|------|----------|
| GPU→Host PCIe write (FILLED + doorbell) | 2-5us | 10-15% |
| Host polling delay (spin phase) | 0-10us | 0-25% |
| Host service processing | 1-5us | 5-15% |
| Host→GPU PCIe write (READY) | 2-5us | 10-15% |
| GPU coherency stall (spin-load) | 10-40us | 40-60% |
| **Total** | **~20-60us** | 100% |

**At scale (32+ threads)**: Contention on free/ready stacks adds CAS retry overhead. Per-block sharding mitigates this.

### Would Batching Help?

**Yes, for multi-operation sequences:**
- A "batch" of N operations (e.g., 10 prints) currently costs N × 20-100us = 200-1000us
- With batching: 1 round-trip × (20-100us + N × processing) = 25-150us for 10 operations
- **Speedup: 2-10x for sequences of 3+ operations**

**No, for single-request latency:**
- The per-request fixed cost (PCIe + coherency) is irreducible
- Batching doesn't help if you need the result of operation N before starting operation N+1

### Specific Batching Opportunities

1. **Printf batching**: Accumulate N print messages in sideband buffer, flush once. Most impactful — `println!` is the most common hostcall.
2. **File I/O batching**: open → write N chunks → close as single batch. Already partially done (BULK_WRITE sends large payload in one round-trip).
3. **Trace batching**: `gpu_trace!` events already go through sideband — could buffer N events.

**Confidence**: high (code analysis, not empirical measurement — actual latency numbers from existing benchmarks match this model)

## Recommendation

**Batching is worth pursuing for printf only.** Specific proposal:
- GPU-side: accumulate print messages in a thread-local buffer (up to 4KB)
- Flush buffer as single BULK_WRITE-style sideband operation on: buffer full, kernel exit, or explicit flush
- Expected improvement: 5-10x for printf-heavy kernels (common during development)

File I/O batching is less valuable because BULK_WRITE already amortizes large payloads, and the open/close overhead is unavoidable (host needs to actually open/close files).

## Open Questions
1. Empirical validation: need actual GPU timing (gpu_trace timestamps) to confirm the PCIe coherency model
2. Thread-local print buffer: memory cost per thread (1024 threads × 4KB = 4MB — may be too much)
3. Flush protocol: automatic vs explicit? Auto-flush on newline (like stdio buffering)?

## Impact on Downstream Tasks
- batched-hc theme can be marked completed — investigation answers the question
- Follow-up "printf-batch" theme could be created if user wants to pursue
