# atomics.2: Stress-test GPU-CPU atomic communication
**Cycle**: 12 | **Theme**: atomics | **Kind**: experiment | **Status**: done

## Summary

Implemented and ran two stress tests for GPU-CPU atomic communication via
pinned (mapped) memory. Both tests passed on the first attempt with zero
races observed, confirming that system-scope atomics provide reliable
cross-device communication.

## Findings

### Q: Can GPU and CPU reliably exchange data via atomic flags in pinned memory?
A: **Yes.** Both the multi-thread counter test (1024 threads across 32 blocks)
and the bidirectional ping-pong test (100 round trips) completed with perfect
correctness. System-scope atomic operations (`atom.add.sys.global.u32`,
`st.release.sys.global.u32`, `ld.acquire.sys.global.u32`) on pinned
`CU_MEMHOSTALLOC_DEVICEMAP` memory work reliably for GPU-CPU communication.

### Q: Does a counter incremented by GPU and read by CPU show correct values?
A: **Yes.** 1024 GPU threads (32 blocks x 32 threads) each performed one
`atom.add.sys.global.u32` on a shared mapped counter. The final counter value
was exactly 1024 — no increments were lost. The `sys` scope ensures the atomic
add is visible across the PCIe bus to the host CPU.

### Q: Are there observable races under heavy concurrent access?
A: **No races observed.** The atomic counter test had 1024 threads contending on
a single u32, with all 32 warps potentially executing the atomic add
simultaneously. The result was exact. The ping-pong test ran 100 sequential
round trips with no missed or corrupted exchanges. System-scope atomics
provide correct serialization even under heavy contention.

### Q: What is the minimum fence required for correctness?
A: **Release/acquire pairs are sufficient.** The ping-pong test uses only:
- GPU side: `st.release.sys.global.u32` for writes, `ld.acquire.sys.global.u32`
  (via `sys_spin_load_acquire_u32`) for reads
- Host side: `AtomicU32::store(Release)` for writes, `AtomicU32::load(Acquire)`
  for reads

No explicit `membar.sys` was needed in the ping-pong protocol. The
release-on-flag-store / acquire-on-flag-load pair is the minimum fence for
correct GPU-CPU communication. For the multi-thread counter test, a `membar.sys`
was added before the flag store to ensure all atomic adds from other blocks are
globally visible before signaling completion, but the atomic adds themselves
(`atom.add.sys`) provide their own atomicity.

## Test Results

### Stress Atomic Counter
- Config: 32 blocks x 32 threads = 1024 total, single mapped u32 counter
- Kernel: each thread does `sys_fetch_add_u32(counter_ptr, 1)`
- Signaling: thread 0 of block 0 does `membar.sys` then `st.release.sys(flag, 1)`
- Host poll iterations until flag: 1240
- Result: counter == 1024 (exact match). **PASSED**

### Stress Ping-Pong
- Config: 1 GPU thread, 100 round trips
- Protocol per iteration i (1..=100):
  - Host: `AtomicU32::store(i, Release)` to request_ptr
  - GPU: `sys_spin_load_acquire_u32(request_ptr)` until == i
  - GPU: `st.release.sys(response_ptr, i*2)`
  - GPU: `st.release.sys(flag_ptr, i)`
  - Host: poll `flag_ptr` with `Acquire` until == i
  - Host: read `response_ptr` with `Acquire`, verify == i*2
- Duration: 544.4us total, 5.444us per round trip
- Result: all 100 responses correct. **PASSED**

## Unexpected Discoveries

1. **Low latency**: The ~5.4us round-trip latency for GPU-CPU ping-pong via
   mapped memory is surprisingly fast. This is well within acceptable bounds
   for a hostcall protocol (the hostcall design targets <10us per call).

2. **Flag polling convergence**: The atomic counter's flag was detected after
   only 1240 host poll iterations despite 32 blocks needing to complete. The
   `membar.sys` + `st.release.sys` on the flag is very quickly visible to the
   host CPU.

3. **nanosleep in spin loop**: The `nanosleep.u32 64` instruction in
   `sys_spin_load_acquire_u32` yields the warp slot during spinning, which
   avoids starving other warps. This did not noticeably increase latency
   (still ~5.4us per round trip) while being friendlier to the GPU scheduler.

## Open Questions

1. What happens with many MORE threads (e.g., 65536 = 256 blocks x 256 threads)?
   The current test uses 1024 which is modest. Higher contention on a single
   atomic could cause performance degradation due to L2 cache line bouncing.

2. How does latency scale with multiple concurrent ping-pong channels? The
   hostcall protocol needs per-warp channels. Testing with 32+ channels
   would be valuable.

3. What is the raw throughput limit of `atom.add.sys.global.u32` on pinned
   memory? This determines the maximum hostcall rate the system can sustain.

## Impact on Downstream Tasks

- **hostcall**: The release/acquire protocol is confirmed working for both
  one-to-one (ping-pong) and many-to-one (counter) patterns. The hostcall
  protocol design (from hostcall.3) can proceed with confidence that
  sys-scope atomics on mapped memory are reliable and fast enough (~5us
  round trip) for practical use.

- **async-runtime**: The ping-pong latency of ~5.4us means a GPU async task
  that needs to make a hostcall (e.g., for I/O) can resume within ~10us
  (request + response). This is fast enough to not be a bottleneck for
  typical async workloads.
