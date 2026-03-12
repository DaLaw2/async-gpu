# per-block-sharding.1: Design per-block packet pool partitioning
**Cycle**: 72 | **Theme**: per-block-sharding | **Kind**: design | **Status**: done

## Summary

CAS contention on the global free_stack and ready_stack is the primary bottleneck at high thread counts. At 128 threads with 64 packets, CAS retries reach 24-43 per call; at 512 threads, ~70% starve. This document designs a per-block sharding scheme that partitions the packet pool across CUDA blocks, giving each block its own free_stack and ready_stack to eliminate cross-block CAS contention entirely.

## Findings

### Q: What buffer header layout changes are needed for per-block sharding?

A: The current 64-byte header is:
```
Offset  0: free_stack    (u64) — global tagged pointer
Offset  8: ready_stack   (u64) — global tagged pointer
Offset 16: doorbell      (u64) — global counter
Offset 24: shutdown      (u32)
Offset 28: num_packets   (u32)
Offset 32: warp_size     (u32)
Offset 36: [padding to 64]
```

The new **extended header** adds sharding metadata and per-block stack arrays:

```
=== Fixed Header (64 bytes, unchanged offsets for backward compat) ===
Offset  0: free_stack     (u64) — legacy global free stack (fallback)
Offset  8: ready_stack    (u64) — legacy global ready stack (fallback)
Offset 16: doorbell       (u64) — global doorbell (kept for host wakeup)
Offset 24: shutdown       (u32)
Offset 28: num_packets    (u32) — total packet count across all shards
Offset 32: warp_size      (u32)
Offset 36: num_shards     (u32) — 0 = legacy mode (no sharding)
Offset 40: pkts_per_shard (u32) — packets allocated to each shard
Offset 44: shard_array_offset (u32) — byte offset from buffer base to shard array
Offset 48: [reserved, 16 bytes]

=== Shard Array (at shard_array_offset, 16 bytes per shard) ===
Per-shard entry [i]:
  Offset  0: shard_free_stack  (u64) — tagged pointer, shard-local free list
  Offset  8: shard_ready_stack (u64) — tagged pointer, shard-local ready list

Total shard array size: num_shards * 16 bytes

=== Packet Pool (immediately after shard array) ===
Same layout as before: num_packets * PACKET_SIZE (2112 bytes each)
```

**Key design decisions:**

1. **num_shards = 0 means legacy mode.** Old kernels that don't read `num_shards` will use the global `free_stack`/`ready_stack` at offsets 0/8, which still work. The host initializes them as a fallback pool containing all packets when `num_shards == 0`.

2. **Shard array is placed between the fixed header and the packet pool.** The `shard_array_offset` field allows the host to locate it. For backward compatibility, if `num_shards == 0`, `shard_array_offset` can be set to `BUFFER_HEADER_SIZE` (64) and the packet pool starts right after. When sharding is enabled, the packet pool shifts forward by `num_shards * 16` bytes.

3. **Global doorbell is retained.** Even with per-block ready stacks, the host needs a single wakeup signal. The GPU still increments the global doorbell after pushing to its shard's ready_stack. The doorbell tells the host "something is ready somewhere" — the host then scans shard ready stacks.

4. **16-byte alignment per shard entry.** Each shard entry is exactly 16 bytes (two u64 tagged pointers), ensuring natural alignment for atomic operations on both x86 and CUDA.

**Buffer size formula (sharded):**
```
total_size = 64 + (num_shards * 16) + (num_packets * 2112)
```

For 16 shards, 256 packets: `64 + 256 + 540,672 = 540,992 bytes` (~528 KB).

**Confidence**: high

---

### Q: How should packets be partitioned across shards?

A: **Static partitioning with contiguous ranges.**

Shard `s` owns packets `[s * pkts_per_shard .. (s+1) * pkts_per_shard - 1]`. The packet indices are global (0..num_packets-1), so the `packet_offset()` function and tagged-pointer encoding remain unchanged.

**Sizing guidelines:**

| Blocks | Threads/block | Total threads | Recommended pkts_per_shard | Total packets | Shard array |
|--------|--------------|---------------|---------------------------|---------------|-------------|
| 4      | 32           | 128           | 8                         | 32            | 64 bytes    |
| 8      | 32           | 256           | 8                         | 64            | 128 bytes   |
| 16     | 32           | 512           | 16                        | 256           | 256 bytes   |
| 32     | 32           | 1024          | 8                         | 256           | 512 bytes   |

**Rule of thumb:** `pkts_per_shard >= threads_per_block / warp_size` at minimum (one packet per warp in the block), but `2x-4x` that for pipelining (threads can overlap spin-wait with new allocations).

**What if a block needs more packets than its shard has?**

Three options, in order of preference:

1. **Backpressure (recommended for v1).** If the shard's free_stack is empty, the thread spin-waits on it (same as today's global pool exhaustion). Since the thread releases packets back to its *own* shard's free_stack after the host responds, packets naturally recycle within the shard. This is the simplest approach and works well when `pkts_per_shard >= 2 * active_warps_per_block`.

2. **Fallback to global pool (v2 enhancement).** If the shard free_stack is empty, try the global `free_stack` at offset 0 as an overflow pool. The host could maintain a small global overflow pool (e.g., 10% of total packets). The GPU returns borrowed packets to the global free_stack, not the shard. This adds one extra CAS attempt but only under pressure.

3. **Work-stealing from neighboring shards (v3, complex).** A thread whose shard is empty tries shard `(block_idx + 1) % num_shards`, then `+2`, etc. This is complex and likely unnecessary — if a shard is consistently overloaded, the kernel launch configuration should be rebalanced.

**Recommendation:** Start with option 1 (backpressure only). The shard sizes should be set so that starvation within a shard is rare. If benchmarks show intra-shard starvation, add option 2 as a fallback.

**Confidence**: high

---

### Q: How does the GPU-side protocol change?

A: The GPU kernel needs to determine its shard index and use the shard-local stacks instead of the global ones.

**Shard index determination:**
```
shard_idx = blockIdx.x % num_shards
```

Typically `num_shards == gridDim.x` (one shard per block), but if the grid has more blocks than shards, multiple blocks share a shard. This allows the host to cap memory usage.

**Modified `hc_pop_free` (sharded):**
```rust
#[inline(always)]
unsafe fn hc_pop_free_sharded(buf: *mut u8, shard_idx: u32) -> u16 {
    let shard_array_off = read_volatile(buf.add(44) as *const u32) as usize;
    let free_ptr = buf.add(shard_array_off + (shard_idx as usize) * 16) as *mut u64;
    // Same CAS loop as before, but on shard-local free_stack
    loop {
        let old_head = sys_load_acquire_u64(free_ptr as *const u64);
        let idx = tagged_index(old_head);
        if idx == NULL_INDEX {
            return NULL_INDEX; // Shard exhausted — caller decides (spin or fallback)
        }
        let pkt = buf.add(packet_offset(idx));
        let next = read_volatile(pkt.add(PKT_OFF_NEXT) as *const u64);
        if sys_cas_u64(free_ptr, old_head, next) == old_head {
            return idx;
        }
        // CAS failed — retry, but contention is now only within this block
    }
}
```

**Modified `hc_push` (sharded ready stack):**
```rust
#[inline(always)]
unsafe fn hc_push_sharded(buf: *mut u8, shard_idx: u32, pkt_idx: u16) {
    let shard_array_off = read_volatile(buf.add(44) as *const u32) as usize;
    let ready_ptr = buf.add(shard_array_off + (shard_idx as usize) * 16 + 8) as *mut u64;
    // Same CAS push loop, but on shard-local ready_stack
    let pkt = buf.add(packet_offset(pkt_idx));
    loop {
        let old_head = sys_load_acquire_u64(ready_ptr as *const u64);
        write_volatile(pkt.add(PKT_OFF_NEXT) as *mut u64, old_head);
        let new_tag = tagged_tag(old_head).wrapping_add(1);
        let new_tagged = make_tagged(new_tag, pkt_idx);
        if sys_cas_u64(ready_ptr, old_head, new_tagged) == old_head {
            break;
        }
    }
}
```

**Modified `gpu_hostcall_request` flow:**
```
1. shard_idx = blockIdx.x % num_shards
2. Pop from shard_free_stack[shard_idx]         (CAS, low contention)
3. Fill packet (same as before)
4. Push to shard_ready_stack[shard_idx]          (CAS, low contention)
5. Increment global doorbell                     (fetch_add, still global but cheap)
6. Spin-wait on packet control (same as before)
7. Release back to shard_free_stack[shard_idx]   (CAS, low contention)
```

**Contention analysis:**
- Before: All N threads CAS on 1 free_stack + 1 ready_stack
- After: N/num_shards threads CAS on each shard's stacks
- With 16 blocks of 32 threads: 32 threads per shard instead of 512 on global
- Within a block, only thread-0 of each warp typically does hostcalls, so effective contention is `ceil(threads_per_block / 32)` = 1 CAS competitor per shard in the common case

**The global doorbell `fetch_add` remains**, but `fetch_add` is not a CAS retry loop — it always succeeds in O(1) on CUDA hardware. It causes serialization at the memory controller but does not cause retries. This is acceptable.

**Legacy detection on GPU:**
```rust
let num_shards = read_volatile(buf.add(36) as *const u32);
if num_shards == 0 {
    // Legacy path: use global stacks
    hc_pop_free(buf)
} else {
    let shard_idx = block_idx_x() % num_shards;
    hc_pop_free_sharded(buf, shard_idx)
}
```

**Confidence**: high

---

### Q: How does the host-side listener change?

A: The host listener currently does:
```
1. Poll doorbell
2. Atomic swap ready_stack → get linked list of ready packets
3. Walk list, dispatch each packet
4. Set CONTROL_READY on each packet
```

With per-block sharding, the host must poll **multiple** ready stacks.

**Option A: Round-robin scan (recommended)**

```rust
fn listen_sharded(&self) {
    let num_shards = self.num_shards();
    let mut last_doorbell: u64 = 0;

    loop {
        let current_doorbell = self.doorbell().load(Acquire);
        if current_doorbell == last_doorbell {
            // Adaptive sleep (same as today)
            continue;
        }
        last_doorbell = current_doorbell;

        // Scan all shard ready stacks
        for s in 0..num_shards {
            let ready_stack = self.shard_ready_stack(s);
            let head = ready_stack.swap(null_tagged(), AcqRel);
            if tagged_index(head) == NULL_INDEX {
                continue;
            }
            // Walk linked list, dispatch packets (same as today)
            self.process_ready_list(head);
        }
    }
}
```

**Cost analysis:**
- Each shard scan is one atomic swap (8 bytes). For 16 shards, that's 16 atomic swaps per doorbell change.
- Each swap is ~100ns on x86 → 16 swaps = ~1.6 us overhead per batch.
- The doorbell coalesces multiple GPU pushes, so the host doesn't scan per-packet — it scans per-batch.
- This is negligible compared to the service handling time.

**Option B: Bitmap notification (optimization for many shards)**

If `num_shards > 64`, scanning all stacks becomes wasteful. A 64-bit bitmap where each GPU push sets bit `shard_idx % 64` would let the host skip empty shards. However, with typical shard counts (4-32), round-robin is fine.

**Host-side initialization (sharded):**

```rust
fn init_sharded(&self) {
    let num_shards = self.num_shards();
    let pkts_per_shard = self.pkts_per_shard();

    for s in 0..num_shards {
        let base_pkt = s * pkts_per_shard;
        let shard_entry = self.shard_entry(s);

        // Build per-shard free list: chain packets within this shard
        for i in 0..pkts_per_shard {
            let pkt_idx = (base_pkt + i) as u16;
            let pkt = self.packet_ptr(pkt_idx);
            let next = if i + 1 < pkts_per_shard {
                make_tagged(0, (base_pkt + i + 1) as u16)
            } else {
                null_tagged()
            };
            write_volatile(pkt.add(PKT_OFF_NEXT), next);
            write_volatile(pkt.add(PKT_OFF_CONTROL), 0u32);
        }

        // Set shard free_stack head
        shard_entry.free_stack.store(make_tagged(0, base_pkt as u16), Release);
        // Set shard ready_stack to empty
        shard_entry.ready_stack.store(null_tagged(), Release);
    }

    // Global stacks: leave empty (or set as overflow pool)
    self.free_stack().store(null_tagged(), Release);
    self.ready_stack().store(null_tagged(), Release);
}
```

**Packet release:** The GPU returns packets to its own shard's free_stack. The host never moves packets between shards. This means the host does NOT push to any free_stack — the GPU owns the release path. This is the same as the current protocol where the GPU does step 8 (return to free stack).

**Confidence**: high

---

### Q: How is backward compatibility maintained?

A: **Two-layer compatibility:**

1. **Old kernels + new host (sharded buffer):** Old kernels read `free_stack` at offset 0 and `ready_stack` at offset 8. If the host sets `num_shards = 0` and puts all packets in the global free_stack (legacy init path), old kernels work unchanged. The host detects `num_shards == 0` and uses the legacy single-stack listener.

2. **New kernels + old buffer (no sharding):** New kernels read `num_shards` at offset 36. In the old 64-byte header, offset 36 is in the padding region (zeroed). Zero means "legacy mode," so the new kernel falls back to global stacks. No breakage.

3. **Version negotiation (optional):** The host could write a `version` field at offset 48 (currently reserved). Version 0 = legacy, version 1 = sharded. But the `num_shards == 0` sentinel is sufficient for v1.

**The `packet_offset()` function must account for the shard array:**
```rust
// New: packets start after header + shard array
pub const fn packet_offset_sharded(index: u16, shard_array_size: usize) -> usize {
    BUFFER_HEADER_SIZE + shard_array_size + (index as usize) * PACKET_SIZE
}
```

The host writes `shard_array_offset` (offset 44) so the GPU knows where the shard array starts. The GPU also needs to know where packets start. Two options:

- **Option A:** Add a `packet_base_offset` field to the header (e.g., offset 48). GPU uses `buf + packet_base_offset + idx * PACKET_SIZE`.
- **Option B:** GPU computes it: `packet_base = shard_array_offset + num_shards * 16`.

Option B avoids adding another header field. Since the GPU already reads `shard_array_offset` and `num_shards`, this is a trivial computation.

For legacy mode (`num_shards == 0`), `shard_array_offset = 64` and `num_shards * 16 = 0`, so `packet_base = 64 = BUFFER_HEADER_SIZE` — matching the current layout exactly.

**Confidence**: high

---

### Q: What are the trade-offs and edge cases?

A:

**Trade-offs:**

| Aspect | Global (current) | Per-block sharded |
|--------|-----------------|-------------------|
| CAS contention | O(N) threads on 1 stack | O(N/S) threads on S stacks |
| Memory overhead | 64 bytes header | 64 + S*16 bytes header |
| Host polling cost | 1 swap | S swaps per batch |
| Packet utilization | Perfect (global pool) | May have idle packets in low-activity shards |
| Implementation complexity | Simple | Moderate |
| GPU code size | Minimal | +~50 instructions for shard selection |

**Edge cases:**

1. **Uneven block activity.** If block 0 makes 100 hostcalls and block 15 makes 0, block 0's shard may exhaust while block 15's packets sit idle. Mitigation: backpressure (block 0 waits for its packets to recycle). This is acceptable because the host processes packets fast (~1-10 us per NOP), so recycling is rapid.

2. **More blocks than shards.** If `gridDim.x = 64` but `num_shards = 16`, blocks 0,16,32,48 share shard 0. Contention within a shared shard is 4 blocks — still 16x better than 64 blocks on a global stack. The host should set `num_shards = min(gridDim.x, max_shards)` where `max_shards` is limited by memory budget.

3. **Dynamic grid sizes.** The buffer is allocated before kernel launch. If the grid size changes between launches, the host must either:
   - Reallocate the buffer (expensive), or
   - Over-provision shards (e.g., allocate 32 shards for a typical max of 32 blocks), or
   - Reuse the same buffer with `num_shards <= allocated_shards` by only initializing the needed shards.

   Recommendation: over-provision moderately (e.g., 32 shards) since the overhead is only 512 bytes.

4. **Warp-cooperative hostcalls.** Currently, only one thread per warp (lane 0) does the hostcall. With 32 threads per block and warp size 32, that's 1 thread contending per shard — effectively zero CAS retries. If multiple warps per block are used (e.g., 128 threads/block = 4 warps), contention per shard is only 4 threads, still minimal.

5. **Host I/O thread interaction.** The I/O thread currently receives packets via an `mpsc::channel`. With sharding, the listener thread walks multiple ready lists but still dispatches to the same I/O thread via the same channel. No change needed to the I/O thread.

6. **ABA problem.** The tagged pointer ABA protection is unchanged. Each shard has its own tag counter space. Since tags are 32-bit and monotonically increasing per-stack, ABA safety is preserved per-shard.

7. **Packet index space.** Packet indices remain global (0..num_packets-1). The `NULL_INDEX = 0xFFFF` sentinel limits total packets to 65534. With sharding, this is still the global limit. At 16 bytes per shard entry, even 1024 shards only add 16 KB — the packet pool (1024 * 2112 = 2.1 MB) dominates.

**Confidence**: high

---

### Q: What is the expected performance improvement?

A: **Analytical model:**

CAS retry probability = `(N-1)/N` per attempt when N threads contend simultaneously (worst case). Expected retries = `N-1` for a successful CAS.

| Scenario | Threads on stack | Expected CAS retries/call |
|----------|-----------------|--------------------------|
| Global, 128 threads | 128 | ~127 (theoretical worst), 24-43 (measured) |
| Global, 512 threads | 512 | ~511 (theoretical worst) |
| Sharded 16 blocks, 128 threads | 8 per shard | ~7 |
| Sharded 16 blocks, 512 threads | 32 per shard | ~31 |
| Sharded, 1 warp/block | 1 per shard | ~0 |

The measured 24-43 retries at 128 threads (vs theoretical 127) suggests partial temporal spreading. Sharding should reduce this proportionally.

**Throughput estimate:** Current 8K calls/s at 512 threads is bottlenecked by CAS retries. With 16 shards (32 threads/shard), throughput should scale roughly linearly with shard count: ~128K calls/s (16x improvement). In practice, the host listener becomes the bottleneck when GPU-side contention is eliminated, so actual improvement is likely 4-8x until host-side optimizations are applied.

**Latency:** P50 latency at 128 threads with sharding should approach single-thread latency (~38 us measured in benchmark.2), down from the current contention-inflated values.

**Confidence**: medium (estimates based on analytical model, needs benchmark validation)

## Impact on Downstream Tasks

1. **gpu-protocol crate:** Add new constants for shard entry size, header field offsets (36, 40, 44), and sharded `packet_offset` helper. Add `buffer_size_sharded(num_packets, num_shards)` function.

2. **gpu-kernel crate:** Update `hc_pop_free`, `hc_push`, and `gpu_hostcall_request` to accept/compute shard index. Add legacy detection (`num_shards == 0` fallback).

3. **gpu-host crate:** Update `HostcallBuffer::new` to accept shard count. Update `init()` to build per-shard free lists. Update `listen_unified` to scan shard ready stacks round-robin.

4. **Benchmark task:** Re-run `hostcall_latency_bench` with sharded buffer at 128/256/512 threads to measure actual CAS retry reduction and throughput improvement.

5. **Potential follow-up tasks:**
   - Global overflow pool (option 2 from sizing question)
   - Host-side parallel shard processing (multiple host threads, each handling a subset of shards)
   - Bitmap-based shard notification for large shard counts
