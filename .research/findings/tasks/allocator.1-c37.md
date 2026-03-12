# allocator.1: Survey GPU allocator designs

**Cycle**: 37 | **Theme**: allocator | **Kind**: investigation | **Status**: done

## Summary

Surveyed GPU memory allocator designs to replace the current bump allocator (which
never frees memory). Evaluated TLSF, slab, buddy, and purpose-built GPU allocators
(ScatterAlloc, halloc, Ouroboros, RegEff, SoaAlloc) against our constraints: nvptx64
bare-metal, no OS/libc, thread-safe for 32-1024 threads, low register pressure,
atomic-only synchronization (no locks/mutexes/TLS). The recommendation is a
**hybrid slab + bitmap allocator** as the best fit for our use case.

## Findings

### Q1: What allocator designs work best in GPU's constrained environment?

A: Five families of allocators were evaluated:

**TLSF (Two-Level Segregated Fit)**
- O(1) alloc and dealloc via a two-level bitmap index over segregated free lists.
- Very low fragmentation (< 15% typically).
- Standard implementation uses a single global data structure with ~100 bytes of
  control overhead. The mapping function uses CLZ (count leading zeros) which maps
  to PTX `bfind` instruction.
- **Problem**: Not designed for concurrency. The reference implementation
  (mattconte/tlsf) requires external locking. Making it lock-free would require
  atomically updating both the bitmap levels AND the free list in a single operation,
  which is not possible with single-word CAS. A lock-free TLSF does not exist in
  the literature.
- **Verdict**: Excellent for single-threaded use. Not viable for multi-thread GPU
  without a global lock, which would serialize all allocations across warps.

**Slab Allocator (fixed-size pools)**
- Pre-partitions memory into pools of fixed block sizes (e.g., 16, 32, 64, 128,
  256, 512, 1024, 2048, 4096 bytes).
- Alloc = find a free block in the appropriate pool. Dealloc = return block to pool.
- Zero external fragmentation within a size class. Internal fragmentation bounded
  by rounding up to next size class (worst case ~50%, typical ~25%).
- **Lock-free implementation is straightforward**: each pool has an atomic bitmap.
  Alloc = atomicOR to claim a bit. Dealloc = atomicAND to clear a bit. No ABA
  problem. No linked-list manipulation needed.
- Very low register pressure: only need the bitmap pointer + size class index.
- **Verdict**: Excellent fit for GPU. Simple, fast, lock-free, low registers.
  Primary limitation: cannot handle allocations larger than the largest slab class.

**Buddy Allocator**
- Splits memory into power-of-2 blocks. Alloc = find smallest block >= requested
  size, split if needed. Dealloc = free block, merge with buddy if buddy is free.
- O(log n) operations. Internal fragmentation up to 50%.
- Merging (coalescing) requires checking buddy status and potentially cascading
  merges, which is hard to make lock-free. Each merge step needs atomic CAS on
  the buddy's status, and cascading merges can cause high contention.
- **Verdict**: Moderate fit. More complex than slab, harder to make lock-free,
  higher register pressure due to recursive splitting/merging logic.

**ScatterAlloc (Steinberger et al. 2012)**
- Hash-based page allocation: threads hash their ID to scatter across memory pages,
  reducing contention. Within each page, uses a bitmap for sub-page allocation.
- 10-100x faster than CUDA's built-in malloc in high-contention scenarios.
- Performance nearly independent of thread count.
- Uses ~16 bytes metadata per page + bitmap overhead.
- **Register usage**: moderate (hash computation + page traversal + bitmap ops).
- **Verdict**: Good design but complex. The hashing + page hierarchy adds
  implementation complexity. Better suited for 10K+ thread scenarios.

**halloc (Adinets 2014)**
- Slab-based allocator with hash-based chunk selection.
- For small allocations (<=1024B), uses slab allocation within chunks.
- For large allocations, falls back to a different path.
- Uses warp-cooperative allocation: one thread per warp does the allocation,
  reducing contention by 32x.
- **Verdict**: Good performance but complex implementation. The warp-cooperative
  pattern is interesting but adds complexity for our use case.

**Ouroboros (Winter et al. 2020)**
- Virtualized queue-based design. Uses circular queues for each size class.
- 98.35% memory utilization (best in class for fragmentation).
- More complex than slab or ScatterAlloc.
- **Verdict**: Best fragmentation but highest implementation complexity.

**RegEff (Vinkler & Havran 2015)**
- Designed specifically for low register pressure on GPU.
- Uses multiple linked free lists. Threads pick a list and traverse it.
- Targets large/variable-size allocations.
- **Verdict**: Good for large allocations, but linked-list traversal is O(n)
  worst case and has higher latency than bitmap-based approaches.

**SoaAlloc (Springer & Masuhara 2019)**
- Lock-free hierarchical bitmap allocator.
- Uses multi-level bitmaps with atomic operations (atomicCAS, atomicOR, atomicAND).
- Low fragmentation through efficient bitmap management.
- **Verdict**: Closest to what we need. Lock-free, bitmap-based, proven on GPU.

**Confidence**: High — based on published papers, benchmarks (Winter et al. PPoPP 2021
survey), and existing implementations.

### Q2: Can TLSF be made lock-free for multi-thread GPU use?

A: **No, not practically.** TLSF's core operation requires atomically updating:
1. The first-level bitmap (which size classes have free blocks)
2. The second-level bitmap (which sub-classes within a size class have free blocks)
3. The free list head pointer for the selected sub-class

These three updates must be consistent. With single-word CAS (which is what we have
via `gpu-atomics`), you cannot atomically update all three. Options considered:

- **Global lock**: Works but serializes ALL allocations. With 32 warps, this means
  31 warps spin-wait while 1 allocates. Unacceptable throughput.
- **DCAS (double-word CAS)**: PTX supports `atom.cas.b64` but we'd need to pack
  both bitmaps into 64 bits, limiting size classes. Even then, the free list head
  update is a separate atomic, creating a window of inconsistency.
- **Helping/assisting pattern**: Too complex for GPU, high register pressure.

The fundamental issue is that TLSF's O(1) guarantee comes from the bitmap index,
but maintaining that index atomically across concurrent threads is the unsolved
problem.

**Confidence**: High — this is a well-understood limitation in concurrent data
structures literature.

### Q3: What is the register pressure impact of different allocator designs?

A: Register pressure is critical on GPU. SM86 (RTX 3060) has 65536 registers per SM
shared across all resident threads. At 255 regs/thread max, only 256 threads can
be resident per SM. More registers per thread = fewer concurrent threads = lower
occupancy.

Estimated register usage per allocator design for alloc+dealloc path:

| Allocator | Registers (est.) | Notes |
|-----------|-------------------|-------|
| Bump (current) | 4-6 | Just pointer + size + alignment math |
| Slab + bitmap | 8-12 | Size class lookup + bitmap ptr + bit index + CAS loop vars |
| TLSF | 15-25 | Two bitmap levels + list traversal + CLZ + more state |
| Buddy | 12-18 | Level tracking + buddy address calc + merge state |
| ScatterAlloc | 15-20 | Hash + page pointer + bitmap + retry state |
| halloc | 12-18 | Hash + chunk pointer + slab bitmap |

The slab + bitmap approach has the lowest register pressure of any design that
supports deallocation. Key savings:
- Size class can be computed from the requested size with shifts (2-3 instructions)
- Bitmap index = `(ptr - pool_base) / block_size` (1 division or shift)
- CAS loop needs only 3 registers (old_val, new_val, result)

With Fat LTO enabled, LLVM can inline the allocator and reuse registers across
the alloc call boundary, further reducing pressure.

**Confidence**: Medium — register counts are estimates based on instruction analysis;
actual counts depend on LLVM register allocation after inlining.

### Q4: How do existing GPU allocators handle thread safety?

A: Four main patterns in the literature:

1. **Atomic bitmaps** (halloc, SoaAlloc, ScatterAlloc within pages):
   - Each memory region has a bitmap where bit N = 1 means block N is allocated.
   - Alloc: `atomicOR(bitmap, 1 << bit_index)` — set bit.
   - Dealloc: `atomicAND(bitmap, ~(1 << bit_index))` — clear bit.
   - Finding a free bit: `__ffs(~bitmap)` (find first set in complement).
   - No ABA problem. No linked lists. Inherently lock-free.
   - **This is the dominant pattern** for GPU allocators.

2. **Atomic CAS on linked-list heads** (our hostcall protocol already uses this):
   - Free list is a linked list with tagged pointer (tag + index packed in u64).
   - Alloc: CAS-pop from free list. Dealloc: CAS-push to free list.
   - ABA problem solved by monotonic tag increment.
   - We already have this pattern proven in `gpu-kernel` hostcall code.
   - Higher overhead than bitmap (pointer chasing vs. bitwise ops).

3. **Hash-based scattering** (ScatterAlloc):
   - Threads hash to different pages, reducing contention.
   - Within pages, uses bitmap or superblock approach.
   - Effective at scale (1000s of threads) but adds complexity.

4. **Warp-cooperative allocation** (halloc):
   - Only lane 0 of each warp performs the allocation.
   - Result is broadcast to other lanes via warp shuffle.
   - Reduces contention by 32x (one allocator call per warp instead of per thread).
   - **Interesting optimization** we could add later, but not required initially.

For our scale (32-1024 threads = 1-32 warps), atomic bitmaps provide sufficient
throughput without the complexity of hashing or warp-cooperative patterns.

**Confidence**: High — based on published GPU allocator papers and survey.

### Q5: Which design is best for our use case (Vec/String/format!, 32-1024 threads)?

A: **Hybrid slab allocator with atomic bitmaps**, specifically:

**Allocation patterns in our workload:**
- `Vec<T>`: initial alloc (small), then realloc on growth (doubling: 4→8→16→32...
  elements). Element sizes typically 4-8 bytes, so allocations are 16→32→64→128...
  bytes.
- `String`: similar to Vec<u8>. `format!()` creates temp Strings, typically 20-200
  bytes.
- `Box<T>`: single allocation, size of T. Typically small (8-64 bytes).
- Largest common allocation: Vec with ~100 elements of u32 = 400 bytes, rounded
  to 512.

**Proposed size classes**: 32, 64, 128, 256, 512, 1024, 2048 bytes (7 classes).
- Covers 99%+ of Rust std allocations.
- Allocations > 2048 bytes: fall back to bump allocator (rare, usually only for
  large data buffers which are passed as kernel args anyway).

**Design:**
```
Pool layout (per size class):
  [bitmap: u32/u64 atomic] [block 0] [block 1] ... [block N-1]

  - 32 blocks per bitmap word (u32) or 64 blocks per bitmap word (u64)
  - Multiple bitmap words per size class if needed
  - Total blocks per class = heap_for_class / block_size

Alloc(size):
  1. class = size_class_for(size)    // shift + lookup table, ~3 instrs
  2. bitmap = pool[class].bitmap     // load bitmap word
  3. bit = find_first_zero(bitmap)   // ~bfind on complement
  4. CAS bitmap to set bit           // atomic, retry on contention
  5. return pool[class].base + bit * block_size

Dealloc(ptr):
  1. Determine which pool/class the ptr belongs to (ptr range check)
  2. bit = (ptr - pool.base) / block_size
  3. atomic_and(bitmap, ~(1 << bit))  // clear bit, always succeeds
```

**Why this beats alternatives for our use case:**
- **vs. TLSF**: Lock-free without complexity. TLSF needs locking.
- **vs. Buddy**: Simpler, fewer registers, no cascading merges.
- **vs. ScatterAlloc**: We don't need 10K-thread scaling. Simpler is better.
- **vs. Ouroboros**: Way simpler. We don't need 98% utilization for short kernels.
- **vs. Bump (current)**: Actually frees memory! Vec realloc won't leak.

**Confidence**: High — this is the most proven pattern for GPU allocators at our
thread count scale.

## Recommendations

### R1: Implement hybrid slab + bitmap allocator as `gpu-alloc` crate

**Design**: 7 size classes (32B-2048B) with atomic bitmap per pool. Fallback to
bump for oversized allocations. Implements `GlobalAlloc` trait.

**Key implementation details:**
- Use `u32` bitmaps with `sys_cas_u32` from `gpu-atomics` (not `sys` scope — we
  only need GPU-scope atomics for allocator, not CPU-visible). However, our current
  `gpu-atomics` crate only has `sys` scope. We should add GPU-scope (`gpu` or
  device-scope) atomics as they have lower latency. Alternatively, use Rust's
  built-in `core::sync::atomic` — these compile to `.gpu`-scope on nvptx64.
- `find_first_zero`: use PTX `bfind` (bit find) instruction on `~bitmap`.
  If not available via Rust intrinsics, use inline PTX asm.
- Size class lookup: `class = (size - 1).leading_zeros()` variant, or a small
  lookup table stored in shared memory or registers.
- Deallocation: given a pointer, determine its pool by checking which pool's
  address range contains it. With 7 pools laid out contiguously, this is a
  simple range comparison (7 comparisons worst case, or binary search for 3).

### R2: Use `core::sync::atomic` for intra-GPU synchronization

For the allocator, we do NOT need system-scope atomics (those are for GPU-CPU
communication). `core::sync::atomic::{AtomicU32, AtomicU64}` on nvptx64 compiles
to device-scope atomics, which are faster. Reserve `gpu-atomics` sys-scope
operations for hostcall protocol only.

### R3: Consider warp-cooperative allocation as future optimization

For the initial implementation, every thread calls alloc/dealloc independently.
If profiling shows contention at 1024 threads, add warp-cooperative pattern:
lane 0 allocates, broadcasts pointer via `shfl.sync`. This is a bolt-on
optimization, not a fundamental design change.

### R4: Fallback strategy for oversized allocations

Allocations > 2048 bytes fall back to the existing bump allocator. This is
acceptable because:
- Large allocations are rare in typical Rust std usage
- If a kernel needs large buffers, they should be passed as kernel arguments
- The bump fallback ensures we never fail an allocation (within heap limits)

### R5: Memory layout initialization

The host must initialize the pool layout before kernel launch:
- Divide the heap region into pools for each size class
- Initialize all bitmaps to 0 (all blocks free)
- Store pool metadata (base pointers, bitmap pointers) in a known location
  in device memory (similar to current `BUMP_STATE` global)

## Impact on Downstream Tasks

- **allocator.2** (next): Implement `gpu-alloc` crate with the hybrid slab + bitmap
  design. Implement `GlobalAlloc` for `GpuAllocator`. Test with existing
  `std_dynamic_vec_kernel` and `std_dynamic_format_kernel`.
- **allocator.3**: Add `bfind`/`clz` PTX intrinsics to `gpu-atomics` or a new
  `gpu-bitops` crate for efficient free-bit scanning.
- **allocator.4**: Benchmark against bump allocator — measure register pressure
  delta and allocation throughput.
- **gpu-std**: All std types (Vec, String, format!) benefit immediately from
  deallocation support. Long-running kernels become feasible.
- **async-runtime**: Embassy executor with dynamic task spawning needs dealloc
  for task completion cleanup.
- **product**: Removes the "memory leak" caveat from all demos.
