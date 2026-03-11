# product.1: Dynamic allocation stress test with runtime data
**Cycle**: 30 | **Theme**: product | **Kind**: experiment | **Status**: done

## Summary
All 5 dynamic allocation tests pass on GPU. The bump allocator (cuda.rs) correctly handles
runtime (non-constant-folded) data including Vec growth with multiple reallocations, format!
with runtime values, multiple simultaneous Vecs, and Vec::with_capacity pre-allocation.

## Findings

### Q: Does Vec::push with runtime data trigger the bump allocator?
A: **Yes.** PTX output for `std_dynamic_vec_kernel` contains 16 `atom.relaxed.global.cas.b64`
instructions operating on `GPU_HEAP_POS` — the bump allocator's CAS loop is active code.
In contrast, the original `std_hello_kernel` (constant data) had zero allocator instructions
(LLVM constant-folded everything).

**Confidence**: high (verified with PTX inspection + runtime execution)

### Q: Can Vec grow beyond its initial capacity?
A: **Yes.** A Vec growing from 0 to 100 elements (pushing one at a time) works correctly.
This requires approximately 7 reallocations (capacity: 0→4→8→16→32→64→128), each leaking
the old buffer. Sum of 1..=100 = 5050, result confirmed correct.

**Confidence**: high

### Q: How much heap does Vec growth consume?
A: For 100 u32 elements pushed one at a time:
- Active allocation: 128 × 4 = 512 bytes (final capacity × element size)
- Leaked allocations: (4+8+16+32+64) × 4 = 496 bytes
- Total heap consumed: ~1,008 bytes for 400 bytes of useful data (2.5× overhead)
- With `Vec::with_capacity(100)`: single allocation of 400 bytes, zero waste

**Confidence**: medium (calculated from Vec growth pattern, not directly measured)

### Q: What happens on OOM?
A: Not tested — the 1 MB heap is sufficient for all our test cases. The bump allocator
returns `null_mut()` on OOM, which would cause std's allocator to call `handle_alloc_error()`,
likely resulting in a panic (→ `trap; exit;` on GPU). For production use, pre-sizing
collections with `Vec::with_capacity` is recommended.

**Confidence**: low (architectural analysis, not verified)

### Q: Does format!() with runtime values correctly use the allocator?
A: **Yes.** `format!("result = {}", 12345u32)` produces a String of length 14 ("result = 12345").
The PTX contains allocator CAS instructions, confirming the String allocation is not constant-folded.

**Confidence**: high

## Unexpected Discoveries

1. **LLVM 21 emits `atom.relaxed.global.cas.b64` for `compare_exchange_weak(Relaxed, Relaxed)`.**
   This partially overturns atomics.5's finding that "atomicrmw is still broken." The CAS
   operation (`cmpxchg`) includes the `.relaxed` ordering qualifier on LLVM 21. Only
   `fetch_add` and `exchange` (true `atomicrmw` instructions) remain broken.

2. **Multiple kernel launches share the same GPU_HEAP_POS global.** The bump allocator's
   position counter persists across kernel invocations. Running multiple allocation-heavy
   kernels sequentially will accumulate heap usage. For multi-test scenarios, the host should
   consider resetting `GPU_HEAP_POS` to 0 between kernel launches (not implemented yet).

3. **Vec::with_capacity is the recommended pattern for GPU.** Since the bump allocator never
   deallocates, pre-sizing collections eliminates reallocation waste entirely.

## Open Questions
- How much total heap is consumed across all 5 tests in a single session?
- Should we add a `gpu_heap_reset()` hostcall to reset the bump allocator between kernels?
- What is the maximum practical Vec size before OOM (for 1 MB heap)?

## Impact on Downstream Tasks
- **product.2** (async pipeline): Can safely use Vec/String with runtime data
- **product.4** (showcase): Dynamic std types confirmed working — showcase can use real data
- **std-pal.1**: Allocator validation increases confidence in PAL stdout routing
