# showcase-demo.2: Implement parallel file search kernel + host
**Cycle**: 290 | **Theme**: showcase-demo | **Kind**: experiment | **Status**: done

## Summary
Implemented "GPU grep" demo: 32 lanes search different chunks of a 4KB file for a byte pattern. Thread 0 handles file I/O via sideband bulk read, then all 32 threads search in parallel, then lane 0 gathers results via `shfl.sync.idx`. Exact match with CPU reference count (168 occurrences).

## Implementation

### Kernel (`parallel-search/kernel/src/lib.rs`)
- Phase 1: Thread 0 opens file, bulk reads 4096 bytes via sideband, closes
- Phase 2: `syncwarp()` + `shfl_sync_idx_u32()` broadcasts file_size to all lanes
- Phase 3: Each lane searches its 1/32 chunk (128 bytes + overlap for boundary matches)
- Phase 4: All lanes participate in `shfl.sync.idx` reduction loop (lane 0 accumulates)
- Phase 5: Thread 0 writes result count to `search_result.txt`

### Host (`parallel-search/host/src/main.rs`)
- Creates 4096-byte text file with known pattern ("GPU") occurrences
- Launches kernel with `(1,1,1)` grid, `(32,1,1)` block — FULL WARP
- Verifies GPU count matches CPU count

### Key Technical Details
- **shfl.sync.idx requires ALL lanes in mask to participate** — discovered when initial version had only lane 0 calling shfl (got 5 instead of 168)
- **File size must fit in MAX_DATA** — bulk read returns min(file_size, MAX_DATA) bytes
- PTX has 33 `shfl.sync.idx` instructions (1 broadcast + 32 unrolled reduction)
- Input: "The GPU can search files in parallel using all 32 GPU lanes. GPU power!" × 56 + padding

### Test Results
```
GPU result: 168
CPU count:  168
Verification: PASSED (exact match)
```

**Confidence**: high

## Impact
- showcase-demo theme: ALL 3 criteria met → theme completed
  - C1: Kernel searches in parallel across 32 lanes ✓
  - C2: End-to-end: create files, search, report results ✓
  - C3: Measurably uses GPU parallelism (shfl reduction, per-lane chunks) ✓
