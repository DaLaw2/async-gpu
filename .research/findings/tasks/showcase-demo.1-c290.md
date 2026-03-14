# showcase-demo.1: Design parallel search kernel architecture
**Cycle**: 290 | **Theme**: showcase-demo | **Kind**: design | **Status**: done

## Summary
Designed a "GPU grep" demo where 32 lanes search different chunks of a file for a byte pattern. Uses async bulk I/O for file loading and warp parallelism for the search phase.

## Architecture

### Concept
Read a file into GPU memory via sideband bulk read, then each of 32 lanes searches 1/32 of the data for a pattern. Results aggregated via warp reduction. This demonstrates genuine GPU parallelism for a practical task.

### Kernel Design (`parallel_search`)

```
#[warp_cooperative]
pub async fn parallel_search(buf, sideband, pattern, pattern_len) -> u32 {
    // Phase 1: Open + bulk read (warp-cooperative, single I/O)
    let fd = GpuOpenFuture::new(buf, b"search_input.txt", READ).await?;
    let mut data = [0u8; MAX_SIZE];  // e.g., 4096 bytes
    let n = GpuBulkReadFuture::new(buf, sideband, fd, data, MAX_SIZE).await?;
    GpuCloseFuture::new(buf, fd).await?;

    // Phase 2: Parallel search (each lane searches its chunk)
    let lane = lane_id();
    let chunk_size = n / 32;
    let start = lane * chunk_size;
    let end = if lane == 31 { n } else { start + chunk_size + pattern_len - 1 };
    // (overlap by pattern_len-1 to catch matches at chunk boundaries)

    let mut count = 0u32;
    for i in start..end-pattern_len+1 {
        if data[i..i+pattern_len] == pattern[..pattern_len] {
            count += 1;
        }
    }

    // Phase 3: Warp reduction (sum counts from all 32 lanes)
    // Use butterfly reduction via shfl.sync
    for offset in [16, 8, 4, 2, 1] {
        count += shfl_xor_sync(0xFFFFFFFF, count, offset);
    }

    // Phase 4: Report result (lane 0 writes output)
    if lane == 0 {
        // Write result via hostcall or output buffer
        GpuOpenFuture::new(buf, b"search_result.txt", WRITE_CREATE).await?;
        // ... write count
    }

    count
}
```

### Key Design Decisions

1. **Single bulk read**: All 32 lanes participate in the warp-cooperative I/O (only lane 0 actually submits, results broadcast). This is correct because `#[warp_cooperative]` handles it.

2. **Chunk overlap**: At chunk boundaries, a match could span two chunks. Each lane extends its search range by `pattern_len - 1` bytes into the next chunk. This may double-count matches at boundaries, but for a demo it's acceptable. (A production version would have boundary dedup.)

3. **Warp reduction**: Instead of writing 32 separate results and reducing on host, use `shfl_xor_sync` for O(log N) in-warp reduction. This is textbook GPU parallelism.

4. **File size limit**: 4096 bytes per search (sideband can handle up to 1MB, but we keep it small for demo). Each lane searches ~128 bytes.

### Host Design

1. Create a large-ish text file (4KB of repeated text with known pattern occurrences)
2. Launch kernel with pattern as kernel argument
3. Read result
4. Verify against CPU `grep -c` equivalent

### Constraints
- `shfl_xor_sync` is available in `gpu-atomics` as `shfl_sync_xor_u32`
- Need to check: does the `#[warp_cooperative]` MIR pass handle the per-lane divergent search loop? Each lane processes a different range but the loop structure is the same.
- Answer: Yes — the MIR pass only inserts barriers at `.await` points. The search loop has no `.await`, so lanes can diverge freely within it. They reconverge at the next `.await`.

**Confidence**: high

## Impact
- showcase-demo.2: Can proceed with implementation. Key risk: PTX codegen for the per-lane search loop + shfl reduction.
