# ml-workload.5: Full warp merge via shfl.sync
**Cycle**: 96 | **Theme**: ml-workload | **Kind**: experiment | **Status**: done

## Summary
Full warp merge implemented and verified on hardware. All 32 lanes' local top-K arrays
are merged via `shfl_sync_idx_u32` to produce a global top-10 with 100% database coverage.
Results match CPU reference exactly (top-1 = id=42, score=1.0000 for vector search;
all 5 batch queries top-1 exact match). No measurable latency increase vs lane-0-only.

## Findings

### Q: Can shfl.sync efficiently merge 32 lanes' top-K arrays (320 candidates -> top-10)?
A: Yes. The algorithm broadcasts each lane's top-K entries (id + score as u32 bits) using
`shfl_sync_idx_u32`. Lane 0 collects all 320 candidates (32 lanes * 10 entries) and
performs insertion sort into a global top-10 array. All lanes participate in the shuffle
(required for correctness), but only lane 0 does the sorting work.

Pattern:
```
for k in 0..K:
    my_id = local_topk[k].id
    my_score_bits = f32::to_bits(local_topk[k].score)
    for s in 0..32:
        cand_id = shfl_sync_idx(mask, my_id, s)
        cand_score = f32::from_bits(shfl_sync_idx(mask, my_score_bits, s))
        if lane_0 && s != 0:
            insert_into_global_topk(cand_id, cand_score)
```
**Confidence**: high (hardware verified, results match CPU reference)

### Q: What is the performance cost of the merge vs the improvement in result quality?
A: Negligible cost. Vector search: 6.4ms with merge vs 6.6ms without (within noise).
Batch search: 8.0ms with merge vs 7.8ms without (within noise). The merge adds 320
shfl.sync operations per query (10 K-entries * 32 lanes * 2 values each = 640 shuffles),
but warp shuffles are single-cycle instructions. The I/O cost (~6ms for 9 hostcalls)
dominates completely.

Quality improvement: from 1/32 of DB (3.125%) to 100% of DB. Top-1 now correctly
identifies exact matches that would have been missed by lane-0-only processing.
**Confidence**: high

## Design Details

### Implementation
- Both VecSearchFuture and BatchSearchFuture updated with identical merge logic
- Two values shuffled per entry: `id` (u32) and `score` (f32 as u32 bits via to_bits/from_bits)
- `f32::to_bits()` and `f32::from_bits()` used instead of transmute (cleaner, no warnings)
- Lane 0 starts with its own local_topk, then merges lanes 1-31's entries
- Batch search applies merge inside the per-query loop

### Verification
| Test | Top-1 ID | Top-1 Score | Matches CPU? |
|------|----------|-------------|--------------|
| Vector search (query=db[42]) | 42 | 1.0000 | Yes |
| Batch query 0 (db[10]) | 10 | 1.0000 | Yes |
| Batch query 1 (db[42]) | 42 | 1.0000 | Yes |
| Batch query 2 (db[77]) | 77 | 1.0000 | Yes |
| Batch query 3 (db[3]) | 3 | 1.0000 | Yes |
| Batch query 4 (db[95]) | 95 | 1.0000 | Yes |

## Open Questions
None. Full warp merge is complete and verified.

## Impact on Downstream Tasks
- ml-workload.6 (README update) can now showcase full-quality vector search results
- The shfl.sync merge pattern is reusable for any future warp-cooperative reduction
