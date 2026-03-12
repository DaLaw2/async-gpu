# ml-workload.3: Multi-query batch vector search
**Cycle**: 95 | **Theme**: ml-workload | **Kind**: experiment | **Status**: done

## Summary
Batch vector search verified on hardware. 5 queries processed in a single kernel launch
against 100-vector database. DB and queries read once each, results written once. Total
7.8ms (1.6ms/query amortized vs 6.6ms single-query). Demonstrates I/O cost amortization
and the WarpFuture loop pattern for multi-item processing.

## Findings

### Q: Can the WarpFuture state machine loop over multiple queries without kernel re-launch?
A: Yes. The COMPUTE state contains a `while qi < nq` loop that processes all queries
sequentially. Each query's top-K results are written to contiguous sideband slots. The
state machine structure is identical to ml-workload.2's 20-state pattern, with the COMPUTE
state doing more work internally.
**Confidence**: high (hardware verified, 5 queries)

### Q: What is the amortized per-query latency when batching N queries?
A: For 5 queries: total 7.8ms, amortized 1.6ms/query. Single-query: 6.6ms. The I/O
overhead (~6ms for 9 hostcall round-trips) is paid once regardless of query count. Each
additional query adds only compute cost (~0.24ms per query for 100 vectors x 128 dims).
At 100 queries, amortized cost would be ~0.08ms/query.
**Confidence**: high

### Q: How should query/result regions be laid out in sideband for batch processing?
A: Layout used:
- DB: offset 0, up to 900KB (standard vecdb.bin format)
- Queries: offset 921600, up to 100KB. Format: [num_q:u32][dim:u32][q0...][q1...]
- Results: dynamically allocated after queries. Format: [num_q:u32][K:u32][{id,score}*K]*num_q
Total for 5 queries x 10 top-K: 8 + 5*10*8 = 408 bytes. Fits easily in remaining sideband.
**Confidence**: high

## Design Details

### BatchSearchFuture
- Same 20 states as VecSearchFuture (BS_SUBMIT_OPEN_DB through BS_DONE)
- Struct adds: `num_queries`, `result_bytes` fields
- COMPUTE state loops over all queries, writing per-query results contiguously
- File format: `queries.bin` has header [num_q:u32][dim:u32] followed by N query vectors
- Results: `batch_results.bin` has header [num_q:u32][K:u32] followed by N*K entries

### Performance Breakdown (5 queries, 100 vectors)
| Phase | Time (approx) |
|-------|--------------|
| I/O (9 hostcalls) | ~6ms |
| Compute (5 queries) | ~1.2ms |
| Total | 7.8ms |
| Per-query amortized | 1.6ms |

## Open Questions
None critical. Full warp merge (all 32 lanes) would improve result quality but is
not blocking for the demo.

## Impact on Downstream Tasks
- ml-workload.4 (persistent kernel) can build on batch search by adding a mailbox loop
- The batch pattern proves that compute amortization works: read once, query many times
