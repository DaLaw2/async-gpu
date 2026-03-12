# ml-workload.6: README update with vector search demos
**Cycle**: 97 | **Theme**: ml-workload | **Kind**: design | **Status**: done

## Summary
README.md updated with vector similarity search and batch vector search demos.
Added actual test output showing GPU-autonomous vector search with full warp merge
(top-1 exact match, 6.4ms) and batch search (5 queries, 1.6ms/query amortized).
Updated "What Works" table with two new entries.

## Findings

### Q: What is the clearest way to present the vector search demo in the README?
A: Two subsections under "The Demo": single-query vector search and batch search.
Each shows actual test output with annotations. Key selling points:
1. 20-state WarpFuture self-coordinating 9 hostcalls
2. Full warp merge via shfl.sync for 100% DB coverage
3. Batch amortization: 1.6ms/query vs 6.4ms single-query
**Confidence**: high

## Impact on Downstream Tasks
- ml-workload theme is now complete (all 6 tasks done)
- README showcases the full capability stack: file I/O pipeline + ML workloads
