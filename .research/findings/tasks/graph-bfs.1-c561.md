# graph-bfs.1: CSR Graph + RMAT Generator + CPU BFS Reference

**Status**: Done
**Date**: 2026-03-17

## What was built

Created `examples/std/graph-algorithms/` with:

1. **CSR graph representation** — `CsrGraph` with `row_ptr` / `col_idx` arrays,
   constructed from edge list with self-loop removal and deduplication.

2. **RMAT synthetic graph generator** — Kronecker-style generator with standard
   probabilities (a=0.57, b=0.19, c=0.19, d=0.05). Uses xorshift64 RNG (no
   external dependencies). Generates directed edges for `2^scale` vertices.

3. **CPU BFS reference** — Level-synchronous BFS using `VecDeque`. Returns
   distance array (`u32::MAX` for unreachable vertices).

## Results (scale=17, edge_factor=16)

| Metric | Value |
|--------|-------|
| Vertices | 131,072 |
| Edges (after dedup) | 1,942,966 |
| Avg degree | 14.82 |
| Max degree | 9,957 |
| RMAT generation | 252 ms |
| CSR build | 45 ms |
| CPU BFS time | 3.6 ms |
| Reachable from v0 | 77,527 / 131,072 (59%) |
| Max BFS depth | 4 |

## Observations

- RMAT produces a highly skewed degree distribution (power-law). Max degree
  ~10K vs avg ~15 — classic hub structure.
- BFS depth of only 4 reflects the "small world" property of RMAT graphs.
- ~59% reachability from vertex 0 is expected for directed RMAT at this density;
  the graph has a giant weakly-connected component but not all vertices are
  reachable via directed paths from a single source.
- CPU BFS at 3.6 ms for 131K vertices is the baseline to beat with GPU BFS.

## Next steps

- Task graph-bfs.2: Implement GPU BFS kernel (level-synchronous, one thread per
  frontier vertex) and verify against this CPU reference.
