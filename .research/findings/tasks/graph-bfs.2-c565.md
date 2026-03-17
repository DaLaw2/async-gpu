# graph-bfs.2: GPU BFS (level-synchronous) + CPU reference comparison

## Summary

Implemented level-synchronous GPU BFS using cudarc + NVRTC-compiled CUDA C kernel,
with full verification against the existing CPU BFS reference.

## Implementation

### GPU BFS Kernel (`bfs_level_expand`)
- Each thread processes one vertex
- If `dist[v] == current_level`, thread explores all neighbors via CSR adjacency
- Uses `atomicCAS` to set `dist[neighbor] = level + 1` only if unvisited (`0xFFFFFFFF`)
- Uses `atomicAdd` on a global counter to track frontier size per level
- Host iterates levels 0, 1, 2, ... launching one kernel per level until frontier is empty

### Results (scale=17, 131K vertices, ~1.9M edges)

| Metric | Value |
|---|---|
| CPU BFS | 3.45 ms |
| GPU BFS | 33.27 ms (includes 4 kernel launches + sync) |
| Correctness | PASS (all 131072 vertices match) |
| CSR round-trip | PASS (row_ptr + col_idx identical after GPU upload/download) |
| BFS levels | 4 (very shallow graph due to RMAT power-law structure) |
| Reachable | 77,527 / 131,072 vertices |

### Why GPU is slower here

1. **Kernel launch overhead**: 4 level iterations, each with kernel launch + `synchronize()`
2. **Small graph**: 131K vertices is too small to saturate GPU parallelism
3. **NVRTC compile cost**: ~2 seconds first time (amortized via warmup)
4. **Memory transfer**: CSR upload + distance array download adds latency

### When GPU BFS wins

- Scale >= 20 (1M+ vertices) with high edge factor
- Graphs with more vertices per BFS level (wider frontiers)
- When kernel is already compiled and data is already on GPU
- Batched BFS from multiple sources

## Files Modified

- `examples/std/graph-algorithms/Cargo.toml` — added `cudarc` dependency
- `examples/std/graph-algorithms/src/main.rs` — added GPU BFS kernel + comparison

## Future Work

- Frontier-based (edge-parallel) BFS kernel for better load balancing
- Direction-optimizing BFS (top-down vs bottom-up switching)
- Persistent kernel approach to eliminate per-level launch overhead
- Larger graph scales (20+) to demonstrate GPU advantage
