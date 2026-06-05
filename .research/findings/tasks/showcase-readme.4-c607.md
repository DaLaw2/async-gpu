# showcase-readme.4: Progressive code snippets

## Result: DONE

Added "Progressive Examples" section to README.md between Quick Start and Feature Matrix.
Three snippets of increasing complexity, each 7-14 lines:

1. **Hello GPU** (14 lines) — `thread::spawn` + `join` on GPU, plus one-line host launch.
   Source: `thread_spawn_test` kernel from `thread_test.rs`, host from `hello-gpu`.
2. **Cooperative Compute** (11 lines) — `thread::cooperative()` matmul with warp-parallel row partitioning.
   Source: hero snippet pattern, verified against `warp-cooperative` example.
3. **Structured Concurrency Pipeline** (13 lines) — `block_scope`, `scope.alloc`, `block_oneshot`, `scope.spawn`.
   Source: `sc_producer_consumer` kernel from `sc_demo.rs`.

## Design Decisions
- Placed after Quick Start, before Feature Matrix: readers see hero vision, get running, then learn the capability progression before the exhaustive feature list.
- Each snippet links to the runnable example directory.
- Snippets are simplified extractions (removed volatile writes, atomic boilerplate, SendPtr wrappers) to show the conceptual API surface, not raw implementation details.
- Snippet 1 includes both kernel and host sides; snippets 2-3 show kernel-side only since they build on the same host pattern.

## Files Changed
- `README.md` — added Progressive Examples section (lines 124-180)
