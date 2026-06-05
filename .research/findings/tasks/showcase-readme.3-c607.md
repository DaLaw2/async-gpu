# showcase-readme.3: North Star Hero Example

## What Changed
Replaced the README hero snippet (lines 10-47) with the Project North Star vision:
`File::read -> matmul -> File::write` in a single GPU kernel.

## Before
- Kernel `unified_io_compute` showed File I/O but had a placeholder comment
  for compute: `// ... each warp processes its portion of data ...`
- No matmul — the compute section was empty
- Host side showed two separate launch calls (run + launch)

## After
- Kernel `matmul_pipeline`: reads two matrices via `std::fs::File`, runs
  cooperative multi-warp matrix multiply partitioned by row, writes result
  back to a file. 20 lines of kernel code.
- Host side: single `gpu::run("matmul_pipeline")` one-liner.
- Zero GPU concepts leak: no cuda, no thread IDs beyond `thread::current_id()`,
  no kernel config, no manual buffer management. Uses `std::thread`,
  `std::fs::File`, `std::io::Write`, `println!`, `Vec`.

## Design Decisions
- Used `thread::cooperative()` + `thread::current_id()` / `thread::available_parallelism()`
  for the matmul — these are the actual GPU runtime APIs.
- Row-partitioned matmul (each warp handles a stripe of rows) — simplest pattern
  that clearly shows cooperative parallelism.
- Helper functions `read_matrix()` and `as_bytes()` assumed but not defined —
  keeps the hero snippet focused on the pipeline, not boilerplate.
- Snippet is aspirational (uses `std::thread::cooperative` which is the
  `gpu_runtime::thread::cooperative` API) but realistic — every piece exists.

## Verification
- Prose before/after the snippet still flows naturally.
- Snippet is 20 lines of kernel code (within 15-25 target).
- Host side is a single expression (one-liner).
