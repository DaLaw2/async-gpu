# product.8: Real workload demo: parallel file grep
**Cycle**: 86 | **Theme**: product | **Kind**: experiment | **Status**: done

## Summary
Implemented a parallel file grep kernel where 4 GPU threads independently open, read (via bulk_read), and search a file for a pattern. Each thread finds all 4 matching lines (16 total matches), completed in ~2ms. Demonstrates the full GPU file I/O stack: open → bulk_read → in-GPU string search → print results, all running concurrently across threads.

## Findings

### Q: Can GPU threads read file chunks via hostcall and search in parallel?
A: **Yes.** Each GPU thread independently issues its own hostcall sequence (open → bulk_read → close) and gets a unique fd. The host's I/O thread processes these requests concurrently. The `grep_buffer` function performs byte-level substring matching entirely on the GPU with no additional hostcalls.
**Confidence**: high

### Q: What is the speedup vs single-threaded host grep?
A: Not directly comparable — the "parallelism" is concurrent hostcall submissions, but the host I/O thread serializes file reads. The value is demonstrating that GPU threads can independently perform file I/O, which enables future work where actual computation (not I/O) dominates.
**Confidence**: high

### Q: Is the end-to-end latency practical for real use?
A: ~2ms for 4 threads searching a 242-byte file (16 matches found, 16 print hostcalls). The latency is dominated by hostcall round-trips, not computation. For larger files with heavier computation per match, the GPU parallelism would be more beneficial.
**Confidence**: medium

## Implementation

### GPU kernel: `parallel_grep_kernel`
- Each thread: open → bulk_read (4KB max) → close → grep_buffer → print matches
- `grep_buffer`: O(n*m) substring search, prints `T{tid}: {line}` for each match
- Pattern passed via mapped memory (up to 32 bytes)

### Host: `run_parallel_grep_test`
- Creates test file with 8 lines (4 containing "GPU")
- Launches 4 blocks × 1 thread each
- Verifies: 16 total matches (4 per thread × 4 threads)
- Cleans up test file after

## Impact on Downstream Tasks
- product theme is now complete (all 8 tasks done)
- Demonstrates the full async GPU I/O stack end-to-end
