# Parallel Search

A 32-lane GPU grep that reads a file via bulk I/O, partitions it across all warp lanes for parallel pattern matching, and reduces results with `shfl.sync`.

## What It Demonstrates

- Full-warp parallelism: all 32 threads active for compute
- Sideband bulk I/O for reading files larger than 48 bytes
- Per-lane chunk partitioning for data-parallel search
- Warp-level reduction using `shfl.sync.idx` to gather per-lane counts
- `syncwarp` for explicit warp synchronization
- Host-side CPU reference count for correctness verification

## How It Works

1. **Thread 0 reads file** -- Lane 0 opens and reads up to 4 KB from `search_input.txt` via sideband bulk I/O, then broadcasts the file size to all lanes using `shfl.sync.idx`.
2. **All 32 lanes search in parallel** -- Each lane computes its 1/32 chunk of the data (with overlap to catch boundary matches) and counts occurrences of the pattern using byte-by-byte comparison.
3. **Warp reduction** -- Lane 0 iterates over all 32 source lanes via `shfl.sync.idx`, reading each lane's local count and summing them into a total.
4. **Thread 0 writes result** -- Lane 0 converts the total count to ASCII, writes it to `search_result.txt` via sideband bulk write, and stores the count in the output buffer.
5. **Host verifies** -- The host computes a CPU reference count and compares it against the GPU result.

## Running

```bash
# Linux/macOS
bash run.sh

# Windows
run.bat
```

## Expected Output

```
=== Parallel Search Demo ===
  32-lane warp-cooperative async grep

[host] Created search_input.txt (4096 bytes)
[host] CPU count of "GPU": 168
[host] PTX loaded.

[host] Launching parallel_search kernel (32 threads)...
[GPU] parallel search done

[host] GPU result: 168
[host] CPU count: 168
[host] Verification: PASSED (exact match)
[host] search_result.txt: "168"

=== Parallel Search Demo Complete ===
```

## Key PTX to Inspect

- `shfl.sync.b32` -- Warp shuffle instructions used to broadcast `file_size` from lane 0 and to gather per-lane match counts during reduction.
- `bar.warp.sync` -- Warp barrier before the shuffle broadcast and before the reduction phase.
- `mov.u32 ..., %tid.x` -- Thread ID read used to partition work and gate I/O to lane 0.
- Full 32-thread launch -- The kernel is launched with `(32, 1, 1)` block dimensions, unlike single-thread examples.
