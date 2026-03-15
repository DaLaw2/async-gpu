# Async Pipeline

A warp-cooperative async data pipeline that performs file I/O and data transformation entirely from GPU kernel code using `#[warp_cooperative] async fn` and hostcall Futures.

## What It Demonstrates

- `#[warp_cooperative] async fn` with real file I/O (open, read, write, close)
- `block_on` executor driving async Futures on the GPU
- Packet-payload I/O (up to 48 bytes) and sideband bulk I/O (up to 1 MB)
- Warp convergence barriers inserted automatically at each `.await` point
- Host-side verification of GPU-produced output files

## How It Works

1. **Host setup** -- The host creates an input text file and launches the GPU kernel with a hostcall buffer.
2. **GPU reads file** -- Thread 0 opens and reads the input file via async hostcall Futures (`GpuOpenFuture`, `GpuReadFuture` / `GpuBulkReadFuture`).
3. **GPU transforms data** -- The kernel converts lowercase ASCII letters to uppercase (Demo 1) or swaps case entirely (Demo 2), running in SIMT lockstep between await points.
4. **GPU writes file** -- The transformed data is written to an output file via async hostcall Futures (`GpuWriteFuture` / `GpuBulkWriteFuture`), then the file is closed.
5. **Host verifies** -- The host reads the output file and compares it byte-by-byte against the expected transformation.

## Running

```bash
# Linux/macOS
bash run.sh

# Windows
run.bat
```

## Expected Output

```
=== Demo 1: Small I/O Pipeline ===
  #[warp_cooperative] async fn with packet-payload I/O

[host] Created pipeline_input.txt (30 bytes)
[host] Launching async_data_pipeline kernel...
[GPU] async pipeline done

[host] Kernel result: 0x1E (30)
[host] pipeline_output.txt: 30 bytes
[host] Verification: PASSED
[host]   Input:  "Hello from GPU async pipeline!"
[host]   Output: "HELLO FROM GPU ASYNC PIPELINE!"

=== Demo 2: Bulk I/O Pipeline ===
  #[warp_cooperative] async fn with sideband bulk I/O

[host] Created bulk_input.txt (77 bytes)
[host] Launching async_bulk_pipeline kernel...
[GPU] bulk pipeline done

[host] Kernel result: 0x4D (77)
[host] bulk_output.txt: 77 bytes
[host] Verification: PASSED
[host]   Input:  "The quick brown fox jumps over the lazy dog. Bulk sideband I/O test from GPU!"
[host]   Output: "tHE QUICK BROWN FOX JUMPS OVER THE LAZY DOG. bULK SIDEBAND i/o TEST FROM gpu!"

=== All Async Pipeline Demos Complete ===
```

## Key PTX to Inspect

- `bar.warp.sync` -- Warp convergence barriers inserted by the MIR pass at each `.await` point (expect 6-7 per async function).
- `nanosleep.u32` -- Yield instruction used by `block_on` between poll attempts.
- Hostcall packet protocol -- Look for volatile stores to the hostcall buffer setting up service IDs (`SERVICE_OPEN`, `SERVICE_READ`, `SERVICE_CLOSE`, etc.).
