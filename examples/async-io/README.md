# Async I/O

GPU-initiated multi-step file I/O pipelines demonstrating sequential writes and a read-transform-write workflow.

## What It Demonstrates

- Writing multiple files from a single GPU kernel in a loop
- A complete read-transform-write pipeline executed entirely from GPU code
- On-GPU data transformation (lowercase to uppercase conversion)
- Sideband bulk read for data exceeding the 56-byte hostcall payload limit
- Error handling at each hostcall step with graceful continuation

## How It Works

1. **write_pipeline** -- Thread 0 iterates over three filenames (`gpu_file_0.txt` through `gpu_file_2.txt`). For each file, it performs OPEN (write+create mode), WRITE (the file's content string), and CLOSE via hostcall requests. A success counter tracks how many files were written.
2. **transform_pipeline** -- Thread 0 opens `gpu_file_0.txt` for reading, performs a sideband bulk read of up to 256 bytes into a stack buffer, then closes the source file.
3. The kernel transforms the data on the GPU by converting each ASCII lowercase letter to uppercase (subtracting 32 from bytes in the `a`-`z` range).
4. It opens `gpu_upper.txt` for writing, writes the transformed data, and closes the output file.
5. The host verifies results by reading back each created file and printing its contents, then cleans up all temporary files.

## Running

```bash
# Linux/macOS
bash run.sh

# Windows
run.bat
```

## Expected Output

```
=== Async I/O Example ===

[host] CUDA device initialized.
[host] PTX module loaded.

--- Demo 1: write_pipeline (3 files from GPU) ---
[GPU] Write pipeline done
[host] write_pipeline: 3/3 files written
[host]   gpu_file_0.txt: "GPU wrote file 0"
[host]   gpu_file_1.txt: "GPU wrote file 1"
[host]   gpu_file_2.txt: "GPU wrote file 2"

--- Demo 2: transform_pipeline (read -> uppercase -> write) ---
[GPU] Transform pipeline done
[host] transform_pipeline: PASSED
[host]   gpu_upper.txt: "GPU WROTE FILE 0"

=== Async I/O example complete! ===
```

## Key PTX to Inspect

- **Loop structure**: The `write_pipeline` kernel contains a loop over three iterations with hostcall sequences inside, producing repeated patterns of volatile store/load pairs in PTX.
- **Sideband protocol**: In `transform_pipeline`, look for accesses to the second pointer argument (sideband buffer) used by `gpu_bulk_read` -- distinct from the primary hostcall buffer.
- **In-register transformation**: The uppercase conversion compiles to simple integer arithmetic (`sub.u32` or `add.s32` with constant 32) with conditional branches for range checking, all operating on registers.
- **Multiple file descriptors**: Both kernels manage file descriptor values returned in hostcall response packets, visible as `ld.volatile` reads from the packet payload offset.
