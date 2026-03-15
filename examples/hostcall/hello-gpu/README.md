# Hello GPU

A comprehensive introduction to the async_gpu hostcall stack, progressing from pure compute to GPU-initiated file I/O.

## What It Demonstrates

- Launching a `#![no_std]` Rust kernel on an NVIDIA GPU via PTX
- Pure GPU compute (vector addition) with no host interaction
- GPU-to-host printing via the PRINT hostcall service
- Sequential file I/O from GPU: OPEN, WRITE, CLOSE hostcall sequence
- Bulk reading a file back from disk via the sideband buffer mechanism
- Using the `gpu-host` SDK types: `GpuRuntime`, `HostcallBuffer`, `MappedBuffer`

## How It Works

1. **vector_add** -- Each GPU thread computes `c[i] = a[i] + b[i]` with bounds checking. No hostcall involved.
2. **hello_gpu** -- Thread 0 sends the string "Hello from GPU via gpu-runtime!" to host stdout through the PRINT hostcall service.
3. **file_io_demo** -- Thread 0 opens `gpu_output.txt` for writing via SERVICE_OPEN, writes "Written by GPU kernel!\n" via SERVICE_WRITE, then closes the file descriptor via SERVICE_CLOSE.
4. **bulk_read_demo** -- Thread 0 re-opens `gpu_output.txt` for reading, performs a bulk read of up to 256 bytes through the sideband buffer (for data exceeding the 56-byte packet payload limit), closes the file, and prints the contents back via PRINT.
5. The host verifies each kernel's result through a `MappedBuffer<u32>` flag, then cleans up the temporary file.

## Running

```bash
# Linux/macOS
bash run.sh

# Windows
run.bat
```

## Expected Output

```
=== Hello GPU Example ===

[host] CUDA device initialized.
[host] PTX module loaded.

--- Demo 1: vector_add ---
[host] vector_add: PASSED

--- Demo 2: hello_gpu (PRINT hostcall) ---
[GPU] Hello from GPU via gpu-runtime!
[host] hello_gpu: PASSED

--- Demo 3: file_io_demo (file I/O from GPU) ---
[host] file_io_demo: PASSED
[host] Verified file content: "Written by GPU kernel!"

--- Demo 4: bulk_read_demo (sideband bulk read) ---
[GPU] Written by GPU kernel!
[host] bulk_read_demo: PASSED (22 bytes read)

=== All demos complete! ===
```

## Key PTX to Inspect

- **Hostcall protocol**: Look for `st.volatile` / `ld.volatile` sequences that implement the GPU-host packet exchange (write payload, set ready flag, spin on completion flag).
- **Sideband buffer access**: The `bulk_read_demo` kernel uses a separate device-mapped buffer pointer for transferring data larger than the 56-byte inline payload.
- **Thread gating**: Kernels 2-4 use `mov.u32 %r, %tid.x` followed by a conditional branch to restrict hostcall execution to thread 0 only.
- **Memory ordering**: `st.release` and `ld.acquire` patterns (or `membar.sys`) ensure correct visibility between GPU and host CPU.
