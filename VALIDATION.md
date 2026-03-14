# First-Run Hardware Validation Checklist

This document guides a first-time tester through validating async_gpu on real NVIDIA GPU hardware. All development to date has been compile-verified only — no actual GPU execution has been performed.

For system design details, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Prerequisites

### Hardware

- NVIDIA GPU with **SM 70+** (Volta or newer)
  - Volta: V100
  - Turing: RTX 2060/2070/2080, T4
  - Ampere: RTX 3060/3070/3080/3090, A100
  - Ada Lovelace: RTX 4060/4070/4080/4090
  - Hopper: H100

Check your GPU's compute capability:
```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

### Software

| Requirement | Install Command |
|-------------|----------------|
| CUDA 12.x driver | Install from [NVIDIA](https://developer.nvidia.com/cuda-downloads) |
| Rust nightly-2026-03-11 | `rustup toolchain install nightly-2026-03-11` |
| nvptx64 target | `rustup target add nvptx64-nvidia-cuda --toolchain nightly-2026-03-11` |
| rust-src component | `rustup component add rust-src --toolchain nightly-2026-03-11` |

Verify CUDA is accessible:
```bash
nvidia-smi
# Should show driver version and GPU name
```

### SM Target Adjustment

The build scripts hardcode `.target sm_86` (Ampere). If your GPU has a different compute capability, you must patch all `build.rs` files:

```
examples/hello-gpu/host/build.rs
examples/async-io/host/build.rs
examples/vector-math/host/build.rs
examples/parallel-search/host/build.rs
examples/tcp-echo/host/build.rs
examples/async-pipeline/host/build.rs
```

In each file, find the line:
```rust
.replace(".target sm_30", ".target sm_86")
```
And change `sm_86` to your GPU's SM version (e.g., `sm_70` for V100, `sm_80` for A100, `sm_89` for RTX 4090).

### Patched Toolchain (optional)

Only needed for `async-pipeline` and `warp-cooperative` examples. Build time: ~30 minutes, ~30 GB disk space.

```bash
# Linux
bash scripts/build-toolchain.sh

# Windows (cmd)
.\scripts\build-toolchain.bat
```

---

## Phase 1: Smoke Test — hello-gpu

This is the simplest validation. Run this first.

```bash
cargo run --manifest-path examples/hello-gpu/host/Cargo.toml
```

### Expected Output

```
=== Hello GPU Example ===

[host] CUDA device initialized.
[host] PTX module loaded.

--- Demo 1: vector_add ---
[host] vector_add: PASSED

--- Demo 2: hello_gpu (PRINT hostcall) ---
[GPU] Hello from GPU!
[host] hello_gpu: PASSED

--- Demo 3: file_io_demo (file I/O from GPU) ---
[host] file_io_demo: PASSED
[host] Verified file content: "Hello from GPU file I/O!"

--- Demo 4: bulk_read_demo (sideband bulk read) ---
[host] bulk_read_demo: PASSED (N bytes read)

=== All demos complete! ===
```

### What Each Demo Proves

| Demo | What it validates |
|------|-------------------|
| vector_add | PTX loading, kernel launch, device memory transfer — no hostcall needed |
| hello_gpu | PRINT hostcall: GPU writes to packet, host polls doorbell, message appears on stdout |
| file_io_demo | OPEN + WRITE + CLOSE services: GPU creates a file through the hostcall protocol |
| bulk_read_demo | Sideband buffer: bulk data transfer via separate mapped memory region |

### Failure Diagnosis

| Symptom | Likely Cause |
|---------|-------------|
| `Failed to run cargo for kernel compilation` | Nightly toolchain not installed or wrong version |
| `PTX file not found` | Kernel compilation failed — check for `asm_experimental_arch` feature gate errors |
| `CUDA_ERROR_NO_DEVICE` | No NVIDIA GPU detected, or CUDA driver not installed |
| `CUDA_ERROR_INVALID_PTX` | SM target mismatch — patch `build.rs` to match your GPU (see SM Target Adjustment above) |
| vector_add PASSED but hello_gpu FAILED | Kernel launch works but hostcall protocol fails — the doorbell or CAS-based packet pool may have an issue |
| hello_gpu hangs (no output, no timeout) | Host listener not polling doorbell, or GPU spin-wait not reaching the host — likely a mapped memory visibility issue |
| `[GPU PANIC]` message | GPU kernel hit a panic — the panic handler itself works, read the message for details |
| Kernel killed after ~2 seconds (Windows) | TDR timeout — see Known Risks below |

---

## Phase 2: Progressive Validation

Run these in order. Each builds on the capabilities validated by the previous one.

### Step 1: async-io (File I/O Futures)

```bash
cargo run --manifest-path examples/async-io/host/Cargo.toml
```

**Validates:** Async `Future` state machine on GPU (Init -> Waiting -> Done), multiple sequential hostcall round-trips, file create/write/read/close lifecycle.

**Check:**
- [ ] `write_pipeline`: 3/3 files written, each with correct content
- [ ] `transform_pipeline`: PASSED — reads file, uppercases on GPU, writes `gpu_upper.txt`
- [ ] No hangs — each Future completes (does not spin forever)

### Step 2: vector-math (Pure Compute)

```bash
cargo run --manifest-path examples/vector-math/host/Cargo.toml
```

**Validates:** Multi-thread GPU compute (up to 1024 threads), data transfer correctness, floating-point precision.

**Check:**
- [ ] SAXPY (1024 elements): PASSED
- [ ] Dot Product (1024 elements): PASSED, GPU and CPU values match
- [ ] Softmax (256 elements): PASSED, sum ~1.0, max error < 0.001

### Step 3: parallel-search (Warp-Level Reduction)

```bash
cargo run --manifest-path examples/parallel-search/host/Cargo.toml
```

**Validates:** All 32 warp lanes active simultaneously, bulk sideband I/O (4 KB read), `shfl.sync.idx` warp reduction, per-lane parallel work.

**Check:**
- [ ] GPU count matches CPU count (exact match or within 32 from boundary overlap)
- [ ] `search_result.txt` created with result summary
- [ ] No error codes (0xE000+)

### Step 4: tcp-echo (TCP Networking)

```bash
cargo run --manifest-path examples/tcp-echo/host/Cargo.toml
```

The example starts its own TCP echo server automatically — no manual server setup needed.

**Validates:** TCP hostcall services (connect, write, read, close), network fd namespace shared with file I/O, `FdResource` dispatch in host fd table.

**Check:**
- [ ] Echo server accepts connection
- [ ] GPU sends "Hello from GPU!" and receives echo
- [ ] Response length matches (15 bytes)

### Step 5: async-pipeline (Warp-Cooperative Async) — requires patched toolchain

```bash
cargo run --manifest-path examples/async-pipeline/host/Cargo.toml
```

**Validates:** `#[warp_cooperative]` MIR pass output, `bar.warp.sync` at `.await` points, `shfl.sync.idx` discriminant broadcast, real I/O through async Futures with warp convergence.

**Check:**
- [ ] Small I/O demo: read -> transform -> write completes
- [ ] Bulk I/O demo: sideband transfer with warp-cooperative state machine
- [ ] All 32 lanes produce consistent results

### Step 6: warp-cooperative (MIR Pass Verification) — requires patched toolchain

```bash
cargo run --manifest-path examples/warp-cooperative/host/Cargo.toml
```

**Validates:** The MIR pass itself — confirms that the generated PTX contains correct `bar.warp.sync` and `shfl.sync.idx` instructions at the right locations.

**Check:**
- [ ] `simple_add`: 32 lanes compute correctly (no `.await`, barrier only)
- [ ] `multi_await`: 2 `.await` points with discriminant broadcast
- [ ] `async_pipeline`: 6 `.await` points simulating full I/O pipeline

---

## Key Things to Observe

These are the core mechanisms that need real hardware validation:

1. **Doorbell signaling** — Does the GPU's atomic increment on the doorbell counter actually become visible to the host polling thread? This is the fundamental GPU-to-host notification mechanism.

2. **CAS-based packet pool** — Does the lock-free stack (tagged CAS with ABA prevention) work correctly under real GPU memory ordering? The free stack pop and ready stack push are single-CAS operations.

3. **Mapped memory coherence** — Can the GPU and CPU both read/write the hostcall buffer and sideband buffer with correct visibility? CUDA mapped memory should provide this, but it has never been tested on real hardware.

4. **Async Future completion** — Do the three-state Futures (Init -> Waiting -> Done) actually complete, or do they spin forever waiting for `CONTROL_READY` on the packet?

5. **Warp convergence** — Do all 32 lanes stay synchronized through `.await` points when using `#[warp_cooperative]`? Lane divergence would cause incorrect results or hangs.

---

## Known Risks and Expected Issues

### PTX Target Mismatch

The build scripts hardcode `sm_86`. A mismatch causes `CUDA_ERROR_INVALID_PTX` at runtime. Fix by patching all `build.rs` files (see SM Target Adjustment in Prerequisites).

### TDR Timeout (Windows)

Windows has a Timeout Detection and Recovery (TDR) mechanism that kills GPU operations after ~2 seconds. Long-running hostcall kernels will be terminated.

To disable TDR for testing, set this registry key and reboot:
```
HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\GraphicsDrivers
  TdrLevel = 0 (DWORD)
```

**Re-enable TDR after testing** (`TdrLevel = 3`) — leaving it disabled can cause the system to hang on GPU errors.

### PTX Post-Processing

The build pipeline applies text substitutions to the generated PTX:
- Removes `.ptr .align 1` annotations (not supported by all SM versions)
- Stubs `panic_const_async_fn_resumed` symbol
- Patches `.target sm_30` to the desired SM version

If you see PTX-related errors, inspect the generated PTX in the build output directory:
```
examples/<name>/kernel/target/nvptx64-nvidia-cuda/release/<crate_name>.ptx
```

### Spin Timeout

The GPU-side spin loop has a 10M-poll timeout (~640ms at 64ns nanosleep). If the host listener is too slow to process a request, the GPU will timeout and return an error. If you see `0xDEAD` or error sentinels, check that the host listener thread is running.

### Memory Ordering

The hostcall protocol relies on system-scope atomics for GPU-host synchronization. While these are correct per the CUDA memory model, real hardware may exhibit different timing characteristics than expected. If you see intermittent failures, it could indicate a memory ordering issue.

---

## Reporting Results

### Success Report

If all examples pass, report the following:

```
GPU: <model name> (SM <version>)
Driver: CUDA <version>
OS: <operating system>
Toolchain: nightly-2026-03-11

Results:
  hello-gpu:         PASS/FAIL (vector_add, hello_gpu, file_io_demo, bulk_read_demo)
  async-io:          PASS/FAIL (write_pipeline, transform_pipeline)
  vector-math:       PASS/FAIL (saxpy, dot_product, softmax)
  parallel-search:   PASS/FAIL (gpu_count vs cpu_count)
  tcp-echo:          PASS/FAIL (echo response length)
  async-pipeline:    PASS/FAIL / SKIPPED (requires patched toolchain)
  warp-cooperative:  PASS/FAIL / SKIPPED (requires patched toolchain)

Notes: <any unexpected output, warnings, or timing observations>
```

### Failure Report

For failures, include:

1. Which step failed and the full terminal output
2. The GPU model and SM version (`nvidia-smi --query-gpu=name,compute_cap --format=csv`)
3. Whether `build.rs` was patched for the correct SM target
4. The generated PTX file (from `examples/<name>/kernel/target/nvptx64-nvidia-cuda/release/`)
5. On Windows: whether TDR was disabled

File issues at the project repository with the `[hardware-validation]` tag.
