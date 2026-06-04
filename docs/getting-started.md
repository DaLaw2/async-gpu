# Getting Started with async-gpu

Write and run a Rust GPU kernel in under 30 minutes — no CUDA experience required.

This guide walks you through five steps: setup, running a pre-built example,
writing a SAXPY kernel from scratch, writing the host program that drives it,
and verifying the results.

**Prerequisites**: Linux x86_64 (or WSL2), an NVIDIA GPU (GTX 1660 / RTX 2060 or
newer), working `nvidia-smi`, and [rustup](https://rustup.rs/) installed.

---

## Step 1: Prerequisites & Setup (~3 min)

Clone the repository and run the one-command setup script:

```bash
git clone https://github.com/DaLaw2/async-gpu.git
cd async-gpu
./scripts/setup.sh --quick
```

`setup.sh --quick` installs the pinned nightly toolchain, the `nvptx64-nvidia-cuda`
target, required components (`rust-src`, `llvm-tools`, `llvm-bitcode-linker`), and
compiles a smoke-test PTX kernel to verify everything works.

Expected output (last few lines):

```
  ✓ Nightly toolchain installed
  ✓ nvptx64-nvidia-cuda target added
  ✓ Components installed (rust-src, llvm-tools, llvm-bitcode-linker)
  ✓ Smoke test PTX compiled successfully
```

Verify the setup:

```bash
./scripts/setup.sh --check
```

If all checks pass, your toolchain is ready.

---

## Step 2: Run the Hello-GPU Example (~2 min)

Before writing any code, run a pre-built example to see GPU output:

```bash
cargo run --manifest-path examples/hostcall/hello-gpu/host/Cargo.toml
```

The first build takes 60-90 seconds (compiling dependencies). You should see:

```
=== Hello GPU Example ===

--- Demo 1: GPU println ---
[GPU] Hello from GPU!
[host] hostcall_print_hello: PASSED

--- Demo 2: GPU file I/O ---
[host] File created and written from GPU
[host] hostcall_file_test: PASSED

--- Demo 3: GPU threading ---
[host] Thread 1 computed: 42 (expected 42)
[host] Thread 2 computed: 99 (expected 99)
[host] thread_spawn_test: PASSED
```

This example uses the **one-liner API** — each demo is a single function call:

```rust
use async_gpu::gpu;

fn main() -> async_gpu::Result<()> {
    // Launch a hostcall-enabled kernel, get one output value
    let result: Vec<u32> = gpu::run_with_output("hostcall_print_hello", 1)?;

    // Launch a pure compute kernel with 4 output slots
    let result: Vec<u32> = gpu::launch("thread_spawn_test", 4, 128)?;

    Ok(())
}
```

**Key concept**: every async-gpu project has two crates — a **kernel** crate
(compiled for `nvptx64-nvidia-cuda`, runs on GPU) and a **host** crate
(compiled for your CPU, drives the GPU).

---

## Step 3: Write Your First Kernel (~5 min)

Time to write a GPU kernel from scratch. We will implement **SAXPY** — the
canonical GPU compute operation: `y[i] = a * x[i] + y[i]`.

Create the kernel crate directory structure. From the repo root:

```bash
mkdir -p examples/getting-started/kernel/src
mkdir -p examples/getting-started/kernel/.cargo
```

### kernel/Cargo.toml

```toml
[workspace]

[package]
name = "getting-started-kernel"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]

[profile.release]
panic = "abort"
opt-level = 3
lto = "fat"
```

The kernel has no dependencies — it uses only `core` (no standard library on GPU
by default). The `cdylib` crate type tells the compiler to produce a single
output artifact (PTX in this case).

### kernel/.cargo/config.toml

```toml
[build]
target = "nvptx64-nvidia-cuda"

[target.nvptx64-nvidia-cuda]
linker = "llvm-bitcode-linker"
rustflags = ["-C", "target-cpu=sm_75", "-C", "target-feature=+ptx78"]

[unstable]
build-std = ["core"]
build-std-features = ["compiler-builtins-mem"]
```

This tells Cargo to cross-compile for NVIDIA GPUs using `-Zbuild-std=core`.

### kernel/src/lib.rs

```rust
#![no_std]
#![feature(abi_gpu_kernel)]
#![feature(stdarch_nvptx)]
#![feature(asm_experimental_arch)]

use core::arch::nvptx;

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    unsafe {
        core::arch::asm!("trap;");
    }
    loop {}
}

/// SAXPY: y[i] = a * x[i] + y[i]
///
/// Each GPU thread computes one element. Thread index is derived from
/// the block/thread hierarchy: idx = blockIdx.x * blockDim.x + threadIdx.x
#[no_mangle]
pub unsafe extern "gpu-kernel" fn saxpy(
    x: *const f32,
    y: *mut f32,
    a: f32,
    n: u32,
) {
    let idx = nvptx::_block_idx_x() as u32
        * nvptx::_block_dim_x() as u32
        + nvptx::_thread_idx_x() as u32;
    if idx < n {
        let val = *x.add(idx as usize);
        let cur = *y.add(idx as usize);
        *y.add(idx as usize) = a * val + cur;
    }
}
```

**What each part does**:

- `#![no_std]` — GPU kernels do not have access to the standard library by default.
- `#![feature(abi_gpu_kernel)]` — enables the `extern "gpu-kernel"` calling convention.
- `#![feature(stdarch_nvptx)]` — enables `core::arch::nvptx` intrinsics for thread indexing.
- `#[panic_handler]` — required for `no_std`; traps on panic.
- `#[no_mangle]` — preserves the function name so the host can find it by name.
- `extern "gpu-kernel"` — marks this as a GPU entry point (like `__global__` in CUDA C).
- **Thread indexing** — `blockIdx.x * blockDim.x + threadIdx.x` gives each thread a
  unique index. With 256 threads per block and 4 blocks, threads 0-1023 each compute
  one element.
- **Bounds check** — `if idx < n` prevents out-of-bounds access when the total thread
  count exceeds the data size.

---

## Step 4: Write the Host Program (~5 min)

The host program uploads data to the GPU, launches the kernel, and downloads results.

```bash
mkdir -p examples/getting-started/host/src
```

### host/Cargo.toml

```toml
[workspace]

[package]
name = "getting-started-host"
version = "0.1.0"
edition = "2021"

[dependencies]
async-gpu = { path = "../../../crates/async-gpu" }
cudarc = { version = "0.12", features = ["cuda-12060"] }
```

### host/build.rs

The build script compiles the kernel crate to PTX so the host can embed it
at compile time. **Copy this from the vector-math example** — it handles
toolchain detection and PTX patching automatically:

```bash
cp examples/hostcall/vector-math/host/build.rs examples/getting-started/host/build.rs
```

Then update the PTX filename to match your kernel crate name (hyphens become
underscores):

```rust
// In build.rs, change the PTX source path (~line 69):
    let ptx_src = kernel_dir
        .join("target")
        .join("nvptx64-nvidia-cuda")
        .join("release")
        .join("getting_started_kernel.ptx");  // was: vector_math_kernel.ptx
```

The build script: (1) invokes `cargo build --release` on the kernel crate
targeting `nvptx64-nvidia-cuda`, (2) finds the resulting `.ptx` file, (3)
patches the PTX target version if needed, and (4) copies it to `OUT_DIR`.

### host/src/main.rs

```rust
use async_gpu::gpu;

/// Embed the compiled PTX at compile time.
const KERNEL_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernel.ptx"));

fn main() -> async_gpu::Result<()> {
    const N: usize = 1024;

    // Input data
    let x: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let y: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();
    let a = 2.0f32;

    // 1. Create a launch context: load PTX, set 256 threads/block,
    //    auto-compute grid to cover N elements
    let ctx = gpu::custom("saxpy")
        .ptx(KERNEL_PTX)
        .threads(256)
        .elements(N as u32)
        .prepare()?;

    // 2. Upload data to GPU (host-to-device copy)
    let x_dev = ctx.upload(&x)?;
    let mut y_dev = ctx.upload(&y)?;

    // 3. Launch the kernel — arguments must match the kernel signature
    let result = unsafe { ctx.launch((&x_dev, &mut y_dev, a, N as u32))? };

    // 4. Download results (device-to-host copy)
    let y_result = result.download(&y_dev)?;

    // Verify: y[i] should equal a * x[i] + y_original[i]
    for i in 0..N {
        let expected = a * x[i] + y[i];
        assert!(
            (y_result[i] - expected).abs() < 0.001,
            "Mismatch at index {i}: got {}, expected {expected}",
            y_result[i]
        );
    }
    println!("SAXPY computed {N} elements correctly!");

    Ok(())
}
```

**The builder API flow**:

1. `gpu::custom("saxpy")` — start building a launch for the `"saxpy"` kernel.
2. `.ptx(KERNEL_PTX)` — load your compiled PTX (embedded at compile time).
3. `.threads(256)` — 256 threads per block (a common default).
4. `.elements(N as u32)` — auto-compute grid dimensions to cover N elements.
5. `.prepare()?` — initialize the CUDA device and load the kernel.
6. `ctx.upload(&x)?` — copy a host slice to GPU memory.
7. `ctx.launch(args)?` — launch the kernel and wait for completion.
8. `result.download(&y_dev)?` — copy GPU memory back to the host.

---

## Step 5: Run, Verify, and Next Steps (~2 min)

Build and run your SAXPY program:

```bash
cargo run --manifest-path examples/getting-started/host/Cargo.toml --release
```

The first build compiles the kernel to PTX (~30-60 seconds) and the host
binary. Expected output:

```
SAXPY computed 1024 elements correctly!
```

Congratulations — you just ran Rust on a GPU.

### Next Steps

- **More compute patterns**: See `examples/hostcall/vector-math/` for dot product
  and softmax kernels using the same builder API.
- **One-liner API**: For simpler cases, `gpu::run("kernel")` and
  `gpu::launch("kernel", n, threads)` skip the builder pattern entirely.
  See `examples/hostcall/hello-gpu/` for usage.
- **GPU I/O and threading**: Run `./scripts/setup.sh --std` to enable `println!()`,
  `File::read/write`, and `std::thread::spawn` on GPU.
  See `examples/std/thread-demo/`.
- **Neural networks**: The `nn` module provides PyTorch-style layers and autograd.
  See `examples/std/gpt2-inference/` and `examples/std/mnist-train/`.
- **API reference**: `cargo doc --open -p async-gpu`

---

## Troubleshooting

### `error[E0658]: the abi "gpu-kernel" is not stable`

Your nightly toolchain is not set up correctly. Run:

```bash
./scripts/setup.sh --check
```

The project requires a specific nightly pinned in `rust-toolchain.toml`. Run
`./scripts/setup.sh --quick` to install it.

### `error: could not compile ... for target nvptx64-nvidia-cuda`

Check that the nvptx64 target and `llvm-bitcode-linker` are installed:

```bash
rustup target list --toolchain nightly-2026-06-03 | grep nvptx
rustup component list --toolchain nightly-2026-06-03 | grep llvm-bitcode
```

If missing, re-run `./scripts/setup.sh --quick`.

### `PTX file not found at ...`

The kernel crate failed to compile. Check the build output above the error
for the actual compilation failure. Common causes:
- Missing `.cargo/config.toml` in the kernel directory
- Wrong crate name in `build.rs` PTX path (must match the crate name with
  hyphens replaced by underscores)

### `CUDA_ERROR_NO_DEVICE` or `CUDA_ERROR_NOT_INITIALIZED`

No GPU detected. Verify your NVIDIA driver:

```bash
nvidia-smi
```

If this fails, your driver is not installed or the GPU is not available.

### `CUDA_ERROR_INVALID_PTX`

The PTX was compiled for a different GPU architecture. Check that the
`target-cpu` in `kernel/.cargo/config.toml` matches your GPU. Use `sm_75`
for most RTX 20xx/30xx/40xx cards, or `sm_70` for Tesla V100.

---

## Level Up: Patched Standard Library

The examples above use `#![no_std]` kernels with raw nvptx intrinsics. For
a richer experience, async-gpu supports a **patched standard library** that
enables `println!()`, `File` I/O, `Vec`, `String`, `HashMap`, and
`std::thread::spawn` directly on the GPU.

### Setup (~15 min, mostly passive build time)

```bash
./scripts/setup.sh --std
```

This patches the Rust standard library's platform abstraction layer (PAL)
to route system calls through the hostcall protocol — the GPU sends
requests to the host CPU, which executes them and returns results.

### What changes in your kernel

With the patched std, your kernel can use familiar Rust:

```rust
// No more #![no_std] — use the full standard library
#![feature(abi_gpu_kernel)]

#[no_mangle]
pub unsafe extern "gpu-kernel" fn my_kernel(result: *mut u32) {
    println!("[GPU] Hello from Rust std on GPU!");

    let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let sum: f32 = data.iter().sum();
    println!("[GPU] Sum of 0..100 = {sum}");

    let handle = std::thread::spawn(|| {
        42u32
    });
    let value = handle.join().unwrap();
    *result = value;
}
```

The kernel's `.cargo/config.toml` changes to build with `std`:

```toml
[unstable]
build-std = ["std"]
build-std-features = []
```

### What works on GPU with patched std

`println!`, `format!`, `Vec`, `String`, `Box`, `HashMap`, `Mutex`,
`std::fs::File` (create/read/write), `std::io::stdin().read_line()`,
`std::thread::spawn` + `JoinHandle::join()`, `Result<T, E>` with `?`.

See `examples/std/thread-demo/` for a complete working example with
the patched standard library.
