# lib-docs.1: Getting-Started Guide Structure (5-step, under 30 min)

Design task: propose the guide outline, per-step timing, key code snippets,
and prerequisite assumptions for a getting-started guide that takes a new
user from zero to running GPU kernel in under 30 minutes.

## Summary

The guide targets Scenario A users (stock nightly, core-only kernels) as the
primary path, with a callout for Scenario B (patched std) as an optional
upgrade. Five steps: prerequisites, setup, first kernel, host program, run
and iterate. Total estimated wall-clock time: 12-20 minutes for Scenario A,
25-30 minutes for Scenario B.

## Findings

### Q1: What are the prerequisite assumptions?

**Confidence: 95%**

The guide assumes:
- Linux x86_64 (primary) or WSL2. macOS is unsupported (NVIDIA GPU required).
- An NVIDIA GPU with compute capability >= 7.0 (Volta+). This is GTX 1660 or
  newer, any RTX card, or any Tesla V100+.
- NVIDIA driver already installed (`nvidia-smi` works).
- `rustup` already installed (or the user can follow the one-liner from rustup.rs).
- Basic Rust familiarity (Cargo, crate structure, `fn main()`).
- No CUDA programming experience required. The guide must NOT assume any prior
  GPU programming knowledge.

For Scenario B (std features), additionally:
- CUDA toolkit installed (`ptxas` available).
- ~500MB disk for rustc-src clone + patched-std.

### Q2: What is the 5-step structure?

**Confidence: 90%**

#### Step 1: Prerequisites & Setup (~3 min)

**Goal**: Working toolchain, verified by `setup.sh --check`.

**Content**:
- Clone the repo: `git clone https://github.com/DaLaw2/async-gpu.git && cd async-gpu`
- Run setup: `./scripts/setup.sh --quick`
  - This installs the nightly toolchain, components, and nvptx64 target.
  - It builds the core PTX kernel as a smoke test.
- Verify: `./scripts/setup.sh --check`
- The guide should show expected output for each command so users know it worked.

**Time breakdown**: git clone (30s), setup --quick (2 min), --check (30s).

**Key decision**: Use `setup.sh --quick` as the default, not `--std`. The std
path requires CUDA toolkit and takes 15 minutes for build-kernel-std.sh. The
quick path gets the user to a running kernel faster, which is the goal.

#### Step 2: Run the Hello-GPU Example (~2 min)

**Goal**: See a GPU kernel run. Build confidence before writing any code.

**Content**:
- `cargo run --manifest-path examples/hostcall/hello-gpu/host/Cargo.toml`
- Walk through the output: what each demo does (GPU println, file I/O, threading).
- Explain the two-crate model: `host/` (runs on CPU) and `kernel/` (runs on GPU).

**Time breakdown**: First build (60-90s for dependency compilation), run (5s).

**Key snippet to highlight (host side)**:
```rust
use async_gpu::gpu;

fn main() -> async_gpu::Result<()> {
    let result: Vec<u32> = gpu::run_with_output("hostcall_print_hello", 1)?;
    println!("Result: {}", if result[0] == 1 { "PASSED" } else { "FAILED" });
    Ok(())
}
```

#### Step 3: Write Your First Kernel (~5 min)

**Goal**: User writes a GPU kernel from scratch that does actual compute.

**Content**:
- Create a new example directory: `examples/getting-started/`
  - `kernel/Cargo.toml` (depends on `gpu-runtime`)
  - `kernel/src/lib.rs` (the GPU code)
- Write a minimal SAXPY kernel: `y[i] = a * x[i] + y[i]`
  - Explain `#![no_std]`, `#![feature(abi_gpu_kernel)]`, `extern "gpu-kernel"`
  - Explain thread indexing: `_block_idx_x() * _block_dim_x() + _thread_idx_x()`
  - Explain the bounds check pattern: `if idx < n { ... }`

**Key snippet (kernel side)**:
```rust
#![no_std]
#![feature(abi_gpu_kernel)]
#![feature(stdarch_nvptx)]

use core::arch::nvptx;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[no_mangle]
pub unsafe extern "gpu-kernel" fn saxpy(
    x: *const f32, y: *mut f32, a: f32, n: u32,
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

**Decision**: Use SAXPY as the first kernel, not "hello world" printing. Reason:
SAXPY is the canonical GPU compute example (like "hello world" for CUDA). It
demonstrates the thread indexing pattern, the element-wise parallelism model,
and data flow between host and device. Printing requires hostcall complexity
that obscures the fundamental pattern. The hello-gpu example already covers
printing in Step 2.

#### Step 4: Write the Host Program (~5 min)

**Goal**: User writes the host-side code that uploads data, launches the kernel,
and downloads results.

**Content**:
- Create `host/Cargo.toml` (depends on `async-gpu`)
- Create `host/build.rs` (compiles kernel to PTX — copy from vector-math example)
- Create `host/src/main.rs`
- Walk through the `gpu::custom()` builder API:
  1. `.ptx(KERNEL_PTX)` — load the compiled kernel
  2. `.threads(256)` — set threads per block
  3. `.elements(N)` — auto-compute grid dimensions
  4. `.prepare()` — initialize CUDA device
  5. `ctx.upload(&data)` — host-to-device copy
  6. `ctx.launch(args)` — launch kernel
  7. `result.download(&buf)` — device-to-host copy

**Key snippet (host side)**:
```rust
use async_gpu::gpu;

const KERNEL_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernel.ptx"));

fn main() -> async_gpu::Result<()> {
    const N: usize = 1024;
    let x: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let y: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();
    let a = 2.0f32;

    let ctx = gpu::custom("saxpy")
        .ptx(KERNEL_PTX)
        .threads(256)
        .elements(N as u32)
        .prepare()?;

    let x_dev = ctx.upload(&x)?;
    let mut y_dev = ctx.upload(&y)?;

    let result = unsafe { ctx.launch((&x_dev, &mut y_dev, a, N as u32))? };
    let y_result = result.download(&y_dev)?;

    // Verify
    for i in 0..N {
        let expected = a * x[i] + y[i];
        assert!((y_result[i] - expected).abs() < 0.001);
    }
    println!("SAXPY computed {} elements correctly!", N);
    Ok(())
}
```

**Build script**: The guide should provide the full build.rs but explain that it:
1. Compiles the kernel crate for nvptx64 target
2. Copies the resulting PTX to OUT_DIR
3. The host binary includes it at compile time via `include_str!`

This is the most complex part. The build.rs from vector-math is ~90 lines. The
guide should provide it as a "copy this file" with a brief explanation, not ask
the user to write it from scratch.

#### Step 5: Run, Verify, and Next Steps (~2 min)

**Goal**: User runs their program, sees it work, knows where to go next.

**Content**:
- `cd host && cargo run --release`
- Expected output: "SAXPY computed 1024 elements correctly!"
- Celebrate: you just ran Rust on a GPU.
- "Next steps" callout box:
  1. **More compute**: See `examples/hostcall/vector-math/` for dot product and softmax
  2. **GPU I/O**: Run `setup.sh --std` to enable `println!()`, `File::read/write`, and
     `std::thread::spawn` on GPU. See `examples/std/thread-demo/`.
  3. **One-liner API**: For simpler use cases, `gpu::run("kernel")` and
     `gpu::launch("kernel", n, threads)` skip the builder pattern entirely
  4. **API reference**: `cargo doc --open -p async-gpu`

**Time breakdown**: Build (60s if deps cached), run (< 1s), reading next steps (60s).

### Q3: What is the total estimated time per scenario?

**Confidence: 85%**

| Step | Scenario A | Scenario B |
|------|-----------|-----------|
| 1. Setup | 3 min | 18 min (+15 for --std) |
| 2. Run hello-gpu | 2 min | 2 min |
| 3. Write kernel | 5 min | 5 min |
| 4. Write host | 5 min | 5 min |
| 5. Run & next | 2 min | 2 min |
| **Total** | **~17 min** | **~32 min** |

Scenario B exceeds 30 minutes. Mitigation: the guide targets Scenario A as the
primary path. Scenario B is a "Level Up" appendix, not part of the main 5 steps.
The 15 minutes of build-kernel-std.sh is passive (the user just waits), so the
active time is still under 20 minutes.

### Q4: What are the key code snippets to include?

**Confidence: 95%**

1. **Kernel boilerplate** (Step 3): `#![no_std]`, features, panic handler, `extern "gpu-kernel"`.
   This is the minimum viable kernel. Users need to see the full file once.

2. **Thread indexing pattern** (Step 3): `block_idx * block_dim + thread_idx`. This is
   THE fundamental GPU concept. Show it, name it, explain it clearly.

3. **Host one-liner API** (Step 2): `gpu::run_with_output("kernel", n)`. Show the
   simplest possible host code first.

4. **Host builder API** (Step 4): `gpu::custom("kernel").ptx(...).threads(256).elements(N)
   .prepare()`. Show the full upload-launch-download flow.

5. **Build script** (Step 4): The full build.rs for compiling kernel to PTX. Provide
   as a copy-paste template, not something the user writes from scratch.

### Q5: What should the guide NOT cover?

**Confidence: 90%**

The guide should explicitly defer these topics to separate docs or advanced guides:
- Hostcall protocol details (packet format, service IDs)
- Patched std internals (sysroot mutation, PAL layer)
- The MIR pass / `#[warp_cooperative]` attribute
- Neural network module (`nn`)
- Async/tokio integration
- Multi-block launches and grid configuration beyond 1D
- Shared memory usage
- CUDA streams and concurrent kernels
- Performance tuning (occupancy, memory coalescing)

### Q6: What is the biggest risk to the 30-minute target?

**Confidence: 80%**

Three risks:
1. **First build time**: If the user has no Cargo cache, the first `cargo run` pulls
   all dependencies (~50 crates) and compiles them. This can take 2-3 minutes even
   on a fast machine. Mitigated by running hello-gpu first (warms the cache).

2. **Build script PTX compilation**: The build.rs that compiles kernel to PTX can take
   30-60 seconds on first run. The user should be warned.

3. **Confusing error messages**: If the nightly toolchain is not set up correctly,
   the error messages from `cargo +nightly build --target nvptx64-nvidia-cuda` are
   cryptic. The guide must include a "Troubleshooting" section with common errors.

## Unexpected Discoveries

1. **Two host API styles compete for "first impression"**: The hello-gpu example uses
   `gpu::run_with_output()` (one-liner), while vector-math uses `gpu::custom()`
   (builder). The guide should show the one-liner first (Step 2) and the builder
   second (Step 4), with a clear explanation of when to use which.

2. **The build.rs for kernel compilation is the hardest part**: The build.rs from
   vector-math is 90 lines of non-trivial code (toolchain detection, env cleaning,
   PTX patching). This is a significant friction point. The guide should provide it
   as a ready-made template and explain it as a one-time copy, not something to
   understand deeply.

3. **hello-gpu uses async-gpu but most other examples use gpu-host directly**: Only 1
   of 24 examples imports `async_gpu`. The guide should consistently use `async_gpu`
   (the facade crate) and note that examples may import `gpu_host` directly due to
   missing feature propagation in the facade.

4. **The kernel crate needs `#![feature(abi_gpu_kernel)]`**: This is a nightly-only
   feature. The guide should explicitly mention this and explain that the project
   requires a specific nightly (pinned in `rust-toolchain.toml`).

## Open Questions

1. **Should the guide include a pre-made "template" directory?** A `examples/template/`
   with kernel + host + build.rs that users can copy and modify would eliminate the
   "write from scratch" friction in Steps 3-4. lib-docs.2 should decide this.

2. **Should Step 3 use `gpu-runtime` or raw `core::arch::nvptx`?** The SAXPY kernel
   uses raw nvptx intrinsics (matching vector-math). But gpu-runtime provides
   `gpu_runtime::index::thread_idx_x()` etc. For a getting-started guide,
   raw intrinsics are more educational (shows what is actually happening), but
   gpu-runtime is more ergonomic. Recommend: raw intrinsics in the guide with a
   note about `gpu_runtime::index::*` as a convenience alternative.

3. **Should the guide live in the repo root as GETTING_STARTED.md, or in a `docs/`
   directory?** Root is more discoverable for repo visitors. `docs/` is cleaner for
   multiple guides. Recommend: `docs/getting-started.md` with a link from README.md.

## Impact on Downstream Tasks

- **lib-docs.2** (actual writing): This design provides the complete outline, code
  snippets, and timing targets. Writing should be straightforward execution of this
  plan. The build.rs template is the hardest part to get right.

- **lib-cleanup.6**: The guide depends on examples using `async-gpu` (the facade
  crate). If lib-cleanup.6 migrates examples from gpu-host to async-gpu, the guide
  code snippets are correct as-is. If not, the guide should use `async_gpu` anyway
  and note the discrepancy.

- **lib-toolchain.2**: The guide's Step 1 depends on `setup.sh` working correctly.
  This was completed and the script exists. No blockers.
