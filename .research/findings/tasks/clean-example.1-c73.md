# clean-example.1: Self-contained hello-gpu example design

**Cycle**: 73 | **Theme**: clean-example | **Kind**: design | **Status**: done

## Summary

Designed a self-contained `examples/hello-gpu/` example that demonstrates the full
async_gpu stack: GPU kernel compilation, hostcall protocol, host listener, PRINT
service, file I/O, and bulk read via sideband buffer. The existing hello-gpu example
already covers PRINT + vector_add. This design extends it to showcase the complete
hostcall repertoire (PRINT, file write, bulk read) while keeping the code minimal
and well-documented.

## Directory Structure

```
examples/hello-gpu/
├── README.md                          # Usage instructions + expected output
├── kernel/
│   ├── .cargo/
│   │   └── config.toml                # nvptx64 target + llvm-bitcode-linker
│   ├── Cargo.toml                     # cdylib, depends on gpu-runtime
│   └── src/
│       └── lib.rs                     # GPU kernels
└── host/
    ├── Cargo.toml                     # binary, depends on cudarc + gpu-protocol
    ├── build.rs                       # Auto-compiles kernel PTX
    └── src/
        └── main.rs                    # Host binary: load PTX, launch, listen
```

This structure already exists. The design below specifies the content for each file
as a complete, coherent example.

## Kernel Crate: `examples/hello-gpu/kernel/`

### Cargo.toml

```toml
[workspace]

[package]
name = "hello-gpu-kernel"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
gpu-runtime = { path = "../../../crates/gpu-runtime" }

[profile.release]
panic = "abort"
opt-level = 3
lto = false
```

### .cargo/config.toml

```toml
[build]
target = "nvptx64-nvidia-cuda"

[target.nvptx64-nvidia-cuda]
linker = "llvm-bitcode-linker"
rustflags = ["-C", "target-cpu=sm_86"]

[unstable]
build-std = ["core"]
build-std-features = ["compiler-builtins-mem"]
```

### src/lib.rs — GPU Kernels

Three kernels demonstrating increasing hostcall complexity:

```rust
//! Hello GPU — GPU kernels demonstrating the async_gpu hostcall stack.
//!
//! Three kernels of increasing complexity:
//! 1. `hello_gpu` — PRINT hostcall (send a message to host stdout)
//! 2. `file_io_demo` — OPEN + WRITE + CLOSE (create a file from GPU)
//! 3. `bulk_read_demo` — OPEN + BULK_READ + CLOSE (read a file via sideband)

#![no_std]
#![feature(abi_ptx)]
#![feature(stdarch_nvptx)]
#![feature(asm_experimental_arch)]

use core::panic::PanicInfo;
use gpu_runtime::prelude::*;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

// ================================================================
// Kernel 1: PRINT hostcall — "Hello from GPU!"
// ================================================================

/// Send a greeting to the host via the PRINT hostcall service.
///
/// Only thread 0 executes the hostcall; all other threads return immediately.
///
/// # Arguments
/// * `buf` — Device pointer to the hostcall buffer (pinned mapped memory)
/// * `result` — Device pointer to a u32 where thread 0 writes 1 (ok) or 0 (fail)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn hello_gpu(buf: *mut u8, result: *mut u32) {
    let tid = core::arch::nvptx::_thread_idx_x() as u32;
    if tid != 0 {
        return;
    }

    let msg = b"Hello from GPU via gpu-runtime!";
    let ok = gpu_hostcall_print(buf, msg.as_ptr(), msg.len() as u32);
    sys_store_release_u32(result, if ok { 1 } else { 0 });
}

// ================================================================
// Kernel 2: File I/O — write a file from GPU
// ================================================================

/// Create a file and write a message to it, entirely from GPU code.
///
/// Demonstrates the full OPEN → WRITE → CLOSE hostcall sequence.
/// The host listener dispatches these to its I/O thread, which performs
/// the actual filesystem operations.
///
/// # Arguments
/// * `buf` — Hostcall buffer pointer
/// * `result` — Success flag (1 = ok, 0 = fail)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn file_io_demo(buf: *mut u8, result: *mut u32) {
    let tid = core::arch::nvptx::_thread_idx_x() as u32;
    if tid != 0 {
        return;
    }

    // Step 1: Open file for writing (FILE_OPEN_WRITE_CREATE = 1)
    let path = b"gpu_output.txt";
    let (pkt, ok) = gpu_hostcall_request(buf, SERVICE_OPEN, |payload| {
        // Slot 0: low 32 = path_len, high 32 = flags
        let slot0: u64 = (path.len() as u64) | ((FILE_OPEN_WRITE_CREATE as u64) << 32);
        core::ptr::write_volatile(payload as *mut u64, slot0);
        // Slots 1-7: path bytes
        let dst = payload.add(8);
        let mut i = 0;
        while i < path.len() {
            core::ptr::write_volatile(dst.add(i), path[i]);
            i += 1;
        }
    });

    if pkt.is_null() || !ok {
        sys_store_release_u32(result, 0);
        return;
    }

    let fd = core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
    gpu_hostcall_release(buf, pkt);

    if fd == FILE_ERROR_SENTINEL {
        sys_store_release_u32(result, 0);
        return;
    }

    // Step 2: Write data to the file
    let data = b"Written by GPU kernel!\n";
    let (pkt, ok) = gpu_hostcall_request(buf, SERVICE_WRITE, |payload| {
        // Slot 0: fd
        core::ptr::write_volatile(payload as *mut u64, fd);
        // Slot 1: data length
        core::ptr::write_volatile(payload.add(8) as *mut u64, data.len() as u64);
        // Slots 2-7: data bytes (up to 48 bytes)
        let dst = payload.add(16);
        let mut i = 0;
        while i < data.len() {
            core::ptr::write_volatile(dst.add(i), data[i]);
            i += 1;
        }
    });

    if !pkt.is_null() {
        gpu_hostcall_release(buf, pkt);
    }

    // Step 3: Close the file
    let (pkt, _) = gpu_hostcall_request(buf, SERVICE_CLOSE, |payload| {
        core::ptr::write_volatile(payload as *mut u64, fd);
    });
    if !pkt.is_null() {
        gpu_hostcall_release(buf, pkt);
    }

    sys_store_release_u32(result, if ok { 1 } else { 0 });
}

// ================================================================
// Kernel 3: Bulk read — read a file via sideband buffer
// ================================================================

/// Read a file's contents into GPU-accessible memory via the sideband buffer.
///
/// Demonstrates OPEN → BULK_READ → CLOSE using the sideband bump allocator
/// for data larger than the 56-byte packet payload limit.
///
/// # Arguments
/// * `buf` — Hostcall buffer pointer
/// * `sideband` — Sideband buffer pointer (for bulk data transfer)
/// * `result` — Success flag (1 = ok, 0 = fail)
/// * `bytes_read` — Output: number of bytes actually read
#[no_mangle]
pub unsafe extern "ptx-kernel" fn bulk_read_demo(
    buf: *mut u8,
    sideband: *mut u8,
    result: *mut u32,
    bytes_read: *mut u32,
) {
    let tid = core::arch::nvptx::_thread_idx_x() as u32;
    if tid != 0 {
        return;
    }

    // Reset sideband allocator
    sideband_reset(sideband);

    // Step 1: Open the file we wrote in file_io_demo
    let path = b"gpu_output.txt";
    let (pkt, ok) = gpu_hostcall_request(buf, SERVICE_OPEN, |payload| {
        let slot0: u64 = (path.len() as u64) | ((FILE_OPEN_READ as u64) << 32);
        core::ptr::write_volatile(payload as *mut u64, slot0);
        let dst = payload.add(8);
        let mut i = 0;
        while i < path.len() {
            core::ptr::write_volatile(dst.add(i), path[i]);
            i += 1;
        }
    });

    if pkt.is_null() || !ok {
        sys_store_release_u32(result, 0);
        return;
    }

    let fd = core::ptr::read_volatile(pkt.add(PKT_OFF_PAYLOAD) as *const u64);
    gpu_hostcall_release(buf, pkt);

    if fd == FILE_ERROR_SENTINEL {
        sys_store_release_u32(result, 0);
        return;
    }

    // Step 2: Bulk read up to 256 bytes via sideband
    let mut read_buf = [0u8; 256];
    let n = gpu_bulk_read(buf, sideband, fd, read_buf.as_mut_ptr(), 256);

    // Step 3: Close the file
    let (pkt, _) = gpu_hostcall_request(buf, SERVICE_CLOSE, |payload| {
        core::ptr::write_volatile(payload as *mut u64, fd);
    });
    if !pkt.is_null() {
        gpu_hostcall_release(buf, pkt);
    }

    // Step 4: Print what we read back (truncate to 56 bytes for PRINT)
    if n > 0 {
        let print_len = if n > 56 { 56 } else { n };
        gpu_hostcall_print(buf, read_buf.as_ptr(), print_len as u32);
    }

    sys_store_release_u32(bytes_read, n as u32);
    sys_store_release_u32(result, if n > 0 { 1 } else { 0 });
}

// ================================================================
// Kernel 4: Pure compute — vector addition (no hostcall)
// ================================================================

/// Each thread computes `c[i] = a[i] + b[i]`.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn vector_add(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    len: u32,
) {
    let idx = core::arch::nvptx::_block_idx_x() as u32
        * core::arch::nvptx::_block_dim_x() as u32
        + core::arch::nvptx::_thread_idx_x() as u32;
    if idx < len {
        *c.add(idx as usize) = *a.add(idx as usize) + *b.add(idx as usize);
    }
}
```

**Design rationale**: Four kernels demonstrate the full API surface in order of
complexity: pure compute (vector_add), PRINT hostcall, multi-step file I/O via
`gpu_hostcall_request`, and bulk data transfer via the sideband buffer. Each kernel
is thread-0-only (simplest correct pattern) with clear step numbering.

## Host Crate: `examples/hello-gpu/host/`

### Cargo.toml

```toml
[workspace]

[package]
name = "hello-gpu-host"
version = "0.1.0"
edition = "2021"

[dependencies]
cudarc = { version = "0.12", features = ["cuda-12060"] }
gpu-protocol = { path = "../../../crates/gpu-protocol" }
```

**Note**: The host does NOT depend on `gpu-host` to keep the example self-contained
and show the raw protocol. A production example could use `gpu-host::HostcallBuffer`
directly (shown in the README as an alternative).

### build.rs

```rust
//! Build script: compiles the kernel crate to PTX and embeds it.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let kernel_dir = manifest_dir.join("..").join("kernel");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    println!("cargo:rerun-if-changed=../kernel/src/lib.rs");
    println!("cargo:rerun-if-changed=../kernel/Cargo.toml");

    // Build kernel for nvptx64 using the kernel's own .cargo/config.toml
    let status = Command::new("cargo")
        .args(["+nightly-2025-08-25", "build", "--release"])
        .current_dir(&kernel_dir)
        .env_remove("CARGO")
        .env_remove("RUSTC")
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .env_remove("CARGO_TARGET_DIR")
        .env_remove("CARGO_BUILD_TARGET")
        .status()
        .expect("Failed to run cargo for kernel. Is nightly-2025-08-25 installed?");

    if !status.success() {
        panic!("Kernel PTX compilation failed");
    }

    let ptx_src = kernel_dir
        .join("target/nvptx64-nvidia-cuda/release/hello_gpu_kernel.ptx");

    assert!(ptx_src.exists(), "PTX not found at {:?}", ptx_src);

    // Patch sm_30 → sm_86 if llvm-bitcode-linker emits the wrong target
    let ptx = std::fs::read_to_string(&ptx_src).expect("read PTX");
    let ptx = ptx
        .replace(".target sm_30", ".target sm_86")
        .replace(".version 6.0", ".version 7.1");
    std::fs::write(out_dir.join("kernel.ptx"), ptx).expect("write PTX");
}
```

### src/main.rs — Host Binary

The host binary uses the existing inline listener from the current example for
PRINT/NOP, but for the file I/O and bulk read demos it should use `gpu-host`'s
`HostcallBuffer` which already handles all services. However, since the current
`Cargo.toml` only depends on `gpu-protocol` (not `gpu-host`), the design presents
two options:

**Option A (recommended for clean example):** Add `gpu-host` as a dependency and
use `HostcallBuffer::listen()`:

```toml
[dependencies]
cudarc = { version = "0.12", features = ["cuda-12060"] }
gpu-host = { path = "../../../crates/gpu-host" }
```

```rust
//! Hello GPU — host binary demonstrating the full async_gpu stack.
//!
//! Runs four GPU kernels in sequence:
//! 1. vector_add — pure compute, no hostcall
//! 2. hello_gpu — PRINT hostcall
//! 3. file_io_demo — file OPEN + WRITE + CLOSE from GPU
//! 4. bulk_read_demo — bulk READ via sideband buffer
//!
//! Uses gpu-host's HostcallBuffer for the listener, which handles all
//! service types with I/O thread separation.

use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::driver::sys::{self, lib as cuda_lib};
use gpu_host::hostcall::HostcallBuffer;
use gpu_protocol::*;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

const KERNEL_PTX_RAW: &str = include_str!(concat!(env!("OUT_DIR"), "/kernel.ptx"));

/// Allocate pinned, device-mapped memory for a single u32.
unsafe fn alloc_mapped_u32() -> (*mut u32, sys::CUdeviceptr) {
    let cu = cuda_lib();
    let mut host: *mut std::ffi::c_void = std::ptr::null_mut();
    let flags = sys::CU_MEMHOSTALLOC_DEVICEMAP | sys::CU_MEMHOSTALLOC_PORTABLE;
    let r = cu.cuMemHostAlloc(&mut host, std::mem::size_of::<u32>(), flags);
    assert_eq!(r, sys::CUresult::CUDA_SUCCESS);
    let mut dev: sys::CUdeviceptr = 0;
    let r = cu.cuMemHostGetDevicePointer_v2(&mut dev, host, 0);
    assert_eq!(r, sys::CUresult::CUDA_SUCCESS);
    (host as *mut u32, dev)
}

fn main() {
    println!("=== Hello GPU Example ===\n");

    let dev = CudaDevice::new(0).expect("CUDA init failed");
    println!("[host] CUDA device initialized.");

    // Load PTX (auto-compiled by build.rs)
    let ptx = KERNEL_PTX_RAW.replace(".target sm_30", ".target sm_86");
    let ptx = cudarc::nvrtc::Ptx::from_src(&ptx);
    dev.load_ptx(ptx, "hello", &[
        "hello_gpu", "vector_add", "file_io_demo", "bulk_read_demo",
    ]).expect("PTX load failed");
    println!("[host] PTX module loaded.\n");

    // --- Demo 1: vector_add (no hostcall) ---
    println!("--- Demo 1: vector_add ---");
    {
        const N: usize = 64;
        let a: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..N).map(|i| (N - i) as f32).collect();
        let a_dev = dev.htod_sync_copy(&a).unwrap();
        let b_dev = dev.htod_sync_copy(&b).unwrap();
        let mut c_dev = dev.alloc_zeros::<f32>(N).unwrap();

        let f = dev.get_func("hello", "vector_add").unwrap();
        let cfg = LaunchConfig { grid_dim: (1,1,1), block_dim: (N as u32,1,1), shared_mem_bytes: 0 };
        unsafe { f.launch(cfg, (&a_dev, &b_dev, &mut c_dev, N as u32)).unwrap() };
        let result = dev.dtoh_sync_copy(&c_dev).unwrap();
        let ok = result.iter().all(|&v| (v - N as f32).abs() < 0.001);
        println!("[host] vector_add: {}\n", if ok { "PASSED" } else { "FAILED" });
    }

    // --- Demos 2-4: hostcall-based kernels ---
    // Create HostcallBuffer (handles all services: PRINT, FILE, BULK, etc.)
    let hcbuf = HostcallBuffer::new(8).expect("HostcallBuffer alloc failed");
    let (result_ptr, result_dev) = unsafe { alloc_mapped_u32() };
    let (bytes_read_ptr, bytes_read_dev) = unsafe { alloc_mapped_u32() };

    // Spawn listener thread
    let hcbuf_ref = &hcbuf;
    std::thread::scope(|scope| {
        let listener = scope.spawn(|| {
            hcbuf_ref.listen(|msg| {
                let s = std::str::from_utf8(msg).unwrap_or("<invalid utf8>");
                println!("[GPU] {}", s);
            });
        });

        // --- Demo 2: hello_gpu (PRINT) ---
        println!("--- Demo 2: hello_gpu (PRINT hostcall) ---");
        unsafe { std::ptr::write_volatile(result_ptr, 0) };
        {
            let f = dev.get_func("hello", "hello_gpu").unwrap();
            let cfg = LaunchConfig { grid_dim: (1,1,1), block_dim: (32,1,1), shared_mem_bytes: 0 };
            unsafe {
                f.launch(cfg, (hcbuf.dev_ptr as u64, result_dev as u64)).unwrap();
            }
            dev.synchronize().unwrap();
            std::thread::sleep(std::time::Duration::from_millis(50));
            let r = unsafe { std::ptr::read_volatile(result_ptr) };
            println!("[host] hello_gpu: {}\n", if r == 1 { "PASSED" } else { "FAILED" });
        }

        // --- Demo 3: file_io_demo (OPEN + WRITE + CLOSE) ---
        println!("--- Demo 3: file_io_demo (file I/O from GPU) ---");
        unsafe { std::ptr::write_volatile(result_ptr, 0) };
        {
            let f = dev.get_func("hello", "file_io_demo").unwrap();
            let cfg = LaunchConfig { grid_dim: (1,1,1), block_dim: (32,1,1), shared_mem_bytes: 0 };
            unsafe {
                f.launch(cfg, (hcbuf.dev_ptr as u64, result_dev as u64)).unwrap();
            }
            dev.synchronize().unwrap();
            std::thread::sleep(std::time::Duration::from_millis(50));
            let r = unsafe { std::ptr::read_volatile(result_ptr) };
            println!("[host] file_io_demo: {}", if r == 1 { "PASSED" } else { "FAILED" });
            // Verify file on host side
            if let Ok(content) = std::fs::read_to_string("gpu_output.txt") {
                println!("[host] Verified file content: {:?}\n", content.trim());
            }
        }

        // --- Demo 4: bulk_read_demo (OPEN + BULK_READ + CLOSE via sideband) ---
        println!("--- Demo 4: bulk_read_demo (sideband bulk read) ---");
        unsafe {
            std::ptr::write_volatile(result_ptr, 0);
            std::ptr::write_volatile(bytes_read_ptr, 0);
        }
        {
            let f = dev.get_func("hello", "bulk_read_demo").unwrap();
            let cfg = LaunchConfig { grid_dim: (1,1,1), block_dim: (32,1,1), shared_mem_bytes: 0 };
            unsafe {
                f.launch(cfg, (
                    hcbuf.dev_ptr as u64,
                    hcbuf.sideband_dev_ptr as u64,
                    result_dev as u64,
                    bytes_read_dev as u64,
                )).unwrap();
            }
            dev.synchronize().unwrap();
            std::thread::sleep(std::time::Duration::from_millis(50));
            let r = unsafe { std::ptr::read_volatile(result_ptr) };
            let n = unsafe { std::ptr::read_volatile(bytes_read_ptr) };
            println!("[host] bulk_read_demo: {} ({} bytes read)\n",
                if r == 1 { "PASSED" } else { "FAILED" }, n);
        }

        // Shutdown listener
        hcbuf_ref.signal_shutdown();
        // listener joins when scope exits
    });

    // Cleanup mapped u32s
    unsafe {
        let cu = cuda_lib();
        cu.cuMemFreeHost(result_ptr as *mut std::ffi::c_void);
        cu.cuMemFreeHost(bytes_read_ptr as *mut std::ffi::c_void);
        // HostcallBuffer cleans up its own allocations via Drop
    }

    // Clean up temp file
    let _ = std::fs::remove_file("gpu_output.txt");

    println!("=== All demos complete! ===");
}
```

**Option B (current approach):** Keep `gpu-protocol` only, use the inline listener.
This requires the host to manually implement FILE I/O dispatch (as shown in the
existing `main.rs`). Not recommended for the expanded example because it duplicates
~200 lines from `gpu-host`.

**Recommendation**: Option A. Change host `Cargo.toml` to depend on `gpu-host`
instead of `gpu-protocol` (gpu-host re-exports gpu-protocol). This cuts the host
code in half and shows users the intended production API.

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
  [HOST] FILE OPEN: "gpu_output.txt" flags=1 -> fd=1
  [HOST] FILE WRITE: fd=1 22 bytes written
  [HOST] FILE CLOSE: fd=1 closed
[host] file_io_demo: PASSED
[host] Verified file content: "Written by GPU kernel!"

--- Demo 4: bulk_read_demo (sideband bulk read) ---
  [HOST] FILE OPEN: "gpu_output.txt" flags=0 -> fd=2
  [HOST] BULK READ: fd=2 22 bytes read
  [HOST] FILE CLOSE: fd=2 closed
[GPU] Written by GPU kernel!
[host] bulk_read_demo: PASSED (22 bytes read)

=== All demos complete! ===
```

## README.md Content

```markdown
# hello-gpu — Minimal async_gpu Example

Demonstrates the full async_gpu stack: Rust GPU kernel → hostcall protocol → host
listener, all compiled and linked automatically.

## What it shows

| Demo | Kernel | Hostcall services used |
|------|--------|----------------------|
| 1. vector_add | Pure compute (a+b=c) | None |
| 2. hello_gpu | Print message | PRINT |
| 3. file_io_demo | Create + write file | OPEN, WRITE, CLOSE |
| 4. bulk_read_demo | Read file via sideband | OPEN, BULK_READ, CLOSE, PRINT |

## Prerequisites

- NVIDIA GPU (sm_86+ recommended, e.g. RTX 3090/4090)
- CUDA Toolkit 12.6+
- Rust nightly-2025-08-25 with `nvptx64-nvidia-cuda` target
- `llvm-bitcode-linker` (install via `cargo install llvm-bitcode-linker`)

## Build & Run

```sh
cd examples/hello-gpu/host
cargo run --release
```

The host's `build.rs` automatically compiles the kernel crate to PTX.
No manual steps needed.

## How it works

1. **build.rs** invokes `cargo +nightly-2025-08-25 build --release` on the kernel
   crate, which targets `nvptx64-nvidia-cuda` via its `.cargo/config.toml`.
2. The resulting PTX file is copied to `OUT_DIR` and embedded via `include_str!`.
3. At runtime, the host initializes CUDA, loads the PTX module, allocates a
   `HostcallBuffer` (pinned device-mapped memory), and spawns a listener thread.
4. Each kernel launch passes the buffer's device pointer. The GPU writes hostcall
   packets; the host listener reads and dispatches them (PRINT to stdout, FILE ops
   to the filesystem, BULK transfers via the sideband buffer).
5. After all kernels complete, the listener is shut down and resources freed.
```

## Key Design Decisions

### D1: Use `gpu-host::HostcallBuffer` instead of inline listener

The current example has a ~100-line inline listener that only handles PRINT and NOP.
The file I/O and bulk read demos require OPEN, WRITE, READ, BULK_READ, CLOSE — all
already implemented in `gpu-host::HostcallBuffer::listen_unified()` with proper I/O
thread separation. Duplicating that code defeats the purpose of having `gpu-host`.

### D2: Thread-0-only kernels

All hostcall-using kernels gate on `tid == 0`. This is the simplest correct pattern:
one warp, one active thread, no warp-level coordination needed. Multi-thread hostcall
is demonstrated in `crates/multi-warp-test/`.

### D3: Sequential kernel launches with shared listener

A single `HostcallBuffer` and listener thread serves all three hostcall kernels.
Each kernel launches, synchronizes, and verifies before the next starts. This avoids
complexity while showing that the buffer is reusable across multiple launches.

### D4: Self-verifying output

- vector_add checks `c[i] == N` for all elements
- hello_gpu checks the `result` flag set by `sys_store_release_u32`
- file_io_demo verifies the file exists on the host after kernel completion
- bulk_read_demo checks `bytes_read > 0` and re-prints the content via PRINT

### D5: build.rs PTX patching

The `llvm-bitcode-linker` on nightly-2025-08-25 emits `.target sm_30` in the PTX
header even when compiled with `-C target-cpu=sm_86`. Both build.rs and the host
`main.rs` patch this at load time. This is a known toolchain quirk documented in
the project's technical notes.

## Implementation Checklist

- [ ] Update `kernel/src/lib.rs` with `file_io_demo` and `bulk_read_demo` kernels
- [ ] Update `host/Cargo.toml` to depend on `gpu-host` instead of `gpu-protocol`
- [ ] Update `host/src/main.rs` to use `HostcallBuffer` and run all 4 demos
- [ ] Update `build.rs` to register `file_io_demo` and `bulk_read_demo` as kernel exports
- [ ] Add `examples/hello-gpu/README.md`
- [ ] Test on hardware: verify all 4 demos pass
- [ ] Clean up temp file (`gpu_output.txt`) after run
