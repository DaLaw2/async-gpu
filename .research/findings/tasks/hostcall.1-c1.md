# hostcall.1: GPU-Host Shared Memory Mechanisms
**Date**: 2026-03-11
**Cycle**: 1
**Theme**: hostcall
**Kind**: investigation
**Status**: done

## Summary

Mapped pinned memory (`cudaHostAllocMapped` / `cuMemHostAlloc` with `CU_MEMHOSTALLOC_DEVICEMAP`) is the correct foundation for the hostcall buffer. It gives GPU kernels a stable device-side pointer into host RAM without migration overhead, and allows both the CPU and GPU to access the same physical bytes directly via PCIe. Unified memory is unsuitable for synchronous GPU-CPU RPC because its demand-paging migration introduces unpredictable latency and can cause page faults mid-kernel. System-scope atomics (PTX `.sys` scope, C++ `cuda::thread_scope_system`) are required for correctness of any shared flag or stack pointer in the hostcall buffer; they are natively supported on Ampere (SM 8.6, RTX 3060) and above. The `cudarc` Rust crate does wrap `cuMemHostAlloc` via `result::malloc_host`, but it does not expose `cuMemHostGetDevicePointer` or the `CU_MEMHOSTALLOC_DEVICEMAP` flag at the safe API level — these must be called through the `sys` (raw FFI) layer.

---

## Detailed Findings

### Q1: cudaHostAlloc + cudaHostAllocMapped Semantics

#### Runtime API

`cudaHostAlloc(void** pHost, size_t size, unsigned int flags)` allocates page-locked (pinned) host memory. Pinned memory cannot be swapped to disk, which allows the GPU DMA engine to transfer to and from it reliably. The following flags are composable:

| Flag | Value | Effect |
|---|---|---|
| `cudaHostAllocDefault` | 0x00 | Identical to `cudaMallocHost`; pinned but not specially mapped. |
| `cudaHostAllocPortable` | 0x01 | The allocation is considered pinned by all CUDA contexts in the process, not just the one that created it. |
| `cudaHostAllocMapped` | 0x02 | Maps the allocation into the CUDA address space. A device-side pointer is obtained via `cudaHostGetDevicePointer()`. Both CPU and GPU can then read/write the same physical pages. |
| `cudaHostAllocWriteCombined` | 0x04 | Uses write-combining CPU cache policy: fast for CPU writes and PCIe DMA reads, but slow for CPU reads (avoids L1/L2 fill). Useful for staging CPU → GPU data, not for a bidirectional signaling buffer. |

For a hostcall buffer that requires bidirectional CPU ↔ GPU polling, use `cudaHostAllocMapped | cudaHostAllocPortable`. Do **not** use `cudaHostAllocWriteCombined` — the host listener must read the buffer frequently, and write-combining caching would hurt host-side read latency.

#### Enabling Mapped Memory

Before calling `cudaHostAlloc` with `cudaHostAllocMapped`, the device must be configured to support host mapping:

```c
// Runtime API
cudaSetDeviceFlags(cudaDeviceMapHost);
// Verify device support
int can_map;
cudaDeviceGetAttribute(&can_map, cudaDevAttrCanMapHostMemory, device_id);
```

All modern NVIDIA GPUs (including RTX 3060) support `cudaDevAttrCanMapHostMemory`. On SM 8.6 this is always 1.

#### Obtaining Device Pointer

After allocation with `cudaHostAllocMapped`:

```c
void* host_ptr;
cudaHostAlloc(&host_ptr, size, cudaHostAllocMapped | cudaHostAllocPortable);

void* device_ptr;
cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
// Pass device_ptr to kernel; kernel dereferences it to reach host RAM
```

On systems with Unified Virtual Addressing (UVA), which is enabled by default on 64-bit Linux/Windows with compute capability >= 2.0, `host_ptr == device_ptr`. The function `cudaHostGetDevicePointer` is still required for correctness portability, but on UVA-enabled systems it simply returns the same value.

#### Driver API Equivalents

The driver API (`cudarc` uses the driver API, not the runtime API):

| Runtime API | Driver API |
|---|---|
| `cudaHostAlloc` with `cudaHostAllocMapped` | `cuMemHostAlloc` with `CU_MEMHOSTALLOC_DEVICEMAP` |
| `cudaHostGetDevicePointer` | `cuMemHostGetDevicePointer` |
| `cudaSetDeviceFlags(cudaDeviceMapHost)` | `cuCtxCreate` with `CU_CTX_MAP_HOST` |

`CU_MEMHOSTALLOC_DEVICEMAP = 0x02` and `CU_MEMHOSTALLOC_PORTABLE = 0x01` match the runtime values. The context must be created with `CU_CTX_MAP_HOST` before any mapped allocation can succeed.

#### Zero-Copy / Mapped Memory Access Model

"Zero-copy" memory is synonymous with `cudaHostAllocMapped` allocations. When a GPU kernel dereferences the device pointer, each access crosses the PCIe bus to reach host DRAM. Consequences:

- **Bandwidth**: PCIe Gen4 x16 ≈ 32 GB/s bidirectional, versus GPU DRAM ≈ 400–900 GB/s. Bulk data access patterns are severely bandwidth-limited.
- **Latency**: Each PCIe round trip adds hundreds of nanoseconds — orders of magnitude slower than on-chip SRAM or even device DRAM.
- **No GPU caching**: The GPU does not L2-cache mapped host memory (unless the device explicitly supports it). Each access goes through PCIe.
- **CPU caching**: The CPU accesses this memory through its normal cache hierarchy (write-back), which means the GPU write to the mapped buffer may sit in the GPU write buffer before becoming visible to the CPU — this is why `.sys`-scoped atomics with `fence.sc.sys` are required (see Q3).

For a hostcall buffer, these properties are acceptable because:
1. Access is infrequent (one RPC at a time per warp, not continuous bulk reads).
2. The critical path is signaling (a few atomic 64-bit words), not bulk data movement.
3. The packet payload (8 × u64 per work-item × 32 work-items = 2 KB per packet) is read once per RPC, not repeatedly.

---

### Q2: Unified Memory vs Mapped Pinned Memory

#### Unified Memory Overview

`cudaMallocManaged` (runtime) / `cuMemAllocManaged` (driver) allocates **managed memory**: a single virtual address accessible by both host and device. The CUDA runtime (with hardware support on Pascal+) migrates pages on demand:

- On CPU access: pages in CPU DRAM; GPU accesses trigger page faults and migration to GPU memory.
- On GPU kernel launch: pages start where last used; first GPU access may fault and pull pages from CPU.
- Pages can be prefetched with `cudaMemPrefetchAsync` to avoid runtime faults.

On Pascal and newer GPUs (SM 6.x+), page fault hardware is present. On pre-Pascal, all managed pages are migrated to GPU before kernel launch. RTX 3060 (SM 8.6, Ampere) has full hardware page fault and migration support.

#### Comparison for GPU-CPU RPC

| Criterion | Mapped Pinned Memory | Unified Memory |
|---|---|---|
| **Latency for signaling** | Deterministic PCIe access, no migration | Unpredictable: first access can cause a page fault |
| **Migration overhead** | None (memory stays in host RAM) | Pages migrate CPU↔GPU; kernel execution may pause on fault |
| **CPU visibility** | CPU always reads from host DRAM (no migration needed) | Page may be on GPU; CPU access causes migration back |
| **Concurrent CPU+GPU access** | Fully supported (with system-scope atomics) | Supported on Pascal+, but requires care to avoid race during migration |
| **Memory location** | Permanently in host DRAM | Dynamically decided by runtime (can move to GPU DRAM) |
| **Programming complexity** | Slightly higher (need `cudaHostGetDevicePointer`) | Simpler (same pointer for host and device) |
| **Suitability for hostcall buffer** | Excellent | Poor |

**Why unified memory is wrong for the hostcall buffer:**

1. The hostcall buffer contains control flags (`ready_stack_`, `free_stack_`, `control_`) that the host listener thread reads in a tight polling loop. With unified memory, if the GPU last wrote these flags, the pages containing them may have migrated to GPU DRAM. Every time the host reads them, it triggers a page fault and migration back to CPU — this adds milliseconds of latency per poll cycle.

2. The doorbell value and stack pointers are written by the GPU and read by the CPU. With unified memory, these accesses can cause ping-pong migration if both sides access them regularly.

3. Mapped pinned memory guarantees the data stays in host DRAM permanently. The CPU access is always local (host DRAM bandwidth). The GPU access is always a PCIe read/write. Costs are predictable.

**When unified memory is appropriate**: bulk computation results, arrays that are written once by GPU and read once by CPU (or vice versa), and memory-oversubscription scenarios where the working set exceeds GPU VRAM. It is an excellent general-purpose tool but the wrong choice for a low-latency signaling buffer.

---

### Q3: System-Scope Atomics from Rust on GPU

#### Why System-Scope Atomics Are Required

The GPU and CPU are separate processors with distinct cache hierarchies. When a GPU thread writes to mapped pinned memory, the write may remain in the GPU's L2 cache or write buffer and not be immediately visible to the CPU (which has its own caches). Conversely, when the CPU writes to pinned memory, the GPU may read a stale cached value.

Standard CUDA `atomicAdd`, `atomicCAS`, etc. operate at **device scope** by default — they guarantee visibility across GPU threads on the same device but not across GPU ↔ CPU. To make writes visible across the PCIe bus to the CPU, **system-scope (`_system`) atomics must be used**.

#### C++ Interface: `atomicXxx_system` Functions

CUDA provides `_system` suffixed variants for key atomic operations:

```c
// Arithmetic atomics (system scope)
int atomicAdd_system(int* address, int val);
unsigned atomicAdd_system(unsigned* address, unsigned val);
unsigned long long atomicAdd_system(unsigned long long* address, unsigned long long val);
unsigned long long atomicCAS_system(unsigned long long* address,
                                    unsigned long long compare,
                                    unsigned long long val);
int atomicExch_system(int* address, int val);
unsigned long long atomicExch_system(unsigned long long* address, unsigned long long val);
int atomicMin_system(int* address, int val);
int atomicMax_system(int* address, int val);
int atomicAnd_system(int* address, int val);
int atomicOr_system(int* address, int val);
int atomicXor_system(int* address, int val);
```

These functions compile to PTX `atom.sys` instructions, which include a full memory fence visible to all processors sharing the memory.

**Compute capability requirement**: System-scope atomics (`_system` suffix) require **compute capability 6.0** (Pascal) or higher for `atomicAdd_system` on 64-bit integers, and **compute capability 7.0** or higher for full support. RTX 3060 is SM 8.6 (Ampere), well above both thresholds.

#### PTX Level: `.sys` Scope

At the PTX assembly level, system-scope atomics use the `.sys` scope qualifier:

```ptx
// System-scope compare-and-swap on 64-bit value
atom.sys.global.cas.b64 %rd1, [%rd2], %rd3, %rd4;

// System-scope add
atom.sys.global.add.u64 %rd1, [%rd2], %rd3;

// System-scope exchange
atom.sys.global.exch.b64 %rd1, [%rd2], %rd3;
```

The `.sys` scope means the atomic operation (and implied memory fence) is visible to **all CUDA threads on all GPUs and all CPU threads in the system** that have access to the same memory. This is the scope needed for GPU → CPU and CPU → GPU signaling.

Memory fence function:

```c
// Available in CUDA device code
__threadfence_system();  // Full system-scope fence: ensures all prior
                          // stores/atomics are visible to all processors
```

PTX equivalent:
```ptx
membar.sys;  // System memory barrier
```

#### libcu++ C++ Atomics with System Scope

The `libcu++` library (bundled with CUDA Toolkit) provides standard C++ `<cuda/atomic>` with explicit scopes:

```cpp
#include <cuda/atomic>
#include <cuda/std/atomic>

// System-scope atomic on pinned memory, accessible from both GPU and CPU
__device__ void gpu_push_ready(cuda::atomic<uint64_t, cuda::thread_scope_system>* ready_stack,
                                uint64_t new_head) {
    uint64_t old_head = ready_stack->load(cuda::memory_order_relaxed);
    while (!ready_stack->compare_exchange_weak(old_head, new_head,
                                               cuda::memory_order_release,
                                               cuda::memory_order_relaxed)) {}
}

// Host side (CPU) with system scope
void host_poll(cuda::atomic<uint64_t, cuda::thread_scope_system>* ready_stack) {
    uint64_t head = ready_stack->load(cuda::memory_order_acquire);
    // ...
}
```

This is the cleanest API: `cuda::thread_scope_system` + `cuda::memory_order_release/acquire` generates the correct `.sys` PTX atomics.

#### Rust on GPU: PTX Inline Assembly

Rust GPU (rust-gpu, `spirv-std`) compiles to SPIR-V, not PTX — system-scope CUDA atomics are not directly expressible in SPIR-V. For CUDA target (`nvptx64-nvidia-cuda`), Rust can emit PTX via `llvm-ptx` or inline PTX assembly using the `asm!` macro.

In a CUDA-targeted Rust kernel (compiled with `rustc --target nvptx64-nvidia-cuda`), system-scope atomics can be expressed as inline PTX:

```rust
// In a #[no_std] Rust kernel compiled for nvptx64
pub unsafe fn atomic_cas_system(addr: *mut u64, expected: u64, desired: u64) -> u64 {
    let result: u64;
    core::arch::asm!(
        "atom.sys.global.cas.b64 {0}, [{1}], {2}, {3};",
        out(reg64) result,
        in(reg64) addr,
        in(reg64) expected,
        in(reg64) desired,
    );
    result
}

pub unsafe fn fence_system() {
    core::arch::asm!("membar.sys;");
}
```

This compiles correctly for the `nvptx64-nvidia-cuda` target. The key requirement is that the address must point into mapped pinned memory (allocated with `CU_MEMHOSTALLOC_DEVICEMAP`) so the `.sys`-scoped access physically reaches host DRAM.

**Important limitation**: Standard Rust `core::sync::atomic::AtomicU64` does NOT generate `.sys`-scoped PTX atomics when compiled for `nvptx64`. The compiler emits `.gpu`-scoped (or `.cta`-scoped) atomics, which are not visible to the host CPU. Inline PTX assembly is required for system-scope atomics in Rust GPU code.

#### SM 8.6 Specifics

RTX 3060 (Ampere SM 8.6) fully supports:
- `atom.sys.global.cas.b64` — 64-bit CAS at system scope
- `atom.sys.global.add.u64` — 64-bit add at system scope
- `atom.sys.global.exch.b64` — 64-bit exchange at system scope
- `membar.sys` — full system memory barrier
- Concurrent CPU + GPU access to mapped pinned memory with cache coherency enforced through PCIe snoop traffic

The Ampere architecture improved the efficiency of system-scope operations compared to Volta/Turing by reducing the number of redundant invalidation messages in multi-GPU configurations. For a single-GPU + CPU configuration (our target), the semantics are the same as on Pascal+.

---

### Q4: cudarc Mapped Memory API

#### What cudarc Wraps

`cudarc` provides three abstraction layers over the CUDA Driver API:
- `sys`: raw auto-generated FFI bindings (`libcuda.so` symbols)
- `result`: thin wrappers returning `Result<T, DriverError>` without safety abstractions
- `safe`: high-level safe Rust types (`CudaContext`, `CudaStream`, `CudaSlice<T>`)

Examining the source at `src/driver/result.rs`:

```rust
pub unsafe fn malloc_host(num_bytes: usize, flags: c_uint)
    -> Result<*mut c_void, DriverError>
{
    let mut host_ptr = MaybeUninit::uninit();
    sys::cuMemHostAlloc(host_ptr.as_mut_ptr(), num_bytes, flags).result()?;
    Ok(host_ptr.assume_init())
}

pub unsafe fn free_host(host_ptr: *mut c_void) -> Result<(), DriverError> {
    sys::cuMemFreeHost(host_ptr).result()
}
```

The `flags` parameter accepts raw `c_uint`, so `CU_MEMHOSTALLOC_DEVICEMAP (0x02)` can be passed:

```rust
use cudarc::driver::sys::{CU_MEMHOSTALLOC_DEVICEMAP, CU_MEMHOSTALLOC_PORTABLE};

let host_ptr = unsafe {
    cudarc::driver::result::malloc_host(
        buffer_size,
        CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_PORTABLE,
    )?
};
```

#### What cudarc Does NOT Wrap

1. **`cuMemHostGetDevicePointer`**: Not wrapped at any level in `result.rs`. Must be called via `sys::cuMemHostGetDevicePointer` directly:

```rust
use cudarc::driver::sys;

let mut device_ptr: sys::CUdeviceptr = 0;
unsafe {
    sys::cuMemHostGetDevicePointer_v2(&mut device_ptr, host_ptr, 0)
        .result()
        .map_err(|e| /* ... */)?;
}
// device_ptr is now a CUdeviceptr (u64) pointing to the mapped host memory
// Pass it to kernels as a pointer argument
```

2. **`CU_CTX_MAP_HOST` for context creation**: The `safe` API creates contexts via `CudaContext::new(device_ordinal)` without exposing `CU_CTX_MAP_HOST`. To enable mapped memory, either:
   - Use `sys::cuCtxCreate_v2` directly with `CU_CTX_MAP_HOST` before calling any cudarc safe API, OR
   - In practice on modern CUDA (>= 4.0), `cuCtxCreate` without `CU_CTX_MAP_HOST` may still allow `cuMemHostAlloc` with `CU_MEMHOSTALLOC_DEVICEMAP` if UVA is enabled (behavior is driver-version-dependent and not guaranteed by the spec).

3. **`PinnedHostSlice<T>` in safe layer**: The safe API does expose `alloc_pinned<T>()` returning a `PinnedHostSlice<T>`, but it uses `CU_MEMHOSTALLOC_WRITECOMBINED` (optimized for host→device writes), which is wrong for a bidirectional signaling buffer, and it does not set `CU_MEMHOSTALLOC_DEVICEMAP`.

#### Practical Integration Strategy

For the hostcall buffer in our Rust implementation:

```rust
use cudarc::driver::{result, sys, CudaContext};

struct HostcallBuffer {
    host_ptr: *mut u8,
    device_ptr: u64,  // CUdeviceptr
    size: usize,
}

impl HostcallBuffer {
    pub fn allocate(size: usize) -> Result<Self, Box<dyn std::error::Error>> {
        // 1. Allocate mapped pinned host memory via cudarc result layer
        let host_ptr = unsafe {
            result::malloc_host(
                size,
                sys::CU_MEMHOSTALLOC_DEVICEMAP | sys::CU_MEMHOSTALLOC_PORTABLE,
            )? as *mut u8
        };

        // 2. Get device-side pointer via sys layer (not exposed in result)
        let mut device_ptr: sys::CUdeviceptr = 0;
        unsafe {
            let err = sys::cuMemHostGetDevicePointer_v2(
                &mut device_ptr,
                host_ptr as *mut _,
                0,
            );
            if err != sys::CUresult::CUDA_SUCCESS {
                result::free_host(host_ptr as *mut _).ok();
                return Err("cuMemHostGetDevicePointer_v2 failed".into());
            }
        }

        Ok(Self { host_ptr, device_ptr, size })
    }
}
```

The `sys` module is part of the public API of `cudarc` (exposed as `cudarc::driver::sys`), so using raw FFI for `cuMemHostGetDevicePointer_v2` is within the library's intended use model, not a hack.

---

## Unexpected Discoveries

1. **UVA makes `cudaHostGetDevicePointer` a no-op on 64-bit systems**: On modern 64-bit CUDA setups with UVA enabled, `host_ptr == device_ptr` for mapped allocations. The call is still required by the spec for portability, but in practice the RTX 3060 + 64-bit Linux/Windows environment returns identical values. This simplifies pointer management.

2. **`cudarc`'s `PinnedHostSlice` uses `WRITECOMBINED`, not `DEVICEMAP`**: The safe API's "pinned" type is designed for staging host-to-device DMA, not for GPU polling. Using it for the hostcall buffer would be silently incorrect — the GPU could not access the memory by pointer.

3. **`CU_CTX_MAP_HOST` vs modern CUDA**: The CUDA documentation says `CU_CTX_MAP_HOST` is required, but in practice with CUDA 11+ and UVA, many applications skip this flag and `cuMemHostAlloc` with `CU_MEMHOSTALLOC_DEVICEMAP` still works. This is an undocumented behavior difference that should be handled defensively (explicitly check for errors and document the requirement).

4. **Zero-copy guidance is for bulk access; signaling is fine**: NVIDIA documentation consistently warns that zero-copy is bad for bulk data access on discrete GPUs. However, this warning is for loops that read/write large arrays — not for atomic signaling of a few 8-byte words. The latency for a single PCIe atomic is ~1–2 µs, acceptable for an RPC system (not for compute kernels).

5. **Write-combining memory is specifically bad for the doorbell**: `CU_MEMHOSTALLOC_WRITECOMBINED` uses a non-temporal store path that bypasses CPU caches — while this speeds up CPU writes, it slows CPU reads. The doorbell is primarily read by the host listener (CPU), so this flag would hurt performance in exactly the critical path.

---

## Key Conclusions

1. **Use `cuMemHostAlloc` with `CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_PORTABLE`** for the hostcall buffer. This gives both CPU and GPU stable, coherent access to the same physical bytes in host DRAM.

2. **Do NOT use unified memory** for the hostcall buffer. Migration-on-fault semantics make polling latency non-deterministic — exactly the wrong property for a signaling buffer.

3. **System-scope PTX atomics (`.sys`) are mandatory** for the `ready_stack_`, `free_stack_`, and `control_` fields. Standard Rust `AtomicU64` targeting `nvptx64` generates device-scope atomics and is insufficient. Inline PTX assembly using `atom.sys.global.cas.b64` / `atom.sys.global.exch.b64` and `membar.sys` is required.

4. **`cudarc` partially supports mapped memory**: `result::malloc_host` wraps `cuMemHostAlloc` and accepts raw flag integers, so `CU_MEMHOSTALLOC_DEVICEMAP` can be passed. However, `cuMemHostGetDevicePointer` is not wrapped and must be called through `cudarc::driver::sys` directly.

5. **RTX 3060 (SM 8.6) fully supports** all required mechanisms: `CAN_MAP_HOST_MEMORY`, system-scope 64-bit atomics, UVA, and concurrent CPU+GPU access to mapped pinned memory.

---

## Open Questions

1. **Does `cudarc`'s `CudaContext::new()` silently enable `CU_CTX_MAP_HOST`?** Testing is needed to confirm whether the default cudarc context allows `cuMemHostAlloc` with `CU_MEMHOSTALLOC_DEVICEMAP` without explicitly passing `CU_CTX_MAP_HOST`. If not, we need to create the context via `sys::cuCtxCreate_v2` before any cudarc initialization.

2. **Will Rust's inline PTX `asm!` macro on `nvptx64` compile correctly with register constraints for 64-bit addresses?** The `nvptx64` target's register constraints are less documented than x86_64. Verification in a small test kernel is needed before the hostcall buffer code is written.

3. **What is the actual end-to-end latency for a `.sys`-scoped `atom.sys.global.exch.b64` on RTX 3060 (PCIe 4.0 x16)?** Benchmarking a round-trip (GPU writes doorbell → CPU detects → CPU writes response flag → GPU detects) will establish the baseline RPC latency for the hostcall protocol.

4. **Are there alignment requirements for `atom.sys.global.cas.b64`?** PTX requires natural alignment (8-byte alignment for 64-bit atomics). The hostcall buffer struct must be aligned accordingly. Needs validation.

5. **Does `libcudacxx` `cuda::atomic<T, cuda::thread_scope_system>` work in a no_std Rust-invoked PTX kernel?** If we emit C++ via bindgen or cudarc's NVRTC support, using `libcudacxx` would be cleaner than raw PTX inline assembly. But if the kernel code is pure Rust→PTX, inline asm is the only path.

---

## Impact on Downstream Tasks

- **hostcall.2** (existing implementations): Confirmed that pinned mapped memory is the right buffer location, consistent with what the ROCm analysis recommended. The `cuMemHostAlloc` + `cuMemHostGetDevicePointer` pattern directly maps to what our protocol needs.

- **hostcall.3** (design and implement protocol): Buffer allocation is now specified. The design must allocate via `cuMemHostAlloc(CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_PORTABLE)`, retrieve `device_ptr` via `cuMemHostGetDevicePointer_v2`, and pass `device_ptr` as a kernel argument. The `ready_stack_` and `free_stack_` atomic operations must use inline PTX `.sys`-scope atomics.

- **atomics.1** (PTX scope verification): This investigation confirms the exact PTX instruction forms needed (`atom.sys.global.cas.b64`, `atom.sys.global.exch.b64`, `membar.sys`) and the Rust inline asm pattern to express them. The `atomics.1` experiment should verify these generate correct PTX and produce visible cross-processor effects.

- **toolchain.1** (CUDA toolchain setup): The `cuMemHostGetDevicePointer_v2` symbol must be present in `libcuda.so`. This is a standard CUDA driver function (present since CUDA 3.2) and should be available on any supported CUDA installation, but the toolchain task should verify the cudarc `sys` bindings include it.

---

## Theme Progress

**hostcall** theme: `hostcall.1` done. The shared memory mechanism question is resolved — mapped pinned memory via the driver API is the correct choice, and cudarc's partial coverage is identified with a clear workaround path. Together with `hostcall.2` (protocol design), the foundation for `hostcall.3` (implementation) is complete. The remaining blocker is `atomics.1` (PTX scope verification), which must be confirmed experimentally before the hostcall experiment phase begins.
