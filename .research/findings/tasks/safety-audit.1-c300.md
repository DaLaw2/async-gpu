# safety-audit.1: Catalog unsafe blocks in hostcall/CAS/PAL code
**Cycle**: 300 | **Theme**: safety-audit | **Kind**: investigation | **Status**: done

## Summary

Comprehensive audit of all `unsafe` usage across 7 crate groups. Found **~1,251 unsafe
occurrences** across **31+ source files** (excluding tests and patched-std upstream code).
The codebase is heavily unsafe by necessity (GPU kernel code, inline PTX, raw pointer
protocols), but SAFETY comment coverage is **extremely sparse** -- only **7 SAFETY comments**
exist across all project-authored code.

## Findings

### Q: How many unsafe blocks and where?

| Crate | Files | Unsafe occurrences | Notes |
|-------|-------|--------------------|-------|
| gpu-host (non-test) | 4 | ~90 | hostcall.rs dominates (67) |
| gpu-host (tests) | 10 | ~741 | Test harness, kernel launches |
| gpu-runtime | 1 | 124 | Single 3600+ line file |
| gpu-atomics | 1 | 17 | All inline PTX, well-structured |
| gpu-libc | 5 | 52 | Stub FFI + hostcall I/O |
| gpu-kernel | 11 | 227 | Kernel entry points + GPU ops |
| patched-std (gpu-specific) | 1 | ~12 | gpu_threads.rs thread-local |
| patched-std (upstream) | 126 | ~431 | Not project-authored |
| examples | ~12 | ~65 | Kernel launches + entry points |

**Total project-authored unsafe: ~1,328 occurrences**
(Excluding patched-std upstream code which has 431 occurrences across 126 files.)

### Q: Which are highest risk?

#### Critical (Lock-free CAS, atomic ordering, concurrent data structures)

1. **`gpu-runtime/src/lib.rs` -- hostcall CAS stack operations** (~20 blocks)
   - `hc_pop_free()`, `hc_pop_free_from()`, `hc_push()`, `hc_push_with()`
   - Lock-free Treiber stack using tagged pointers (ABA prevention via epoch tag)
   - Uses `sys_cas_u64` for atomic compare-and-swap at system scope
   - **No SAFETY comments.** Key invariant: tagged pointer epoch must increment monotonically

2. **`gpu-host/src/hostcall.rs` -- listener CAS operations** (~15 blocks)
   - `ready_stack().swap()` -- AtomicU64 AcqRel swap to drain ready queue
   - Shard-level ready stack draining with per-shard AtomicU64
   - `reinit_packets()` -- reinitializes lock-free stacks between launches
   - **Only 2 SAFETY comments** (one for Send/Sync, one for reinit precondition)

3. **`gpu-atomics/src/lib.rs` -- `sys_cas_u32`/`sys_cas_u64`** (2 blocks)
   - Foundation of all lock-free operations
   - Well-documented with doc comments but **no inline SAFETY comments**
   - Correctness depends on callers providing valid mapped memory pointers

4. **`gpu-libc/src/memory.rs` -- bump allocator CAS** (2 blocks)
   - `malloc()` and `posix_memalign()` use `compare_exchange_weak` in CAS loops
   - Thread-safe via Rust's `AtomicU64`, but Relaxed ordering means no fence
   - **No SAFETY comments.** Key invariant: BUMP_STATE must be initialized before use

#### High (Raw pointer arithmetic, memory mapping, GPU/host boundary)

5. **`gpu-host/src/hostcall.rs` -- packet buffer manipulation** (~30 blocks)
   - `init()`, `reinit_packets()` -- raw pointer writes to initialize protocol buffers
   - `handle_*()` service handlers -- raw `read_volatile`/`write_volatile` on packet payloads
   - `packet_ptr()` -- computes raw pointers from base + offset arithmetic
   - **No SAFETY comments on handlers.** Invariant: offsets must be in bounds of allocation

6. **`gpu-host/src/memory.rs` -- MappedBuffer<T>** (~10 blocks)
   - `read()`/`write()` -- volatile access to pinned memory
   - `as_slice()`/`as_mut_slice()` -- creates references from raw pointers
   - `Drop` -- calls `cuMemFreeHost`
   - **1 SAFETY comment** on Send/Sync impls. Doc comments exist on unsafe methods.

7. **`gpu-host/src/mapped_mem.rs` -- alloc_mapped_*()** (5 blocks)
   - `alloc_mapped_u32()`, `alloc_mapped_result_array()`, `alloc_mapped_u64_array()`
   - Returns raw pointers from CUDA allocator
   - **No SAFETY comments.** Doc-level `# Safety` sections present.

8. **`gpu-runtime/src/lib.rs` -- hostcall request/response** (~15 blocks)
   - `gpu_hostcall_request()` -- fills payload via closure, spin-waits for response
   - `gpu_hostcall_request_with_timeout()` -- same with timeout
   - Raw pointer arithmetic into shared buffer with system-scope atomics
   - **No SAFETY comments.**

9. **`gpu-libc/src/hostcall_io.rs` -- static mut HC_BUF** (5 blocks)
   - `static mut HC_BUF: *mut u8` -- global mutable state, set once at init
   - All I/O functions read this global without synchronization
   - **No SAFETY comments.** Safe only because GPU "threads" within a block share address space

10. **`gpu-libc/src/errno.rs` -- static mut ERRNO_ARRAY** (4 blocks)
    - `static mut ERRNO_ARRAY: [c_int; 1024]` -- per-thread errno via thread ID indexing
    - Thread safety relies on each thread accessing a unique index
    - **No SAFETY comments.** Graceful degradation for >1024 threads (shares slot 0)

11. **`gpu-runtime/src/lib.rs` -- static mut PANIC_BUF / RESULT_BUF** (4 blocks)
    - Global mutable pointers initialized once at kernel entry
    - `write_panic_to_result()` writes to RESULT_BUF without synchronization
    - **No SAFETY comments.**

#### Medium (Inline PTX assembly)

12. **`gpu-atomics/src/lib.rs` -- all 17 unsafe functions** (17 blocks)
    - System-scope loads, stores, CAS, fetch-add, exchange, fences
    - Warp intrinsics: activemask, lane_id, syncwarp, shfl_sync_idx
    - Well-understood PTX patterns with comprehensive doc comments
    - **No `// SAFETY` comments**, but function-level docs describe safety requirements

13. **`gpu-libc/src/errno.rs` -- inline PTX for thread ID** (1 block)
    - `thread_id_in_block()` reads `%tid.x/y/z` and `%ntid.x/y`
    - Standard PTX pattern, low risk

14. **`patched-std gpu_threads.rs` -- inline PTX for thread ID** (1 block)
    - `gpu_tid()` -- same pattern as errno.rs
    - **5 SAFETY comments** -- best coverage in the project

15. **`gpu-runtime/src/lib.rs` -- nvptx intrinsics** (6 blocks)
    - `_block_idx_x()`, `_thread_idx_x/y/z()`, `_block_dim_x/y()`
    - Calls to `core::arch::nvptx::*` -- thin wrappers, low risk

#### Low (FFI calls to cudarc/CUDA)

16. **`gpu-host/src/hostcall.rs` -- CUDA API calls** (~12 blocks)
    - `cuMemHostAlloc`, `cuMemHostGetDevicePointer_v2`, `cuMemFreeHost`
    - Upstream cudarc responsibility for FFI correctness
    - Pattern is consistent: allocate, check result, return error on failure

17. **`gpu-host/src/memory.rs` -- CUDA API calls** (~8 blocks)
    - Same pattern as hostcall.rs for MappedBuffer allocation

18. **`gpu-host/src/mapped_mem.rs` -- CUDA API calls** (5 blocks)
    - Same pattern

#### Trivial (unsafe impl Send/Sync, extern "C" fn declarations)

19. **`unsafe impl Send/Sync`** -- 18 total across codebase:
    - `MappedBuffer<T>`: Send + Sync (1 SAFETY comment)
    - `HostcallBuffer`: Send + Sync (1 SAFETY comment)
    - `CommandBuffer`: Send + Sync (no SAFETY comment)
    - `FlightRecorder`: Send + Sync (no SAFETY comment)
    - `GpuPrintFuture`: Send (1 SAFETY comment)
    - `GpuOpenFuture`: Send (1 SAFETY comment)
    - `GpuWriteFuture`, `GpuReadFuture`, `GpuCloseFuture`: Send (1 each)
    - `GpuBulkWriteFuture`, `GpuBulkReadFuture`: Send (1 each)
    - `GpuTcpConnectFuture`, `GpuTcpWriteFuture`, etc.: Send (1 each)
    - `EagerStorage<T>`, `LazyStorage<T>`, `LocalPointer`: Sync (3 SAFETY comments in gpu_threads.rs)

20. **`unsafe extern "C" fn` declarations** in gpu-libc -- 30 functions:
    - Stubs (stub.rs): 24 functions returning ENOSYS or aborting
    - Memory (memory.rs): 7 functions (memcpy, memset, malloc, etc.)
    - String (string.rs): 3 functions (strlen, strcmp, strncmp)
    - I/O (hostcall_io.rs): 4 functions (open, read, write, close)

21. **`unsafe extern "ptx-kernel" fn` declarations** in gpu-kernel -- ~50 functions:
    - Kernel entry points; inherently unsafe (raw pointer args, GPU execution)

### Q: Which already have SAFETY comments?

Only **7 distinct SAFETY comment locations** exist across all project-authored code:

| File | Line | Comment |
|------|------|---------|
| `gpu-host/src/hostcall.rs` | 168 | `// SAFETY: The buffer is pinned memory...` (Send/Sync) |
| `gpu-host/src/hostcall.rs` | 382 | `/// SAFETY: Must only be called after cuCtxSynchronize()` |
| `gpu-host/src/memory.rs` | 31 | `// SAFETY: The buffer is pinned memory...` (Send/Sync) |
| `gpu-runtime/src/lib.rs` | 1664 | `// SAFETY: On GPU, all threads access the same global memory.` |
| `gpu-runtime/src/lib.rs` | 2442 | `// SAFETY: On GPU, all threads access the same global memory.` |
| `patched-std gpu_threads.rs` | 93,101,156,162,253 | 5 SAFETY comments for Sync impls and MaybeUninit patterns |

**Coverage: ~7 SAFETY comments vs ~1,251 unsafe occurrences = <1% coverage.**

### Q: What safety invariants are undocumented?

1. **Tagged pointer ABA prevention** -- `hc_pop_free`/`hc_push` use epoch tags in the high 16 bits
   of the u64 stack head to prevent ABA. This invariant is not documented in any SAFETY comment.

2. **Packet buffer bounds** -- All `pkt.add(PKT_OFF_*)` pointer arithmetic assumes the buffer is
   large enough. No runtime bounds checking or documented size invariant.

3. **System-scope memory ordering** -- The interplay between GPU `st.release.sys` and host
   `AtomicU32::load(Acquire)` is the fundamental correctness mechanism but is never documented
   as a SAFETY invariant at usage sites.

4. **static mut initialization ordering** -- `HC_BUF`, `PANIC_BUF`, `RESULT_BUF`, `ERRNO_ARRAY`
   must be initialized before use. No compile-time enforcement; relies on kernel entry protocol.

5. **Sideband buffer ownership** -- The sideband bump allocator (`sideband_alloc`) hands out
   offsets that must not overlap. Thread safety relies on CAS but the lifetime/validity of
   returned offsets is not documented.

6. **CommandBuffer/FlightRecorder Send+Sync** -- These have `unsafe impl Send/Sync` but no
   SAFETY comments explaining why they are safe to share. The invariant is the same as
   HostcallBuffer (pinned memory, protocol-enforced access patterns).

7. **Future `get_unchecked_mut()` usage** -- All GPU async Futures use
   `Pin::get_unchecked_mut()` in their `poll()` implementations. The structural pinning invariant
   (future must not be moved after first poll) is never documented.

8. **Warp-cooperative correctness** -- `syncwarp(mask)` will deadlock if `mask` includes inactive
   lanes. This is documented in gpu-atomics doc comments but not at call sites in gpu-runtime.

9. **`realloc` copies `new_size` bytes from old allocation** -- If growing, this reads beyond
   the old allocation's original size. Comment acknowledges this as a known limitation of the
   bump allocator but it is technically UB.

## Catalog (by file)

### gpu-atomics (`crates/core/gpu-atomics/src/lib.rs`)
- **17 unsafe functions** (all `pub unsafe fn`)
- **0 `// SAFETY` comments** (but comprehensive doc comments with `Safety:` sections)
- Risk: **Medium** (inline PTX, well-understood patterns)
- Functions: `membar_sys`, `sys_store_release_u32/u64`, `sys_load_acquire_u32/u64`,
  `sys_cas_u32/u64`, `sys_fetch_add_u32/u64`, `sys_exchange_u64`,
  `sys_spin_load_acquire_u32/u64`, `activemask`, `lane_id`, `syncwarp`,
  `shfl_sync_idx_u32`, `st_global_u32`

### gpu-runtime (`crates/core/gpu-runtime/src/lib.rs`)
- **124 unsafe occurrences** in a single ~3,600 line file
- **2 `// SAFETY` comments** (both for `unsafe impl Send`)
- Breakdown:
  - 6 nvptx intrinsic wrappers (Low)
  - ~15 hostcall protocol functions (Critical: CAS stacks)
  - ~10 print/trace/assert/panic functions (High: raw pointer payloads)
  - ~5 sideband/bulk transfer functions (High: raw pointer + size calculations)
  - ~5 print buffer functions (Medium)
  - ~5 panic/result global state functions (High: static mut)
  - ~30 GPU Future poll() implementations (High: Pin::get_unchecked_mut)
  - 14 `unsafe impl Send` for Futures (Trivial: all same pattern)
  - ~5 warp-cooperative runtime functions (Critical: warp synchronization)
  - ~5 command buffer functions (Medium)
  - ~5 flight recorder functions (Medium)
  - ~5 async executor functions (Critical: block_on, warp_run_future)

### gpu-host (`crates/core/gpu-host/src/hostcall.rs`)
- **67 unsafe occurrences** (non-test code)
- **2 `// SAFETY` comments**
- Breakdown:
  - 2 `unsafe impl Send/Sync` for HostcallBuffer (Trivial, has SAFETY)
  - 2 `unsafe impl Send/Sync` for CommandBuffer (Trivial, no SAFETY)
  - 2 `unsafe impl Send/Sync` for FlightRecorder (Trivial, no SAFETY)
  - ~12 CUDA API calls (Low: FFI to cudarc)
  - ~15 buffer init/reinit volatile writes (High: raw pointer arithmetic)
  - ~5 atomic cast helpers (doorbell, ready_stack, shutdown) (Medium)
  - ~10 packet processing in listener (High: raw pointer reads)
  - ~15 service handler functions (High: raw pointer payload manipulation)
  - ~5 CommandBuffer submit/reset (High: volatile pointer writes)
  - ~5 FlightRecorder methods (Medium: volatile reads)

### gpu-host (`crates/core/gpu-host/src/memory.rs`)
- **18 unsafe occurrences**
- **1 `// SAFETY` comment** (Send/Sync impl)
- All methods have `# Safety` doc sections
- Risk: **High** (volatile reads/writes on mapped memory, CUDA FFI)

### gpu-host (`crates/core/gpu-host/src/mapped_mem.rs`)
- **5 unsafe occurrences**
- **0 `// SAFETY` comments** (but `# Safety` doc sections exist)
- Risk: **Low** (CUDA allocation FFI with proper error handling)

### gpu-libc (`crates/core/gpu-libc/src/`)
- **52 unsafe occurrences** across 5 files
- **0 `// SAFETY` comments**
- `memory.rs` (10): memcpy/memset/memcmp/memmove + bump allocator (High: CAS + pointer arith)
- `stub.rs` (30): 24 libc stubs returning ENOSYS (Trivial: extern "C" declarations)
- `hostcall_io.rs` (5): open/read/write/close via hostcall (High: static mut + raw pointers)
- `errno.rs` (4): static mut ERRNO_ARRAY + PTX asm (High: mutable static, Medium: PTX)
- `string.rs` (3): strlen/strcmp/strncmp (Medium: raw pointer iteration)

### gpu-kernel (`crates/kernel/gpu-kernel/src/`)
- **227 unsafe occurrences** across 11 files
- **0 `// SAFETY` comments**
- Predominantly `pub unsafe extern "ptx-kernel" fn` entry points (~50)
- Kernel bodies contain calls to gpu-atomics and gpu-runtime unsafe functions
- Risk: **Medium-High** (kernel code, but patterns are repetitive and well-tested)

### patched-std (`patched-std/src/sys/thread_local/gpu_threads.rs`)
- **~12 unsafe occurrences**
- **5 `// SAFETY` comments** -- best coverage in the project
- Risk: **High** (UnsafeCell + MaybeUninit + thread ID indexing)
- Key invariants well-documented: per-thread slot isolation via hardware thread ID

### examples (`examples/*/`)
- **~65 unsafe occurrences** across ~12 files
- **0 `// SAFETY` comments**
- Predominantly kernel launches and `extern "ptx-kernel"` entry points
- Risk: **Low** (example code, follows established patterns)

## Risk Summary

| Risk Level | Count (approx) | SAFETY comments |
|------------|----------------|-----------------|
| Critical | ~45 | 0 |
| High | ~120 | 4 |
| Medium | ~60 | 0 |
| Low | ~30 | 0 |
| Trivial | ~80 | 3 |
| Tests/Examples | ~800+ | 0 |

## Impact on Downstream Tasks

1. **SAFETY comment debt is severe.** The most critical code paths (CAS stack operations,
   hostcall protocol, GPU-host memory ordering) have zero SAFETY documentation.

2. **`static mut` usage** in gpu-libc and gpu-runtime is an immediate code quality concern.
   These should migrate to safer patterns (e.g., `core::cell::UnsafeCell` or `SyncUnsafeCell`).

3. **`realloc()` in gpu-libc has a known UB** -- copies `new_size` bytes from old allocation
   without knowing old size. This needs a fix or at minimum a documented limitation.

4. **The lock-free Treiber stack** (tagged pointer CAS) is the highest-risk code in the entire
   project. A formal correctness argument or at least a detailed SAFETY comment block
   explaining the ABA prevention strategy is essential.

5. **Future pin safety** -- all GPU futures use `get_unchecked_mut()` without documenting the
   structural pinning invariant. This is a common pattern but should be documented.

6. **Recommended priority for adding SAFETY comments:**
   - P0: `hc_pop_free`, `hc_push`, `gpu_hostcall_request` (CAS protocol correctness)
   - P0: `static mut` globals (HC_BUF, PANIC_BUF, RESULT_BUF, ERRNO_ARRAY)
   - P1: `HostcallBuffer::init/reinit_packets` (buffer layout invariants)
   - P1: All `unsafe impl Send` for GPU Futures (pin safety)
   - P2: Service handlers in hostcall.rs (payload offset bounds)
   - P2: gpu-atomics functions (already have doc comments, just needs `// SAFETY` at call sites)
   - P3: Test and example code (lower priority, patterns are repetitive)
