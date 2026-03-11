# toolchain.2: Investigate rust-cuda (rustc_codegen_nvvm)
**Cycle**: 12 | **Theme**: toolchain | **Kind**: investigation | **Status**: done

## Summary

rust-cuda (Rust-GPU/Rust-CUDA) provides a custom rustc codegen backend (`rustc_codegen_nvvm`) that compiles Rust to NVVM IR, which NVIDIA's proprietary `libnvvm` then lowers to optimized PTX. The project was rebooted in January 2025 after years of dormancy, currently supports nightly-2025-06-23 and CUDA 12.x (experimental), and is actively maintained but still early-stage. While it offers significantly better PTX quality than the LLVM PTX backend (nvptx64-nvidia-cuda) and broader language feature support (closures, dynamic dispatch, iterators, alloc), it requires a specific nightly toolchain, depends on LLVM 7-era libnvvm, and lacks atomics support — making it a viable fallback but divergent from VectorWare's upstream-rustc approach.

## Findings

### Q: What is the detailed compilation pipeline of codegen_nvvm?

A: The pipeline has these stages:

1. **Standard rustc frontend**: Source → HIR → Type checking → MIR (identical to normal compilation)
2. **Backend loading**: rustc loads `rustc_codegen_nvvm` as a shared library (dylib), creates `NvvmCodegenBackend`, invokes `codegen_crate`
3. **MIR → LLVM IR (NVVM IR)**: Uses `rustc_codegen_ssa` traits (the same SSA abstraction that `rustc_codegen_llvm` uses). The backend implements `BuilderMethods`, `TypeMethods`, etc. to emit LLVM IR that conforms to NVVM IR restrictions. Key differences from normal LLVM codegen:
   - Address spaces: global(1), shared(3), constant(4), local(5), generic(0)
   - Kernel functions annotated via `nvvm.annotations` named metadata with `kernel` property
   - Integer types restricted to i1, i8, i16, i32, i64 (no i128 natively; emulated since Aug 2025)
   - No atomic load/store instructions (must use NVVM intrinsics)
   - No fence instructions
   - No fp128, x86_fp80 types
   - Arguments <32 bits require `zeroext`/`signext` attributes
4. **LLVM optimization**: Each codegen unit is optimized by LLVM independently (parallel)
5. **LTO / Module merge**: All bitcode modules merged into a single module. If `-C lto` not specified, thin local LTO across codegen units
6. **NVVM verification**: Merged IR verified against NVVM IR rules
7. **libnvvm compilation**: NVIDIA's proprietary `libnvvm` (based on LLVM 7.0.1) compiles NVVM IR → PTX with proprietary optimizations not available in the open-source LLVM PTX backend
8. **PTX post-processing**: PTX string parsed, function-level dead code elimination (DCE) removes unused functions to reduce bloat
9. **Final PTX**: Embedded in host binary or loaded at runtime via CUDA Driver API (cust)

Key architectural note: The project bundles prebuilt LLVM 7.1.0 libraries because libnvvm speaks LLVM 7 bitcode. The codegen backend must emit LLVM 7-compatible IR despite being built against modern rustc internals. NVVM IR supports two dialects: LLVM 7 (current) and a modern dialect based on LLVM 21.1.0 (Blackwell+ GPUs only).

**Confidence**: high

### Q: What GPU-side APIs does cuda_std provide?

A: `cuda_std` provides a comprehensive GPU-side standard library:

**Threading & Indexing:**
- `thread` module: `threadIdx_x/y/z`, `blockIdx_x/y/z`, `blockDim_x/y/z`, `gridDim_x/y/z`
- `thread::index_1d()` for globally-unique linear thread index

**Synchronization:**
- `sync_threads()` — block-level barrier
- `sync_threads_count()`, `sync_threads_and()`, `sync_threads_or()` — predicated barriers

**Warp Operations:**
- `warp` module: `shuffle_down`, `shuffle_up`, `shuffle_idx`, `shuffle_xor`
- `vote_all`, `vote_any`, `vote_ballot` — warp-level voting
- Warp-level reductions

**Memory:**
- `shared_array!` macro for static shared memory
- `dynamic_shared_memory()` for runtime-sized shared memory
- `mem` module: CUDA memory allocation via system calls (enables `alloc` support)
- `ptr` module: CUDA-specific pointer operations
- Address space checking functions (global, shared, constant, local)

**Intrinsics & Math:**
- `intrinsics` module: raw libdevice math functions
- `GpuFloat` / `FloatExt` traits for GPU math intrinsics
- `f16` and `bf16` types (via `half` crate re-export)

**I/O & Debugging:**
- `print!` / `println!` macros (GPU-to-stdout via vprintf)
- `assert_eq!` / `assert_ne!` macros

**Attributes:**
- `#[kernel]` — marks function as GPU kernel entry point (adds `nvvm.annotations`)
- `#[gpu_only]` — creates CPU/GPU function variants
- `#[externally_visible]` — prevents DCE from removing a function
- `#[address_space(...)]` — manual memory address space specification

**Compute Capability:**
- `#[target_feature(enable = "compute_75")]` — architecture-specific compilation
- `cfg` module for capability-gated code

**Confidence**: high

### Q: Where does it diverge from upstream rustc?

A: Major divergences from upstream `nvptx64-nvidia-cuda`:

1. **Custom codegen backend**: Replaces `rustc_codegen_llvm` entirely. Not a target triple change — it's a whole backend swap loaded as a dylib
2. **LLVM version**: Targets LLVM 7.0.1 (via libnvvm) vs upstream rustc's LLVM 19+ for the LLVM PTX backend
3. **Pinned nightly**: Must use a specific nightly version (currently nightly-2025-06-23). Cannot use arbitrary nightly or stable
4. **Build system**: Uses `cuda_builder` crate (wraps cargo) instead of standard `-Zbuild-std`. The builder handles architecture selection, optimization settings, and PTX embedding
5. **Optimization pipeline**: NVIDIA's proprietary optimizations in libnvvm produce significantly better PTX than the open-source LLVM PTX backend (especially for complex kernels)
6. **Address space handling**: Explicit address space management with opt-in constant memory (`--use-constant-memory-space` flag), automatic spillover to global memory
7. **No `extern "ptx-kernel"`**: Uses `#[kernel]` attribute macro instead, which adds `nvvm.annotations` metadata
8. **Host-side library**: Bundles `cust` (forked from RustaCUDA) for host-side CUDA Driver API, rather than relying on external crates
9. **Atomics**: Currently traps on atomic operations (intentionally disabled pending design work), whereas upstream nvptx64 emits atomics but without scope qualifiers (LLVM bug #173993)
10. **alloc support**: Actually works (via CUDA malloc system calls), whereas upstream nvptx64 cannot link `alloc` at all

**Confidence**: high

### Q: What Rust language features are currently supported?

A: Feature support matrix based on the project's documentation:

| Feature | Status | Notes |
|---------|--------|-------|
| Control flow (if/match/loops) | Fully supported | |
| Closures | Fully supported | PTX is more expressive than CUDA C here |
| Enums | Fully supported | |
| Unions | Fully supported | |
| Unsized slices | Fully supported | |
| Dynamic dispatch (dyn Trait) | Fully supported | vtable-based dispatch works |
| Pointer casts | Fully supported | |
| Iterators | Fully supported | |
| Try operator (?) | Fully supported | |
| Alloc (heap allocation) | Supported | via `extern crate alloc` + CUDA malloc |
| Unified memory | Supported | |
| Printing (println!) | Supported | via GPU vprintf |
| Panicking | Supported | currently traps/aborts |
| Proc macros | Supported | run on host at compile time |
| Opt-levels | Supported | |
| Codegen-units | Supported | parallel compilation |
| 128-bit integers (i128/u128) | Partial | basic ops emulated; ctpop, rotate unsupported |
| f16 math | Partial | less common operations unsupported |
| Atomics | Not supported | intentionally traps; design in progress |
| Async/await | Not supported | no runtime, no waker infrastructure |
| Recursion | Unclear | GPU stack is limited; deep recursion will overflow |
| Inline assembly | Not documented | likely unsupported (NVVM IR restriction) |

Compared to upstream nvptx64-nvidia-cuda, rust-cuda supports significantly more features: closures work reliably, dynamic dispatch works, alloc is available, iterators work, and panicking is handled. The upstream LLVM PTX backend "often generates completely invalid PTX for trivial programs" according to the project FAQ.

**Confidence**: high (for documented features), medium (for undocumented ones)

### Q: What are the known limitations and workarounds?

A: **Critical Limitations:**

1. **Pinned nightly requirement**: Must use exactly the specified nightly (currently nightly-2025-06-23). The backend deeply integrates with rustc internals that change frequently between nightlies. No stable compiler support.

2. **Atomics completely disabled**: All atomic operations trap. Issue #8 tracks the design discussion. CUDA has 32-bit and 64-bit atomics but Rust core expects 8-bit atomics too. GPU atomics also have scope variants (_system, _block) with no Rust equivalent. For our project, this is actually similar to the upstream nvptx64 problem (ADR-1 amendment) — both paths need custom atomic implementations.

3. **LLVM 7 dependency**: libnvvm is based on LLVM 7.0.1. The codegen must emit LLVM 7-compatible bitcode despite being compiled against modern rustc. This creates fragility when rustc internals change. A modern dialect (LLVM 21.1.0) exists but only for Blackwell+ GPUs.

4. **Constant memory limit**: Static variables can exceed the 64KB constant memory limit, causing runtime `IllegalAddress` errors. Workaround: as of May 2025, statics default to global memory with opt-in constant memory via `--use-constant-memory-space` flag.

5. **Mutable reference restriction**: `&mut [T]` cannot be passed as kernel arguments because it implies exclusive access, which is incorrect in GPU context. Workaround: use raw pointers (`*mut T`).

6. **No GPU testing in CI**: GitHub Actions lacks NVIDIA hardware, so GPU tests cannot run in CI.

7. **CUDA version constraints**: Officially supports CUDA 11.2-11.8; CUDA 12.x is experimental.

8. **Minimum compute capability**: Requires compute capability 3.0+.

9. **Early development status**: Despite the 2025 reboot, the project warns to "expect bugs, safety issues, and things that don't work."

**Workarounds:**
- Constant memory overflow → use `--use-constant-memory-space` flag or `#[address_space()]` annotations
- Mutable refs → use raw pointers
- Atomics → not yet resolved (design discussion ongoing)
- i128 → emulated since Aug 2025 update
- Nightly pinning → Docker images provided for reproducible environments

**Confidence**: high

## Unexpected Discoveries

1. **NVVM modern dialect**: NVIDIA has introduced an LLVM 21.1.0-based NVVM IR dialect for Blackwell+ GPUs. This could eventually eliminate the LLVM 7 dependency problem, but only for newest hardware.

2. **Rust GPU Chimera**: There is active work on a "Chimera demo" that combines rust-cuda (NVVM/PTX for NVIDIA) with Rust GPU (SPIR-V for Vulkan) for cross-vendor GPU support. This indicates the ecosystem is consolidating.

3. **Coordination with rustc PTX backend team**: The rust-cuda maintainers mention plans to coordinate with the upstream rustc PTX backend team. This could improve the upstream nvptx64 target quality over time.

4. **i128 emulation**: The Aug 2025 update added i128 emulation specifically to support crates like `sha2`. This suggests the project is actively working on crate ecosystem compatibility.

5. **Atomics are disabled by design, not by bug**: Unlike upstream nvptx64 where atomics silently emit incorrect code (missing scope qualifiers), rust-cuda intentionally traps on atomics until they design a proper solution. This is arguably safer.

## Open Questions

1. Does the modern NVVM dialect (LLVM 21.1.0) remove the need for LLVM 7 bitcode emission? Could this simplify the codegen?
2. Will the rust-cuda team's atomics design align with our `gpu-atomics` crate approach (inline PTX for system-scope atomics)?
3. How much of cuda_std's GPU-side API could be reused in our approach (even if we stick with upstream nvptx64)?
4. What is the performance delta between libnvvm-optimized PTX and LLVM PTX backend output for real workloads?
5. Could we use cuda_std's `#[kernel]` and shared memory macros as reference implementations for our own GPU abstractions?

## Impact on Downstream Tasks

- **ADR-1 confirmed**: Our choice of upstream `nvptx64-nvidia-cuda` with `extern "ptx-kernel"` remains correct for reproducing VectorWare's approach. rust-cuda diverges significantly from VectorWare's method (custom backend vs upstream target).
- **Fallback viability**: rust-cuda is a viable fallback if the LLVM PTX backend proves too buggy for our needs. It supports more Rust features (closures, alloc, dynamic dispatch) and produces better PTX. However, the pinned-nightly requirement and disabled atomics are significant constraints.
- **gpu-atomics crate**: Both paths (upstream nvptx64 and rust-cuda) need custom atomic implementations. Our `gpu-atomics` crate with inline PTX / NVVM intrinsics is the right approach regardless of backend choice.
- **cuda_std as reference**: The `cuda_std` crate's API design (thread indexing, shared memory, warp intrinsics, print macros) is valuable reference material for building our own GPU-side abstractions on the upstream target.
- **hostcall relevance**: rust-cuda's `print!`/`println!` implementation (via GPU vprintf) is a simpler form of host communication that could inform our hostcall design, though our lock-free two-stack protocol (hostcall.3) is more general.
