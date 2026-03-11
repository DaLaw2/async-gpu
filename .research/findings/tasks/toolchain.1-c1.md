# toolchain.1: nvptx64 Built-in Target Investigation
**Date**: 2026-03-11
**Cycle**: 1
**Theme**: toolchain
**Kind**: investigation
**Status**: done

## Summary

The `nvptx64-nvidia-cuda` target is a Rust Tier 2 target for compiling GPU kernels as PTX
(Parallel Thread Execution) assembly for NVIDIA GPUs. It is strictly `no_std`, requires the
nightly toolchain with `llvm-bitcode-linker`, and exposes only `core` (not `alloc`) via
`-Zbuild-std`. The `extern "ptx-kernel"` ABI is the entry-point convention for GPU kernels
but carries significant restrictions: kernels must return `()` or `!`, cannot use inline
assembly, and must avoid cyclic static initializers. Numerous LLVM PTX backend limitations
make several common Rust patterns unreliable or outright broken on this target.

## Detailed Findings

### Q1: Supported Rust Language Features

**What works:**
- Basic control flow: `if`/`else`, `loop`, `for`, `while`, `match`
- Primitive types: `u8`–`u64`, `i8`–`i64`, `f32`, `f64`, `bool`, `usize`, `isize`
- Structs and enums (with `repr(C)` recommended for kernel parameters)
- References and raw pointers
- Generics and monomorphization
- Traits (static dispatch via monomorphization)
- Closures (if they compile to zero or known-size data)
- Arrays and slices (as parameters, via pointer+length pair)
- `core` library (`-Zbuild-std=core`)
- `core::arch::nvptx` intrinsics (behind `#![feature(stdarch_nvptx)]`)
- Thread/block index queries, synchronization barriers
- `f16x2` vector operations (via `core::arch::nvptx`)
- Dynamic memory: `malloc`/`free` from a fixed-size global heap (via `core::arch::nvptx`)
- `vprintf` for debug output from GPU kernel
- `trap()` instruction

**What is restricted or broken:**
- `no_std` only — the standard library is not available
- `alloc` crate: not officially supported via `-Zbuild-std`; requires a custom global allocator
  backed by the fixed-size heap (via `core::arch::nvptx::malloc`/`free`)
- Panic with unwinding: panic strategy is forced to `abort`; stack unwinding is not supported
- `#[target_feature(enable = "...")]` attribute: explicitly unsupported for `ptx*`/`sm_*`
  features; may cause undefined behavior if used
- Inline assembly (`asm!`): nvptx64 is NOT in the list of targets with stable or even unstable
  `asm!` support in the Rust reference
- `i128`/`u128`: historically caused LLVM assertion failures (issue #38824, fixed by PR #40257),
  but these types pass as byte arrays in the current ABI implementation; practical reliability
  may vary
- Cyclic static initializers: the compiler explicitly rejects cycles in static initializer
  graphs (e.g., `static A: Foo = Foo(&A)` is rejected)
- Dynamic dispatch (`dyn Trait`): not inherently supported; virtual dispatch requires function
  pointers which carry stack/recursion risks on GPU
- Thread-local storage (`#[thread_local]`): not supported
- Recursion: technically allowed by PTX but stack depth is severely limited (~16 MB GPU stack);
  overflows produce opaque `InvalidAddress` errors rather than readable diagnostics
- Deriving `Debug` on GPU structs: causes dramatically increased compile times and PTX size due
  to insufficient dead code elimination in the codegen backend

**Target spec details (from `nvptx64_nvidia_cuda.rs`):**
- Architecture: `nvptx64`, OS: `Cuda`, Vendor: `nvidia`
- LLVM target triple: `nvptx64-nvidia-cuda`
- Linker flavor: `Llbc` (LLVM bitcode linker)
- Data layout: `e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64`
- Pointer width: 64 bits
- Default CPU: `sm_30` (very old; must override with `-C target-cpu=sm_XY`)
- Maximum atomic width: 64 bits
- Panic strategy: Abort
- MergeFunctions optimization: Disabled
- Stack protector: Disabled
- DLL suffix: `.ptx`

### Q2: ptx-kernel ABI Limitations

The `extern "ptx-kernel"` calling convention (`compute_ptx_kernel_abi_info` in
`rustc_target/src/callconv/nvptx64.rs`) enforces the following rules:

**Return type restriction:**
- Kernels may only return `()` (unit) or `!` (never/diverge). Any other return type is a
  hard compile error. This is fundamental to PTX: GPU kernels are "entry points" with no
  caller to receive a return value.

**Parameter passing rules:**
- Primitives (`u8`–`u64`, `i8`–`i64`, `f32`, `f64`): passed directly by value; values
  smaller than 32 bits are zero-extended to 32 bits before passing
- `u128`/`i128`: passed as byte arrays (not as a single 128-bit register, since PTX lacks
  native i128 registers)
- Structs/aggregates: passed as byte arrays at the LLVM IR level (`classify_arg_kernel`),
  not as single integers. This matches the PTX/CUDA ABI expectation
- Slices (`&[T]`): decomposed into a pointer and a length word — two separate kernel
  parameters at the PTX level
- Mutable slices (`&mut [T]`): prohibited because they imply exclusive ownership, which is
  invalid in a parallel context where multiple threads may receive the same argument
- Zero-sized types (ZSTs): completely elided from the PTX parameter list
- `repr(Rust)` structs: technically allowed but strongly discouraged since Rust may change
  layout across compiler versions; `repr(C)` is required for ABI stability
- References and raw pointers: passed as PTX pointer parameters; must point to device memory

**Aggregate classification (`classify_aggregate`):**
The implementation selects a PTX register type based on the aggregate's alignment:
- align 1 → `i8`, align 2 → `i16`, align 4 → `i32`, align 8 → `i64`, align 16 → `i128`
- If `size == alignment`, uses a `cast` mode (prefixed); otherwise uses `uniform` integer cast

**`extern "C"` ABI (device functions, not kernels):**
`compute_abi_info` handles device-side function calls between GPU functions. These also
extend sub-32-bit integers to 32 bits, use the same aggregate classification, but do allow
non-unit return types. There is no CUDA-style Rust calling convention; only `"C"` is
reliable for device functions.

**What is NOT part of the ptx-kernel ABI:**
- No stack protector
- No function prologue/epilogue in the traditional sense
- No support for `extern "Rust"` ABI on kernel entry points
- No variadic argument support
- No support for `#[target_feature]` per-function override (features must be set crate-wide)

### Q3: core::arch::nvptx Intrinsics

All intrinsics are behind `#![feature(stdarch_nvptx)]` (tracking issue #111199, opened
2023-05-04, not yet stabilized as of 2026-03-11).

**Thread and block index queries (all return `i32`):**
- `_thread_idx_x()`, `_thread_idx_y()`, `_thread_idx_z()` — thread index within the block
- `_block_idx_x()`, `_block_idx_y()`, `_block_idx_z()` — block index within the grid
- `_block_dim_x()`, `_block_dim_y()`, `_block_dim_z()` — dimensions of the block
- `_grid_dim_x()`, `_grid_dim_y()`, `_grid_dim_z()` — dimensions of the grid

**Synchronization:**
- `_syncthreads()` — barrier synchronization for all threads in the block (equivalent to
  CUDA's `__syncthreads()`)

**f16x2 vector arithmetic (struct `f16x2`, 32-bit wide = two packed f16 values):**
- `f16x2_add(a, b)` — add, round to nearest even
- `f16x2_sub(a, b)` — subtract, round to nearest even
- `f16x2_mul(a, b)` — multiply, round to nearest even
- `f16x2_fma(a, b, c)` — fused multiply-add, round to nearest even
- `f16x2_neg(a)` — negate
- `f16x2_min(a, b)` — minimum (NaN-propagating)
- `f16x2_max(a, b)` — maximum (NaN-propagating)
- `f16x2_min_nan(a, b)` — minimum, NaNs pass through
- `f16x2_max_nan(a, b)` — maximum, NaNs pass through

**Dynamic memory (fixed-size global heap):**
- `malloc(size: usize) -> *mut u8` — allocate from a fixed-size heap in global memory
- `free(ptr: *mut u8)` — free previously allocated memory

**Output and debugging:**
- `vprintf(fmt: *const u8, args: *const c_void) -> i32` — formatted output to host stream
- `__assert_fail(...)` — assert failure syscall (invoked when an assert produces `false`)

**Trap:**
- `trap()` — emits the PTX `TRAP` instruction; terminates the thread

**Unresolved design question (per tracking issue #111199):**
Whether the API should follow the CUDA API naming conventions remains an open question.
The intrinsics are nightly-only and the final API surface may change before stabilization.

**What is NOT in `core::arch::nvptx`:**
- No warp-level primitives (`__shfl_sync`, `__ballot_sync`, `__match_any_sync`, etc.)
- No tensor core (WMMA) intrinsics
- No cooperative group operations
- No atomic operations beyond what LLVM infers from `core::sync::atomic`
- No clock/timer intrinsics
- No texture/surface memory intrinsics
- No shared memory declaration helpers (shared memory must be declared via `static` or
  external assembly workarounds)
- No cluster-level intrinsics (sm_90+)

### Q4: -Zbuild-std with alloc

**Current state:**
The official documentation and build commands for nvptx64 only show `-Zbuild-std=core`.
The `alloc` crate is NOT officially supported in the same way as `core` for this target.

**Why `alloc` cannot be used naively:**
`alloc` requires a `#[global_allocator]` implementation. On a bare-metal GPU target
there is no OS-provided heap; the only available dynamic memory is the fixed-size heap
managed by `core::arch::nvptx::malloc`/`free`. The user must:
1. Implement a `GlobalAlloc` type that forwards to `malloc`/`free` from `core::arch::nvptx`
2. Annotate it with `#[global_allocator]`
3. Then compile with `-Zbuild-std=core,alloc`

**Practical status:**
Some community projects (e.g., `cuda_std` in Rust-CUDA) have successfully used `alloc` on
GPU by providing a custom allocator wrapping the CUDA device heap. The Rust-CUDA guide
states "the alloc crate can be used if explicitly included," confirming this path works but
requires manual setup.

**Risks and limitations of alloc on GPU:**
- The GPU device heap is fixed-size and small by default (configurable via CUDA API
  `cudaDeviceSetLimit(cudaLimitMallocHeapSize, ...)` on the host before kernel launch)
- Heap fragmentation and allocation failures are not recoverable on GPU (no exception
  handling, abort-only panic)
- `Vec`, `String`, `Box`, `Arc` will work if the global allocator is set up, but
  performance will be much worse than stack-based or shared-memory operations
- `alloc::collections` (e.g., `HashMap`, `BTreeMap`) involve dynamic dispatch and complex
  code that may trigger LLVM PTX backend bugs
- Dynamic trait objects (`Box<dyn Trait>`) introduce vtables and function pointers that
  may produce invalid PTX for complex hierarchies

**Technical path for `-Zbuild-std=core,alloc`:**
```toml
# .cargo/config.toml
[unstable]
build-std = ["core", "alloc"]
build-std-features = ["compiler-builtins-mem"]
```
This requires nightly and `rust-src` component. Compiling `alloc` for nvptx64 is
technically possible but not officially tested or maintained.

### Q5: Known LLVM PTX Backend Bugs

**Historical confirmed bugs (in the Rust/LLVM ecosystem):**

1. **i128 support (Rust issue #38824, fixed PR #40257):**
   The NVPTX backend originally could not handle `i128` return values or parameters,
   triggering LLVM assertions: "Bad return value decomposition" and "LowerFormalArguments
   didn't emit the correct number of values!". This was resolved, but i128 is still passed
   as byte arrays rather than a native register type.

2. **Invalid PTX for many common Rust operations (Rust-CUDA project):**
   The Rust-CUDA guide documents that "the LLVM PTX backend does not always work and would
   generate invalid PTX for many common Rust operations." This is a broad, ongoing issue
   rather than a specific tracked bug.

3. **Cross-crate function calls (rust-ptx-linker project):**
   Non-inlined functions could not be used across crate boundaries without a linker that
   performs LLVM bitcode linking. The abandoned `rust-ptx-linker` and the new in-tree
   `llvm-bitcode-linker` (PR #117458, merged 2024-03-11) exist to solve this. Prior to
   `llvm-bitcode-linker`, the nvptx64 test suite was entirely disabled.

4. **Address space 0 restriction:**
   LLVM requires that global variables NOT reside in address space 0 (the generic space).
   Violating this produces incorrect PTX. Rust's target spec handles this, but custom
   unsafe code using raw LLVM IR tricks may encounter this.

5. **Stack usage with recursion:**
   LLVM cannot statically determine stack usage for recursive functions on PTX. `ptxas`
   (NVIDIA's PTX assembler) emits warnings in this case, and runtime behavior produces
   `InvalidAddress` errors instead of readable stack overflow messages.

6. **`__nvvm_reflect` undefined function:**
   When compiling CUDA code through LLVM, `ptxas` may complain about an undefined
   function `__nvvm_reflect`. This is a known LLVM NVPTX issue where the NVVM IR
   reflection function is not properly resolved when using the raw PTX backend path
   (as opposed to the NVVM IR path used by Rust-CUDA).

7. **Dead code elimination deficiency:**
   The LLVM PTX backend does not eliminate dead code as aggressively as CPU backends.
   Deriving `Debug` (or other trait impls with unused code) dramatically increases PTX
   output size and compile time because dead paths are not pruned.

8. **Warp-synchronous code and Volta+ independent thread scheduling:**
   NVIDIA changed the thread scheduling model in Volta (sm_70+). Code relying on implicit
   warp-level synchronization (writing to shared memory without `__syncwarp`) produces
   undefined behavior. This is a PTX ISA semantic issue, not purely a Rust/LLVM bug.

9. **`read_volatile` intrinsic bug (Rust-CUDA issue #216, fixed 2025-08-07):**
   The `read_volatile` intrinsic had an implementation bug in the Rust-CUDA codegen backend
   (NVVM IR path) that was only recently fixed. This suggests the LLVM PTX path may have
   analogous issues with memory intrinsics.

10. **llvm-bitcode-linker dependency (PR #117458):**
    The previous `rust-ptx-linker` was abandoned for 3+ years and incompatible with newer
    LLVM. While the in-tree `llvm-bitcode-linker` resolves this, it is still an unstable
    component that requires `rustup component add llvm-bitcode-linker --toolchain nightly`.

**LLVM backend-specific restrictions (from LLVM NVPTX docs):**
- Cache hint and multicast flags in bulk copy operations must be compile-time constants
- Certain tensor memory operations only support specific sizes (e.g., 128-byte operations)
- Cluster-level features (`nvvm.maxclusterrank`, cluster synchronization) are Hopper+ only
  (sm_90+); using them on older targets silently produces incorrect behavior or compiler errors
- Grid-constant parameters: writing to a grid-constant is undefined behavior; the address
  is shared across all kernel invocations in a grid

## Unexpected Discoveries

1. **No inline assembly support.** The Rust reference's inline assembly documentation lists
   stable and unstable `asm!` targets; nvptx64 is not among them. This means GPU developers
   cannot write raw PTX instructions inline — a significant limitation for performance-critical
   intrinsics or unsupported hardware features.

2. **`llvm-bitcode-linker` is required but only merged in 2024.** Before PR #117458 landed
   (March 2024), the nvptx64 target had no maintained linker and all nvptx codegen tests were
   disabled. The target was effectively broken for over 3 years before this fix.

3. **`core::arch::nvptx` is still unstable as of 2026.** The tracking issue #111199 was opened
   in May 2023 and has had only one comment. There is no FCP or stabilization PR. GPU
   developers must use nightly indefinitely for these intrinsics.

4. **Windows is not supported for GPU Rust development** via the `rust-ptx-linker` path. The
   `llvm-bitcode-linker` may change this, but it has not been confirmed.

5. **No warp-level intrinsics exist in `core::arch::nvptx`.** Warp shuffles, ballot, and
   match operations are critical for GPU performance. They are entirely absent from the
   standard intrinsics, forcing developers to use unsafe `asm!`-style workarounds or external
   crates — except that `asm!` is also unsupported on nvptx64 (see point 1 above).

6. **NVVM IR vs. raw PTX are two distinct codegen paths.** Rust's built-in nvptx64 target
   uses the LLVM PTX backend (raw PTX), while Rust-CUDA uses libNVVM (NVVM IR). These have
   different bugs, capabilities, and toolchain requirements. Findings from one do not
   necessarily apply to the other.

## Key Conclusions

1. **nvptx64 is viable for basic GPU kernels** that use primitive types, core control flow,
   and `core` library functions. The toolchain is functional since `llvm-bitcode-linker` was
   merged in 2024.

2. **`alloc` is possible but requires manual setup.** There is no out-of-the-box support;
   a custom `GlobalAlloc` backed by `core::arch::nvptx::malloc`/`free` is required.

3. **`extern "ptx-kernel"` is very restrictive.** Kernels must return unit/never, cannot
   use inline assembly, and have limited type support. Complex Rust abstractions (vtables,
   recursive closures, `dyn Trait`) may generate invalid PTX.

4. **The intrinsics set (`core::arch::nvptx`) is thin.** Only thread/block indices, a sync
   barrier, f16x2 ops, malloc/free, vprintf, trap, and assert are provided. Critical GPU
   programming primitives (warp shuffles, shared memory, atomics, tensor cores) are absent.

5. **Multiple LLVM PTX backend bugs exist** that affect real-world Rust GPU programs. The
   most severe are: no dead code elimination, no inline asm, invalid PTX for complex Rust
   operations, and the historical i128 issue.

6. **Feature granularity is crate-level, not function-level.** `#[target_feature]` does not
   work on nvptx64, which means a single crate must target one specific GPU architecture.
   Multi-architecture GPU libraries require separate compilation units.

## Open Questions

1. Can `alloc` be made reliably usable by providing a custom allocator, and what are the
   performance/reliability trade-offs in practice?

2. Are there any plans to stabilize `core::arch::nvptx` (tracking issue #111199 shows
   no activity)?

3. Can warp-level primitives (shuffle, ballot, etc.) be added to `core::arch::nvptx`, and
   if so, through what mechanism given the lack of inline assembly support?

4. Does the `llvm-bitcode-linker` fully enable Windows development, or are there remaining
   Windows-specific issues?

5. What is the gap between the LLVM PTX backend (used by nvptx64 Rust target) and the
   NVVM IR backend (used by Rust-CUDA)? Which is more stable for real programs?

6. Can `async`/`await` state machines be compiled to valid PTX? The state machines are
   ordinary Rust structs with `poll` methods — but if they involve dynamic dispatch, heap
   allocation, or recursion, they may trigger PTX backend bugs.

7. Are atomic operations (`core::sync::atomic`) reliably lowered to PTX atomic instructions?

## Impact on Downstream Tasks

- **hostcall**: The `vprintf` intrinsic is the only built-in hostcall mechanism. Any richer
  host communication (memory allocation callbacks, OS services) must be implemented via
  PTX hostcall instructions, which are not yet exposed through `core::arch::nvptx`.

- **gpu-std**: Building a GPU `std` requires `alloc` and a global allocator. The
  `malloc`/`free` intrinsics provide the necessary foundation but require a wrapper
  implementing `GlobalAlloc`. This is feasible but fragile given LLVM dead-code issues.

- **async-runtime**: Async state machines are Rust structs — they should compile to PTX in
  principle. However: (a) if the executor uses dynamic dispatch it may generate invalid PTX;
  (b) if the runtime needs a heap (for `Box<dyn Future>`) it requires the `alloc` path;
  (c) the absence of warp-level primitives means cooperative scheduling must be cooperative
  in a GPU-specific way (thread-based, not warp-based). This is the highest-risk area.

- **integration**: The toolchain is now functional (llvm-bitcode-linker merged 2024), but
  nightly-only. Any integration work must track nightly compatibility carefully.

- **toolchain (remaining tasks)**: The lack of `asm!` support and thin intrinsics set means
  that any VectorWare-style `std` reimplementation likely requires either: (a) a custom
  codegen backend (like Rust-CUDA's NVVM path), or (b) waiting for upstream intrinsics
  expansion. This should be investigated in `toolchain.2` or a new task.

## Theme Progress

The `toolchain` theme has made progress in understanding the baseline capabilities and
limitations of the built-in `nvptx64-nvidia-cuda` target. The investigation confirms that
the target is functional for basic kernels but has significant gaps for the project goal of
running Rust `std` and `async/await` on GPU. Key next steps:

- Investigate whether a custom codegen backend (NVVM IR path, Rust-CUDA style) closes the
  gaps identified here (thin intrinsics, no asm!, LLVM PTX bugs)
- Prototype a minimal `GlobalAlloc` implementation using `core::arch::nvptx::malloc`/`free`
  to validate the `alloc` path
- Investigate `async` state machine compilation to PTX with a minimal executor to determine
  feasibility before investing in runtime design
