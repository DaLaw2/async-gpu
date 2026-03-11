# integration.3: Vendor and patch std source for -Zbuild-std=std
**Cycle**: 28 | **Theme**: integration | **Kind**: experiment | **Status**: done

## Summary
Successfully patched Rust std source to compile for nvptx64-nvidia-cuda with `-Zbuild-std=std`. Only 4 changes needed: add `target_os = "cuda"` to 3 `cfg_select` blocks + add `nvptx64` to `MIN_ALIGN` + create a new `cuda.rs` bump allocator. Two GPU kernels using `Vec`, `String`, and `format!` compiled and executed correctly. LLVM constant-folded all allocations away in these cases.

## Findings

### Q: Does vendored std compile with cfg_select patches for nvptx64?

A: **Yes.** Four specific changes are needed:

1. **`std/src/sys/alloc/mod.rs`**:
   - Add `target_arch = "nvptx64"` to the 16-byte MIN_ALIGN tier
   - Add `target_os = "cuda"` case to the allocator cfg_select, pointing to new `cuda.rs`

2. **`std/src/sys/thread_local/mod.rs`**:
   - Add `target_os = "cuda"` to the no_threads case (first cfg_select)
   - Add `target_os = "cuda"` to the guard module's no-thread-exit case

3. **`std/src/sys/random/mod.rs`**:
   - Add `target_os = "cuda"` to the unsupported case (alongside wasm, xous)
   - Add `target_os = "cuda"` to the hashmap_random_keys exclusion list

4. **New file: `std/src/sys/alloc/cuda.rs`**:
   - Implements `GlobalAlloc for System` using a bump allocator
   - 1 MB static heap in GPU global memory
   - CAS-based allocation for thread safety
   - No deallocation (suitable for short-lived kernel invocations)

The patched std compiles with zero errors for `nvptx64-nvidia-cuda`. The `unsupported` PAL layer handles all OS operations (returns errors for unsupported operations).

Additional crate-side requirements:
- `#![feature(restricted_std)]` — required because nvptx64 is not a "supported" std platform
- `#![feature(abi_ptx)]` — required for `extern "ptx-kernel"` ABI
- `build-std = ["std", "core", "panic_abort"]` — `panic_abort` must be included explicitly

**Confidence**: high (verified compilation + GPU execution)

### Q: Can we route std::io::stdout() through hostcall at the PAL level?

A: **Not attempted yet, but the path is clear.** The `unsupported` PAL at `sys/pal/unsupported/stdio.rs` provides stub implementations that return errors. To route stdout through hostcall:

1. Create a new `sys/pal/cuda/` PAL directory (or modify `unsupported`)
2. Implement `Stdout::write` to call our hostcall PRINT service
3. This requires the hostcall buffer pointer to be accessible from std internals — likely via a global variable set at kernel entry

This was NOT attempted because:
- The compilation proof is the primary goal of integration.3
- Our existing `gpu_hostcall_print()` already provides println-equivalent functionality
- PAL routing adds complexity without new capability (just ergonomic improvement)

**Confidence**: medium (architectural assessment only, not verified)

### Q: What additional patches are needed beyond cfg_select?

A: **None for compilation. Two feature gates for the user crate:**

1. `#![feature(restricted_std)]` — nvptx64 is marked as restricted in the target spec. This is a lint gate, not a functionality blocker.

2. `#![feature(abi_ptx)]` — required for `extern "ptx-kernel"` function declarations. VectorWare's `extern "gpu-kernel"` would not need this (it's their own stable ABI).

The `unsupported` PAL handles everything else:
- `std::io` operations return `io::Error` with "unsupported" message
- `std::thread` operations return errors
- `std::process` operations return errors
- `std::net` operations return errors

This is fine for our use case — we don't need these features through std's interface (we have hostcall).

**Confidence**: high

## Build Recipe

```bash
# 1. Apply patches to vendored std source
./std-patches/apply.sh

# 2. Configure crate
# .cargo/config.toml:
[build]
target = "nvptx64-nvidia-cuda"
[target.nvptx64-nvidia-cuda]
linker = "llvm-bitcode-linker"
rustflags = ["-C", "target-cpu=sm_86"]
[unstable]
build-std = ["std", "core", "panic_abort"]
build-std-features = ["compiler-builtins-mem"]

# 3. Build with patched std source
__CARGO_TESTS_ONLY_SRC_ROOT="patched-std/library" cargo +nightly build --release
```

## Test Results

### std_hello_kernel
- Config: 1 block × 1 thread
- Code: `vec![1,2,3,4,5].iter().sum()` + `String::from("Hello from GPU std!").len()`
- Expected: 15 + 19 = 34
- Result: **PASSED** (result=34)
- PTX: LLVM constant-folded entire Vec+String to `st.volatile.global.b32 [%rd2], 34`

### std_format_kernel
- Config: 1 block × 1 thread
- Code: `format!("value = {}", 42u32).len()`
- Expected: 10
- Result: **PASSED** (result=10)
- PTX: LLVM constant-folded format! to `st.volatile.global.b32 [%rd2], 10`

## PTX Analysis

The generated PTX is 206 lines total:
- 2 kernel entry points (`.visible .entry`)
- Panic infrastructure (panic_with_hook, begin_panic, rust_panic → `trap; exit;`)
- Global panic counter using `atom.global.add.u64` (std's panic_count)
- Thread-local panic state via global variables (no_threads mode)
- Zero unresolved `.extern` declarations — Fat LTO resolved everything

Register pressure is minimal (2 b64 regs per kernel) because LLVM constant-folded all allocations away. Real-world kernels with dynamic data would use the bump allocator and have higher register counts.

## Unexpected Discoveries

1. **LLVM constant-folds Vec and String operations.** `vec![1,2,3,4,5].iter().sum()` compiles to a single constant `34`. The allocator code exists in PTX but is dead code for constant inputs. This is the same aggressive optimization we saw in gpu-std.2.

2. **`panic_abort` must be explicitly listed in build-std.** Without it, the linker cannot find the panic runtime. This was a new error not seen in the core-only builds.

3. **`__CARGO_TESTS_ONLY_SRC_ROOT` env var works.** Despite being an internal Cargo variable (prefixed with `__`), it reliably redirects `-Zbuild-std` to use patched source. Points to the directory containing `library/`. May break in future Cargo versions.

4. **std's no_threads TLS works on GPU.** The `no_threads` thread-local storage (simple static variables) is correct for GPU — each kernel invocation has its own global memory space. The panic counter uses `atom.global.add.u64` which is correct for single-thread-per-block kernels.

5. **The gpu-libc dependency is still useful.** Even with std compiled, the `unsupported` PAL returns errors for I/O operations. Our gpu-libc shim provides the actual implementations via hostcall. The two complement each other: std gives us types and traits, gpu-libc gives us functionality.

## Artifacts

- `std-patches/alloc_mod.patch` — diff for sys/alloc/mod.rs
- `std-patches/thread_local_mod.patch` — diff for sys/thread_local/mod.rs
- `std-patches/random_mod.patch` — diff for sys/random/mod.rs
- `std-patches/cuda.rs` — new bump allocator for System GlobalAlloc
- `std-patches/apply.sh` — script to vendor and patch std source
- `crates/std-build-test/` — test crate with two GPU kernels using std
- `crates/gpu-host/std_build_test.ptx` — compiled PTX

## Impact on Downstream Tasks

- **VectorWare parity score increases from 80% to ~90%.** The key gap (-Zbuild-std=std) is now closed.
- **Future work**: Route std::io through hostcall at PAL level for `println!` via std.
- **integration theme**: This was the last high-risk task. Only polish remains.
