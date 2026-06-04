# kernel-entry.1: Implicit Hostcall Buffer Injection — Investigation

STATUS: done
SUMMARY: Approach B (device global via cuModuleGetGlobal_v2) is the recommended path. It requires zero compiler changes, zero kernel parameter changes, and maps directly to existing infrastructure — the three global AtomicU64 pointers (STDIO_HOSTCALL_BUF, HC_BUF, PANIC_BUF) already exist in the compiled PTX as `.global .align 8 .u64` symbols. The host writes the buffer address to these globals after module load, and the kernel reads them at runtime. This is almost certainly what VectorWare does.
FILES_CHANGED: none (investigation only)

---

## Current State

Today a typical gpu-kernel-std kernel looks like:

```rust
pub unsafe extern "gpu-kernel" fn unified_io_compute(buf: *mut u8, result: *mut u32) {
    stdio_init(buf);                         // stores buf into STDIO_HOSTCALL_BUF
    gpu_libc::gpu_libc_io_init(buf);         // stores buf into HC_BUF
    gpu_runtime::thread::gpu_main_poll(|| {  // warp thread pool setup
        // user code here
    });
}
```

Three subsystems each maintain their own `static` pointer to the hostcall buffer:
- `gpu-kernel-std::STDIO_HOSTCALL_BUF` (AtomicU64) — for println!/stdin
- `gpu-libc::HC_BUF` (AtomicU64) — for open/read/write/close
- `gpu-runtime::panic::PANIC_BUF` (*mut u8) — for panic handler

The host passes the buffer as a kernel parameter:
```rust
func.launch(config, (session.dev_ptr(),))
```

**Goal**: Eliminate the `buf` parameter and all manual init calls so the user writes:
```rust
pub extern "gpu-kernel" fn kernel_main() {
    println!("Hello from GPU!");
}
```

---

## Approach A: Compiler Transform (MIR/codegen pass)

**Mechanism**: A rustc MIR pass rewrites `extern "gpu-kernel" fn kernel_main()` to inject `buf: *mut u8` as a hidden first parameter, then inserts `stdio_init(buf)`, `gpu_libc_io_init(buf)`, and `gpu_main_poll(|| { original_body })` at the function entry.

**Existing infrastructure**:
- `rustc-patches/warp_cooperative.rs` — a MIR pass that transforms coroutine bodies (inserts `shfl.sync` discriminant broadcast + `bar.warp.sync` barriers). Runs after `StateTransform` on nvptx64 targets.
- The existing pass modifies basic blocks, inserts inline asm terminators, and rewrites control flow. However, it operates on coroutine bodies, not on `extern "gpu-kernel"` entry points.

**Feasibility**: Possible but very invasive:
1. Changing function signatures at MIR level is non-trivial — it affects calling convention, ABI, and the PTX codegen. The `extern "gpu-kernel"` ABI has special handling in the LLVM backend (parameters map to PTX `.param` entries).
2. Injecting calls to `stdio_init()` etc. means the MIR pass needs to resolve those functions — cross-crate function resolution in MIR is complex.
3. Wrapping the body in `gpu_main_poll(|| { ... })` requires generating a closure at MIR level, which is extremely difficult (closures are desugared early in HIR→MIR).
4. Every kernel would need to be recompiled if the init sequence changes — tight coupling.

**Verdict**: **Not recommended.** Too much compiler complexity for something achievable at the library level.

---

## Approach B: Device Global via CUDA Driver API ★ RECOMMENDED ★

**Mechanism**: The host writes the hostcall buffer's device address directly into the kernel's global variables after loading the PTX module, before launching the kernel. The kernel reads the global at runtime — no parameter needed.

**Key discovery**: The compiled `kernel_std.ptx` already contains these device globals:

```ptx
.global .align 8 .u64 _RNvCsfWITU4pBCoI_14gpu_kernel_std18STDIO_HOSTCALL_BUF_$_0;
.global .align 8 .u64 _RNvCsfWITU4pBCoI_14gpu_kernel_std18STDIO_SIDEBAND_PTR_$_0;
```

These are the `static STDIO_HOSTCALL_BUF: AtomicU64` and `static STDIO_SIDEBAND_PTR: AtomicU64` from `gpu-kernel-std/src/lib.rs`, compiled by the nvptx64 backend into PTX global variables. Similarly, `gpu-libc::HC_BUF` and `gpu-runtime::panic::PANIC_BUF` appear as globals in the PTX.

**CUDA API**: `cuModuleGetGlobal_v2(dptr, bytes, module, name)` returns the device address of a named global variable in a loaded module. Then `cuMemcpyHtoD` writes the hostcall buffer address into it.

**cudarc support**: cudarc 0.12.1 exposes `cuModuleGetGlobal_v2` via the raw driver FFI (`sys::Lib::cuModuleGetGlobal_v2`). There is no safe wrapper, but we already use raw CUDA driver calls extensively in `hostcall.rs` (e.g., `cuMemHostAlloc`, `cuMemHostGetDevicePointer_v2`).

**Implementation sketch**:

Host side (in `gpu::run()` or `GpuContext::prepare()`):
```rust
// After loading PTX module:
let cu = unsafe { cuda_lib() };
let mut dptr: CUdeviceptr = 0;
let mut size: usize = 0;
let name = CString::new("_RNvCsfWITU4pBCoI_14gpu_kernel_std18STDIO_HOSTCALL_BUF_$_0").unwrap();
unsafe {
    cu.cuModuleGetGlobal_v2(&mut dptr, &mut size, module, name.as_ptr());
    cu.cuMemcpyHtoD_v2(dptr, &session.dev_ptr() as *const _ as *const _, 8);
}
// Repeat for HC_BUF, PANIC_BUF, SIDEBAND_PTR
```

Kernel side: **No change needed** — the globals are already read by the existing code paths (`STDIO_HOSTCALL_BUF.load(Relaxed)`, `HC_BUF.load(Relaxed)`, etc.). The `stdio_init(buf)` / `gpu_libc_io_init(buf)` calls become unnecessary.

**Symbol name challenge**: The mangled Rust symbol names (`_RNvCsfWITU4pBCoI_...`) are unstable and crate-hash-dependent. Solutions:
1. **`#[no_mangle]` + `#[used]` on the statics** — rename them to stable well-known names like `__async_gpu_hostcall_buf`. This is the cleanest approach.
2. **Convention**: Define a single canonical global `__HOSTCALL_BUF` that all subsystems read from. Consolidate the three separate globals into one.
3. **PTX scanning**: Parse the PTX at load time to find the symbol. Fragile but possible.

Option 1 is best: add `#[no_mangle] #[used]` to a single canonical `__HOSTCALL_BUF: AtomicU64` in gpu-runtime, and have stdio/libc/panic all read from it.

**Advantages**:
- Zero compiler changes
- Zero kernel parameter changes — immediate DX improvement
- Works with existing PTX compilation pipeline
- Backward compatible (kernels that still pass `buf` explicitly continue working)
- The pattern is standard CUDA (CUDA C uses `__constant__` or `__device__` globals this way)

**Disadvantages**:
- Requires the host to know the symbol name (solvable with `#[no_mangle]`)
- Module-scoped: each fresh module needs its own write (but we already create fresh modules per launch)
- Requires cudarc raw FFI (no safe wrapper for `cuModuleGetGlobal_v2`)

---

## Approach C: Fixed Address / Reserved Memory

**Mechanism**: Host and device agree on a fixed virtual address for the hostcall buffer. The host allocates at that address; the kernel has a compile-time constant.

**Feasibility**: CUDA does not support allocating at a specific virtual address in the unified address space without CUDA Virtual Memory Management APIs (`cuMemMap`/`cuMemAddressReserve`). Even with VMM, there's no guarantee a specific address is available. On older GPUs, VMM may not be supported.

**Verdict**: **Not recommended.** Fragile, not portable, and unnecessary when Approach B exists.

---

## Approach D: Wrapper Kernel Pattern

**Mechanism**: A proc macro or build.rs generates a wrapper kernel that takes `buf` and calls the user's zero-param kernel:

```rust
// User writes:
#[gpu_kernel]
pub fn kernel_main() { ... }

// Proc macro generates:
#[no_mangle]
pub unsafe extern "gpu-kernel" fn __kernel_main_wrapper(buf: *mut u8) {
    stdio_init(buf);
    gpu_libc_io_init(buf);
    gpu_main_poll(|| { kernel_main() });
}
```

**Feasibility**: Straightforward, but:
1. Requires a proc macro crate (adds build complexity)
2. The wrapper needs access to `stdio_init`, `gpu_libc_io_init`, `gpu_main_poll` — cross-crate resolution in proc macros means emitting fully-qualified paths
3. The host must know to launch `__kernel_main_wrapper` instead of `kernel_main`
4. User sees different function name in error messages / PTX

**Verdict**: **Acceptable fallback** if Approach B proves problematic. But Approach B is strictly better because it requires no code generation at all.

---

## Approach B+: Unified Global + gpu_main_poll Injection

The full VectorWare-style DX requires two things:
1. **Hostcall buffer injection** — Approach B solves this
2. **gpu_main_poll wrapping** — the user shouldn't need to call `gpu_main_poll()` either

For (2), two options:
- **Compiler attribute**: Extend `extern "gpu-kernel"` handling in rustc so that nvptx64 `extern "gpu-kernel"` functions automatically get `gpu_main_poll` wrapping. This is a codegen-level change (emit call to `gpu_main_poll` trampoline in the function prologue).
- **Linker/build trick**: The `build.rs` or `cargo xtask` that compiles the kernel crate could inject the wrapper at the PTX level.
- **Convention**: Provide a `#[gpu_entry]` proc macro that generates the `gpu_main_poll` wrapper. The macro is simple because it doesn't need to handle hostcall injection (Approach B does that).

The cleanest path is: Approach B for hostcall + a simple `#[gpu_entry]` attribute macro for `gpu_main_poll` wrapping. The attribute macro is trivial:

```rust
#[gpu_entry]
fn kernel_main() {
    println!("Hello!");
}

// Expands to:
#[no_mangle]
pub unsafe extern "gpu-kernel" fn kernel_main() {
    gpu_runtime::thread::gpu_main_poll(|| {
        println!("Hello!");
    });
}
```

---

## What VectorWare Likely Does

VectorWare almost certainly uses Approach B (device global). Evidence:
- They use a modified LLVM/Rust toolchain (like us)
- CUDA's `__device__` global pattern is the standard way to pass implicit state to kernels
- Their runtime likely does `cuModuleGetGlobal_v2` to write the hostcall buffer address after loading the module
- The fact that their kernels have zero parameters strongly suggests the buffer comes from a global, not a parameter

---

## Recommendation

**Phase 1 (Approach B — device global)**:
1. Add a single `#[no_mangle] #[used] static __HOSTCALL_BUF: AtomicU64` to gpu-runtime
2. Have `stdio_init`, `gpu_libc_io_init`, and `gpu_panic_init` all read from `__HOSTCALL_BUF` (or alias it)
3. In gpu-host, after `dev.load_ptx()`, use `cuModuleGetGlobal_v2` + `cuMemcpyHtoD` to write the session's `dev_ptr()` into `__HOSTCALL_BUF`
4. Similarly write `__SIDEBAND_BUF` for the sideband pointer
5. Kernels no longer need `buf` parameter or manual init calls

**Phase 2 (proc macro wrapper)**:
1. Create a `#[gpu_entry]` attribute macro that wraps the body in `gpu_main_poll(|| { ... })`
2. User experience becomes: `#[gpu_entry] fn kernel_main() { println!("Hello!"); }`

**Phase 3 (compiler integration, optional)**:
1. Make `extern "gpu-kernel"` on nvptx64 automatically imply `gpu_main_poll` wrapping in the codegen backend
2. Final DX: `pub extern "gpu-kernel" fn kernel_main() { println!("Hello!"); }`

Phase 1 alone delivers 80% of the DX improvement (no manual init, no `buf` parameter for hostcall-only kernels). Phase 2 eliminates `gpu_main_poll`. Phase 3 is pure polish.

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| cuModuleGetGlobal_v2 not in cudarc safe API | Low | Already using raw CUDA driver calls; pattern is established |
| Symbol name instability | Low | Use `#[no_mangle]` for stable names |
| Module scope (fresh module per launch) | None | Already creating fresh modules in `gpu::run()` |
| Thread safety of global write | None | Write happens before kernel launch; device sync guarantees visibility |
| Backward compat | None | Old-style kernels with explicit `buf` param still work |
