# native-api.3: Investigation — gpu-kernel ABI availability + migration plan

## Summary
`extern "gpu-kernel"` is fully available on nightly-2026-06-03 and works on nvptx64-nvidia-cuda. It uses feature gate `#![feature(abi_gpu_kernel)]`. The ABI is functionally identical to `extern "ptx-kernel"` on NVPTX (same LLVM calling convention, same `.visible .entry` PTX output). The project already has a conditional `gpu_kernel_abi` feature in gpu-kernel crate with a working demo. Migration is a mechanical find-and-replace across 213 call sites in 34 files.

## Findings

**Q1: Is `extern "gpu-kernel"` available in nightly-2026-06-03?**
A: Yes. Confirmed by examining rustc source (`compiler/rustc_abi/src/extern_abi.rs` line 81: `GpuKernel`) and by successfully compiling test programs. Confidence: 100%.

**Q2: What feature gate does it require?**
A: `#![feature(abi_gpu_kernel)]`. Defined in `compiler/rustc_feature/src/unstable.rs` line 374: `(unstable, abi_gpu_kernel, "1.86.0", Some(135467))`. The old `abi_ptx` (since 1.15.0) is still available and not removed. Confidence: 100%.

**Q3: Does it work on nvptx64-nvidia-cuda?**
A: Yes. Tested by compiling standalone `.rs` files targeting nvptx64-nvidia-cuda AND by building the gpu-kernel crate with `--features gpu_kernel_abi`. Both produce correct `.visible .entry` PTX. The ABI mapping in `compiler/rustc_target/src/spec/abi_map.rs` lines 147-149 confirms both `PtxKernel` (NVPTX-only) and `GpuKernel` (NVPTX + AMDGPU) map to `CanonAbi::GpuKernel`. Both use `nvptx64::compute_ptx_kernel_abi_info()` on NVPTX. Confidence: 100%.

**Q4: Minimal compilation test?**
A: Confirmed. `extern "gpu-kernel" fn test(result: *mut u32) { ... }` compiles to `.visible .entry test(...)` in PTX, identical to `extern "ptx-kernel"`. Both safe and unsafe versions compile; both produce the same PTX. Confidence: 100%.

**Q5: Differences between ptx-kernel and gpu-kernel in practice?**
A: On NVPTX they are functionally identical — same LLVM calling convention, same PTX output, same restrictions (no return values, no async/gen). The only semantic difference: `gpu-kernel` also targets AMDGPU, while `ptx-kernel` is NVPTX-only. The rustc lint module (`compiler/rustc_lint/src/gpukernel_abi.rs`) adds two helpful warnings for `gpu-kernel` functions: `improper_gpu_kernel_arg` (warns on non-primitive argument types) and `missing_gpu_kernel_export_name` (warns if `#[no_mangle]` or `#[export_name]` is missing). Confidence: 100%.

**Q6: Does gpu-kernel eliminate the need for `unsafe` or `#[no_mangle]`?**
A: `unsafe` was NEVER required by either ABI — it was used because kernel bodies dereference raw pointers, not because the ABI demands it. Neither `PtxKernel` nor `GpuKernel` has `reject_safe_fn()` in `ast_validation.rs`. The function signature CAN be `extern "gpu-kernel" fn foo(...)` without `unsafe`. However, `#[unsafe(no_mangle)]` (or `#[unsafe(export_name = "...")]`) is still needed and recommended — the gpu-kernel lint explicitly warns when it's missing, because the kernel name needs to be discoverable in the PTX. Confidence: 100%.

**Q7: Does the function name have to be `main()`?**
A: No. Any function name works. Tested `main()`, `my_custom_kernel()`, and functions with `#[unsafe(export_name = "...")]`. All compile to `.visible .entry <name>(...)` in PTX. The function name in the PTX equals the `#[no_mangle]` name or `#[export_name]` value. The epic's success criterion says `extern "gpu-kernel" fn main()`, but this is about the ABI, not a name restriction. Confidence: 100%.

**Q8: Migration plan**
A: See detailed migration plan below.

## Migration Plan

### Phase 1: Feature gate swap (1 file)
- `crates/kernel/gpu-kernel/src/lib.rs`: Change `#![feature(abi_ptx)]` to `#![feature(abi_gpu_kernel)]`. Remove the `cfg_attr` for `gpu_kernel_abi` since it becomes the default.

### Phase 2: ABI string replacement (34 files, 213 call sites)
Mechanical find-and-replace of `extern "ptx-kernel"` → `extern "gpu-kernel"` in:

**Kernel crates** (most changes):
- `crates/kernel/gpu-kernel/src/basic.rs`
- `crates/kernel/gpu-kernel/src/compute_*.rs` (7 files)
- `crates/kernel/gpu-kernel/src/hostcall_kernels.rs`
- `crates/kernel/gpu-kernel/src/hybrid.rs`
- `crates/kernel/gpu-kernel/src/pipeline.rs`
- `crates/kernel/gpu-kernel/src/thread_test.rs`
- `crates/kernel/gpu-kernel/src/warp.rs`
- `crates/kernel/gpu-kernel-std/src/lib.rs`

**Test crates** (6 files):
- `crates/test/async-hostcall-test/src/lib.rs`
- `crates/test/async-pipeline-test/src/lib.rs`
- `crates/test/embassy-test/src/lib.rs`
- `crates/test/gpu-std-test/src/lib.rs`
- `crates/test/multi-warp-test/src/lib.rs`
- `crates/test/std-build-test/src/lib.rs`

**Example kernels** (7 files):
- `examples/hostcall/*/kernel/src/lib.rs` (6 files)
- `examples/hostcall/warp-cooperative/src/lib.rs`

**Runtime/macro** (3 files, doc comments only):
- `crates/core/gpu-runtime/src/lib.rs` (doc example)
- `crates/core/gpu-runtime/src/thread.rs` (doc example)
- `crates/macro/warp-macro/src/lib.rs` (code generation)

**Documentation** (1 file):
- `examples/std/thread-demo/src/main.rs` (doc comment)

### Phase 3: Feature gate in all kernel crates (16 files)
Replace `#![feature(abi_ptx)]` → `#![feature(abi_gpu_kernel)]` in every kernel/test crate's `lib.rs`.

### Phase 4: Remove conditional `gpu_kernel_abi` feature
- `crates/kernel/gpu-kernel/Cargo.toml`: Remove the `gpu_kernel_abi = []` feature.
- `crates/kernel/gpu-kernel/src/lib.rs`: Remove `#![cfg_attr(feature = "gpu_kernel_abi", ...)]`.
- `crates/kernel/gpu-kernel/src/thread_test.rs`: Remove `#[cfg(feature = "gpu_kernel_abi")]` from `gpu_kernel_demo`.

### Phase 5: Optional cleanup
- Remove `unsafe` from kernel function signatures where bodies already use `unsafe { ... }` blocks internally (cosmetic, low priority — the bodies still need `unsafe` blocks for raw pointer ops).
- Update `#[no_mangle]` → `#[unsafe(no_mangle)]` on any remaining old-style attributes.

### Risks
1. **Build breakage**: The standard toolchain nightly-2026-06-03 has `abi_gpu_kernel` — verified. No toolchain rebuild needed.
2. **PTX compatibility**: Both ABIs produce identical `.visible .entry` PTX. Host-side CUDA driver doesn't care about the Rust ABI string — it sees the same PTX. Zero runtime risk.
3. **Scope**: 213 mechanical replacements across 34 files. High volume but zero complexity. Can be done with `sed` + manual review.
4. **warp-macro codegen**: The proc macro at `crates/macro/warp-macro/src/lib.rs:1874` generates `extern "ptx-kernel"` in a quote! block — must be updated.

## Unexpected Discoveries
- The project already had `gpu_kernel_abi` as a cargo feature in gpu-kernel crate with a working demo (`gpu_kernel_demo` in thread_test.rs). Someone started this migration path but left it conditional.
- The `gpu-kernel` ABI adds two lints not present for `ptx-kernel`: argument type checking and export name checking. These are helpful guardrails that make the migration a net positive for code quality.
- `extern "gpu-kernel"` also works on AMDGPU, which could be relevant for future multi-vendor GPU support.

## Open Questions
- Should the `unsafe` keyword be removed from kernel signatures during migration? This is cosmetic but aligns with the "looks like normal Rust" goal. The function bodies still need `unsafe {}` blocks for raw pointer operations.
- The warp-macro generates `#[no_mangle]` (old style). Should it be updated to `#[unsafe(no_mangle)]` at the same time?

## Impact on Downstream Tasks
- Unblocks the full native-api theme: once migrated, all kernels use `extern "gpu-kernel"`, achieving the epic's success criterion.
- The `gpu_kernel_abi` feature flag in gpu-kernel Cargo.toml becomes unnecessary.
- The build.rs in gpu-host doesn't need changes — it already builds gpu-kernel and the PTX output is identical.
