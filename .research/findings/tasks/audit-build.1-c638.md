# audit-build.1: Build Audit — Compile All Examples and Test Crates

## Summary

All 5 workspace members, 14 std examples, 10 hostcall host examples, 7 hostcall kernel
subcrates, 4 kernel crates, and 6 of 7 non-workspace test crates compile successfully on
`rustc 1.98.0-nightly (d595fce01 2026-06-02)`. The single failure is `std-build-test`,
which hits a linker-level symbol collision (`gpu_stdin_read` multiply defined) between its
own `#[no_mangle]` definition and the one from `gpu-runtime`. Several crates emit warnings
(unused fields, imports, variables) but no other compilation errors exist.

## Findings

### Build Results Summary

| Category | Crate | Status | Notes |
|----------|-------|--------|-------|
| **Workspace** | gpu-protocol | OK | clean |
| **Workspace** | gpu-host | OK | 2 warnings (dead field `layer_norm_eps` x2, unused import) |
| **Workspace** | async-gpu | OK | clean |
| **Workspace** | gpu-test-harness | OK | 1 warning (dead field `kernel_name`) |
| **Workspace** | gpu-test-macro | OK | clean |
| **Std Example** | benchmark | OK | clean |
| **Std Example** | cifar-train | OK | 1 warning (unused variable `chw`) |
| **Std Example** | diff-physics | OK | 2 warnings (unused import, unused variable) |
| **Std Example** | dynamic-control | OK | clean |
| **Std Example** | gpt2-inference | OK | clean |
| **Std Example** | gpt2-lora | OK | 4 warnings (unused imports, unused variable) |
| **Std Example** | gpu-rag | OK | clean |
| **Std Example** | graph-algorithms | OK | clean |
| **Std Example** | mnist-cnn | OK | clean |
| **Std Example** | mnist-train | OK | clean |
| **Std Example** | monte-carlo | OK | clean |
| **Std Example** | resnet-cifar | OK | clean |
| **Std Example** | thread-demo | OK | clean |
| **Std Example** | yolo-detect | OK | clean |
| **Hostcall Host** | async-io-host | OK | clean |
| **Hostcall Host** | async-pipeline-host | OK | clean |
| **Hostcall Host** | gpu-channels | OK | clean |
| **Hostcall Host** | hello-gpu-host | OK | clean |
| **Hostcall Host** | parallel-search-host | OK | clean |
| **Hostcall Host** | structured-concurrency | OK | clean |
| **Hostcall Host** | tcp-echo-host | OK | clean |
| **Hostcall Host** | tokio-offload | OK | clean |
| **Hostcall Host** | vector-math-host | OK | clean |
| **Hostcall Host** | warp-cooperative | OK | clean |
| **Hostcall Kernel** | async-io-kernel | OK | 2 warnings (unused Result) |
| **Hostcall Kernel** | async-pipeline-kernel | OK | clean |
| **Hostcall Kernel** | hello-gpu-kernel | OK | clean |
| **Hostcall Kernel** | parallel-search-kernel | OK | clean |
| **Hostcall Kernel** | tcp-echo-kernel | OK | clean |
| **Hostcall Kernel** | vector-math-kernel | OK | clean |
| **Kernel** | gpu-kernel-core | OK | clean |
| **Kernel** | gpu-kernel-compute | OK | 12 warnings (unused constants, variables) |
| **Kernel** | gpu-kernel-io | OK | clean |
| **Kernel** | gpu-kernel-test | OK | 5 warnings (unnecessary unsafe, unused variable) |
| **Test** | async-hostcall-test | OK | kernel target, builds with nvptx64 |
| **Test** | async-pipeline-test | OK | kernel target, builds with nvptx64 |
| **Test** | embassy-test | OK | kernel target, builds with nvptx64 |
| **Test** | gpu-critical-section | OK | kernel target, builds with nvptx64 |
| **Test** | gpu-std-test | OK | kernel target, builds with nvptx64 |
| **Test** | multi-warp-test | OK | kernel target, builds with nvptx64 |
| **Test** | std-build-test | **FAIL** | linker error: `gpu_stdin_read` multiply defined |

### Failure Detail: std-build-test

- **Category**: linker (symbol collision)
- **Error**: `Linking globals named 'gpu_stdin_read': symbol multiply defined!`
  followed by `failed to load bitcode of module "gpu_runtime-*.o"`
- **Root cause**: `std-build-test/src/lib.rs:65` defines `#[no_mangle] pub fn gpu_stdin_read()`
  while also depending on `gpu-runtime` which provides the same `#[no_mangle]` symbol at
  `gpu-runtime/src/stdio.rs:90`. The llvm-bitcode-linker rejects the duplicate.
- **Impact**: This test crate cannot produce PTX. It was likely broken by a recent change
  that added `gpu_stdin_read` to `gpu-runtime` (or vice versa, the test crate added its
  own copy after the runtime already had one).

### Warning Inventory

| Source | Warning | Location |
|--------|---------|----------|
| gpu-host | field `layer_norm_eps` never read (TransformerBlock) | `crates/core/gpu-host/src/nn/models/gpt2.rs:105` |
| gpu-host | field `layer_norm_eps` never read (Int4TransformerBlock) | `crates/core/gpu-host/src/nn/models/gpt2.rs:1226` |
| gpu-host | unused import `cudarc::nvrtc::compile_ptx` | `crates/core/gpu-host/src/nn/ops/gemm.rs:361` |
| gpu-test-harness | field `kernel_name` never read | `crates/test/gpu-test-harness/src/panic_unwrap_verify.rs:18` |
| gpu-kernel-compute | constants `STATUS_FREE`, `SEQ_OFFSET` never used | `crates/kernel/gpu-kernel-compute/src/compute_persistent.rs` |
| gpu-kernel-test | unnecessary unsafe block | `crates/kernel/gpu-kernel-test/src/lib.rs:583` |
| gpu-kernel-test | unused variable `buf` | `crates/kernel/gpu-kernel-test/src/lib.rs:829` |
| cifar-train | unused variable `chw` | `examples/std/cifar-train/src/main.rs:63` |
| diff-physics | unused import + unused variable | `examples/std/diff-physics/src/main.rs:280,91` |
| gpt2-lora | unused imports + unused variable | `examples/std/gpt2-lora/src/main.rs` |
| async-io-kernel | unused Result | `examples/hostcall/async-io/kernel/src/lib.rs:236` |
| all nvptx64 crates | unstable feature `ptx78` | `.cargo/config.toml` rustflags |

## Unexpected Discoveries

1. **All std examples compile clean** — 14/14 with zero errors. This is excellent health.
2. **All hostcall examples compile clean** — both host (10/10) and kernel (7/7 tested)
   subcrates build without errors.
3. **All 4 kernel crates compile clean** — gpu-kernel-{core,compute,io,test} all link
   successfully for nvptx64.
4. **The `ptx78` feature warning is universal** — every nvptx64 crate emits
   `unstable feature specified for -Ctarget-feature: ptx78`. This is cosmetic and expected
   on nightly.
5. **gpu-host dead code warnings** — `layer_norm_eps` fields in gpt2.rs are stored but
   never read. These violate the CLAUDE.md "no dead_code" policy.

## Open Questions

1. Was `std-build-test` ever working, or was it broken from introduction? (Check git log
   for when `gpu_stdin_read` was added to `gpu-runtime::stdio`.)
2. Should `std-build-test` remove its local `gpu_stdin_read` and use the one from
   `gpu-runtime`, or does it intentionally provide a different implementation?
3. The `layer_norm_eps` dead fields in gpt2.rs — are they needed for future use or should
   they be removed per the project's dead-code policy?

## Impact on Downstream Tasks

- **1 broken crate** (`std-build-test`) needs a fix task — either remove the duplicate
  symbol or restructure the dependency so only one definition exists.
- **Warning cleanup** is a separate task: ~12 warnings across the codebase, mostly unused
  variables/imports and dead fields.
- **All examples are buildable**, so the next audit task (runtime verification) can proceed
  on all 24 examples without build blockers.
- **Kernel build pipeline is healthy** — `build-kernels.sh` covers the 4 main kernel
  crates and they all compile.
