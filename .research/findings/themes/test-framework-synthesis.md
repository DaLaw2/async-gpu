# Theme Synthesis: test-framework

## Status
Active — proc macro implemented and verified (test-framework.1, .2 complete).

## Key Findings
- **`#[gpu_test]` proc macro works end-to-end.** Expands stub to `#[test]` that
  launches the eponymous kernel via `run_zero_param_with_cubin`. Three GPU tests
  pass in 2.74s with cubin fast-loading.
- **assert!/assert_eq!/assert_ne! work on GPU** via patched std panic handler.
  No custom GPU assert needed. Panic messages include thread/block coordinates.
- **Cubin fast-loading is essential.** PTX JIT takes 15-30 min for the 10MB
  unified PTX. The proc macro generates runtime cubin loading with PTX fallback.

## Architecture
Kernel test code: `gpu-kernel-std/src/lib.rs` (zero-param entry, `stdio_auto_init`).
Proc macro: `crates/test/gpu-test-macro/` (syn/quote, parses threads/grid attrs).
Host tests: `gpu-test-harness/tests/gpu_tests.rs` (standard `cargo test` discovery).
New API: `gpu::run_zero_param_with_cubin` in gpu-host for cubin-accelerated launch.

## Next Tasks
- test-framework.3: Write 10+ GPU test kernels covering existing features
- test-framework.4: Create dedicated host-side test crate with `#[gpu_test]` stubs
- test-framework.5: Warp/lane ID decoration in GPU assert failure messages
- test-framework.6: PTX module caching + parallel test serialization
