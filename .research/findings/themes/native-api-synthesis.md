# native-api: gpu::run() one-liner launcher + clean examples
**Epic**: native-rust-dx | **Status**: active | **Updated**: 2026-06-04

## Progress
- gpu::run(), gpu::run_with_output(), gpu::launch() implemented in gpu.rs
- gpu::compute() deprecated alias for gpu::launch()
- 4 examples now use native API: thread-demo, hello-gpu, async-io, async-pipeline
- 3 examples rewrote from 100-170 lines of boilerplate to 30-50 lines of one-liners
- Per-example kernel compilation (build.rs) eliminated for 3 examples

## Verified Conclusions
- gpu::launch(name, n, threads) wraps device init + PTX load + launch + sync + dtoh
- gpu::run_with_output(name, n) adds hostcall session for kernels needing I/O
- kernel.ptx (2.5MB) loads in ~1s — practical for runtime JIT
- kernel_std.ptx (4.5MB) too large for runtime JIT (>180s) — NOT viable for run_std()
- Examples with custom multi-arg kernel signatures cannot use the one-liner API
- examples/std/* already use nn-level API — higher abstraction, no conversion needed

## Rejected Approaches
- gpu::run_std() loading kernel_std.ptx at runtime — JIT compilation too slow (4.5MB PTX)

## Open Questions
- Pre-compile kernel_std.ptx to cubin at build time to enable run_std()?
- Add gpu API variants with custom argument passing for remaining examples?

## Key Metrics
- Converted examples: 3/7 hostcall examples (hello-gpu, async-io, async-pipeline)
- Not convertible: 4/7 (vector-math, tcp-echo, parallel-search: custom kernel args; warp-cooperative: raw test)
- examples/std: 0 need conversion (already high-level nn API)
- Lines saved: ~350 lines of boilerplate removed across 3 examples

## Next Steps
- Consider adding flexible gpu::run_custom() API for multi-arg kernels
- Consider cubin pre-compilation for kernel_std to enable std-based examples
