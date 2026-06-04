# native-api.2: Rewrite examples to use gpu::run() native Rust style

## Summary
Rewrote 3 hostcall examples (hello-gpu, async-io, async-pipeline) from verbose manual PTX+launch boilerplate to one-liner `gpu::run_with_output()` / `gpu::launch()` calls. Removed per-example kernel compilation (build.rs) and direct cudarc dependencies. Examples that demonstrate unique capabilities (vector-math, tcp-echo, parallel-search, warp-cooperative, tokio-offload) were assessed but intentionally kept as-is because their kernels have multi-argument signatures incompatible with the one-liner API.

## Findings

**Q: Which examples already use the native API?**
A: Only `examples/std/thread-demo` used `gpu::launch()`. All other hostcall examples used the old verbose style with GpuRuntime, HostcallSession, MappedBuffer, cudarc::driver::LaunchAsync, and per-example build.rs kernel compilation. Confidence: 100%.

**Q: Which examples can be converted to the one-liner API?**
A: hello-gpu, async-io, and async-pipeline can be converted because suitable kernels exist in the embedded kernel.ptx with matching signatures (`(buf, result)` for `gpu::run_with_output()`, `(result)` for `gpu::launch()`). Confidence: 100%.

**Q: Why can't vector-math, tcp-echo, parallel-search be converted?**
A: Their kernels have multi-argument custom signatures (e.g., saxpy takes `(x, y, a, n)`, tcp_echo_kernel takes `(buf, sideband, port, output)`) that don't match the fixed signatures of gpu::run/run_with_output/launch. These would require new API variants or kernel rewrites. Confidence: 100%.

**Q: Can kernel_std.ptx be used for run_std() API?**
A: No, not practically. kernel_std.ptx is 4.5MB / 122K lines (includes full Rust std for GPU). Runtime JIT compilation takes too long (>180s timeout). The regular kernel.ptx (2.5MB) loads in ~1s. A run_std() API was prototyped but reverted due to this limitation. Confidence: 100%.

**Q: What about the examples/std/ directory?**
A: These (benchmark, cifar-train, gpt2-inference, etc.) use the nn-level API (KernelRegistry, nn::ops, nn::models) which is already a high-level Rust abstraction. Converting them to gpu::run() would be a downgrade — they're already "native Rust style" at a higher level. Confidence: 95%.

## Unexpected Discoveries
- kernel_std.ptx (4.5MB) is too large for practical runtime JIT. A future optimization could pre-compile kernel_std.ptx to cubin at build time.
- The `[HOST] FILE OPEN/WRITE/CLOSE` log messages appear even when using HostcallSession (they come from the hostcall listener's I/O thread, not from the on_print callback).
- The `branching_pipeline` kernel's behavior changed between runs — first run creates files, subsequent runs find existing files. This is correct behavior (the kernel checks if file exists before creating).

## Open Questions
- Should the gpu module support custom argument signatures beyond the three fixed patterns? This would enable converting more examples.
- Should kernel_std.ptx be pre-compiled to cubin at build time to avoid JIT overhead?
- Should the remaining hostcall examples (vector-math, tcp-echo, parallel-search) get new API variants or new wrapper kernels?

## Impact on Downstream Tasks
- hello-gpu, async-io, async-pipeline now depend only on `gpu-host` (no direct cudarc dependency)
- Build.rs files are now no-ops — per-example kernel compilation is no longer needed for these 3 examples
- The kernel/ subdirectories of these examples are now dead code (kernels are from embedded PTX instead)
