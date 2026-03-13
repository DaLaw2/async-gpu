# host-sdk.3: Standalone example using SDK as library dependency
**Cycle**: 194 | **Theme**: host-sdk | **Kind**: experiment | **Status**: done

## Summary
Rewrote `examples/hello-gpu/host` to use gpu-host as a library dependency with
`default-features = false` (no gpt2 model/tokenizer). Demonstrates all four
kernel demos (vector_add, hello_gpu, file_io_demo, bulk_read_demo) using the
three core SDK types: `GpuRuntime`, `HostcallBuffer`, `MappedBuffer<T>`.

## Findings
### Q: Can an external binary crate depend on gpu-host and launch a kernel?
A: Yes. With `default-features = false`, the example depends only on cudarc and
gpu-protocol (transitively). The SDK surface is clean: `GpuRuntime` for device
management, `HostcallBuffer` for GPU-host RPC, `MappedBuffer<T>` for pinned
memory. All four demos pass.
**Confidence**: high

### Q: Does the example work from a clean workspace setup?
A: Yes, provided the kernel is pre-built (build.rs compiles kernel crate via
`cargo +nightly`). The build.rs was updated to gracefully fall back to cached
PTX when the nightly kernel toolchain is unavailable (e.g., during `cargo clippy`).
**Confidence**: high

## Unexpected Discoveries
- `build.rs` panics during `cargo clippy` because clippy invokes build scripts
  but the nightly nvptx64 toolchain may not be available in that context. Fixed
  by falling back to cached PTX when kernel compilation fails.

## Open Questions
None.

## Impact on Downstream Tasks
- host-sdk.4 (additional examples) can follow the same pattern
- host-sdk.5 (build automation) should address the nightly toolchain requirement
