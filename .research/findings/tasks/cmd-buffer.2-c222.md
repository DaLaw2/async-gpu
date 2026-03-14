# cmd-buffer.2: Host-side CommandBuffer + GPU-side polling kernel
**Cycle**: 222 | **Theme**: cmd-buffer | **Kind**: experiment | **Status**: done

## Summary
Implemented the full command buffer stack: host-side CommandBuffer API (alloc, submit, reset, Drop), GPU-side cmd module (cmd_poll, cmd_ack, cmd_yield), and multi_cmd_kernel that processes COMPUTE/PRINT/EXIT commands from a mapped-memory ring buffer. Verified on GPU.

## Findings

### Q: Does the ring buffer protocol work correctly with sys-scope atomics?
A: Yes. Host writes with Release semantics on write_idx, GPU reads with Acquire via sys_load_acquire_u64. The GPU correctly sees command payloads after observing the updated write_idx. Backpressure (GPU ack via read_idx) also works.
**Confidence**: high

### Q: Can the kernel process multiple command types in a single launch?
A: Yes. The multi_cmd_kernel successfully processes COMPUTE (doubles 4 values → [20,40,60,80]), PRINT (hostcall message), and EXIT in sequence. All within a single kernel launch, TDR-safe.
**Confidence**: high

## Files Modified
- `crates/gpu-protocol/src/lib.rs` — CMD_* constants (done in cmd-buffer.1)
- `crates/gpu-host/src/hostcall.rs` — CommandBuffer struct, Command enum
- `crates/gpu-runtime/src/lib.rs` — `cmd` module (cmd_poll, cmd_ack, cmd_yield)
- `crates/gpu-kernel/src/hostcall_kernels.rs` — multi_cmd_kernel
- `crates/gpu-host/src/tests_hostcall.rs` — run_multi_cmd_test
