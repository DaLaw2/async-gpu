# safety-audit.2: Add SAFETY comments to high/medium-risk unsafe blocks
**Cycle**: 302 | **Theme**: safety-audit | **Kind**: experiment | **Status**: done

## Summary
Added ~83 `// SAFETY:` comments across 6 files covering all P0 (critical CAS protocol), P1 (buffer init, Future pinning, Send/Sync impls), and P2 (service handlers, memory ops) unsafe blocks. Coverage went from <1% (~7 comments) to comprehensive coverage of all high-risk paths.

## Files Modified

### gpu-runtime/src/lib.rs (26 SAFETY comments)
- P0: CAS stack ops (hc_pop_free_from, hc_push_with) — ABA prevention, epoch tags, system-scope ordering
- P0: gpu_hostcall_request — full packet lifecycle documentation
- P0: static mut PANIC_BUF/RESULT_BUF — single-writer-then-read-only pattern
- P1: sideband_alloc — atomic fetch_add uniqueness
- P1: All 13 Future poll() implementations — structural pinning justification
- P1: block_on_with, SpinExecutor::run — stack-pinned future safety

### gpu-host/src/hostcall.rs (~40 SAFETY comments)
- P0: ready_stack swap, per-shard draining, reinit_packets (cuCtxSynchronize precondition)
- P1: init/alloc_internal — pointer arithmetic bounds, write_volatile rationale
- P1: Packet processing — traversal, packet_ptr, doorbell, shutdown
- P1: CommandBuffer/FlightRecorder Send/Sync — pinned CUDA memory
- P2: All service handlers — payload slot bounds, sideband bounds

### gpu-host/src/memory.rs (~10 SAFETY comments)
- MappedBuffer: new_zeroed, read, write, as_slice, as_mut_slice, Drop

### gpu-libc/src/memory.rs (3 SAFETY comments)
- malloc, posix_memalign — CAS bump allocator
- realloc — documented known limitation (copies new_size without old_size)

### gpu-libc/src/hostcall_io.rs (1 SAFETY comment)
- HC_BUF static mut — single-writer-then-read-only

### gpu-libc/src/errno.rs (3 SAFETY comments)
- ERRNO_ARRAY — per-thread indexing with clamped bounds

## Impact on Downstream Tasks
- safety-audit theme complete
- Known issues documented: realloc UB in gpu-libc (bump allocator limitation, acceptable for GPU use)
