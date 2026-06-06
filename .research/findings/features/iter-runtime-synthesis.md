# iter-runtime — Iterator runtime synthesis

## Status: 2/N tasks done

## Completed tasks

### iter-runtime.1 — Warp-parallel input partitioning
Warp-striped round-robin partitioning is correct and optimal for the
1-warp-per-logical-thread execution model. Four demo kernels verified
(map+collect, map+sum, enumerate+collect, zip+collect). Cross-warp fold
reduction via WARP_RESULT slots is correct.

### iter-runtime.2 — Chained iterator fusion
Chained `.map()` calls produce zero intermediate buffers. PTX inspection
confirms that GpuMap<GpuMap<...>> nesting is fully inlined by LLVM into
a single register-to-register expression per element. Verified patterns:
dual map, triple map+sum, map+filter+count. No MIR pass needed — Rust
monomorphization handles all practical fusion cases.

## Key evidence
- Dual map: two `add.rn.f32` instructions between one load and one store
- Triple map: three arithmetic instructions fused with fold accumulation
- Map+filter: `mul` + `setp` + branchless `selp` count, no intermediate buffer
- All ZST closures: iterator chain is 16 bytes regardless of depth

## Open questions
- Closure captures with non-ZST data: verified to fit in 256-byte scratch
  buffer but no stress test for large captured arrays
- Performance at scale (>10K elements): no benchmark yet, only correctness
