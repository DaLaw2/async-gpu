# cost-analysis.1: Register + Shared Memory Estimation from MIR/PTX

## Summary

Investigated three approaches for compile-time GPU resource estimation: MIR analysis, PTX
analysis, and ptxas -v parsing. The hybrid approach (Option D) is recommended: ptxas -v for
register/occupancy (most accurate, already in the build pipeline), supplemented by PTX
parsing for shared memory declarations, and MIR analysis for bank conflict detection.

## Findings

### Q1: MIR-level register estimation feasibility

**Feasibility: Low for register counts, useful for structural analysis.**

MIR locals do not correspond to physical registers. The mapping is:

- MIR locals → LLVM IR virtual registers → PTX virtual registers → ptxas physical registers
- Each transformation drastically changes register count: LLVM performs SSA construction,
  copy propagation, dead code elimination; ptxas performs register allocation with coalescing.

Evidence from the codebase:
- `gemm_f32_v3` has 64 named accumulator locals in Rust source → 568 PTX virtual regs
  → 111 ptxas physical regs (5.1:1 compression ratio)
- `vector_add` has ~5 meaningful locals → 22 PTX virtual → 12 ptxas physical (1.8:1)
- Compression ratio varies from 0.2x to 11.7x — no stable correlation

**Conclusion**: MIR local count cannot predict physical register usage. Too many
optimization passes (LLVM + ptxas) stand between MIR and hardware. A MIR-only register
estimator would have >3x error margin, making occupancy predictions unreliable.

However, MIR-level analysis IS useful for:
- Counting explicit shared memory allocations (before LLVM lowers them)
- Detecting access stride patterns for bank conflict analysis
- Identifying high-register-pressure code patterns (e.g., many live f32 accumulators)

### Q2: PTX-level register estimation

**PTX .reg declarations provide virtual register counts — NOT physical.**

`.reg` syntax in PTX: `.reg .b32 %r<301>` means up to 301 32-bit virtual registers.
These are SSA-form, not allocated. ptxas performs its own register allocation.

Key observations from the codebase:
- PTX virtual regs consistently overcount: 568 virtual → 111 physical for gemm_f32_v3
- Predicate registers (`.reg .pred`) and 64-bit registers (`.reg .b64`) complicate
  the mapping further (1 b64 reg = 2 physical 32-bit regs)
- No stable ratio between virtual and physical counts

**ptxas -v output format** (confirmed by running on all 4 PTX files):
```
ptxas info : Compiling entry function 'gemm_f32_v3' for 'sm_75'
ptxas info : Function properties for gemm_f32_v3
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info : Used 111 registers, used 1 barriers, 400 bytes cmem[0]
```

This provides per-kernel: register count, barrier count, stack frame, spill bytes,
constant memory, cumulative stack size. Easily parseable with regex:
`Used (\d+) registers.*?(\d+) bytes cmem\[0\]`

**ptxas -v is the only source of truth for physical register allocation.**

### Q3: cuobjdump approach

**cuobjdump --dump-resource-usage works and provides similar data, but requires cubins.**

Format (from kernel_std.cubin analysis):
```
Function gemm_f32_v3:
  REG:111 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:400 TEXTURE:0 SURFACE:0
Function std_pipeline_test:
  REG:255 STACK:520 SHARED:0 LOCAL:0 CONSTANT[2]:2316 CONSTANT[0]:352
```

Provides: REG, STACK, SHARED, LOCAL, CONSTANT[n], TEXTURE, SURFACE, SAMPLER per function.

**Compared to ptxas -v**: same register data, but also shows SHARED (static), LOCAL,
TEXTURE, SURFACE. However, requires pre-compiled cubins (.cubin files), which are only
built in `--prod` mode. ptxas -v can be run as a one-off analysis step without producing
a cubin (though it still does full compilation).

**Verdict**: cuobjdump is useful for post-build analysis, but ptxas -v is better for
integration into the build pipeline since it runs during cubin compilation anyway.

### Q4: Shared memory estimation

**All shared memory in this codebase is dynamically allocated.**

Evidence:
- Every PTX file declares: `.extern .shared .align 4 .b8 dynamic_smem[];`
- Zero static `.shared` declarations with fixed sizes
- Shared memory size is specified at kernel launch: `shared_mem_bytes: N` in Rust host code

This means:
- **PTX analysis**: Can detect THAT a kernel uses shared memory (via `cvta.shared` instructions),
  but NOT how much. The `dynamic_smem` array has no declared size.
- **Host-side analysis**: The `shared_mem_bytes` parameter in `KernelConfig` struct is the
  only source of actual shared memory size. Can be extracted by analyzing host Rust code.
- **MIR analysis**: Could detect shared memory intent at the kernel source level if we
  track calls to `get_dynamic_smem_ptr()` and subsequent index calculations.

For static shared memory (not used here but common in other CUDA code):
- PTX: `.shared .align 4 .b32 smem[256]` — size is directly available
- MIR: track `alloca` in shared address space (LLVM addrspace(3))

### Q5: Bank conflict detection

**Partially feasible at compile time — stride analysis can catch common patterns.**

Bank conflicts on sm_75: 32 banks, 4-byte stride. Thread i in warp accesses bank
`(addr/4) % 32`. Conflict when multiple threads hit same bank with different addresses.

Common patterns detectable at compile time:
1. **Column-major access to row-major array**: e.g., `smem[tid * STRIDE]` where
   `STRIDE % 32 == 0` → all threads hit same bank. Detectable from MIR index expressions.
2. **Padding detection**: `gemm_f32_v3` uses `A_STRIDE = BM + 4 = 132` (padding 4 to
   avoid bank conflicts with stride 128). This pattern IS visible in MIR/source.
3. **Sequential 4-byte access**: `smem[tid]` → no conflicts. Easy to verify.

What's NOT detectable:
- Data-dependent access patterns (e.g., hash table lookups into shared memory)
- Runtime-computed indices
- Indirect shared memory access through pointers

**Recommendation**: Implement stride analysis for the common patterns:
- Flag `smem[tid * K]` where K is a compile-time constant and K % 32 == 0
- Flag missing padding in shared memory tile declarations
- This catches the most impactful bank conflict patterns in tiled GEMM/convolution kernels

### Q6: Recommended approach — Hybrid (Option D)

| Resource        | Approach                     | Why                                    |
|-----------------|------------------------------|----------------------------------------|
| Registers       | ptxas -v (parsed)            | Only source of truth for physical regs |
| Occupancy       | ptxas -v + occupancy formula | Register count → occupancy table       |
| Shared memory   | Host analysis + PTX          | Dynamic smem size from launch config   |
| Bank conflicts  | MIR stride analysis          | Best access to high-level index exprs  |
| Spill detection | ptxas -v (spill bytes)       | Reports spill stores/loads directly    |

**Why not pure MIR (Option A)?** MIR local count has no stable relationship to physical
registers. Error margin would be >3x, making warnings useless or misleading.

**Why not pure PTX parsing (Option B)?** PTX virtual registers also don't predict physical
allocation. Close, but still off by 2-5x.

**Why not pure ptxas -v (Option C)?** It's the most accurate, but it misses structural
information. Bank conflicts and shared memory intent are better analyzed from MIR/source.

**Hybrid (Option D)**: Best of all worlds. ptxas -v is already run during `--prod` builds;
we just need to capture and parse its output. MIR analysis adds bank conflict detection
that ptxas can't provide.

### Q7: Integration point

**Two integration points needed:**

1. **ptxas -v parsing**: In `scripts/build-kernels.sh` during `--prod` builds.
   - Currently: `ptxas --gpu-name sm_75 -o kernel.cubin kernel.ptx`
   - Change to: `ptxas -v --gpu-name sm_75 -o kernel.cubin kernel.ptx 2>kernel_resources.txt`
   - Parse `kernel_resources.txt` to generate per-kernel resource report
   - Could also add a `--check` mode that runs ptxas -v without producing cubin (faster)

2. **MIR bank conflict analysis**: As a rustc MIR pass, following the `warp_cooperative.rs`
   pattern. Registered in `rustc_mir_transform/src/lib.rs` after `WarpCooperativeTransform`.
   - Only runs on nvptx64 target
   - Analyzes shared memory index expressions for stride patterns
   - Emits `span_warn` diagnostics for detected bank conflict patterns

For dev builds (no ptxas): could offer a "lightweight estimate" using PTX virtual reg
count as a rough upper bound, with explicit caveat that it's imprecise.

## Unexpected Discoveries

1. **Device function register inflation**: Many kernels in `kernel_compute.ptx` show 112
   registers even though they're simple (e.g., `bias_add` with 28 virtual regs → 112
   physical). This is because all device functions (`.func`) in the same compilation unit
   share the register file. The 112-reg floor is set by the most register-hungry device
   function. This is a known ptxas limitation — separate compilation or `__launch_bounds__`
   can mitigate it.

2. **std_pipeline_test hits 255 registers** (the hardware maximum for sm_75), giving 25%
   occupancy. This is a real example where a compile-time warning would have caught a
   performance issue — validating the epic's premise.

3. **matmul_io_compute uses 216 registers** (also 25% occupancy with 464 bytes stack).
   Another real case where compile-time occupancy warning would help.

4. **All shared memory is dynamic** in this codebase. Static shared memory analysis is
   therefore unnecessary for NOW, but the analysis framework should support both for
   generality.

5. **cuobjdump reports different register counts than ptxas -v** for some kernels in the
   combined cubin (e.g., bias_add shows REG:160 in cuobjdump vs 112 in ptxas -v). This is
   because the cubin was compiled with the full kernel_std.ptx (which includes IO kernels
   with higher register pressure), while ptxas -v was run on kernel_compute.ptx alone.

## Open Questions

1. **Performance of ptxas -v**: Running ptxas -v adds ~10-30 seconds to the build. Is this
   acceptable for dev builds? Should we only run it in `--prod` mode?

2. **Per-kernel vs per-module register counts**: The device function inflation issue means
   per-kernel register counts from ptxas depend on what OTHER functions are in the same PTX
   file. Should we split PTX files per-kernel for accurate analysis?

3. **Occupancy formula complexity**: The simple `regs/thread → occupancy` table assumes
   256-thread blocks and no shared memory. Real occupancy depends on block size and shared
   memory, which are runtime parameters. How much complexity should the formula handle?

4. **Bank conflict false positive rate**: Stride analysis may flag patterns that are
   intentional (e.g., broadcast reads). Need to calibrate sensitivity.

## Impact on Downstream

- **cost-analysis.2** (occupancy warnings): The ptxas -v approach is ready to implement.
  The format is simple to parse, and the occupancy formula is straightforward for sm_75.
  Two real kernels (std_pipeline_test=25%, matmul_io_compute=25%) are immediate test cases.

- **cost-analysis.3** (bank conflict detection): MIR-level stride analysis is feasible for
  common patterns. The gemm_f32_v3 padding pattern (A_STRIDE=132 to avoid 128-stride
  conflicts) provides a concrete test case.

- **Epic success criterion #4** ("at least one real kernel where compile-time warning
  catches a performance issue"): Already identified — std_pipeline_test with 255 registers
  and 25% occupancy is a clear candidate.
