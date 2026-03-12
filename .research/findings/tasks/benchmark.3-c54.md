# benchmark.3: Register and occupancy profiling with cuobjdump/PTX parsing
**Cycle**: 54 | **Theme**: benchmark | **Kind**: experiment | **Status**: done

## Summary
Parsed PTX output for all kernel variants to extract virtual register counts and
local memory usage. Calculated theoretical occupancy on SM_86 (RTX 3060).
Key finding: async kernels use local memory (stack spilling) due to Embassy executor
state, but virtual register counts are moderate. The file I/O kernel
(hostcall_file_test) has the highest register pressure at ~888 virtual registers.

## Findings

### Q: What is the hardware register count for each kernel variant?

**Note**: These are PTX virtual register counts, NOT hardware register counts.
cuobjdump (not available in PATH) would give actual hardware register allocation.
ptxas maps virtual → physical and may reduce counts significantly via register
allocation optimization. Estimates below use: total_regs ≈ b32 + 2*b64.

| Kernel | b32 | b64 | pred | b16 | Est. HW regs | .local? |
|--------|-----|-----|------|-----|-------------|---------|
| vector_add | 9 | 11 | 2 | — | 31 | No |
| hostcall_print_hello | 18 | 37 | 10 | — | 92 | No |
| hostcall_print_multi | 35 | * | 16 | 18 | ~90 | Yes (SP) |
| error_propagation_test | 132 | 201 | 82 | 24 | 534 | No |
| hostcall_file_test | 296 | 296† | 247 | 40 | ~888 | Yes (SP) |
| hostcall_stdin_time_test | 103 | * | 58 | 10 | ~200 | Yes (SP) |
| multi_warp_sync_kernel | 8 | * | 2 | 21 | ~20 | Yes (SP) |
| multi_block_sync_kernel | 17 | * | 2 | 25 | ~30 | Yes (SP) |
| **async_hostcall_single** | 9 | 24 | 9 | — | 57 | Yes |
| **async_hostcall_two** | 10 | 36 | 13 | — | 82 | Yes |
| **futures_join** | 9 | 24 | 9 | — | 57 | Yes |
| **multi_block_async** | 11 | * | 8 | 22 | ~30 | Yes |
| **pipeline_kernel** | 9 | 24 | 9 | — | 57 | Yes |

*SP/SPL present but b64 count not separately extracted for some kernels.

†hostcall_file_test has very high counts because it includes all file I/O wrappers
(open, write, read, close) with their hostcall protocol code.

**Confidence**: medium (virtual regs, not hardware regs)

### Q: What is the actual occupancy achieved on hardware?
A: **Theoretical occupancy** on SM_86 (RTX 3060, 65536 regs/SM, 48 warps max):

| Kernel | Est. regs/thread | Max threads/SM | Warps | Occupancy |
|--------|-----------------|----------------|-------|-----------|
| vector_add | 31 | 1536 (capped) | 48 | **100%** |
| hostcall_print_hello | 92 | 712 | 22 | **46%** |
| async_hostcall_single | 57 | 1149 | 35 | **73%** |
| async_hostcall_two | 82 | 799 | 24 | **50%** |
| pipeline_kernel | 57 | 1149 | 35 | **73%** |
| error_propagation_test | 534 | 122 | 3 | **6%** |
| hostcall_file_test | 888 | 73 | 2 | **4%** |

**IMPORTANT**: These are rough estimates. ptxas typically reduces virtual register
counts by 2-3x via register allocation. Actual hardware counts require
`cuobjdump --function-reg-count` or Nsight Compute.

**Confidence**: low (virtual regs overestimate hardware regs significantly)

### Q: Is register spilling occurring to local memory?
A: **Yes, for all kernels with SP/SPL registers.** The presence of `.reg .b64 %SP`
and `.reg .b64 %SPL` indicates a stack pointer, which means the kernel uses local
memory for spilling. All async kernels, multi-warp kernels, and file I/O kernels
use local memory. Only the simplest kernels (vector_add, test kernels, basic
hostcall_print_hello) avoid spilling.

**Confidence**: high (SP/SPL is definitive evidence of stack usage)

### Q: How does async state machine size affect occupancy?
A: Comparing sync vs async for the same task (single hostcall print):
- **Sync**: hostcall_print_hello = 92 virtual regs → 46% occupancy
- **Async single**: async_hostcall_single = 57 virtual regs → 73% occupancy
- **Async two tasks**: async_hostcall_two = 82 virtual regs → 50% occupancy

Surprisingly, the async single-task kernel uses FEWER virtual registers than the
sync version. This is because the async version's control flow is structured
differently (state machine vs inline code). The two-task async kernel is comparable
to the sync version.

However, all async kernels use local memory (stack spilling), which adds latency
when accessing spilled values. The occupancy numbers above don't account for the
local memory bandwidth impact.

**Confidence**: medium (virtual regs may not reflect hardware reality)

## Unexpected Discoveries
- async_hostcall_single (57 regs) is lighter than sync hostcall_print_hello (92 regs)
- The file I/O kernel has extreme register pressure (888 virtual regs) due to
  including all file operation wrappers in one kernel
- All async kernels use stack spilling — this is expected for Embassy's executor
  which stores task state on the stack

## Open Questions
- What are the ACTUAL hardware register counts? Need cuobjdump.
- How much latency does local memory spilling add per access?
- Would `__launch_bounds__` help constrain register usage?

## Impact on Downstream Tasks
- benchmark.4 needs cuobjdump for accurate comparison with CUDA C++
- warp-coop should consider the spilling impact when designing shared executors
