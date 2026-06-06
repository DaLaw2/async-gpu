# dyn-perf synthesis

## Status: done — overhead measured, well within <3x target

Dynamic dispatch (`&dyn Trait`) on nvptx64 has near-zero per-call overhead.
PTX analysis shows identical instruction count per call (4 insns) for both
static and dynamic paths. The only cost: one vtable load (amortized) and
indirect vs direct jump.

## Key findings
1. **Per-call: 0 extra instructions** — param setup + call + retval read = 4 insns both paths
2. **One-time: +2 instructions** — extra param load + vtable fn ptr load (amortized over N calls)
3. **LLVM treats both identically** — same 4x unrolling, same register pressure (within +2 b64)
4. **Real cost is inlining prevention** — static can inline small methods; dyn cannot
5. **Overhead ratio: ~1.0x-1.15x** — well within the <3x success criteria

## Assessment
For compute-heavy trait methods (the typical GPU use case), dyn dispatch
is practically free. The overhead only matters for trivially small methods
that would benefit from inlining — and those should use generics anyway.

## Files
- Kernel: `crates/kernel/gpu-kernel-test/src/lib.rs` (test_gpu_dyn_perf_benchmark)
- Findings: `.research/findings/tasks/dyn-perf.1-c642.md`
