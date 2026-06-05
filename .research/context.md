## Current Focus
**Cycle 627 — gpu-generics epic in progress** (2026-06-06). gen-mono theme complete: Rust generics compile to type-specific PTX via standard monomorphization. gen-traits next.

## Recent Decisions
- 2026-06-06: generic_map<f32> → mul.rn.f32 + add.rn.f32, generic_map<i32> → mad.lo.s32 (LLVM fuses int FMA)
- 2026-06-06: Pattern: concrete `extern "gpu-kernel"` entry → `#[inline(always)]` generic body
- 2026-06-06: PTX monomorphization works via standard Rust monomorphization — no special GPU pass needed
- 2026-06-06: dyn Trait works on GPU via indirect calls (vtable)
- 2026-06-06: cooperative_indexed() uses HRTB `for<'coop>` to create fresh WarpIndex lifetime

## Tried & Rejected
- Round-robin DisjointSlice partitioning: can't return contiguous &mut [T]
- Lifetime-locked get_mut (WarpIndex<'scope> only): blocked cooperative_indexed cross-scope usage

## Active Constraints
- GTX 1660 (sm_75): 192 GB/s, 5 TFLOPS FP32, 48KB smem
- Max 2 concurrent heavy subagents

## Key Metrics
- gen-mono.2: 9 new kernel entry points (4 generic + 5 test), all produce correct results on GPU
- gen-mono.2: LLVM applies type-specific optimizations — int FMA fusion (mad.lo.s32), float separate ops
- gen-mono theme: generics work identically on nvptx64 as on CPU targets
- 787 tasks completed, 53 epics

## Next
1. gen-traits.1: Experiment — user-defined trait with where bounds in GPU kernel
2. gen-demo.1: Generic parallel_reduce<T: Reducible> for f32, i32, custom types
