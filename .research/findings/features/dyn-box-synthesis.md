# dyn-box — Box<dyn Trait> end-to-end on GPU

## Status: done (PTX compilation verified; runtime pending cubin rebuild)

## What works
- `Box<dyn Trait>` compiles to valid PTX via hostcall allocator
- Vtables in `.global` memory, indirect calls via `.callprototype`
- Structs with data fields (Parrot.word_count) read correctly from heap
- `Vec<Box<dyn Animal>>` compiles (heterogeneous collections)
- Runtime-chosen `Box<dyn>` via conditionals compiles
- Zero special GPU handling needed — standard Rust patterns work

## Key finding
Box<dyn Trait> = heap allocation + same dispatch as &dyn Trait.
The allocator is the only new ingredient; dispatch is identical.

## Files
- Kernel: `crates/kernel/gpu-kernel-test/src/lib.rs` (test_gpu_box_dyn_trait)
- Host test: `crates/test/gpu-test-harness/tests/gpu_tests.rs`
- Findings: `.research/findings/tasks/dyn-box.1-c642.md`

## Runtime
Cubin rebuild needed to include new kernel. PTX JIT fallback works but takes ~25 min.
