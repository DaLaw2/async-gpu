# dyn-probe synthesis

## Status: promising — compilation works, runtime test needed

`&dyn Trait` compiles to valid PTX on nvptx64 with zero issues. The LLVM
NVPTX backend emits real vtables in `.global` memory and indirect calls via
register-based `call` with `.callprototype`. The fat pointer layout
(data_ptr + vtable_ptr) matches standard Rust exactly. Vtable entries
use the standard `[drop, size, align, methods...]` layout.

## Key finding
The indirect call mechanism is: load function pointer from vtable via
`ld.b64 %reg, [vtable_ptr+offset]`, then `call (ret), %reg, (args), proto`.
Vtable pointers are converted from `.global` to generic address space via
`cvta.global.u64` before use. No LLVM warnings or unsupported-feature errors.

## Next steps
1. Runtime test: launch `test_gpu_dyn_trait` kernel and verify results
2. `Box<dyn Trait>` test: heap-allocated trait objects
3. `&dyn Fn()` closure test: indirect call through closure vtable
4. Performance measurement: indirect vs direct call overhead

## Risk
ptxas or GPU driver may reject indirect calls at JIT/execution time even
though PTX generation succeeds. This is the critical unknown for dyn-probe.2.
